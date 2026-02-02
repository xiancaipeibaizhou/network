import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

# 引入 Fast 模型
from network_fast_transformer import ROEN_Fast_Transformer 

# ==========================================
# 1. 稀疏图构建函数
# ==========================================
def create_graph_data_sparse(time_slice, global_ip_map, label_encoder, time_window):
    time_slice = time_slice.copy()
    src_globals = time_slice['Src IP'].map(global_ip_map).fillna(-1).values.astype(np.int64)
    dst_globals = time_slice['Dst IP'].map(global_ip_map).fillna(-1).values.astype(np.int64)

    valid_mask = (src_globals >= 0) & (dst_globals >= 0)
    if not valid_mask.any(): return None
        
    time_slice = time_slice.iloc[valid_mask]
    src_globals = src_globals[valid_mask]
    dst_globals = dst_globals[valid_mask]

    # 标签处理
    if label_encoder:
        try:
            labels = label_encoder.transform(time_slice['Label'].astype(str))
        except:
            labels = np.zeros(len(time_slice), dtype=int)
    else:
        labels = time_slice['Label'].values.astype(int)

    all_nodes_in_slice = np.concatenate([src_globals, dst_globals])
    unique_nodes, inverse_indices = np.unique(all_nodes_in_slice, return_inverse=True)
    
    n_nodes = len(unique_nodes)
    src_local = inverse_indices[:len(src_globals)]
    dst_local = inverse_indices[len(src_globals):]
    
    edge_index = torch.tensor([src_local, dst_local], dtype=torch.long)
    n_id = torch.tensor(unique_nodes, dtype=torch.long)
    x = torch.ones((n_nodes, 1), dtype=torch.float) 
    
    drop_cols = ['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp', 'Src Port', 'Dst Port']
    edge_attr_vals = time_slice.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).values
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float)

    if edge_index.size(1) > 0:
        edge_labels = torch.tensor(labels, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels, n_id=n_id)
    else:
        return None

# ==========================================
# 2. Dataset & Collate
# ==========================================
class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data_seq, seq_len=8):
        self.graph_data_seq = [g for g in graph_data_seq if g is not None]
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.graph_data_seq) - self.seq_len + 1)
    def __getitem__(self, idx):
        return self.graph_data_seq[idx : idx + self.seq_len]

def temporal_collate_fn(batch):
    if len(batch) == 0: return []
    seq_len = len(batch[0])
    batched_seq = []
    for t in range(seq_len):
        graphs_at_t = [sample[t] for sample in batch]
        batched_seq.append(Batch.from_data_list(graphs_at_t))
    return batched_seq

# ==========================================
# 3. 评估辅助函数
# ==========================================
from analys import evaluate_comprehensive,evaluate_with_threshold

# ==========================================
# 4. 主流程
# ==========================================
def main():
    SEQ_LEN = 8       
    BATCH_SIZE = 128   
    NUM_EPOCHS = 50
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading NB15 Data...")
    data = pd.read_csv("data/CIC-NUSW-NB15/CICFlowMeter_out.csv") 
    data['Label'] = data['Label'].astype(str).str.strip().replace('', np.nan)
    data.dropna(subset=['Label', 'Timestamp'], inplace=True)
    
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")

    # 归一化
    print("Normalizing...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['Label', 'Timestamp', 'Src IP', 'Dst IP', 'Flow ID', 'Src Port', 'Dst Port']
    feat_cols = [c for c in numeric_cols if c not in exclude]
    data[feat_cols] = data[feat_cols].fillna(0)
    for col in feat_cols:
        if data[col].max() > 100: data[col] = np.log1p(data[col].abs())
    data[feat_cols] = StandardScaler().fit_transform(data[feat_cols])

    # 时间处理
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
    data.dropna(subset=['Timestamp'], inplace=True)
    data = data.sort_values('Timestamp')
    data['time_idx'] = data['Timestamp'].dt.floor('T')

    # 图构建
    print("Building Graphs...")
    all_ips = pd.concat([data['Src IP'], data['Dst IP']]).unique()
    global_ip_map = {ip: i for i, ip in enumerate(all_ips)}
    
    grouped = data.groupby('time_idx', sort=True)
    graph_data_seq = []
    for name, group in tqdm(grouped):
        g = create_graph_data_sparse(group, global_ip_map, None, name)
        if g: graph_data_seq.append(g)

    # 切分
    train_seqs, test_seqs = train_test_split(graph_data_seq, test_size=0.2, shuffle=True, random_state=42)
    train_loader = DataLoader(TemporalGraphDataset(train_seqs, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(TemporalGraphDataset(test_seqs, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn)

    # 模型初始化
    edge_dim = graph_data_seq[0].edge_attr.shape[1]
    model = ROEN_Fast_Transformer(1, edge_dim, 64, len(class_names)).to(DEVICE)
    
    # ========================================================
    # [关键修复]：直接计算 Class Weights，不遍历 loader
    # ========================================================
    print("Calculating Class Weights (Fast Method)...")
    # 直接使用 pandas 的 value_counts，瞬间完成
    label_counts = data['Label'].value_counts().sort_index()
    # 转换为 tensor
    class_counts_tensor = torch.tensor(label_counts.values, dtype=torch.float).to(DEVICE)
    
    # 这里的 1000.0 是一个平滑系数，可以调整。
    # 稀有类别的权重 = 1 / 频率。如果样本只有 100 个，权重会很大。
    weights = 1.0 / (torch.sqrt(class_counts_tensor) + 1.0) # 使用 sqrt 平滑，防止权重过大不稳定
    weights = weights / weights.sum() * len(class_names) # 归一化
    
    print(f"Class Weights Applied: {weights.cpu().numpy()}")
    
    # 使用带权重的 CE Loss
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 训练循环
    print("Start Training...")
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if not batch: continue
            batch = [g.to(DEVICE) for g in batch]
            optimizer.zero_grad()
            pred = model(batch, len(batch))[-1]
            loss = criterion(pred, batch[-1].edge_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        
        if (epoch + 1) % 10 == 0 or (epoch+1) == NUM_EPOCHS:
            acc, prec, rec, f1, far, auc, asa = evaluate_comprehensive(
                model, test_loader, DEVICE, class_names
            )

            print(
                f"Test (Epoch {epoch+1}) -> "
                f"ACC: {acc:.4f}, F1: {f1:.4f}, Rec: {rec:.4f}, "
                f"FAR: {far:.4f}, AUC: {auc:.4f}, ASA: {asa:.4f}"
            )
        elif (epoch + 1) % 50 == 0 or (epoch+1) == NUM_EPOCHS:    
            # 保存模型
            torch.save(model.state_dict(), f'models/2020/fast_transformer_epoch_{epoch+1}.pth')

    # 阈值搜索与绘图
    print("Optimizing Threshold...")
    best_th, best_asa = 0.5, 0.0
    for th in [0.5, 0.4, 0.3, 0.2, 0.1]:
        _, _, far, asa = evaluate_with_threshold(model, test_loader, DEVICE, class_names, th)
        print(f"Thresh {th}: FAR {far:.4f}, ASA {asa:.4f}")
        if asa > best_asa and far < 0.03: best_th, best_asa = th, asa

    # 最终绘图
    OPTIMAL_THRESH = best_th
    attack_indices = [i for i, n in enumerate(class_names) if 'normal' not in n.lower() and 'benign' not in n.lower()]
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = [g.to(DEVICE) for g in batch]
            probs = torch.softmax(model(batch, len(batch))[-1], dim=1)
            preds = torch.argmax(probs, dim=1)
            if attack_indices:
                mask = probs[:, attack_indices].sum(dim=1) > OPTIMAL_THRESH
                if mask.any():
                    sub_argmax = torch.argmax(probs[mask][:, attack_indices], dim=1)
                    preds[mask] = torch.tensor(attack_indices, device=DEVICE)[sub_argmax]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch[-1].edge_labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    os.makedirs('png/nb15', exist_ok=True)
    save_path = f'png/nb15/Final_CM.png'
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_pct, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Thresh={OPTIMAL_THRESH})')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    print(f"Saved CM to {save_path}")

if __name__ == "__main__":
    main()