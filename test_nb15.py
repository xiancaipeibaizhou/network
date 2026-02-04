import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from analys import FocalLoss

# 引入 Fast 模型
from network_fast_transformer import ROEN_Fast_Transformer 
from network_advanced import ROEN_Advanced
from ROEN_Final import ROEN_Final
# ==========================================
# 辅助函数：子网键生成
# ==========================================
def _subnet_key(ip):
    try:
        parts = str(ip).split(".")
        if len(parts) < 3:
            return (0, 0, 0)
        a = int(parts[0])
        b = int(parts[1])
        c = int(parts[2])
        return (a, b, c)
    except Exception:
        return (0, 0, 0)

# ==========================================
# 1. 稀疏图构建函数 (对齐 train_2020_fast)
# ==========================================
def create_graph_data_sparse(time_slice, global_ip_map, global_subnet_ids, label_encoder, time_window):
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
    
    # === [改进] 节点特征：从全1改为度特征 (Degree Features) ===
    ones = torch.ones(edge_index.size(1), dtype=torch.float)
    in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees.scatter_add_(0, edge_index[0], ones)
    in_degrees.scatter_add_(0, edge_index[1], ones)
    # [N, 2]
    x = torch.stack([torch.log1p(in_degrees), torch.log1p(out_degrees)], dim=-1).float()

    # === [改进] 子网ID ===
    subnet_id = None
    if global_subnet_ids is not None:
        subnet_id = torch.tensor(global_subnet_ids[unique_nodes], dtype=torch.long)
    
    drop_cols = ['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp', 'Src Port', 'Dst Port']
    edge_attr_vals = time_slice.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).values
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float)

    if edge_index.size(1) > 0:
        edge_labels = torch.tensor(labels, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels, n_id=n_id)
        if subnet_id is not None:
            data.subnet_id = subnet_id
        return data
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
from analys import evaluate_comprehensive, evaluate_with_threshold

# ==========================================
# 4. 主流程
# ==========================================
def temporal_split(data_list, test_size=0.2):
    """严禁Shuffle的时序切分"""
    split_idx = int(len(data_list) * (1 - test_size))
    return data_list[:split_idx], data_list[split_idx:]

def main():
    SEQ_LEN = 10       
    BATCH_SIZE = 32   
    NUM_EPOCHS = 150
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading NB15 Data (CICFlowMeter Format)...")
    # 假设使用 CICFlowMeter 提取的 NB15 特征文件
    data = pd.read_csv("data/CIC-NUSW-NB15/CICFlowMeter_out.csv") 
    
    # 清洗标签
    data['Label'] = data['Label'].astype(str).str.strip().replace('', np.nan)
    data.dropna(subset=['Label', 'Timestamp'], inplace=True)
    
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")

    # 归一化
    print("Normalizing..." )
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['Label', 'Timestamp', 'Src IP', 'Dst IP', 'Flow ID', 'Src Port', 'Dst Port' ]
    feat_cols = [c for c in numeric_cols if c not in  exclude]
    
    data[feat_cols] = data[feat_cols].fillna(0)
    for col in feat_cols:
        if data[col].max() > 100:
            data[col] = np.log1p(data[col].abs())
    
    scaler = StandardScaler()
    data[feat_cols] = scaler.fit_transform(data[feat_cols])

    # === 修改 1: 稳健的时间处理 ===
    print("Processing Time..." )
    # 移除硬编码的 format，让 pandas 自动尝试解析
    # data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce' )
    print("Parsing NB15 Timestamps (dayfirst=True)...")
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], dayfirst=True, errors='coerce')
    # 打印诊断信息
    before_drop = len (data)
    data.dropna(subset=['Timestamp'], inplace=True )
    after_drop = len (data)
    if after_drop < before_drop * 0.1 :
        print(f"CRITICAL WARNING: {before_drop - after_drop} rows dropped due to timestamp parsing failure!" )
    
    data = data.sort_values('Timestamp' )
    # 修正 FutureWarning
    data['time_idx'] = data['Timestamp'].dt.floor('min' )

    # === 修改 2: 稳健的 IP 处理 ===
    print("Building Global Maps..." )
    # 强制转为字符串，防止因类型问题导致 map 失败
    data['Src IP'] = data['Src IP'].astype(str).str .strip()
    data['Dst IP'] = data['Dst IP'].astype(str).str .strip()

    all_ips = pd.concat([data['Src IP'], data['Dst IP' ]]).unique()
    global_ip_map = {ip: i for i, ip in enumerate (all_ips)}
    NUM_GLOBAL_NODES = len (all_ips)
    
    print(f"Total Nodes: {NUM_GLOBAL_NODES}" )
    if NUM_GLOBAL_NODES < 10 :
        print("ERROR: Total nodes is still suspiciously low. Check input CSV content." )
        return # 提前终止，避免跑出无意义的结果
    
    subnet_to_idx = {}
    global_subnet_ids = np.zeros(NUM_GLOBAL_NODES, dtype=np.int64)
    for ip, gid in global_ip_map.items():
        key = _subnet_key(ip)
        sid = subnet_to_idx.get(key)
        if sid is None:
            sid = len(subnet_to_idx)
            subnet_to_idx[key] = sid
        global_subnet_ids[gid] = sid
    num_subnets = len(subnet_to_idx)
    print(f"Total Nodes: {NUM_GLOBAL_NODES}, Total Subnets: {num_subnets}")

    # 图构建
    print("Building Graphs...")
    grouped = data.groupby('time_idx', sort=True)
    graph_data_seq = []
    for name, group in tqdm(grouped):
        # 传入 global_subnet_ids
        g = create_graph_data_sparse(group, global_ip_map, global_subnet_ids, None, name)
        if g: graph_data_seq.append(g)

    # === [关键修改] 数据切分：Strict Temporal Split ===
    print("Splitting Data (Temporal)...")
    train_seqs, test_seqs = temporal_split(graph_data_seq, test_size=0.2)
    
    train_dataset = TemporalGraphDataset(train_seqs, SEQ_LEN)
    test_dataset = TemporalGraphDataset(test_seqs, SEQ_LEN)

    # 训练集 Shuffle 增强泛化，测试集顺序验证
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn)

    # 模型初始化
    if len(graph_data_seq) > 0:
        edge_dim = graph_data_seq[0].edge_attr.shape[1]
    else:
        edge_dim = 1
        
    print(f"Initializing Model (Node In: 2, Subnets: {num_subnets})...")
    # ROEN_Fast_Transformer ROEN_Advanced
    model = ROEN_Final(
        node_in=2, # 入度+出度
        edge_in=edge_dim, 
        hidden=128, 
        num_classes=len(class_names),
        num_subnets=num_subnets,
        seq_len=SEQ_LEN,
        heads=8
    ).to(DEVICE)
    
    # 计算类别权重 (可选)
    print("Calculating Class Weights...")
    label_counts = data['Label'].value_counts().sort_index()
    class_counts_tensor = torch.tensor(label_counts.values, dtype=torch.float).to(DEVICE)
    weights = 1.0 / (torch.sqrt(class_counts_tensor) + 1.0)
    weights = weights / weights.sum() * len(class_names)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = FocalLoss(alpha=0.25, gamma=2.0) # 使用 Focal Loss 更好
    # criterion = nn.CrossEntropyLoss(weight=weights) 
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
            
            # Forward
            pred_seq = model(batch, len(batch))
            last_pred = pred_seq[-1]
            last_label = batch[-1].edge_labels
            
            loss = criterion(last_pred, last_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        
        # 评估
        if (epoch + 1) % 50 == 0 or (epoch+1) == NUM_EPOCHS:
            acc, prec, rec, f1, far, auc, asa = evaluate_comprehensive(
                model, test_loader, DEVICE, class_names
            )

            print(
                f"Test (Epoch {epoch+1}) -> "
                f"ACC: {acc:.4f}, F1: {f1:.4f}, Rec: {rec:.4f}, "
                f"FAR: {far:.4f}, AUC: {auc:.4f}, ASA: {asa:.4f}"
            )
        elif (epoch + 1) % 150 == 0:    
            torch.save(model.state_dict(), f'models/nb15/fast_transformer_epoch_{epoch+1}.pth')

    # 阈值搜索
    print("Optimizing Threshold...")
    best_th, best_asa = 0.5, 0.0
    for th in [0.5, 0.4, 0.3, 0.2, 0.1]:
        _, _, far, asa = evaluate_with_threshold(model, test_loader, DEVICE, class_names, th)
        print(f"Thresh {th}: FAR {far:.4f}, ASA {asa:.4f}")
        if asa > best_asa and far < 0.03: best_th, best_asa = th, asa

    # 最终绘图
    OPTIMAL_THRESH = best_th
    # 假设 'Normal' 或 'Benign' 是正常流量，其余为攻击
    attack_indices = [i for i, n in enumerate(class_names) if 'normal' not in n.lower() and 'benign' not in n.lower()]
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = [g.to(DEVICE) for g in batch]
            logits = model(batch, len(batch))[-1]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            if attack_indices:
                mask = probs[:, attack_indices].sum(dim=1) > OPTIMAL_THRESH
                if mask.any():
                    sub_argmax = torch.argmax(probs[mask][:, attack_indices], dim=1)
                    preds[mask] = torch.tensor(attack_indices, device=DEVICE)[sub_argmax]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch[-1].edge_labels.cpu().numpy())

    # 绘图
    try:
        os.makedirs('png/nb15', exist_ok=True)
        labels_idx = list(range(len(class_names)))
        cm = confusion_matrix(all_labels, all_preds, labels=labels_idx)
        cm = np.asarray(cm, dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0
        save_path = f'png/nb15/FINAL_BEST_CM_Thresh{OPTIMAL_THRESH}.png'
        
        try:
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm_pct,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.title(f'Confusion Matrix (Threshold={OPTIMAL_THRESH})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
        except Exception:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm_pct, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=100.0)
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(len(class_names)),
                yticks=np.arange(len(class_names)),
                xticklabels=class_names,
                yticklabels=class_names,
                title=f'Confusion Matrix (Threshold={OPTIMAL_THRESH})',
                ylabel='True label',
                xlabel='Predicted label'
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            thresh = cm_pct.max() / 2.0 if cm_pct.size else 0.0
            for i in range(cm_pct.shape[0]):
                for j in range(cm_pct.shape[1]):
                    ax.text(
                        j, i, f'{cm_pct[i, j]:.1f}%',
                        ha='center', va='center',
                        color='white' if cm_pct[i, j] > thresh else 'black'
                    )
            fig.tight_layout()
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
        print(f"Saved CM to {save_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    os.makedirs('models/nb15', exist_ok=True)
    os.makedirs('png/nb15', exist_ok=True)
    main()
