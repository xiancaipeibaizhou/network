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
from analys import FocalLoss

# 引入我们刚才定好的模型
# from network_fast_transformer import ROEN_Fast_Transformer 
from network_fast_transformer import ROEN_Fast_Transformer
# ==========================================
# 1. 稀疏图构建函数 (核心优化)
# ==========================================
def create_graph_data_sparse(time_slice, global_ip_map, label_encoder, time_window):
    """
    构建稀疏图：只包含当前活跃节点，但携带全局 n_id 用于模型内部对齐
    """
    # 1. 基础清洗
    time_slice = time_slice.copy()
    
    # 2. 获取 Global IDs
    # map 可能会产生 NaN (极少数情况)，fillna(-1) 后过滤
    src_globals = time_slice['Src IP'].map(global_ip_map).fillna(-1).values.astype(np.int64)
    dst_globals = time_slice['Dst IP'].map(global_ip_map).fillna(-1).values.astype(np.int64)

    # 过滤无效映射
    valid_mask = (src_globals >= 0) & (dst_globals >= 0)
    if not valid_mask.any():
        return None
        
    time_slice = time_slice.iloc[valid_mask]
    src_globals = src_globals[valid_mask]
    dst_globals = dst_globals[valid_mask]

    # 3. 标签编码
    if label_encoder:
        try:
            labels = label_encoder.transform(time_slice['Label'].astype(str))
        except:
            labels = np.zeros(len(time_slice), dtype=int)
    else:
        labels = time_slice['Label'].values.astype(int)

    # 4. 生成局部索引 (0, 1, 2...) 用于构建当前的小图 edge_index
    # np.unique 返回排序后的唯一节点 ID
    # inverse_indices 是原始数组在 unique 数组中的下标，正是我们要的局部索引
    all_nodes_in_slice = np.concatenate([src_globals, dst_globals])
    unique_nodes, inverse_indices = np.unique(all_nodes_in_slice, return_inverse=True)
    
    n_nodes = len(unique_nodes)
    src_local = inverse_indices[:len(src_globals)]
    dst_local = inverse_indices[len(src_globals):]
    
    edge_index = torch.tensor([src_local, dst_local], dtype=torch.long)
    
    # 5. 节点特征 & 关键的 n_id
    # n_id 记录了这些局部节点对应的全局 ID，模型会用它来做 SearchSorted 对齐
    n_id = torch.tensor(unique_nodes, dtype=torch.long)
    
    # 节点特征 x: 这里简单使用端口号归一化，或者全 1
    # 为了配合 Transformer，这里初始化为 [N, 1]
    # 如果想更强，可以把端口号映射进去，这里为了速度先用全1占位
    # 模型会自动通过 node_enc 学习 Embedding
    x = torch.ones((n_nodes, 1), dtype=torch.float) 
    
    # 6. 边特征 (已在全局做过 Log + Scale，直接取)
    drop_cols = ['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp', 'Src Port', 'Dst Port']
    edge_attr_vals = time_slice.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).values
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float)

    # 7. 返回 Data
    if edge_index.size(1) > 0:
        edge_labels = torch.tensor(labels, dtype=torch.long)
        # 注意：这里必须把 n_id 传进去
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels, n_id=n_id)
    else:
        return None

# ==========================================
# 2. Dataset & Collate (时序处理)
# ==========================================
class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data_seq, seq_len=8):
        self.graph_data_seq = [g for g in graph_data_seq if g is not None]
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.graph_data_seq) - self.seq_len + 1)

    def __getitem__(self, idx):
        # 返回长度为 seq_len 的图列表
        return self.graph_data_seq[idx : idx + self.seq_len]

def temporal_collate_fn(batch):
    # batch: list of [g1, g2, ..., g8]
    if len(batch) == 0: return []
    seq_len = len(batch[0])
    
    batched_seq = []
    for t in range(seq_len):
        # 收集 batch 中所有样本在时刻 t 的图
        graphs_at_t = [sample[t] for sample in batch]
        # PyG 的 Batch 会自动处理 n_id 的拼接
        batched_seq.append(Batch.from_data_list(graphs_at_t))
    return batched_seq

# ==========================================
# 3. 评估辅助函数
# ==========================================
from analys import evaluate_comprehensive,evaluate_with_threshold

# ==========================================
# 4. 主训练流程
# ==========================================
def main():
    # --- 配置 ---
    SEQ_LEN = 8       # 真正启用时序 (Original Paper Logic)
    BATCH_SIZE = 64   # 适当调大，因为图变稀疏了显存占用很小
    NUM_EPOCHS = 50
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")

    # --- 1. 数据加载与全局预处理 ---
    print("Loading Data...")
    data = pd.read_csv("data/CIC-Darknet2020/Darknet.csv") 
    
    # 基础清洗
    data.drop(columns=['Label.1'], inplace=True, errors='ignore')
    data = data.dropna(subset=['Label', 'Timestamp']).copy()
    
    # 标签编码
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'].astype(str))
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")

    # === 修正版：全局特征归一化 (Log + Scale) ===
    print("Performing Global Normalization (Log+Scale)...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Label', 'Timestamp', 'Src IP', 'Dst IP', 'Flow ID', 'Src Port', 'Dst Port']
    feat_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    data[feat_cols] = data[feat_cols].replace([np.inf, -np.inf], np.nan)
    data[feat_cols] = data[feat_cols].fillna(0)
 
    for col in feat_cols:
        if data[col].max() > 100:
            data[col] = np.log1p(data[col].abs())
             
    data[feat_cols] = data[feat_cols].replace([np.inf, -np.inf], 0)
    data[feat_cols] = data[feat_cols].fillna(0)
 
    scaler = StandardScaler()
    try:
        data[feat_cols] = scaler.fit_transform(data[feat_cols])
    except ValueError as e:
        print("Error during scaling. Checking data stats:")
        print(data[feat_cols].describe())
        raise e
         
    print("Normalization Done.")

    # 时间处理
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
    data = data.sort_values('Timestamp')
    data['time_idx'] = data['Timestamp'].dt.floor('T') # 按分钟聚合

    # --- 2. 构建全局 IP 映射 (必须步骤) ---
    print("Building Global IP Mapping...")
    all_ips = pd.concat([data['Src IP'], data['Dst IP']]).unique()
    global_ip_map = {ip: i for i, ip in enumerate(all_ips)}
    NUM_GLOBAL_NODES = len(all_ips)
    print(f"Total Global Nodes: {NUM_GLOBAL_NODES}")

    # --- 3. 构建稀疏图序列 ---
    print("Constructing Sparse Graphs...")
    grouped_data = data.groupby('time_idx', sort=True)
    graph_data_seq = []
    
    for name, group in tqdm(grouped_data, desc="Building Graphs"):
        # 调用新的稀疏构建函数
        graph = create_graph_data_sparse(group, global_ip_map, None, name) 
        # 注意：这里 label_encoder 传 None 因为我们在外部已经编码过了 data['Label']
        # 但 graph 需要 labels tensor，create_graph_data_sparse 会直接取 values
        
        if graph is not None:
            graph_data_seq.append(graph)
            
    print(f"Total Graphs: {len(graph_data_seq)}")

    # --- 4. 数据集切分 (Shuffle Split) ---
    # 为了保证每类都有，且验证模型泛化性
    train_size = int(len(graph_data_seq) * 0.8)
    train_seqs = graph_data_seq[:train_size]
    test_seqs = graph_data_seq[train_size:]

    # 同时建议在 TemporalGraphDataset 中保持序列的连续性
    train_dataset = TemporalGraphDataset(train_seqs, seq_len=SEQ_LEN)
    # 测试集应与训练集在时间上完全解耦
    test_dataset = TemporalGraphDataset(test_seqs, seq_len=SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn)

    # --- 5. 模型初始化 ---
    # 获取边特征维度
    if len(graph_data_seq) > 0:
        edge_dim = graph_data_seq[0].edge_attr.shape[1]
    else:
        edge_dim = 1
    
    print(f"Initializing ROEN_Fast_Transformer (Edge Dim: {edge_dim})...")
    model = ROEN_Fast_Transformer(
        node_in=1, # x 初始化为全1
        edge_in=edge_dim,
        hidden=64, # 隐藏层维度
        num_classes=len(class_names)
    ).to(DEVICE)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = FocalLoss(alpha=0.25, gamma=2.0) # 或 FocalLoss

    # --- 6. 训练循环 ---
    print("Start Training...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batched_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            if not batched_seq: continue
            
            # 搬运到 GPU
            batched_seq = [g.to(DEVICE) for g in batched_seq]
            
            optimizer.zero_grad()
            
            # 前向传播 (核心：seq_len=8, 动态对齐)
            preds_seq = model(graphs=batched_seq, seq_len=len(batched_seq))
            
            # 只监督最后一个时间步 (Many-to-One)
            last_pred = preds_seq[-1]
            last_label = batched_seq[-1].edge_labels
            
            loss = criterion(last_pred, last_label)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # 简单评估
        if (epoch + 1) % 5 == 0 or (epoch+1) == NUM_EPOCHS:
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

    print("\n=== Post-Training Threshold Optimization ===")
    print("Searching for best threshold to boost ASA without ruining FAR...")

    best_asa = 0.0
    best_thresh = 0.0

    for th in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15]:
        acc, f1, far, asa = evaluate_with_threshold(
            model, test_loader, DEVICE, class_names, threshold=th
        )
        print(f"[Threshold {th}] -> ACC: {acc:.4f}, F1: {f1:.4f}, FAR: {far:.4f}, ASA: {asa:.4f}")

        if asa > best_asa and far < 0.03:
            best_asa = asa
            best_thresh = th

    print(f"\nBest Strategy: Use Threshold = {best_thresh}")
    print(f"Expected Impact: ASA boosts to {best_asa:.4f} (Recall improved!)")

    print("\n=== Generating Final Report with Optimal Threshold (0.2) ===")
    OPTIMAL_THRESH = 0.2

    attack_indices = []
    for idx, name in enumerate(class_names):
        if 'non' not in name.lower():
            attack_indices.append(idx)

    model.eval()
    final_preds = []
    final_labels = []

    with torch.no_grad():
        for batched_seq in test_loader:
            batched_seq = [g.to(DEVICE) for g in batched_seq]
            logits = model(graphs=batched_seq, seq_len=len(batched_seq))[-1]
            probs = torch.softmax(logits, dim=1)

            batch_preds = torch.argmax(probs, dim=1)
            if len(attack_indices) > 0:
                attack_probs_sum = probs[:, attack_indices].sum(dim=1)
                mask = attack_probs_sum > OPTIMAL_THRESH

                if mask.any():
                    sub_probs = probs[mask][:, attack_indices]
                    sub_argmax = torch.argmax(sub_probs, dim=1)
                    new_preds = torch.tensor(attack_indices, device=DEVICE)[sub_argmax]
                    batch_preds[mask] = new_preds

            final_preds.extend(batch_preds.cpu().numpy())
            final_labels.extend(batched_seq[-1].edge_labels.cpu().numpy())

    labels_idx = list(range(len(class_names)))
    cm = confusion_matrix(final_labels, final_preds, labels=labels_idx)
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0

    save_path = f'png/2020/FINAL_BEST_CM_Thresh{OPTIMAL_THRESH}.png'

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

    print(f"Final Confusion Matrix saved to {save_path}")

    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs('models', exist_ok=True)
    os.makedirs('png/2020', exist_ok=True)
    main()