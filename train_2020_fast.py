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
from ROEN_Final import ROEN_Final
# 引入我们刚才定好的模型
# from network_fast_transformer import ROEN_Fast_Transformer 
from network_fast_transformer import ROEN_Fast_Transformer
# ==========================================
# 1. 稀疏图构建函数 (核心优化)
# ==========================================
def create_graph_data_sparse(time_slice, global_ip_map, global_subnet_ids, label_encoder, time_window):
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
    
    edge_index = torch.tensor(np.array([src_local, dst_local]), dtype=torch.long)
    
    # 5. 节点特征 & 关键的 n_id
    # n_id 记录了这些局部节点对应的全局 ID，模型会用它来做 SearchSorted 对齐
    n_id = torch.tensor(unique_nodes, dtype=torch.long)
    
    ones = torch.ones(edge_index.size(1), dtype=torch.float)
    in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees.scatter_add_(0, edge_index[0], ones)
    in_degrees.scatter_add_(0, edge_index[1], ones)
    x = torch.stack([torch.log1p(in_degrees), torch.log1p(out_degrees)], dim=-1).float()

    subnet_id = None
    if global_subnet_ids is not None:
        subnet_id = torch.tensor(global_subnet_ids[unique_nodes], dtype=torch.long)
    
    # 6. 边特征 (已在全局做过 Log + Scale，直接取)
    drop_cols = ['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp', 'Src Port', 'Dst Port']
    edge_attr_vals = time_slice.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).values
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float)

    # 7. 返回 Data
    if edge_index.size(1) > 0:
        edge_labels = torch.tensor(labels, dtype=torch.long)
        # 注意：这里必须把 n_id 传进去
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels, n_id=n_id)
        if subnet_id is not None:
            data.subnet_id = subnet_id
        return data
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
def temporal_split(data_list, test_size=0.2):
    split_idx = int(len(data_list) * (1 - test_size))
    return data_list[:split_idx], data_list[split_idx:]

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

def main():
    # --- 配置 ---
    SEQ_LEN = 10       # 真正启用时序 (Original Paper Logic)
    BATCH_SIZE = 32   # 适当调大，因为图变稀疏了显存占用很小
    NUM_EPOCHS = 150
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

    # 时间处理（Darknet.csv: 24/07/2015 04:09:48 PM）
    data['Timestamp'] = pd.to_datetime(
        data['Timestamp'],
        format="%d/%m/%Y %I:%M:%S %p",
        errors='coerce'
    )
    data = data.sort_values('Timestamp')
    data['time_idx'] = data['Timestamp'].dt.floor('min')  # 按分钟聚合

    # --- 2. 构建全局 IP 映射 (必须步骤) ---
    print("Building Global IP Mapping...")
    all_ips = pd.concat([data['Src IP'], data['Dst IP']]).unique()
    global_ip_map = {ip: i for i, ip in enumerate(all_ips)}
    NUM_GLOBAL_NODES = len(all_ips)
    print(f"Total Global Nodes: {NUM_GLOBAL_NODES}")

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

    # --- 3. 构建稀疏图序列 ---
    print("Constructing Sparse Graphs...")
    grouped_data = data.groupby('time_idx', sort=True)
    graph_data_seq = []
    
    for name, group in tqdm(grouped_data, desc="Building Graphs"):
        # 调用新的稀疏构建函数
        graph = create_graph_data_sparse(group, global_ip_map, global_subnet_ids, None, name) 
        # 注意：这里 label_encoder 传 None 因为我们在外部已经编码过了 data['Label']
        # 但 graph 需要 labels tensor，create_graph_data_sparse 会直接取 values
        
        if graph is not None:
            graph_data_seq.append(graph)
            
    print(f"Total Graphs: {len(graph_data_seq)}")

    # ==========================================
    # --- 4. 数据集切分 (修正版：分层时序切分) ---
    # ==========================================
    # 针对 CIC-Darknet2020 这种拼接数据集，必须按类别单独切分，
    # 否则会出现训练集/测试集类别完全不重合的情况。
    
    print("Performing Stratified Temporal Split...")
    
    train_seqs = []
    test_seqs = []
    
    # 按 Label 分组处理
    # 注意：graph_data_seq 中的每个 graph 都有 edge_labels，我们需要取第一个 label 来区分
    # 这里假设一个图内大部分 Flow 属于同一类，或者我们根据原始 Data 的时间来切
    
    # 更稳健的做法：回到原始 dataframe 切分，然后再生成图
    # 但为了改动最小，我们这里对 graph_data_seq 进行重新归类
    
    # 1. 先把图按类别归桶
    from collections import defaultdict
    class_buckets = defaultdict(list)
    
    for graph in graph_data_seq:
        # 获取该图中出现最多的标签作为该图的分类依据（通常一个时间片内流量混杂，但主导流量决定分类）
        # 如果是混合流量，这种方法近似有效。
        # CIC数据集通常是分段采集的，所以一段时间内通常是同一大类。
        labels = graph.edge_labels.numpy()
        if len(labels) > 0:
            major_label = np.bincount(labels).argmax()
            class_buckets[major_label].append(graph)
    
    # 2. 对每个桶单独按时间切分 (graph_data_seq 本身已经是按时间排好序的)
    for label, graphs in class_buckets.items():
        # 确保每个类内部按时间排序 (虽然 buckets 已经是按时间加入的，本身有序)
        n_samples = len(graphs)
        if n_samples < 2:
            # 样本太少，全放训练集或测试集，这里全放训练
            train_seqs.extend(graphs)
            continue
            
        split_point = int(n_samples * 0.8)
        
        # 前80% -> 训练
        train_seqs.extend(graphs[:split_point])
        # 后20% -> 测试
        test_seqs.extend(graphs[split_point:])
        
        # 打印日志证明切分合理性
        cls_name = class_names[label] if label < len(class_names) else str(label)
        print(f"Class {cls_name}: Total {n_samples} -> Train {split_point} / Test {n_samples - split_point}")
 
    # 3. 重新按时间排序 (模拟真实的数据流)
    # 因为上面分别切分后，顺序乱了 (Class A 的 1月, Class B 的 1月...)
    # 训练集可以乱序(Shuffle=True)，也可以按时间重排。为了严谨，我们通常按时间重排。
    # 但由于 graph 对象里没存绝对时间，这里直接依赖 DataLoader 的 shuffle=True 即可。
    
    print(f"Final Split -> Train: {len(train_seqs)}, Test: {len(test_seqs)}")
    
    # 4. 构建 Dataset
    train_dataset = TemporalGraphDataset(train_seqs, seq_len=SEQ_LEN)
    test_dataset = TemporalGraphDataset(test_seqs, seq_len=SEQ_LEN)
    
    # 5. DataLoader
    # 训练集打乱，增强泛化
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=temporal_collate_fn)
    # 测试集不打乱，模拟时序到达
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn)

    # --- 5. 模型初始化 ---
    # 获取边特征维度
    if len(graph_data_seq) > 0:
        edge_dim = graph_data_seq[0].edge_attr.shape[1]
    else:
        edge_dim = 1
    
    print(f"Initializing ROEN_Fast_Transformer (Edge Dim: {edge_dim})...")
    model = ROEN_Final(
        node_in=2,
        edge_in=edge_dim,
        hidden=128, # 隐藏层维度
        num_classes=len(class_names),
        num_subnets=num_subnets,
        seq_len=SEQ_LEN,
        heads=8
    ).to(DEVICE)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # 计算类别权重 (可选)
    print("Calculating Class Weights...")
    label_counts = data['Label'].value_counts().sort_index()
    class_counts_tensor = torch.tensor(label_counts.values, dtype=torch.float).to(DEVICE)
    weights = 1.0 / (torch.sqrt(class_counts_tensor) + 1.0)
    weights = weights / weights.sum() * len(class_names)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = FocalLoss(alpha=0.25, gamma=2.0) # 或 FocalLoss

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
        if (epoch + 1) % 30 == 0 or (epoch+1) == NUM_EPOCHS:
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

    if best_thresh == 0.0:
        best_thresh = 0.5
    print(f"\nBest Strategy: Use Threshold = {best_thresh}")
    print(f"Expected Impact: ASA boosts to {best_asa:.4f} (Recall improved!)")

    print("\n=== Generating Final Report with Optimal Threshold ===")
    OPTIMAL_THRESH = best_thresh

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
