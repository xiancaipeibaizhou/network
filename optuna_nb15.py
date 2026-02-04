import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import gc
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 导入模型和工具函数
from ROEN_Final import ROEN_Final
from test_nb15 import (
    create_graph_data_sparse, 
    TemporalGraphDataset, 
    temporal_collate_fn, 
    _subnet_key
)
from analys import evaluate_comprehensive

# ==========================================
# 1. 预加载数据（在 objective 外部，避免重复计算）
# ==========================================
def prepare_static_data():
    # 参考 test_nb15.py 的加载逻辑
    data = pd.read_csv("data/CIC-NUSW-NB15/CICFlowMeter_out.csv") 
    data['Label'] = data['Label'].astype(str).str.strip()
    
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])
    class_names = list(label_encoder.classes_)
    
    # 归一化和时间处理
    numeric_cols = data.select_dtypes(include=[np.number]).columns.difference(['Label', 'Timestamp'])
    scaler = StandardScaler()
    data[list(numeric_cols)] = scaler.fit_transform(data[list(numeric_cols)].fillna(0))
    print("Parsing NB15 Timestamps (dayfirst=True)...")
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], dayfirst=True, errors='coerce')
    data = data.sort_values('Timestamp').dropna(subset=['Timestamp'])
    data['time_idx'] = data['Timestamp'].dt.floor('min')

    # 构建全局映射
    all_ips = pd.concat([data['Src IP'], data['Dst IP']]).unique()
    global_ip_map = {ip: i for i, ip in enumerate(all_ips)}
    
    subnet_to_idx = {}
    global_subnet_ids = np.zeros(len(all_ips), dtype=np.int64)
    for ip, gid in global_ip_map.items():
        key = _subnet_key(ip)
        if key not in subnet_to_idx:
            subnet_to_idx[key] = len(subnet_to_idx)
        global_subnet_ids[gid] = subnet_to_idx[key]
    
    # 构建图序列
    graph_data_seq = []
    for _, group in data.groupby('time_idx'):
        g = create_graph_data_sparse(group, global_ip_map, global_subnet_ids, None, None)
        if g: graph_data_seq.append(g)
        
    return graph_data_seq, class_names, len(subnet_to_idx)

# 全局加载数据
GRAPH_DATA_SEQ, CLASS_NAMES, NUM_SUBNETS = prepare_static_data()

def objective(trial):
    # --- 2. 推荐的超参数搜索空间 ---
    # 核心参数
    SEQ_LEN = trial.suggest_int("SEQ_LEN", 8, 11)
    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [32])
    LR = trial.suggest_float("LR", 1e-4, 5e-3, log=True)
    
    # 模型架构参数
    hidden = trial.suggest_categorical("hidden", [128])
    # 约束：hidden 必须能被 heads 整除
    possible_heads = [h for h in [4, 8] if hidden % h == 0]
    heads = trial.suggest_categorical("heads", possible_heads)
    
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 3. 数据集划分（此时可以访问全局变量） ---
    split_idx = int(len(GRAPH_DATA_SEQ) * 0.8)
    train_seqs = GRAPH_DATA_SEQ[:split_idx]
    test_seqs = GRAPH_DATA_SEQ[split_idx:]
    
    train_loader = DataLoader(TemporalGraphDataset(train_seqs, SEQ_LEN), 
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(TemporalGraphDataset(test_seqs, SEQ_LEN), 
                             batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn)

    try:
        edge_dim = GRAPH_DATA_SEQ[0].edge_attr.shape[1]
        model = ROEN_Final(
            node_in=2,
            edge_in=edge_dim,
            hidden=hidden,
            num_classes=len(CLASS_NAMES),
            num_subnets=NUM_SUBNETS,
            seq_len=SEQ_LEN,
            heads=heads,
            dropout=dropout,
        ).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(15):
            for batch in train_loader:
                if not batch:
                    continue
                batch = [g.to(DEVICE) for g in batch]
                optimizer.zero_grad()
                pred_seq = model(batch, len(batch))
                loss = criterion(pred_seq[-1], batch[-1].edge_labels)
                loss.backward()
                optimizer.step()

        acc, prec, rec, f1, far, auc, asa = evaluate_comprehensive(
            model, test_loader, DEVICE, CLASS_NAMES
        )
        return f1
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("-inf")
    finally:
        if "model" in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, catch=(torch.cuda.OutOfMemoryError,))

    print("Best parameters:", study.best_params)
    print("Best F1 Score:", study.best_value)
