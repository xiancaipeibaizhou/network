import os
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from analys import FocalLoss, evaluate_comprehensive_with_binary_auc
from network_fast_transformer import ROEN_Fast_Transformer
from ROEN_Final import ROEN_Final

def _subnet_key(ip):
    try:
        parts = str(ip).split(".")
        if len(parts) < 3:
            return (0, 0, 0)
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return (0, 0, 0)


def create_graph_data_sparse(time_slice, global_ip_map, global_subnet_ids, label_encoder, time_window):
    time_slice = time_slice.copy()

    src_globals = time_slice["Src IP"].map(global_ip_map).fillna(-1).values.astype(np.int64)
    dst_globals = time_slice["Dst IP"].map(global_ip_map).fillna(-1).values.astype(np.int64)

    valid_mask = (src_globals >= 0) & (dst_globals >= 0)
    if not valid_mask.any():
        return None

    time_slice = time_slice.iloc[valid_mask]
    src_globals = src_globals[valid_mask]
    dst_globals = dst_globals[valid_mask]

    if label_encoder:
        try:
            labels = label_encoder.transform(time_slice["Label"].astype(str))
        except Exception:
            labels = np.zeros(len(time_slice), dtype=int)
    else:
        labels = time_slice["Label"].values.astype(int)

    all_nodes_in_slice = np.concatenate([src_globals, dst_globals])
    unique_nodes, inverse_indices = np.unique(all_nodes_in_slice, return_inverse=True)

    n_nodes = len(unique_nodes)
    src_local = inverse_indices[: len(src_globals)]
    dst_local = inverse_indices[len(src_globals) :]

    edge_index = torch.tensor(np.array([src_local, dst_local]), dtype=torch.long)
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

    drop_cols = ["Src IP", "Dst IP", "Flow ID", "Label", "Timestamp", "Src Port", "Dst Port", "time_idx"]
    edge_attr_vals = (
        time_slice.drop(columns=drop_cols, errors="ignore")
        .select_dtypes(include=[np.number])
        .values
    )
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float)

    if edge_index.size(1) > 0:
        edge_labels = torch.tensor(labels, dtype=torch.long)
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_labels=edge_labels,
            n_id=n_id,
        )
        if subnet_id is not None:
            data.subnet_id = subnet_id
        return data
    return None


class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data_seq, seq_len=8):
        self.graph_data_seq = [g for g in graph_data_seq if g is not None]
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.graph_data_seq) - self.seq_len + 1)

    def __getitem__(self, idx):
        return self.graph_data_seq[idx : idx + self.seq_len]


def temporal_collate_fn(batch):
    if len(batch) == 0:
        return []
    seq_len = len(batch[0])
    batched_seq = []
    for t in range(seq_len):
        graphs_at_t = [sample[t] for sample in batch]
        batched_seq.append(Batch.from_data_list(graphs_at_t))
    return batched_seq


def _get_normal_indices(class_names):
    keywords = ("non", "non-tor", "nonvpn", "normal", "benign")
    normal_indices = []
    for idx, name in enumerate(class_names):
        low = str(name).lower()
        if any(k in low for k in keywords):
            normal_indices.append(idx)
    if len(normal_indices) == 0 and len(class_names) > 0:
        normal_indices = [0]
    return normal_indices


def _window_has_attack(graph, normal_indices):
    if graph is None or not hasattr(graph, "edge_labels") or graph.edge_labels is None:
        return False
    if graph.edge_labels.numel() == 0:
        return False
    y = graph.edge_labels.detach().cpu().numpy().astype(int)
    return bool((~np.isin(y, normal_indices)).any())


def temporal_split(data_list, class_names, seq_len, test_size=0.2):
    data_list = [g for g in data_list if g is not None]
    n = len(data_list)
    if n <= 1:
        return data_list, []

    normal_indices = _get_normal_indices(class_names)
    window_has_attack = np.array(
        [_window_has_attack(g, normal_indices) for g in data_list], dtype=bool
    )
    overall_has_attack = bool(window_has_attack.any())

    min_split = max(1, int(seq_len))
    max_split = min(n - 1, n - int(seq_len))
    if min_split > max_split:
        min_split = 1
        max_split = n - 1

    default_split = int(n * (1 - test_size))
    default_split = max(min_split, min(max_split, default_split))

    trimmed = False
    if overall_has_attack and not window_has_attack[default_split:].any():
        last_attack_idx = int(np.where(window_has_attack)[0][-1])
        data_list = data_list[: last_attack_idx + 1]
        window_has_attack = window_has_attack[: last_attack_idx + 1]
        n = len(data_list)
        trimmed = True

        if n <= 1:
            return data_list, []

        min_split = max(1, int(seq_len))
        max_split = min(n - 1, n - int(seq_len))
        if min_split > max_split:
            min_split = 1
            max_split = n - 1

        default_split = int(n * (1 - test_size))
        default_split = max(min_split, min(max_split, default_split))

    prefix_any = np.maximum.accumulate(window_has_attack)
    suffix_any = np.maximum.accumulate(window_has_attack[::-1])[::-1]

    split_idx = default_split
    if overall_has_attack:
        candidates = []
        for i in range(min_split, max_split + 1):
            train_has_attack = bool(prefix_any[i - 1]) if i - 1 >= 0 else False
            test_has_attack = bool(suffix_any[i]) if i < n else False
            if train_has_attack and test_has_attack:
                candidates.append(i)
        if len(candidates) > 0:
            split_idx = min(candidates, key=lambda i: abs(i - default_split))
        elif not bool(suffix_any[split_idx]):
            split_idx = max(min_split, min(max_split, int(np.where(window_has_attack)[0][-1])))

    if trimmed:
        print(f"Temporal Split: trimmed to {n} windows so test can include attacks")
    print(
        f"Temporal Split: split_idx={split_idx}, train_windows={split_idx}, test_windows={n - split_idx}"
    )

    return data_list[:split_idx], data_list[split_idx:]


def _edge_label_counts(graphs, num_classes):
    ys = []
    for g in graphs:
        if g is None or not hasattr(g, "edge_labels") or g.edge_labels is None:
            continue
        if g.edge_labels.numel() == 0:
            continue
        ys.append(g.edge_labels.detach().cpu().numpy())
    if len(ys) == 0:
        return np.zeros(num_classes, dtype=np.int64)
    y = np.concatenate(ys, axis=0).astype(int)
    return np.bincount(y, minlength=num_classes).astype(np.int64)


def _print_split_labels(train_seqs, test_seqs, class_names):
    num_classes = len(class_names)
    train_counts = _edge_label_counts(train_seqs, num_classes)
    test_counts = _edge_label_counts(test_seqs, num_classes)

    train_present = [class_names[i] for i, c in enumerate(train_counts) if c > 0]
    test_present = [class_names[i] for i, c in enumerate(test_counts) if c > 0]

    train_str = ", ".join([f"{class_names[i]}:{int(train_counts[i])}" for i in range(num_classes) if train_counts[i] > 0])
    test_str = ", ".join([f"{class_names[i]}:{int(test_counts[i])}" for i in range(num_classes) if test_counts[i] > 0])

    print(f"Train Labels Present: {train_present}")
    print(f"Train Label Counts: {train_str if train_str else '[]'}")
    print(f"Test Labels Present: {test_present}")
    print(f"Test Label Counts: {test_str if test_str else '[]'}")


def main():
    SEQ_LEN = int(os.getenv("SEQ_LEN", "8"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "150"))
    LR = float(os.getenv("LR", "0.001"))
    CSV_PATH = os.getenv(
        "CSV_PATH", "data/2017/TrafficLabelling_/Tuesday-WorkingHours.pcap_ISCX.csv"
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(CSV_PATH, encoding="latin1")

    data.columns = data.columns.str.strip()
    data = data.rename(
        columns={
            "Source IP": "Src IP",
            "Destination IP": "Dst IP",
            "Source Port": "Src Port",
            "Destination Port": "Dst Port",
        }
    )

    data["Label"] = data["Label"].astype(str).str.strip().replace("", np.nan)
    data.dropna(subset=["Label", "Timestamp", "Src IP", "Dst IP"], inplace=True)

    label_encoder = LabelEncoder()
    data["Label"] = label_encoder.fit_transform(data["Label"])
    class_names = list(label_encoder.classes_)

    numeric_exclude = {
        "Label",
        "Timestamp",
        "Src IP",
        "Dst IP",
        "Flow ID",
        "Src Port",
        "Dst Port",
    }
    for col in data.columns:
        if col in numeric_exclude:
            continue
        if col == "time_idx":
            continue
        if col == "Label":
            continue
        if col == "Timestamp":
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c not in {"Label"}]

    data[feat_cols] = data[feat_cols].replace([np.inf, -np.inf], 0)
    data[feat_cols] = data[feat_cols].fillna(0)
    for col in feat_cols:
        try:
            if data[col].max() > 100:
                data[col] = np.log1p(data[col].abs())
        except Exception:
            continue

    scaler = StandardScaler()
    data[feat_cols] = scaler.fit_transform(data[feat_cols])

    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    before_drop = len(data)
    data.dropna(subset=["Timestamp"], inplace=True)
    after_drop = len(data)
    if after_drop < before_drop * 0.1:
        print(
            f"CRITICAL WARNING: {before_drop - after_drop} rows dropped due to timestamp parsing failure!"
        )

    data = data.sort_values("Timestamp")
    data["time_idx"] = data["Timestamp"].dt.floor("min")

    data["Src IP"] = data["Src IP"].astype(str).str.strip()
    data["Dst IP"] = data["Dst IP"].astype(str).str.strip()

    all_ips = pd.concat([data["Src IP"], data["Dst IP"]]).unique()
    global_ip_map = {ip: i for i, ip in enumerate(all_ips)}
    num_global_nodes = len(all_ips)
    print(f"Total Global Nodes: {num_global_nodes}")
    if num_global_nodes < 10:
        print("ERROR: Total nodes is still suspiciously low. Check input CSV content.")
        return

    subnet_to_idx = {}
    global_subnet_ids = np.zeros(num_global_nodes, dtype=np.int64)
    for ip, gid in global_ip_map.items():
        key = _subnet_key(ip)
        sid = subnet_to_idx.get(key)
        if sid is None:
            sid = len(subnet_to_idx)
            subnet_to_idx[key] = sid
        global_subnet_ids[gid] = sid
    num_subnets = len(subnet_to_idx)
    print(f"Total Global Nodes: {num_global_nodes}, Total Subnets: {num_subnets}")

    grouped_data = data.groupby("time_idx", sort=True)
    graph_data_seq = []
    for name, group in tqdm(grouped_data, desc="Building Graphs"):
        graph = create_graph_data_sparse(group, global_ip_map, global_subnet_ids, None, name)
        if graph is not None:
            graph_data_seq.append(graph)
    print(f"Total Graphs: {len(graph_data_seq)}")

    train_seqs, test_seqs = temporal_split(graph_data_seq, class_names, SEQ_LEN, test_size=0.2)
    _print_split_labels(train_seqs, test_seqs, class_names)

    train_dataset = TemporalGraphDataset(train_seqs, seq_len=SEQ_LEN)
    test_dataset = TemporalGraphDataset(test_seqs, seq_len=SEQ_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=temporal_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn
    )

    if len(graph_data_seq) > 0:
        edge_dim = graph_data_seq[0].edge_attr.shape[1]
    else:
        edge_dim = 1

    model = ROEN_Final(
        node_in=2,
        edge_in=edge_dim,
        hidden=128, # 隐藏层维度
        num_classes=len(class_names),
        num_subnets=num_subnets,
        seq_len=SEQ_LEN,
        heads=8
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    # 计算类别权重 (可选)
    print("Calculating Class Weights...")
    label_counts = data['Label'].value_counts().sort_index()
    class_counts_tensor = torch.tensor(label_counts.values, dtype=torch.float).to(DEVICE)
    weights = 1.0 / (torch.sqrt(class_counts_tensor) + 1.0)
    weights = weights / weights.sum() * len(class_names)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = FocalLoss(alpha=0.25, gamma=2.0)

    os.makedirs("models/2017", exist_ok=True)

    print("Start Training...")
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batched_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            if not batched_seq:
                continue
            batched_seq = [g.to(DEVICE) for g in batched_seq]

            optimizer.zero_grad()
            preds_seq = model(graphs=batched_seq, seq_len=len(batched_seq))
            last_pred = preds_seq[-1]
            last_label = batched_seq[-1].edge_labels

            loss = criterion(last_pred, last_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS:
            acc, prec, rec, f1, far, auc, asa, auc_bin = evaluate_comprehensive_with_binary_auc(
                model, test_loader, DEVICE, class_names
            )
            print(
                f"Test (Epoch {epoch+1}) -> "
                f"ACC: {acc:.4f}, F1: {f1:.4f}, Rec: {rec:.4f}, "
                f"FAR: {far:.4f}, AUC: {auc:.4f}, AUC_BIN: {auc_bin:.4f}, ASA: {asa:.4f}"
            )

        if (epoch + 1) % 50 == 0 or (epoch + 1) == NUM_EPOCHS:
            save_path = os.path.join("models/2017", f"model_T_W_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    end_time = time.time()
    print(f"运行时间为{end_time - start_time}")


if __name__ == "__main__":
    main()
