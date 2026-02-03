import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix
from utils import create_ip_mapping_nb15, create_graph_data_nb15, GraphDataset
from network import G_TCN, MultiScaleG_TCN
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch_geometric.loader import DataLoader 
import torch
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
import time
from analysis_utils import plot_confusion_matrix, visualize_tsne

data_path = os.getenv("NB15_CSV_PATH", "data/CIC-NUSW-NB15/CICFlowMeter_out.csv")
data_nrows = int(os.getenv("DATA_NB15_NROWS", "0"))
read_kwargs = {}
if data_nrows > 0:
    read_kwargs["nrows"] = data_nrows

print("Loading data...")
data = pd.read_csv(data_path, **read_kwargs)
data.ffill(inplace=True)
data.bfill(inplace=True)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(data['Label'])
num_edge_classes = len(label_encoder.classes_)

# Parse the timestamps
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')

# Check for invalid timestamps
invalid_timestamps = data['Timestamp'].isna().sum()
if invalid_timestamps > 0:
    print(f"Warning: Found {invalid_timestamps} invalid timestamps. Dropping these rows.")
    data = data.dropna(subset=['Timestamp'])

# Fit a global scaler (sampled) then build graphs with global normalization
scaler_nrows = int(os.getenv("SCALER_NROWS", "50000"))
if len(data) > scaler_nrows:
    sample_df = data.sample(n=scaler_nrows, random_state=42).copy()
else:
    sample_df = data.copy()
sample_df.replace([np.inf, -np.inf], np.nan, inplace=True)
sample_df.ffill(inplace=True)
sample_df.bfill(inplace=True)
sample_df = sample_df.dropna(subset=["Label", "Timestamp"])
numeric_cols = sample_df.select_dtypes(include=["float64", "int64"]).columns.difference(
    ["Src IP", "Dst IP", "Flow ID", "Label", "Timestamp"]
)
if len(numeric_cols) == 0:
    raise RuntimeError("No numeric columns found to fit scaler.")
print("[Preprocessing] Using RobustScaler to preserve attack outliers...")
global_scaler = RobustScaler()
global_scaler.fit(sample_df[list(numeric_cols)])

# Create time windows by minute
data['Timestamp'] = data['Timestamp'].dt.floor('T')

# Group data by time window
grouped_data = data.groupby('Timestamp')

# Dynamically build graphs for each time window and store them in a list
graph_data_seq = []
for name, group in grouped_data:
    
    # Dynamically generate IP mapping for each time window
    ip_to_id = create_ip_mapping_nb15(group)
    
    # Build graph for the current time window and pass the time window name
    graph_data_seq.append(create_graph_data_nb15(group, ip_to_id, label_encoder, time_window=name, scaler=global_scaler))

graph_data_seq = [g for g in graph_data_seq if g is not None]
if len(graph_data_seq) == 0:
    raise ValueError("No valid graph data created. Please check the data preprocessing.")

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Split the dataset into training set and test set
print(f"Total graphs: {len(graph_data_seq)}")
all_labels_flat = []
for g in graph_data_seq:
    all_labels_flat.extend(g.edge_labels.detach().cpu().numpy())
label_counts = np.bincount(all_labels_flat, minlength=int(num_edge_classes))
graph_stratify_labels = []
for g in graph_data_seq:
    g_labels = g.edge_labels.detach().cpu().numpy()
    unique_labels = np.unique(g_labels)
    rarest_label = min(unique_labels, key=lambda l: label_counts[int(l)])
    graph_stratify_labels.append(int(rarest_label))
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
try:
    train_idx, test_idx = next(splitter.split(graph_data_seq, graph_stratify_labels))
    print("成功执行基于稀有类别的分层划分！")
except ValueError as e:
    print(f"[Warning] 样本极度不平衡，无法严格分层: {e}")
    print("回退到随机划分...")
    indices = np.arange(len(graph_data_seq))
    np.random.shuffle(indices)
    split_point = int(0.8 * len(graph_data_seq))
    train_idx, test_idx = indices[:split_point], indices[split_point:]
train_data_seq = [graph_data_seq[i] for i in train_idx]
test_data_seq = [graph_data_seq[i] for i in test_idx]
print(f"Training samples: {len(train_data_seq)}")
print(f"Test samples: {len(test_data_seq)}")

# Create DataLoader for training and test sets
train_dataset = GraphDataset(train_data_seq, device=device)
test_dataset = GraphDataset(test_data_seq, device=device)
batch_size = int(os.getenv("BATCH_SIZE", "248"))
if batch_size < 1:
    batch_size = 248
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

train_edge_labels = []
for g in train_data_seq:
    if g is None:
        continue
    if hasattr(g, "edge_labels") and g.edge_labels is not None:
        train_edge_labels.append(g.edge_labels.detach().cpu())
if train_edge_labels:
    train_edge_labels = torch.cat(train_edge_labels, dim=0).long()
    counts = torch.bincount(train_edge_labels, minlength=int(num_edge_classes)).float()
    total = float(counts.sum().item())
    denom = float(num_edge_classes) * counts
    class_weights = torch.zeros(int(num_edge_classes), dtype=torch.float)
    mask = counts > 0
    class_weights[mask] = total / denom[mask]
else:
    class_weights = torch.ones(int(num_edge_classes), dtype=torch.float)

def evaluate1(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_labels = []
    all_preds = []
    all_probabilities = []

    with torch.no_grad():  # Turn off gradient calculation
        for graph_data in dataloader:
            graph_data = graph_data.to(device)

            # Get model predictions
            edge_predictions = model(graphs=[graph_data], seq_len=1)
            edge_labels_batch = graph_data.edge_labels.to(device)
            
            edge_probs = torch.softmax(edge_predictions[0], dim=1)
            _, predicted = torch.max(edge_probs, dim=1)

            # Store true labels, predictions, and probabilities
            all_labels.extend(edge_labels_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probabilities.extend(edge_probs.cpu().numpy())
    
    # Calculate accuracy
    accuracy = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)

    # Calculate precision
    precision = precision_score(all_labels, all_preds, average='weighted')

    # Calculate recall
    recall = recall_score(all_labels, all_preds, average='weighted')

    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    y_true = np.asarray(all_labels)
    y_prob = np.asarray(all_probabilities)

    try:
        present = np.unique(y_true)
        if present.size < 2:
            auc = float('nan')
            auc1 = auc
        elif present.size == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            auc1 = roc_auc_score(y_true, y_prob[:, 0])
        else:
            if y_prob.ndim != 2:
                raise ValueError("y_prob must be 2D for multiclass AUC")
            if y_prob.shape[1] != present.size:
                if np.any(present < 0) or np.any(present >= y_prob.shape[1]):
                    raise ValueError("y_true labels out of range for y_prob")
                y_prob = y_prob[:, present]
                row_sums = y_prob.sum(axis=1, keepdims=True)
                if np.any(row_sums == 0):
                    raise ValueError("y_prob rows sum to zero after slicing")
                y_prob = y_prob / row_sums
                remap = {int(old): int(new) for new, old in enumerate(present)}
                y_true = np.vectorize(lambda x: remap[int(x)])(y_true)
            auc = roc_auc_score(y_true, y_prob, multi_class='ovo')
            auc1 = auc
    except ValueError:
        auc = float('nan')
        auc1 = auc
        
    model.train()  # Revert model to training mode
 
    return accuracy, precision, recall, f1, auc, all_probabilities, all_labels, all_preds, auc1


def train(model, train_dataloader, test_dataloader, optimizer, criterion, num_epochs, eval_interval, save_dir, plot_dir=None, class_names=None, tsne_samples=5000):
    model.train()  # Set the model to training mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()  
    hpo_mode = os.getenv("HPO_MODE", "0").strip().lower() in {"1", "true", "yes", "y"}
    loss_print_interval = int(os.getenv("LOSS_PRINT_INTERVAL", "1"))
    if loss_print_interval < 1:
        loss_print_interval = 1
    last_objective = None
    if not hpo_mode:
        os.makedirs(save_dir, exist_ok=True)
    for epoch in range(num_epochs):
        total_loss = 0

        # Training phase
        for graph_data in train_dataloader:  # Iterate through training data
            graph_data = graph_data.to(device)

            optimizer.zero_grad()

            # Get predictions
            edge_predictions = model(graphs=[graph_data], seq_len=1)

            # Get edge labels for current batch
            edge_labels_batch = graph_data.edge_labels.to(device)

            # Calculate loss
            loss = criterion(edge_predictions[0], edge_labels_batch)

            # Backpropagation
            loss.backward()

            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            # Update model parameters
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for current epoch
        if loss_print_interval == 1 or (epoch + 1) % loss_print_interval == 0 or (epoch + 1) == num_epochs:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader):.4f}')
        
        # Evaluate every eval_interval epochs
        if (epoch + 1) % eval_interval == 0:
            print(f"Evaluating at epoch {epoch+1}/{num_epochs}...")
            accuracy, precision, recall, f1, auc, all_probabilities, all_labels, all_preds, auc1 = evaluate1(model, test_dataloader)
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(int(num_edge_classes))))
            row_sums = cm.sum(axis=1)
            diag = np.diag(cm).astype(float)
            per_class_recall = np.divide(diag, row_sums, out=np.zeros_like(diag), where=row_sums != 0)
            objective = float(per_class_recall.sum())
            last_objective = objective
            print(f'Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}, AUC1: {auc1:.4f}')
            print(f"判断正确个数{(np.array(all_preds) == np.array(all_labels)).sum()}")
            print(f"总数{len(all_labels)}")
            names = class_names if class_names else [str(i) for i in range(len(per_class_recall))]
            print(f"RECALL_SUM={objective:.6f}")
            for i, (name, val) in enumerate(zip(names, per_class_recall.tolist())):
                print(f"RECALL[{i}]={float(val):.6f} name={name}")
            if hpo_mode:
                print(f"OBJECTIVE={objective:.6f}")
            else:
                save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), save_path)
                print(f'Model saved to {save_path}')
            if (not hpo_mode) and plot_dir and class_names and (epoch + 1) == num_epochs:
                os.makedirs(plot_dir, exist_ok=True)
                plot_confusion_matrix(
                    y_true=all_labels,
                    y_pred=all_preds,
                    class_names=class_names,
                    save_path=os.path.join(plot_dir, "confusion_matrix.png"),
                )
                visualize_tsne(
                    model=model,
                    dataloader=test_dataloader,
                    device=device,
                    num_samples=tsne_samples,
                    save_path=os.path.join(plot_dir, "tsne_features.png"),
                )
    if last_objective is None and hpo_mode:
        accuracy, precision, recall, f1, auc, all_probabilities, all_labels, all_preds, auc1 = evaluate1(model, test_dataloader)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(int(num_edge_classes))))
        row_sums = cm.sum(axis=1)
        diag = np.diag(cm).astype(float)
        per_class_recall = np.divide(diag, row_sums, out=np.zeros_like(diag), where=row_sums != 0)
        last_objective = float(per_class_recall.sum())
        names = class_names if class_names else [str(i) for i in range(len(per_class_recall))]
        print(f"RECALL_SUM={last_objective:.6f}")
        for i, (name, val) in enumerate(zip(names, per_class_recall.tolist())):
            print(f"RECALL[{i}]={float(val):.6f} name={name}")
        print(f"OBJECTIVE={last_objective:.6f}")
    end_time = time.time()  
    print(f"运行时间为{end_time-start_time}")            
    return last_objective

# Start training
print("\n" + "="*50)
print("Starting training...")
print("="*50)

# Initialize and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
edge_in_channels = train_data_seq[0].edge_attr.shape[1]
model_name = os.getenv("MODEL", "G_TCN").strip()
heads_env = os.getenv("HEADS")
if heads_env is not None and heads_env.strip() != "":
    heads = int(heads_env)
else:
    heads = 4 if model_name.lower() in {"multiscaleg_tcn", "multiscale", "ms"} else 2

if model_name.lower() in {"multiscaleg_tcn", "multiscale", "ms"}:
    kernel_sizes_raw = os.getenv("TCN_KERNEL_SIZES", "3,5,7")
    tcn_kernel_sizes = tuple(int(x.strip()) for x in kernel_sizes_raw.split(",") if x.strip() != "")
    mlp_hidden_channels = int(os.getenv("MLP_HIDDEN_CHANNELS", "256"))
    hidden_channels_node = int(os.getenv("HIDDEN_CHANNELS_NODE", "256"))
    hidden_channels_edge = hidden_channels_node
    model = MultiScaleG_TCN(
        node_in_channels=1,
        edge_in_channels=edge_in_channels,
        hidden_channels_node=hidden_channels_node,
        hidden_channels_edge=hidden_channels_edge,
        mlp_hidden_channels=mlp_hidden_channels,
        num_edge_classes=num_edge_classes,
        heads=heads,
        tcn_kernel_sizes=tcn_kernel_sizes,
    ).to(device)
else:
    tcn_kernel_size = int(os.getenv("TCN_KERNEL_SIZE", "3"))
    if tcn_kernel_size < 2:
        tcn_kernel_size = 7
    hidden_channels_node = int(os.getenv("HIDDEN_CHANNELS_NODE", "128"))
    hidden_channels_edge = hidden_channels_node
    mlp_hidden_channels = int(os.getenv("MLP_HIDDEN_CHANNELS", str(hidden_channels_node)))
    model = G_TCN(
        node_in_channels=1,
        edge_in_channels=edge_in_channels,
        hidden_channels_node=hidden_channels_node,
        hidden_channels_edge=hidden_channels_edge,
        mlp_hidden_channels=mlp_hidden_channels,
        num_edge_classes=num_edge_classes,
        heads=heads,
        tcn_kernel_size=tcn_kernel_size,
    ).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
print("Class Weights:", class_weights)
lr = float(os.getenv("LR", "0.001"))
optimizer = optim.Adam(model.parameters(), lr=lr)

num_epochs = int(os.getenv("NUM_EPOCHS", "150"))
eval_interval = int(os.getenv("EVAL_INTERVAL", str(num_epochs)))
tsne_samples = int(os.getenv("TSNE_SAMPLES", "5000"))
train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    criterion,
    num_epochs,
    eval_interval,
    'models/nb15',
    plot_dir='results_nb15_final',
    class_names=list(label_encoder.classes_),
    tsne_samples=tsne_samples,
)
 
# MODEL=multiscale HEADS=4 TCN_KERNEL_SIZES=3,7,11 BATCH_SIZE=128 MLP_HIDDEN_CHANNELS=256 NUM_EPOCHS=150 python train_nb15.py
