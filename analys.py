from torch_geometric.data import Data
import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score, confusion_matrix

# ==========================================
# 3. 评估辅助函数
# ==========================================
def _get_normal_indices(class_names):
    keywords = ("non", "Non-Tor", "NonVPN", "normal", "benign", "non-tor", "nonvpn")
    normal_indices = []
    for idx, name in enumerate(class_names):
        low = str(name).lower()
        if any(k in low for k in keywords):
            normal_indices.append(idx)
    if len(normal_indices) == 0 and len(class_names) > 0:
        normal_indices = [0]
    return normal_indices


def _auc_ovr_macro(y_true, y_probs, present_labels):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs)
    if y_probs.ndim != 2 or y_probs.shape[0] != y_true.shape[0]:
        return 0.5

    aucs = []
    for c in np.asarray(present_labels).astype(int):
        if c < 0 or c >= y_probs.shape[1]:
            continue
        y_bin = (y_true == c).astype(int)
        if np.unique(y_bin).size < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y_bin, y_probs[:, c])))
        except Exception:
            continue
    if len(aucs) == 0:
        return 0.5
    return float(np.mean(aucs))

def evaluate_comprehensive(model, dataloader, device, class_names):
    """
    全指标评估：Acc, Prec, Rec, F1 + FAR, AUC, ASA
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    normal_indices = _get_normal_indices(class_names)

    with torch.no_grad():
        for batched_seq in dataloader:
            batched_seq = [g.to(device) for g in batched_seq]

            preds_seq = model(graphs=batched_seq, seq_len=len(batched_seq))
            logits = preds_seq[-1]

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(batched_seq[-1].edge_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    if len(all_labels) == 0:
        return 0, 0, 0, 0, 0, 0, 0

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    acc = (y_pred == y_true).mean()
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    is_true_normal = np.isin(y_true, normal_indices)
    is_pred_normal = np.isin(y_pred, normal_indices)
    fp = np.logical_and(is_true_normal, ~is_pred_normal).sum()
    tn = np.logical_and(is_true_normal, is_pred_normal).sum()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    is_true_attack = ~is_true_normal
    asa = (y_pred[is_true_attack] == y_true[is_true_attack]).mean() if is_true_attack.any() else 0.0

    try:
        present_labels = np.unique(y_true).astype(int)
        if y_probs.ndim != 2 or y_probs.shape[0] != y_true.shape[0] or present_labels.size < 2:
            auc = float("nan")
        elif y_probs.shape[1] == 2 or present_labels.size == 2:
            present_labels = np.sort(present_labels)
            pos_label = int(present_labels[-1])
            y_true_bin = (y_true == pos_label).astype(int)
            auc = roc_auc_score(y_true_bin, y_probs[:, pos_label])
        else:
            auc = _auc_ovr_macro(y_true, y_probs, present_labels)
    except Exception:
        auc = float("nan")

    return acc, prec, rec, f1, far, auc, asa


def evaluate_comprehensive_with_binary_auc(model, dataloader, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    normal_indices = _get_normal_indices(class_names)

    with torch.no_grad():
        for batched_seq in dataloader:
            batched_seq = [g.to(device) for g in batched_seq]

            preds_seq = model(graphs=batched_seq, seq_len=len(batched_seq))
            logits = preds_seq[-1]

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(batched_seq[-1].edge_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    if len(all_labels) == 0:
        return 0, 0, 0, 0, 0, 0.5, 0, 0.5

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    acc = (y_pred == y_true).mean()
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    is_true_normal = np.isin(y_true, normal_indices)
    is_pred_normal = np.isin(y_pred, normal_indices)
    fp = np.logical_and(is_true_normal, ~is_pred_normal).sum()
    tn = np.logical_and(is_true_normal, is_pred_normal).sum()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    is_true_attack = ~is_true_normal
    asa = (y_pred[is_true_attack] == y_true[is_true_attack]).mean() if is_true_attack.any() else 0.0

    try:
        present_labels = np.unique(y_true).astype(int)
        if y_probs.ndim != 2 or y_probs.shape[0] != y_true.shape[0] or present_labels.size < 2:
            auc = 0.5
        elif y_probs.shape[1] == 2 or present_labels.size == 2:
            present_labels = np.sort(present_labels)
            pos_label = int(present_labels[-1])
            y_true_bin = (y_true == pos_label).astype(int)
            auc = roc_auc_score(y_true_bin, y_probs[:, pos_label])
        else:
            auc = _auc_ovr_macro(y_true, y_probs, present_labels)
    except Exception:
        auc = 0.5

    try:
        y_true_bin = (~np.isin(y_true, normal_indices)).astype(int)
        normal_prob = y_probs[:, normal_indices].sum(axis=1) if len(normal_indices) > 0 else 0.0
        attack_prob = 1.0 - normal_prob
        if np.unique(y_true_bin).size < 2:
            auc_bin = 0.5
        else:
            auc_bin = roc_auc_score(y_true_bin, attack_prob)
    except Exception:
        auc_bin = 0.5

    return acc, prec, rec, f1, far, auc, asa, auc_bin

def evaluate_with_threshold(model, dataloader, device, class_names, threshold=0.4):
    """
    带阈值调整的评估函数：用于提升 ASA (召回率)
    """
    model.eval()
    all_labels = []
    all_preds = []

    normal_indices = _get_normal_indices(class_names)
    attack_indices = [i for i in range(len(class_names)) if i not in set(normal_indices)]

    print(
        f"Threshold Analysis: Normal IDs {normal_indices}, "
        f"Attack IDs {attack_indices}, Threshold={threshold}"
    )

    with torch.no_grad():
        for batched_seq in dataloader:
            batched_seq = [g.to(device) for g in batched_seq]

            preds_seq = model(graphs=batched_seq, seq_len=len(batched_seq))
            logits = preds_seq[-1]
            probs = torch.softmax(logits, dim=1)

            final_preds = torch.argmax(probs, dim=1)

            if len(attack_indices) > 0:
                attack_probs_sum = probs[:, attack_indices].sum(dim=1)
                mask = attack_probs_sum > threshold

                if mask.any():
                    sub_probs = probs[mask][:, attack_indices]
                    sub_argmax = torch.argmax(sub_probs, dim=1)
                    new_preds = torch.tensor(attack_indices, device=device)[sub_argmax]
                    final_preds[mask] = new_preds

            all_labels.extend(batched_seq[-1].edge_labels.cpu().numpy())
            all_preds.extend(final_preds.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    is_true_normal = np.isin(y_true, normal_indices)
    is_pred_normal = np.isin(y_pred, normal_indices)
    fp = np.logical_and(is_true_normal, ~is_pred_normal).sum()
    tn = np.logical_and(is_true_normal, is_pred_normal).sum()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    is_true_attack = ~is_true_normal
    asa = (y_pred[is_true_attack] == y_true[is_true_attack]).mean() if is_true_attack.any() else 0.0

    acc = (y_pred == y_true).mean() if len(y_true) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) if len(y_true) > 0 else 0.0

    return acc, f1, far, asa

# 在创建数据集和数据加载器之前，添加以下重平衡函数
def rebalance_graph_dataset(graph_data_seq, attack_label=0, normal_sample_ratio=0.1, random_seed1=42):
    """
    重平衡图数据集，保留所有攻击边，随机采样一部分正常边
    
    参数:
    graph_data_seq: 图数据序列
    attack_label: 攻击边的标签值，默认为0
    normal_sample_ratio: 正常边的采样比例，默认为0.1（即保留10%的正常边）
    random_seed: 随机种子，确保结果可复现
    
    返回:
    重平衡后的图数据序列
    """
    # 设置随机种子
    np.random.seed(random_seed1)
    torch.manual_seed(random_seed1)
    
    rebalanced_graphs = []
    
    for graph_data in graph_data_seq:
        if not hasattr(graph_data, 'edge_labels') or graph_data.edge_labels is None:
            continue
        
        # 分离攻击边和正常边
        attack_mask = (graph_data.edge_labels == attack_label)
        normal_mask = ~attack_mask
        
        # 获取攻击边的索引
        attack_indices = torch.nonzero(attack_mask).squeeze()
        if attack_indices.dim() == 0:  # 确保是一维张量
            attack_indices = attack_indices.unsqueeze(0)
        
        # 随机采样正常边 
        normal_indices = torch.nonzero(normal_mask).squeeze()
        if normal_indices.dim() > 0:  # 确保有正常边
            # 计算要采样的正常边数量
            num_normal_to_sample = max(1, int(len(normal_indices) * normal_sample_ratio))
            # 随机选择正常边索引
            selected_normal_indices = normal_indices[torch.randperm(len(normal_indices))[:num_normal_to_sample]]
            if selected_normal_indices.dim() == 0:  # 确保是一维张量
                selected_normal_indices = selected_normal_indices.unsqueeze(0)
            
            # 合并攻击边和采样的正常边索引
            combined_indices = torch.cat([attack_indices, selected_normal_indices])
            
            # 重新排序索引，确保顺序正确
            combined_indices, _ = torch.sort(combined_indices)
        else:
            # 如果没有正常边，只保留攻击边
            combined_indices = attack_indices
        
        # 提取选择的边
        selected_edge_index = graph_data.edge_index[:, combined_indices]
        selected_edge_attr = graph_data.edge_attr[combined_indices]
        selected_edge_labels = graph_data.edge_labels[combined_indices]
        
        # 找出在选中边中出现的所有节点
        unique_nodes = torch.unique(selected_edge_index)
        
        # 重新映射节点索引
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(unique_nodes)}
        
        # 重新映射边索引
        remapped_edge_index = torch.zeros_like(selected_edge_index)
        for i in range(selected_edge_index.size(1)):
            remapped_edge_index[0, i] = node_mapping[selected_edge_index[0, i].item()]
            remapped_edge_index[1, i] = node_mapping[selected_edge_index[1, i].item()]
        
        # 只保留相关的节点特征
        node_features = graph_data.x[unique_nodes]
        
        # 创建新的图数据
        rebalanced_graph = Data(
            x=node_features,
            edge_index=remapped_edge_index,
            edge_attr=selected_edge_attr,
            edge_labels=selected_edge_labels
        )
        
        rebalanced_graphs.append(rebalanced_graph)
    
    return rebalanced_graphs


# 修改后的evaluate
def evaluate1(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_labels = []
    all_preds = []
    all_probabilities = []  # 存储预测概率，用于AUC计算

    with torch.no_grad():  # Turn off gradient calculation
        for graph_data in dataloader:
            graph_data = graph_data.to(device)

            # Get model predictions
            edge_predictions = model(graphs=[graph_data], seq_len=1)
            edge_labels_batch = graph_data.edge_labels.to(device)
            
            # Get predicted class and probabilities for each edge
            edge_probs = torch.softmax(edge_predictions[0], dim=1)  # 转换为概率
            _, predicted = torch.max(edge_probs, dim=1)

            # Store true labels, predictions and probabilities
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
       
    # Calculate AUC value using probabilities instead of predictions
    try:
        probs = np.asarray(all_probabilities)
        labels = np.asarray(all_labels)
        present_labels = np.unique(labels)

        if probs.ndim != 2 or probs.shape[1] < 2 or present_labels.size < 2:
            auc = float('nan')
            auc1 = float('nan')
        elif probs.shape[1] == 2 or present_labels.size == 2:
            neg_label, pos_label = int(present_labels[0]), int(present_labels[1])
            y_true_pos = (labels == pos_label).astype(int)
            y_true_neg = (labels == neg_label).astype(int)
            auc = roc_auc_score(y_true_pos, probs[:, pos_label])
            auc1 = roc_auc_score(y_true_neg, probs[:, neg_label])
        else:
            label_list = present_labels.astype(int)
            probs_subset = probs[:, label_list]
            auc = roc_auc_score(labels, probs_subset, multi_class='ovo', labels=label_list)
            auc1 = auc
    except ValueError:
        auc = float('nan')  # If AUC cannot be calculated, return NaN
        auc1 = auc


    model.train()  # Revert model to training mode
 
    # Return evaluation metrics
    return accuracy, precision, recall, f1, auc,all_probabilities,all_labels,all_preds,auc1


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
