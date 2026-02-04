import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_geometric.utils import softmax, scatter
import math


# ==========================================
# 1. 基础组件：Edge-Augmented Attention (Edge -> Node)
# ==========================================
class EdgeAugmentedAttention(MessagePassing):
    """
    利用边特征增强节点特征的注意力层
    """

    def __init__(self, in_dim, out_dim, edge_dim, heads=4, dropout=0.1):
        super().__init__(node_dim=0, aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.dropout = dropout

        assert out_dim % heads == 0, "out_dim must be divisible by heads"

        self.WQ = nn.Linear(in_dim, out_dim, bias=False)
        self.WK = nn.Linear(in_dim, out_dim, bias=False)
        self.WV = nn.Linear(in_dim, out_dim, bias=False)
        self.WE = nn.Linear(edge_dim, out_dim, bias=False)  # 边特征投影

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.bn = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WE.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, edge_index, edge_attr):
        residual = x
        q = self.WQ(x).view(-1, self.heads, self.head_dim)
        k = self.WK(x).view(-1, self.heads, self.head_dim)
        v = self.WV(x).view(-1, self.heads, self.head_dim)
        e_emb = self.WE(edge_attr).view(-1, self.heads, self.head_dim)

        out = self.propagate(edge_index, q=q, k=k, v=v, e_emb=e_emb, size=None)
        out = out.view(-1, self.out_dim)
        out = self.out_proj(out)
        out = self.bn(out + residual)
        return self.act(out)

    def message(self, q_i, k_j, v_j, e_emb, index):
        score = (q_i * (k_j + e_emb)).sum(dim=-1) / (self.head_dim ** 0.5)
        alpha = softmax(score, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * (v_j + e_emb)


# ==========================================
# 2. 基础组件：Edge Updater (Node -> Edge) [关键改进]
# ==========================================
class EdgeUpdaterModule(nn.Module):
    """
    利用更新后的节点特征，反向刷新边特征
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.1):
        super().__init__()
        input_dim = node_dim * 2 + edge_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.res_proj = nn.Linear(edge_dim, hidden_dim) if edge_dim != hidden_dim else None

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x_src, x_dst = x[src], x[dst]
        cat_feat = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        new_edge = self.mlp(cat_feat)

        if self.res_proj is not None:
            edge_attr = self.res_proj(edge_attr)
        return new_edge + edge_attr


# ==========================================
# 3. 基础组件：Subnet Interaction (子网增强)
# ==========================================
class SubnetInteractionBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.subnet_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, subnet_id):
        if subnet_id is None: return x
        # 聚合：计算子网中心
        subnet_features = scatter(x, subnet_id, dim=0, reduce='mean')
        # 变换：提取子网模式
        subnet_features = self.subnet_mlp(subnet_features)
        # 广播：加回节点
        # 检查 subnet_id 是否越界（如果是 batch 拼接的需要注意）
        if subnet_features.size(0) < subnet_id.max() + 1:
            # 安全回退：如果 global subnet id 没对齐，通常不需要处理或直接返回
            return x
        out = x + subnet_features[subnet_id]
        return self.norm(out)


# ==========================================
# 4. 基础组件：Linear Temporal Attention (全局时序)
# ==========================================
class LinearTemporalAttention(nn.Module):
    def __init__(self, feature_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = feature_dim // heads
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, x):
        # x: [Batch, Feature, Time]
        B, C, T = x.shape
        x = x.permute(0, 2, 1)  # [B, T, C]

        q = self.elu(self.q_proj(x).view(B, T, self.heads, self.head_dim)) + 1.0
        k = self.elu(self.k_proj(x).view(B, T, self.heads, self.head_dim)) + 1.0
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim)

        kv = torch.einsum('bthd,bthe->bhde', k, v)
        z = torch.einsum('bthd,bhd->bth', q, k.sum(dim=1))
        num = torch.einsum('bthd,bhde->bthe', q, kv)

        out = num / (z.unsqueeze(-1) + 1e-6)
        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return (out + x).permute(0, 2, 1)


# ==========================================
# 5. 完整整合模型：ROEN_Final
# ==========================================
class ROEN_Final(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_classes, num_subnets=None, seq_len=8, heads=4, dropout=0.1):
        super(ROEN_Final, self).__init__()
        self.hidden = hidden

        # --- 1. 初始编码 ---
        self.node_enc = nn.Linear(node_in, hidden)
        self.edge_enc = nn.Linear(edge_in, hidden)

        # --- 2. 空间交互层 (Spatial Layers) ---
        # 包含两轮 "Node Update <-> Edge Update"
        self.spatial_layers = nn.ModuleList()
        for _ in range(2):  # 2层深度足够
            self.spatial_layers.append(nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(hidden, hidden, hidden, heads, dropout),
                'subnet': SubnetInteractionBlock(hidden, dropout) if num_subnets else None,
                'edge_upd': EdgeUpdaterModule(hidden, hidden, hidden, dropout)
            }))

        # --- 3. 时序层 (Temporal Layers) ---
        self.tpe = nn.Embedding(seq_len, hidden)
        self.temp_conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)  # 局部
        self.temp_global = LinearTemporalAttention(hidden, heads, dropout)  # 全局

        # --- 4. 分类器 (Classifier) ---
        # ⚠️ 直接基于“深层边特征”分类，辅以节点上下文
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),  # Edge + Src_Node + Dst_Node
            nn.LayerNorm(hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.3),  # 稍微降低 dropout 防止欠拟合
            nn.Linear(hidden * 2, num_classes)
        )

    def forward(self, graphs, seq_len):
        """
        graphs: List of PyG Data objects (length = seq_len)
        """
        spatial_node_feats = []  # 用于时序对齐
        spatial_edge_feats = []  # 用于最终分类 (这里的边特征是经过这一帧所有层更新后的)
        batch_global_ids = []

        # === Phase 1: Spatial Evolution (帧内交互) ===
        for t in range(seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch_global_ids.append(data.n_id)

            # Initial Mapping
            x = self.node_enc(x)
            edge_attr = self.edge_enc(edge_attr)

            # Iterative Updates
            for layer in self.spatial_layers:
                # A. Edge -> Node
                x = layer['node_att'](x, edge_index, edge_attr)

                # B. Subnet -> Node (Optional)
                if layer['subnet'] is not None and hasattr(data, 'subnet_id'):
                    x = layer['subnet'](x, data.subnet_id)

                # C. Node -> Edge (关键步骤: 刷新边特征)
                edge_attr = layer['edge_upd'](x, edge_index, edge_attr)

            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_attr)  # 保存的是最终进化的边

        # === Phase 2: Dynamic Alignment (动态对齐) ===
        all_ids = torch.cat(batch_global_ids)
        unique_ids, _ = torch.sort(torch.unique(all_ids))
        num_unique = len(unique_ids)
        device = unique_ids.device

        # [Num_Active_Nodes, Hidden, Seq_Len]
        dense_stack = torch.zeros((num_unique, self.hidden, seq_len), device=device)

        for t in range(seq_len):
            frame_ids = batch_global_ids[t]
            frame_feats = spatial_node_feats[t]  # 取出节点特征进行时序建模
            indices = torch.searchsorted(unique_ids, frame_ids)
            dense_stack[indices, :, t] = frame_feats

        # === Phase 3: Temporal Evolution (时序演化) ===
        # 注入 TPE
        time_indices = torch.arange(seq_len, device=device)
        t_emb = self.tpe(time_indices).permute(1, 0).unsqueeze(0)  # [1, H, T]
        dense_out = dense_stack + t_emb

        # 局部 + 全局
        dense_out = self.temp_conv(dense_out) + dense_out
        dense_out = self.temp_global(dense_out)

        # === Phase 4: Readout & Classification ===
        batch_preds = []
        for t in range(seq_len):
            # 1. 找回当前帧的时序增强节点特征
            frame_ids = batch_global_ids[t]
            indices = torch.searchsorted(unique_ids, frame_ids)
            node_out_t = dense_out[indices, :, t]  # [Num_Frame_Nodes, Hidden]

            edge_index = graphs[t].edge_index
            src, dst = edge_index[0], edge_index[1]

            # 2. 拼接: 深层边特征 + 时序增强的源节点 + 时序增强的宿节点
            # 注意：这里的 spatial_edge_feats[t] 已经是包含丰富上下文的特征了
            edge_rep = torch.cat([
                spatial_edge_feats[t],
                node_out_t[src],
                node_out_t[dst]
            ], dim=1)

            batch_preds.append(self.classifier(edge_rep))

        return batch_preds