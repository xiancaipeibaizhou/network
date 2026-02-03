import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn import BatchNorm
# 复用您原有的 attention 定义
from network_fast_transformer import MultiHeadAttentionLayer 

# ==========================================
# 1. 线性注意力模块 (对应建议4: 核化线性复杂度)
#    替代传统的 Softmax Attention，复杂度从 O(T^2) 降为 O(T)
# ==========================================
class LinearTemporalAttention(nn.Module):
    def __init__(self, feature_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = feature_dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 简单的核函数：ELU + 1，保证非负
        self.elu = nn.ELU()

    def forward(self, x):
        # x input: [Batch, Feature, Time]
        B, C, T = x.shape
        x = x.permute(0, 2, 1) # [B, T, C]
        
        q = self.q_proj(x).view(B, T, self.heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim)
        
        # 核化处理 (Kernel Trick): ELU + 1 保证非负
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        
        # 1. 计算 Linear Attention 的分子: Q * (K^T * V)
        # kv: [B, Heads, Head_Dim, Head_Dim] -> 聚合了整个时间窗口的信息
        kv = torch.einsum('bthd,bthe->bhde', k, v)
        
        # num: [B, Time, Heads, Head_Dim]
        num = torch.einsum('bthd,bhde->bthe', q, kv)
        
        # 2. 计算归一化因子 (分母): Q * (sum(K)^T)
        # k_sum: [B, Heads, Head_Dim] (沿时间轴 t 求和)
        k_sum = k.sum(dim=1)
        
        # z: [B, Time, Heads] -> Q 和 K_sum 的点积
        # [修正点] 使用 bthd, bhd -> bth 避免维度混淆
        z = torch.einsum('bthd,bhd->bth', q, k_sum)
        
        # 3. 归一化与输出
        # z.unsqueeze(-1) -> [B, Time, Heads, 1] 用于广播除法
        out = num / (z.unsqueeze(-1) + 1e-6)
        
        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # 恢复维度 [B, C, T]
        return (out + x).permute(0, 2, 1)

# ==========================================
# 2. 子网交互模块 (对应建议1: 空间维度多尺度)
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

    def forward(self, x, subnet_id, batch_idx=None):
        # x: [N, Hidden]
        # subnet_id: [N]
        
        # 1. 聚合：计算子网中心向量 (Subnet Centroids)
        # 使用 scatter_mean 将属于同一子网的节点特征取平均
        # 必须确保 subnet_id 在 batch 内是连续索引或处理过，否则需要映射
        if subnet_id is None:
            return x
            
        subnet_features = scatter(x, subnet_id, dim=0, reduce='mean') # [Num_Subnets, Hidden]
        
        # 2. 变换：在子网级别提取特征 (例如发现子网内的协同攻击模式)
        subnet_features = self.subnet_mlp(subnet_features)
        
        # 3. 广播：将子网特征加回该子网下的所有节点
        # subnet_features[subnet_id] 自动完成广播
        out = x + subnet_features[subnet_id]
        return self.norm(out)

# ==========================================
# 3. 主模型架构 (ROEN_Advanced)
# ==========================================
class ROEN_Advanced(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_classes, num_subnets=None, seq_len=8):
        super(ROEN_Advanced, self).__init__()
        
        # --- 基础编码 ---
        self.node_enc = nn.Linear(node_in, hidden)
        self.edge_enc = nn.Linear(edge_in, edge_in)
        self.edge_proj = nn.Linear(edge_in, hidden)

        # --- 1. 改进的空间层 (GNN + Subnet Interaction) ---
        self.gnn1 = MultiHeadAttentionLayer(in_dim=hidden, out_dim=hidden, edge_dim=edge_in, n_heads=4)
        self.bn1 = BatchNorm(hidden)
        
        # 新增：子网交互层
        self.subnet_block = SubnetInteractionBlock(hidden) if num_subnets else None
        
        self.gnn2 = MultiHeadAttentionLayer(in_dim=hidden, out_dim=hidden, edge_dim=edge_in, n_heads=4)
        self.bn2 = BatchNorm(hidden)

        # --- 2. 改进的时序层 (TPE + Linear Attention) ---
        # Time-Period Embedding (对应建议2)
        self.tpe = nn.Embedding(seq_len, hidden)
        
        # 混合时序块：1层卷积捕捉局部 + 1层线性Attention捕捉全局
        self.temp_conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.temp_global = LinearTemporalAttention(hidden) # 对应建议4
        
        # --- 3. 关联维度 (Flow Interaction) ---
        # 增强分类器的输入，不仅仅拼接节点，还显式加入边的上下文
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden + hidden, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden * 2, num_classes)
        )

    def forward(self, graphs, seq_len):
        # --- Phase 1: Spatial Encoding ---
        spatial_node_feats = []
        spatial_edge_feats = []
        batch_global_ids = []
        
        for t in range(seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch_global_ids.append(data.n_id) 

            # Node Encode
            x = F.relu(self.node_enc(x))
            
            # GNN Layer 1
            x = self.gnn1(x, edge_index, edge_attr)
            x = self.bn1(x).relu()
            
            # Subnet Interaction (关键改进)
            # 如果存在子网信息，进行子网级特征聚合与增强
            if self.subnet_block is not None and hasattr(data, "subnet_id"):
                x = self.subnet_block(x, data.subnet_id)
            
            # GNN Layer 2
            x = self.gnn2(x, edge_index, edge_attr)
            x = self.bn2(x).relu()
            
            spatial_node_feats.append(x)
            spatial_edge_feats.append(self.edge_proj(edge_attr))

        # --- Phase 2: Dynamic Alignment ---
        all_ids = torch.cat(batch_global_ids)
        unique_ids, _ = torch.sort(torch.unique(all_ids))
        num_unique = len(unique_ids)
        device = unique_ids.device
        
        # [Num_Active, Hidden, Seq_Len]
        dense_stack = torch.zeros((num_unique, spatial_node_feats[0].size(1), seq_len), device=device)
        
        for t in range(seq_len):
            frame_ids = batch_global_ids[t]
            frame_feats = spatial_node_feats[t]
            indices = torch.searchsorted(unique_ids, frame_ids)
            dense_stack[indices, :, t] = frame_feats

        # --- Phase 3: Temporal Evolution (Enhanced) ---
        # Input: [N, C, T]
        
        # 1. 注入 Time-Period Embedding (TPE)
        # 为每个时间步增加位置编码，捕捉周期性
        time_indices = torch.arange(seq_len, device=device) # [T]
        t_emb = self.tpe(time_indices) # [T, Hidden]
        
        # 变换为 [1, Hidden, T] 以便广播
        t_emb = t_emb.permute(1, 0).unsqueeze(0) 
        
        # 扩展到所有节点 [Num_Nodes, Hidden, T]
        t_emb = t_emb.expand(num_unique, -1, -1)
        
        # === [Critical Fix] 必须定义 dense_out ===
        dense_out = dense_stack + t_emb 
        
        # 2. 局部卷积 (Local)
        # 确保 dense_out 在这里已经被定义
        dense_out = self.temp_conv(dense_out) + dense_out
        
        # 3. 全局线性注意力 (Global Persistence)
        dense_out = self.temp_global(dense_out)

        # --- Phase 4: Readout & Classification ---
        batch_preds = []
        for t in range(seq_len):
            frame_ids = batch_global_ids[t]
            indices = torch.searchsorted(unique_ids, frame_ids)
            node_out_t = dense_out[indices, :, t] 
            
            edge_index = graphs[t].edge_index
            src, dst = edge_index[0], edge_index[1]
            
            # 拼接
            edge_rep = torch.cat([
                node_out_t[src], 
                node_out_t[dst], 
                spatial_edge_feats[t] # 这里的 edge feat 是投影后的流特征
            ], dim=1)
            
            batch_preds.append(self.classifier(edge_rep))
            
        return batch_preds