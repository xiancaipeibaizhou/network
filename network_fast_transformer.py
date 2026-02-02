import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_geometric.utils import softmax

# ==========================================
# 1. 自定义边缘增强多头注意力层 (Core Contribution)
# ==========================================
class MultiHeadAttentionLayer(MessagePassing):
    """
    Edge-Augmented Multi-Head Attention Layer
    论文卖点: 显式将边特征(流量属性)融合进注意力机制的 Key 和 Value 中。
    """
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, n_heads: int = 4, dropout: float = 0.1):
        # aggr='add' 表示我们将注意力加权的邻居特征相加
        super().__init__(node_dim=0, aggr='add') 
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.dropout = dropout

        assert out_dim % n_heads == 0, f"out_dim {out_dim} must be divisible by n_heads {n_heads}"

        # 定义 Q, K, V 投影矩阵
        self.WQ = nn.Linear(in_dim, out_dim, bias=False)
        self.WK = nn.Linear(in_dim, out_dim, bias=False)
        self.WV = nn.Linear(in_dim, out_dim, bias=False)
        
        # 边特征投影矩阵 (关键：把边特征映射到与 Head 相同的维度)
        if edge_dim is not None:
            self.WE = nn.Linear(edge_dim, out_dim, bias=False)
        else:
            self.WE = None
            
        # 最终输出融合
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # x: [Num_Nodes, in_dim]
        # edge_index: [2, Num_Edges]
        # edge_attr: [Num_Edges, edge_dim]

        # 1. 线性变换并分头 [N, Heads, Head_Dim]
        q = self.WQ(x).view(-1, self.n_heads, self.head_dim)
        k = self.WK(x).view(-1, self.n_heads, self.head_dim)
        v = self.WV(x).view(-1, self.n_heads, self.head_dim)

        # 2. 处理边特征
        e = None
        if self.WE is not None and edge_attr is not None:
            e = self.WE(edge_attr).view(-1, self.n_heads, self.head_dim)

        # 3. 开始消息传递 (Propagate)
        # 自动调用 message -> aggregate -> update
        out = self.propagate(edge_index, q=q, k=k, v=v, e=e, size=None)

        # 4. 拼接多头并输出
        out = out.view(-1, self.out_dim)
        out = self.out_proj(out)
        return out

    def message(self, q_i, k_j, v_j, e, index):
        # q_i: 目标节点 (Receiver) 的 Query
        # k_j: 源节点 (Sender) 的 Key
        # v_j: 源节点 (Sender) 的 Value
        # e:   边特征
        # index: 目标节点索引 (用于 Softmax)

        # === 核心创新点：Edge-Aware Attention ===
        if e is not None:
            k_j = k_j + e # Key 融合边信息
            v_j = v_j + e # Value 也融合边信息
        
        # 1. 计算注意力分数 (Scaled Dot-Product)
        # (Q * K) / sqrt(d)
        alpha = (q_i * k_j).sum(dim=-1) / (self.head_dim ** 0.5)
        
        # 2. Softmax 归一化 (在邻居范围内)
        alpha = softmax(alpha, index)
        
        # 3. Dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 4. 加权求和
        out = v_j * alpha.unsqueeze(-1)
        
        return out

# ==========================================
# 2. 时序卷积组件 (Causal Dilated Conv)
# ==========================================
class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv2d, self).__init__()
        # kernel_size[1] 是时间维度的卷积核大小
        # padding 设为 (k-1)*d 确保因果性（不看未来）
        self.padding = (kernel_size[1] - 1) * dilation
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=(0, self.padding), 
            dilation=(1, dilation)
        )
        
    def forward(self, x):
        # x: [N, C, H, W_time]
        x = self.conv(x)
        # 裁剪掉多余的 padding (右侧)，只保留历史
        if self.padding > 0:
            x = x[:, :, :, :-self.padding]
        return x

class TemporalInception(nn.Module):
    def __init__(self, in_features, out_features, dilation_factor=2):
        super(TemporalInception, self).__init__()
        self.kernel_set = [1, 2, 3, 5] 
        cout_per_kernel = out_features // len(self.kernel_set)
        
        self.tconv = nn.ModuleList()
        for kern in self.kernel_set:
            self.tconv.append(
                CausalConv2d(
                    in_channels=in_features, 
                    out_channels=cout_per_kernel, 
                    kernel_size=(1, kern), 
                    dilation=dilation_factor
                )
            )
        # 1x1 卷积用于特征融合/残差匹配
        self.project = nn.Conv2d(in_features, out_features, kernel_size=1)

    def forward(self, x):
        outputs = []
        for conv in self.tconv:
            outputs.append(conv(x))
        out = torch.cat(outputs, dim=1)
        
        if x.shape[1] == out.shape[1]:
            out = out + x
        else:
            out = out + self.project(x)
        return F.relu(out)

# ==========================================
# 3. 主模型架构 (ROEN_Fast_Transformer)
# ==========================================
class ROEN_Fast_Transformer(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_classes):
        super(ROEN_Fast_Transformer, self).__init__()
        
        # 1. 基础编码器
        self.node_enc = nn.Linear(node_in, hidden)
        self.edge_enc = nn.Linear(edge_in, edge_in) # 保留原始维度，由 Attention 层投影
        
        # 2. 空间层：使用自定义的多头注意力
        self.gnn1 = MultiHeadAttentionLayer(in_dim=hidden, out_dim=hidden, edge_dim=edge_in, n_heads=4)
        self.bn1 = BatchNorm(hidden)
        
        self.gnn2 = MultiHeadAttentionLayer(in_dim=hidden, out_dim=hidden, edge_dim=edge_in, n_heads=4)
        self.bn2 = BatchNorm(hidden)
        
        # 边特征投影 (用于最后拼接)
        self.edge_proj = nn.Linear(edge_in, hidden)

        # 3. 时序层：Inception
        self.inception = TemporalInception(hidden, hidden)
        
        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, graphs, seq_len):
        # graphs: list of Data objects, length = seq_len
        
        # --- Phase 1: 稀疏空间提取 (Sparse Spatial) ---
        spatial_node_feats = []
        spatial_edge_feats = []
        batch_global_ids = []
        
        for t in range(seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            
            # 记录全局 ID (n_id) 用于后续对齐
            # 注意：data.n_id 是我们在 create_graph_data_sparse 里手动存进去的
            batch_global_ids.append(data.n_id) 

            # 编码
            x = F.relu(self.node_enc(x))
            
            # Layer 1: Custom Attention
            x = self.gnn1(x, edge_index, edge_attr)
            x = self.bn1(x).relu()
            
            # Layer 2: Custom Attention
            x = self.gnn2(x, edge_index, edge_attr)
            x = self.bn2(x).relu()
            
            spatial_node_feats.append(x)
            spatial_edge_feats.append(self.edge_proj(edge_attr))

        # --- Phase 2: 动态对齐 (Dynamic Alignment) ---
        # 这一步是速度的核心：只处理 Batch 内出现过的唯一节点
        all_ids = torch.cat(batch_global_ids)
        unique_ids, _ = torch.sort(torch.unique(all_ids))
        num_unique = len(unique_ids)
        
        device = unique_ids.device
        # 构建紧凑的时序张量 [Num_Active_Nodes, Hidden, Seq_Len]
        # 初始化为0 (Zero Padding)
        dense_stack = torch.zeros((num_unique, spatial_node_feats[0].size(1), seq_len), device=device)
        
        for t in range(seq_len):
            frame_ids = batch_global_ids[t]
            frame_feats = spatial_node_feats[t]
            
            # GPU 极速映射：找到当前帧节点在 unique_ids 中的位置
            # searchsorted 要求 unique_ids 有序
            indices = torch.searchsorted(unique_ids, frame_ids)
            
            # 填入张量
            dense_stack[indices, :, t] = frame_feats

        # --- Phase 3: 时序卷积 (Inception) ---
        # Input format: [N, C, H=1, W=Time]
        dense_in = dense_stack.unsqueeze(2) 
        dense_out = self.inception(dense_in) 
        dense_out = dense_out.squeeze(2) # [N, C, T]

        # --- Phase 4: 映射回边并分类 (Map back & Classify) ---
        batch_preds = []
        for t in range(seq_len):
            frame_ids = batch_global_ids[t]
            # 再次映射，获取当前帧节点在 dense_out 中的位置
            indices = torch.searchsorted(unique_ids, frame_ids)
            
            # 取出对齐且经过时序增强后的节点特征
            node_out_t = dense_out[indices, :, t] # [Num_Frame_Nodes, Hidden]
            
            edge_index = graphs[t].edge_index
            src, dst = edge_index[0], edge_index[1]
            
            # 拼接：源节点状态 + 宿节点状态 + 边特征
            # node_out_t 的索引对应 edge_index 中的局部索引 0..N
            edge_rep = torch.cat([
                node_out_t[src], 
                node_out_t[dst], 
                spatial_edge_feats[t]
            ], dim=1)
            
            batch_preds.append(self.classifier(edge_rep))
            
        return batch_preds