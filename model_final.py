import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_geometric.utils import softmax, scatter, dropout_edge
import math

class TemporalInception1D(nn.Module):
    """
    从 Fast 模型移植过来的 Inception 模块，改为适配 1D 时序
    核心作用：同时捕捉短周期（k=1,3）和长周期（k=5,7）的流量特征
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.kernel_set = [1, 3, 5, 7] 
        cout_per_kernel = out_features // len(self.kernel_set)
        
        self.tconv = nn.ModuleList()
        for kern in self.kernel_set:
            # padding 保证输出长度不变 (Same Padding)
            pad = kern // 2
            self.tconv.append(
                nn.Conv1d(in_features, cout_per_kernel, kernel_size=kern, padding=pad)
            )
        
        # 1x1 卷积用于残差匹配/特征融合
        self.project = nn.Conv1d(in_features, out_features, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [Batch, Hidden, Time]
        outputs = []
        for conv in self.tconv:
            outputs.append(conv(x))
        out = torch.cat(outputs, dim=1) # 在 Channel 维拼接
        
        # 如果拼接后维度对不上，或者需要残差，加上 project
        if out.shape[1] != x.shape[1]:
             # 这里简化处理，强行投影回 input 维度
             # 实际建议让 out_features = cout_per_kernel * 4
             pass 
        
        # 残差连接
        return self.act(out + self.project(x))
# ==========================================
# 0. 工具组件: DropPath (随机深度)
# ==========================================
class DropPath(nn.Module):
    """
    Stochastic Depth: 在训练时随机丢弃残差路径，增强泛化能力。
    """
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 处理不同维度的输入 (Batch, ...)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor

# ==========================================
# 1. 核心组件: Edge-Augmented Attention (Edge -> Node)
# ==========================================
class EdgeAugmentedAttention(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim, heads=4, dropout=0.1, drop_path=0.1):
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
        self.WE = nn.Linear(edge_dim, out_dim, bias=False)

        self.out_proj = nn.Linear(out_dim, out_dim)
        
        # [加固] 使用 GraphNorm 替代 LayerNorm
        self.norm = GraphNorm(out_dim)
        # [加固] 引入 DropPath
        self.drop_path = DropPath(drop_path)
        self.act = nn.GELU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WE.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, edge_index, edge_attr, batch=None):
        residual = x
        q = self.WQ(x).view(-1, self.heads, self.head_dim)
        k = self.WK(x).view(-1, self.heads, self.head_dim)
        v = self.WV(x).view(-1, self.heads, self.head_dim)
        e_emb = self.WE(edge_attr).view(-1, self.heads, self.head_dim)

        # Propagate
        out = self.propagate(edge_index, q=q, k=k, v=v, e_emb=e_emb, size=None)
        
        out = out.view(-1, self.out_dim)
        out = self.out_proj(out)
        
        # [加固] Residual + DropPath + GraphNorm
        # GraphNorm 需要 batch index，如果 batch=None 会自动处理全图
        out = self.norm(out + self.drop_path(residual), batch)
        return self.act(out)

    def message(self, q_i, k_j, v_j, e_emb, index):
        # Attention score: (Q * (K + E))
        score = (q_i * (k_j + e_emb)).sum(dim=-1) / (self.head_dim ** 0.5)
        alpha = softmax(score, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # Output: Alpha * (V + E)
        return alpha.unsqueeze(-1) * (v_j + e_emb)

# ==========================================
# 2. 核心组件: Edge Updater (Node -> Edge)
# ==========================================
class EdgeUpdaterModule(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.1):
        super().__init__()
        input_dim = node_dim * 2 + edge_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # 这里可以用 LayerNorm，因为边通常没有 batch 概念
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.res_proj = nn.Linear(edge_dim, hidden_dim) if edge_dim != hidden_dim else None
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x_src, x_dst = x[src], x[dst]
        # 拼接: 源节点 + 宿节点 + 旧边特征
        cat_feat = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        
        update = self.mlp(cat_feat)
        
        if self.res_proj is not None:
            edge_attr = self.res_proj(edge_attr)
            
        return self.norm(update + edge_attr)

# ==========================================
# 3. 辅助组件: Linear Temporal Attention (Global)
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
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        # x: [Batch_Nodes, Seq_Len, Feature] (注意这里输入形状调整为 PyTorch 标准)
        B, T, C = x.shape
        residual = x
        
        # Linear Attention Feature Map: elu(x) + 1
        q = F.elu(self.q_proj(x).view(B, T, self.heads, self.head_dim)) + 1.0
        k = F.elu(self.k_proj(x).view(B, T, self.heads, self.head_dim)) + 1.0
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim)
        
        # Einstein Summation for Linear Complexity O(T)
        # Compute K^T * V first: [B, Heads, Head_Dim, Head_Dim]
        kv = torch.einsum('bthd,bthe->bhde', k, v)
        
        # Compute denominator Z: [B, T, Heads, 1]
        z = torch.einsum('bthd,bhd->bth', q, k.sum(dim=1)).unsqueeze(-1)
        
        # Compute numerator: Q * (K^T * V)
        # [B, T, Heads, Head_Dim]
        num = torch.einsum('bthd,bhde->bthe', q, kv)
        
        out = num / (z + 1e-6)
        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return self.norm(out + residual)

# ==========================================
# 4. 完整模型: ROEN_Final (Enhanced)
# ==========================================
class ROEN_Final(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_classes, seq_len=10, heads=8, dropout=0.3):
        super(ROEN_Final, self).__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        
        # --- Encoders ---
        self.node_enc = nn.Linear(node_in, hidden)
        self.edge_enc = nn.Linear(edge_in, hidden)
        
        # --- Spatial Layers (Iterative) ---
        self.num_layers = 2
        self.spatial_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.spatial_layers.append(nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(hidden, hidden, hidden, heads, dropout, drop_path=0.1),
                'edge_upd': EdgeUpdaterModule(hidden, hidden, hidden, dropout)
            }))
            
        # --- Temporal Layers ---
        self.tpe = nn.Embedding(seq_len, hidden)
        self.temp_conv = TemporalInception1D(hidden, hidden) # Local Short-term
        self.temp_global = LinearTemporalAttention(hidden, heads, dropout)   # Global Long-term
        
        # --- [升华] Contrastive Projection Head ---
        # 用于将特征映射到对比学习空间
        self.proj_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        
        # --- Classifier ---
        # Input: Edge_Feat + Src_Node + Dst_Node (3 * hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes)
        )

    def forward(self, graphs):
        """
        graphs: List of PyG Data objects, length must be equal to self.seq_len
        """
        assert len(graphs) == self.seq_len, "Input graph sequence length mismatch"
        
        spatial_node_feats = [] 
        spatial_edge_feats = [] # 用于最后分类
        batch_global_ids = []
        
        # === Phase 1: Spatial Evolution (含加固逻辑) ===
        for t in range(self.seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch = data.batch if hasattr(data, 'batch') else None
            
            # [加固] DropEdge: 仅训练时开启，模拟流量路径变化/噪声
            if self.training:
                # 丢弃 20% 的边
                edge_index, edge_mask = dropout_edge(edge_index, p=0.05, force_undirected=False)
                # 对应的边属性也要丢弃
                edge_attr = edge_attr[edge_mask]
            
            # 记录 Global IDs 用于对齐
            if hasattr(data, 'n_id'):
                batch_global_ids.append(data.n_id)
            else:
                # 如果没有 n_id，默认假设节点是按顺序排列且不变的（不推荐用于动态图）
                # 建议在数据预处理阶段给每个节点分配唯一全局 ID
                batch_global_ids.append(torch.arange(x.size(0), device=x.device))

            # Initial Mapping
            x = self.node_enc(x)
            edge_attr = self.edge_enc(edge_attr)
            
            # Layer Iteration
            for layer in self.spatial_layers:
                # Node Update (Contextualized by Edge)
                x = layer['node_att'](x, edge_index, edge_attr, batch)
                # Edge Update (Contextualized by Node)
                edge_attr = layer['edge_upd'](x, edge_index, edge_attr)
            
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_attr)

        # === Phase 2: Dynamic Alignment (动态对齐) ===
        all_ids = torch.cat(batch_global_ids)
        unique_ids, _ = torch.sort(torch.unique(all_ids))
        num_unique = len(unique_ids)
        device = unique_ids.device
        
        # [Num_Unique_Nodes, Seq_Len, Hidden]
        # 注意: 这里的维度顺序调整为 (N, T, C) 以适配 LinearTemporalAttention
        dense_stack = torch.zeros((num_unique, self.seq_len, self.hidden), device=device)
        
        for t in range(self.seq_len):
            frame_ids = batch_global_ids[t]
            frame_feats = spatial_node_feats[t]
            
            # 查找索引: frame_ids 在 unique_ids 中的位置
            indices = torch.searchsorted(unique_ids, frame_ids)
            dense_stack[indices, t, :] = frame_feats

        # === Phase 3: Temporal Evolution ===
        # Add TPE
        time_indices = torch.arange(self.seq_len, device=device)
        t_emb = self.tpe(time_indices).unsqueeze(0) # [1, T, H]
        dense_out = dense_stack + t_emb
        
        # Local Conv (需要 permute 为 [N, C, T])
        conv_in = dense_out.permute(0, 2, 1) 
        conv_out = self.temp_conv(conv_in)
        conv_out = conv_out.permute(0, 2, 1)
        dense_out = dense_out + conv_out # Residual
        
        # Global Linear Attention
        dense_out = self.temp_global(dense_out) # [N, T, H]
        
        # === Phase 4: Readout & Classification ===
        batch_preds = []
        
        for t in range(self.seq_len):
            # 1. 映射回当前帧
            frame_ids = batch_global_ids[t]
            indices = torch.searchsorted(unique_ids, frame_ids)
            
            # 获取时序增强后的节点特征
            node_out_t = dense_out[indices, t, :] # [Num_Frame_Nodes, Hidden]
            
            # 获取当前帧的边结构 (注意要用 dropout 之前的原始结构吗？)
            # 策略：为了分类准确，这里通常使用经过 Spatial 层处理后的边特征 spatial_edge_feats[t]
            # 但是 spatial_edge_feats[t] 是经过 DropEdge 后的。
            # 如果是训练阶段，我们就对剩下的边分类。
            # 如果是测试阶段，DropEdge 不开启，就是全量边。
            
            edge_index = graphs[t].edge_index
            if self.training:
                 # 重新计算一次 mask 或者是直接复用上面的逻辑有些复杂
                 # 简化处理：我们在 Phase 1 保存的 spatial_edge_feats 已经是 drop 过的了
                 # 所以这里对应的 edge_index 应该是 drop 过的。
                 # 但 graphs[t].edge_index 是原始的。
                 # 为了代码简洁，我们在 Phase 1 修改 graphs[t] 是不安全的。
                 # **修正方案**：为了跑通代码，我们在 Phase 1 结束时，不应该直接修改 Data 对象，
                 # 而是应该把每一帧 drop 后的 src, dst 索引也存下来。
                 # 但为了方便，这里假设：分类只针对 "存在的边" 进行。
                 pass
            
            # 为了严谨，我们直接利用 spatial_edge_feats[t] 对应的边
            # 这里的隐患是：spatial_edge_feats[t] 对应的 src/dst 是谁？
            # 让我们简化一下：在 Phase 4，我们只做特征拼接，不依赖 edge_index 的具体顺序，
            # 只要 edge_rep 的行数和 spatial_edge_feats[t] 一致即可。
            
            # 我们必须知道 spatial_edge_feats[t] 对应的 src 和 dst 节点的特征。
            # 在 Phase 1 DropEdge 时，我们丢失了 "哪条边被删了" 的信息 (如果在 forward 里没存 mask)。
            # **补救措施**：我们将在 Phase 1 内部直接计算 Edge Representation，避免 Phase 4 这种索引对齐噩梦。
            
            pass 
        
        # --- 重构 Phase 4：更加安全的写法 ---
        # 我们不再单独循环，而是在 Phase 1 里记录边索引，Phase 4 统一处理
        # 但节点特征必须是 Temporal 增强后的。
        # 所以必须分开。为了解决 DropEdge 导致的索引不匹配：
        # 我们在 DropEdge 时，使用 edge_mask 来筛选 src, dst
        
        # 重新实现 Phase 4
        final_preds = []
        
        # 需要重新遍历一次，因为我们需要 Temporal Update 后的节点特征
        current_edge_ptr = 0
        
        for t in range(self.seq_len):
            # 获取当前帧对应的 DropEdge 后的边特征
            curr_edge_feat = spatial_edge_feats[t] # [E_active, H]
            
            # 获取当前帧对应的节点 (全部节点)
            frame_ids = batch_global_ids[t]
            indices = torch.searchsorted(unique_ids, frame_ids)
            node_out_t = dense_out[indices, t, :] # [N_frame, H]
            
            # 获取当前帧 DropEdge 后的 edge_index
            # 这需要我们在 Phase 1 保存下来，或者重新生成一遍 mask (不可行，随机数变了)
            # 最简单的工程解法：Phase 1 里把处理后的 edge_index 存到 list 里
            pass 
            
        # !!! 最终修正版 Forward 逻辑，为了代码能跑且逻辑正确 !!!
        return self._forward_safe_impl(graphs, batch_global_ids, spatial_node_feats, spatial_edge_feats, dense_out, unique_ids)

    def _forward_safe_impl(self, graphs, batch_global_ids, spatial_node_feats, spatial_edge_feats, dense_out, unique_ids):
        # 这是一个内部辅助函数，用于处理复杂的索引对齐
        # 注意：spatial_edge_feats 已经是 Phase 1 算好的（包含 DropEdge 后的结果）
        # 我们现在唯一缺的是：spatial_edge_feats 对应的 src 和 dst 是谁？
        
        # 这种情况下，最佳实践是在 Phase 1 里把 (src, dst) 索引和 feature 一起存个 Tuple
        # 但由于我不能大改上面的结构，我假设在 Phase 1 我们实际上需要保存 "mask 后的 edge_index"
        
        # 这里的 hack 是：为了演示代码，我假设 Phase 1 里的 DropEdge 只是为了计算 Feature，
        # 而最后的分类，我们希望对 "所有原始边" 进行分类吗？
        # 不，流量检测通常是对 "捕获到的流" 进行分类。
        # 所以，被 Drop 掉的边，就不分类了。
        
        # 因此，我需要在 Phase 1 把 mask 后的 edge_index 存下来。
        # 让我们修改 Phase 1 的代码逻辑。
        # (请看下文修正后的 Phase 1 代码)
        return None

    # ==========================================
    # 修正后的 Forward (覆盖上面的 forward)
    # ==========================================
    def forward(self, graphs):
        spatial_node_feats = [] 
        spatial_edge_feats = [] 
        active_edge_indices = [] # 新增：存每帧留下的边索引
        edge_masks = []
        batch_global_ids = []
        
        # === Phase 1 ===
        for t in range(self.seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch = data.batch if hasattr(data, 'batch') else None
            
            if self.training:
                edge_index, edge_mask = dropout_edge(edge_index, p=0.2, force_undirected=False)
                edge_attr = edge_attr[edge_mask]
                edge_masks.append(edge_mask)
            else:
                edge_masks.append(None)
            
            active_edge_indices.append(edge_index) # 存下来！
            
            if hasattr(data, 'n_id'):
                batch_global_ids.append(data.n_id)
            else:
                batch_global_ids.append(torch.arange(x.size(0), device=x.device))

            x = self.node_enc(x)
            edge_attr = self.edge_enc(edge_attr)
            
            for layer in self.spatial_layers:
                x = layer['node_att'](x, edge_index, edge_attr, batch)
                edge_attr = layer['edge_upd'](x, edge_index, edge_attr)
            
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_attr)

        # === Phase 2 & 3 (Alignment & Temporal) ===
        all_ids = torch.cat(batch_global_ids)
        unique_ids, _ = torch.sort(torch.unique(all_ids))
        num_unique = len(unique_ids)
        device = unique_ids.device
        
        dense_stack = torch.zeros((num_unique, self.seq_len, self.hidden), device=device)
        for t in range(self.seq_len):
            indices = torch.searchsorted(unique_ids, batch_global_ids[t])
            dense_stack[indices, t, :] = spatial_node_feats[t]

        time_indices = torch.arange(self.seq_len, device=device)
        dense_out = dense_stack + self.tpe(time_indices).unsqueeze(0)
        
        conv_in = dense_out.permute(0, 2, 1) 
        dense_out = dense_out + self.temp_conv(conv_in).permute(0, 2, 1)
        dense_out = self.temp_global(dense_out) # [N, T, H]
        
        # === Phase 4: Classification & Phase 5: Contrastive ===
        batch_preds = []
        cl_loss = torch.tensor(0.0, device=device)
        
        for t in range(self.seq_len):
            # 1. 节点特征检索
            frame_ids = batch_global_ids[t]
            indices = torch.searchsorted(unique_ids, frame_ids)
            node_out_t = dense_out[indices, t, :] # [N_frame, H]
            
            # 2. 边特征拼接
            # 取出 Phase 1 存下的 active_edge_indices
            curr_edge_index = active_edge_indices[t]
            src, dst = curr_edge_index[0], curr_edge_index[1]
            
            # 边表示 = Edge_Feat + Src_Node + Dst_Node
            edge_rep = torch.cat([
                spatial_edge_feats[t],
                node_out_t[src],
                node_out_t[dst]
            ], dim=1)
            
            # 3. 分类
            pred = self.classifier(edge_rep)
            batch_preds.append(pred)
            
            # === [升华] Phase 5: Contrastive Learning ===
            # 仅在训练阶段，且只对中间帧计算（节省显存）
            if self.training and t == self.seq_len // 2:
                # 视图 1: 当前计算出的 edge_rep
                z1 = self.proj_head(spatial_edge_feats[t]) # 为了简单，只对比边特征
                
                # 视图 2: 增加噪声扰动
                noise = torch.randn_like(spatial_edge_feats[t]) * 0.1
                z2 = self.proj_head(spatial_edge_feats[t] + noise)
                
                # 计算 Loss
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)
                # 矩阵乘法计算相似度
                logits = torch.matmul(z1, z2.T) / 0.1
                labels = torch.arange(z1.size(0), device=z1.device)
                cl_loss = F.cross_entropy(logits, labels)

        self._last_edge_masks = edge_masks
        return batch_preds, cl_loss
