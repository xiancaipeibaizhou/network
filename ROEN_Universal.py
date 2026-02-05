import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

# 复用已有的组件
from network_fast_transformer import TemporalInception  #
from network_advanced import LinearTemporalAttention, SubnetInteractionBlock #
from ROEN_Final import EdgeAugmentedAttention, EdgeUpdaterModule #

class ROEN_Universal(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_classes, num_subnets=None, seq_len=8, heads=8, dropout=0.3):
        super(ROEN_Universal, self).__init__()
        self.hidden = hidden
        
        # --- 1. 空间层 (保留 ROEN_Final 的强拓扑能力，适配 NB15) ---
        self.node_enc = nn.Linear(node_in, hidden)
        self.edge_enc = nn.Linear(edge_in, hidden)
        
        self.spatial_layers = nn.ModuleList()
        # 保持 2 层深度，足以捕捉 NB15 的子网传播
        for _ in range(2): 
            self.spatial_layers.append(nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(hidden, hidden, hidden, heads, dropout),
                'subnet': SubnetInteractionBlock(hidden, dropout) if num_subnets else None,
                'edge_upd': EdgeUpdaterModule(hidden, hidden, hidden, dropout)
            }))
            
        # --- 2. 时序层 (融合改进：Inception + Linear Attention) ---
        self.tpe = nn.Embedding(seq_len, hidden)
        
        # [关键改进 A] 微观提取器：替换原本简单的 Conv1d，使用 Inception 捕捉多尺度局部指纹 (适配 Darknet)
        self.micro_temporal = TemporalInception(hidden, hidden)
        
        # [关键改进 B] 宏观聚合器：保留 Linear Attention 捕捉长程攻击链 (适配 NB15)
        self.macro_temporal = LinearTemporalAttention(hidden, heads, dropout)
        
        # --- 3. 分类器 ---
        # 增加 Dropout 防止在 Darknet 上过拟合
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1), # 略微增加 Dropout
            nn.Linear(hidden * 2, num_classes)
        )

    def forward(self, graphs, seq_len):
        spatial_node_feats = [] 
        spatial_edge_feats = [] 
        batch_global_ids = []
        
        # === Phase 1: Spatial Evolution (不变) ===
        for t in range(seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch_global_ids.append(data.n_id)
            
            x = self.node_enc(x)
            edge_attr = self.edge_enc(edge_attr)
            
            for layer in self.spatial_layers:
                x = layer['node_att'](x, edge_index, edge_attr)
                if layer['subnet'] is not None and hasattr(data, 'subnet_id'):
                    x = layer['subnet'](x, data.subnet_id)
                edge_attr = layer['edge_upd'](x, edge_index, edge_attr)
            
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_attr)

        # === Phase 2: Dynamic Alignment (不变) ===
        all_ids = torch.cat(batch_global_ids)
        unique_ids, _ = torch.sort(torch.unique(all_ids))
        num_unique = len(unique_ids)
        device = unique_ids.device
        
        dense_stack = torch.zeros((num_unique, self.hidden, seq_len), device=device)
        for t in range(seq_len):
            indices = torch.searchsorted(unique_ids, batch_global_ids[t])
            dense_stack[indices, :, t] = spatial_node_feats[t]

        # === Phase 3: Temporal Evolution (双流融合) ===
        # 1. 注入位置编码
        time_indices = torch.arange(seq_len, device=device)
        t_emb = self.tpe(time_indices).permute(1, 0).unsqueeze(0)
        dense_out = dense_stack + t_emb
        
        # 2. [Micro] Inception 处理：提取多尺度局部特征
        # Inception 期望输入 [N, C, H=1, W=T]，需要维度变换
        # dense_out: [N, C, T] -> [N, C, 1, T]
        dense_in_inception = dense_out.unsqueeze(2)
        dense_micro = self.micro_temporal(dense_in_inception).squeeze(2) # -> [N, C, T]
        
        # 残差连接：叠加微观特征
        dense_out = dense_out + dense_micro
        
        # 3. [Macro] Linear Attention 处理：全局上下文关联
        # 这里的输入已经是包含丰富局部特征的向量了
        dense_out = self.macro_temporal(dense_out) # 内部已有残差连接

        # === Phase 4: Readout & Classification ===
        batch_preds = []
        for t in range(seq_len):
            frame_ids = batch_global_ids[t]
            indices = torch.searchsorted(unique_ids, frame_ids)
            node_out_t = dense_out[indices, :, t]
            
            edge_index = graphs[t].edge_index
            src, dst = edge_index[0], edge_index[1]
            
            # 融合: 深度进化的边 + 双流时序增强的节点
            edge_rep = torch.cat([
                spatial_edge_feats[t], 
                node_out_t[src], 
                node_out_t[dst]
            ], dim=1)
            
            batch_preds.append(self.classifier(edge_rep))
            
        return batch_preds
