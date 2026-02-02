import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv2d, self).__init__()
        self.padding = (kernel_size[1] - 1) * dilation
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=(0, self.padding), 
            dilation=(1, dilation)
        )
        
    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :, :-self.padding]
        return x

class TemporalInception(nn.Module):
    def __init__(self, in_features, out_features, dilation_factor=2):
        super(TemporalInception, self).__init__()
        self.kernel_set = [1, 2, 3, 5] 
        assert out_features % len(self.kernel_set) == 0
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

class ROEN(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels_node, 
                 hidden_channels_edge, mlp_hidden_channels, num_edge_classes, 
                 seq_len=None):
        super(ROEN, self).__init__()
        
        # 1. 基础编码器
        self.mlp_node = nn.Sequential(
            nn.Linear(node_in_channels, hidden_channels_node),
            nn.ReLU(),
            nn.Linear(hidden_channels_node, hidden_channels_node),
            nn.ReLU()
        )
        self.mlp_edge = nn.Sequential(
            nn.Linear(edge_in_channels, hidden_channels_edge),
            nn.ReLU(),
            nn.Linear(hidden_channels_edge, hidden_channels_edge),
            nn.ReLU()
        )

        # 2. 空间 GNN (使用 TransformerConv 以获得更好性能)
        self.gnn1 = TransformerConv(hidden_channels_node, hidden_channels_node, heads=4, edge_dim=hidden_channels_edge, dropout=0.1, concat=False)
        self.gnn2 = TransformerConv(hidden_channels_node, hidden_channels_node, heads=4, edge_dim=hidden_channels_edge, dropout=0.1, concat=False)
        
        self.edge_fc = nn.Sequential(
            nn.Linear(hidden_channels_edge, hidden_channels_edge),
            nn.ReLU()
        )

        # 3. 节点时序提取 (只有节点可以 Stack)
        self.inception_node = TemporalInception(hidden_channels_node, hidden_channels_node)

        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels_node + hidden_channels_edge, mlp_hidden_channels),
            nn.ReLU(),
            nn.Linear(mlp_hidden_channels, num_edge_classes)
        )

    def forward(self, graphs, seq_len=None):
        if seq_len is None: seq_len = len(graphs)
        
        node_seq_list = []
        edge_spatial_features = [] 

        # --- 阶段 1: 空间处理 (逐帧) ---
        for t in range(seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            
            x = self.mlp_node(x)
            edge_attr = self.mlp_edge(edge_attr)

            x = self.gnn1(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.gnn2(x, edge_index, edge_attr)
            
            node_seq_list.append(x)
            edge_attr = self.edge_fc(edge_attr)
            edge_spatial_features.append(edge_attr)

        # --- 阶段 2: 节点时序处理 ---
        # 假设使用了全局IP映射，节点数固定，可以 Stack
        node_stack = torch.stack(node_seq_list, dim=-1) # [N, F, T]
        node_in = node_stack.unsqueeze(2) # [N, F, 1, T]
        node_out = self.inception_node(node_in) 
        node_out = node_out.squeeze(2).permute(2, 0, 1) # [T, N, F]

        # --- 阶段 3: 动态边分类 (逐帧融合) ---
        batch_preds = []
        for t in range(seq_len):
            edge_index = graphs[t].edge_index
            src, dst = edge_index[0], edge_index[1]
            
            # 使用 t 时刻的时序节点特征 + t 时刻的瞬时边特征
            edge_rep = torch.cat([
                node_out[t][src], 
                node_out[t][dst], 
                edge_spatial_features[t]
            ], dim=1)
            
            pred = self.classifier(edge_rep)
            batch_preds.append(pred)

        return batch_preds