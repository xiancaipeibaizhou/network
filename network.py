import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# --- 1. 定义因果卷积层 (Causal Conv1d) ---
# 核心修正：只在左侧填充，确保输出长度 == 输入长度，且不泄露未来信息
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        # 计算需要填充的长度: (K-1) * D
        self.padding = (kernel_size - 1) * dilation
        # 初始化标准卷积，padding设为0 (因为我们要手动padding)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=0, dilation=dilation, **kwargs)

    def forward(self, x):
        # x shape: (Batch, Channels, Seq_Len)
        # F.pad 参数格式: (left, right, top, bottom, ...)
        # 只在左侧填充 self.padding 个 0
        x = F.pad(x, (self.padding, 0)) 
        return self.conv(x)

# --- 2. 定义残差时序块 (Temporal Residual Block) ---
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()
        
        # 第一层因果卷积
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层因果卷积
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.bn2, self.relu2, self.dropout2)
        
        # 1x1 卷积用于调整残差维度 (如果输入输出通道数不一致)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # 核心：残差连接 (现在维度保证一致了)
        return self.relu(out + res)

# --- 3. 定义 TCN 主网络 ---
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # 使用 TemporalBlock 替换普通卷积
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                     dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- 4. 完整的 ROEN-TCN 模型 ---
class G_TCN(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels_node, hidden_channels_edge, mlp_hidden_channels, num_edge_classes, heads=2, tcn_kernel_size=3):
        super(G_TCN, self).__init__()
        self.classifier_dropout = nn.Dropout(0.5)

        # --- 节点/边特征预处理 ---
        self.mlp_node_fc1 = nn.Linear(node_in_channels, hidden_channels_node)
        self.mlp_node_fc2 = nn.Linear(hidden_channels_node, hidden_channels_node)
        
        self.mlp_edge_fc1 = nn.Linear(edge_in_channels, hidden_channels_edge)
        self.mlp_edge_fc2 = nn.Linear(hidden_channels_edge, hidden_channels_edge)

        # --- GAT 层 ---
        self.gat_node_layers1 = GATConv(
            hidden_channels_node,
            hidden_channels_node,
            heads=heads,
            concat=False,
            dropout=0.2,
            edge_dim=hidden_channels_edge,
        )
        self.bn_node1 = nn.BatchNorm1d(hidden_channels_node)

        self.gat_node_layers2 = GATConv(
            hidden_channels_node,
            hidden_channels_node,
            heads=heads,
            concat=False,
            dropout=0.2,
            edge_dim=hidden_channels_edge,
        )
        self.bn_node2 = nn.BatchNorm1d(hidden_channels_node)
        
        self.edge_fc1 = nn.Linear(hidden_channels_edge, hidden_channels_edge)
        self.edge_fc2 = nn.Linear(hidden_channels_edge, hidden_channels_edge)

        # --- TCN 模块 ---
        tcn_channels_node = [hidden_channels_node, hidden_channels_node] # 2层 TCN
        self.tcn_node = TemporalConvNet(hidden_channels_node, tcn_channels_node, kernel_size=tcn_kernel_size)
        
        tcn_channels_edge = [hidden_channels_edge, hidden_channels_edge] # 2层 TCN
        self.tcn_edge = TemporalConvNet(hidden_channels_edge, tcn_channels_edge, kernel_size=tcn_kernel_size)

        # --- 分类器 ---
        self.mlp_classifier_fc1 = nn.Linear(2 * hidden_channels_node + hidden_channels_edge, mlp_hidden_channels)  
        self.mlp_classifier_fc2 = nn.Linear(mlp_hidden_channels, num_edge_classes)
        
    def forward(self, graphs, seq_len):
        batch_edge_predictions = []
        node_features_seq = []
        edge_features_seq = []

        # 1. 空间特征提取 (Spatial)
        for t in range(seq_len):
            x_t = graphs[t].x
            edge_index_t = graphs[t].edge_index
            edge_attr_t = graphs[t].edge_attr
            
            x_t = torch.relu(self.mlp_node_fc1(x_t))
            x_t = torch.relu(self.mlp_node_fc2(x_t)) 
            
            edge_attr_t = torch.relu(self.mlp_edge_fc1(edge_attr_t))
            edge_attr_t = torch.relu(self.mlp_edge_fc2(edge_attr_t))  

            x_t = self.gat_node_layers1(x_t, edge_index_t, edge_attr=edge_attr_t)
            x_t = self.bn_node1(x_t)
            x_t = torch.relu(x_t)
            x_t = F.dropout(x_t, p=0.2, training=self.training)

            x_t = self.gat_node_layers2(x_t, edge_index_t, edge_attr=edge_attr_t)
            x_t = self.bn_node2(x_t)
            x_t = torch.relu(x_t)
            x_t = F.dropout(x_t, p=0.2, training=self.training)
            
            edge_attr_t = torch.relu(self.edge_fc1(edge_attr_t))
            edge_attr_t = torch.relu(self.edge_fc2(edge_attr_t))  
            
            node_features_seq.append(x_t)
            edge_features_seq.append(edge_attr_t)

        # 2. 时序特征提取 (Temporal)
        # Stack -> (Seq_Len, Nodes, Channels)
        node_features_seq = torch.stack(node_features_seq, dim=0) 
        edge_features_seq = torch.stack(edge_features_seq, dim=0)  
        
        # Permute -> (Nodes, Channels, Seq_Len)
        # TCN requires input: (Batch, Channels, Length)
        node_seq_in = node_features_seq.permute(1, 2, 0)
        edge_seq_in = edge_features_seq.permute(1, 2, 0)

        tcn_out_node = self.tcn_node(node_seq_in)
        tcn_out_edge = self.tcn_edge(edge_seq_in)

        # Permute back -> (Seq_Len, Nodes, Channels)
        tcn_out_node = tcn_out_node.permute(2, 0, 1)
        tcn_out_edge = tcn_out_edge.permute(2, 0, 1)

        # 3. 分类
        for t in range(seq_len):
            edge_index_t = graphs[t].edge_index
            edge_src = edge_index_t[0]
            edge_dst = edge_index_t[1]
            
            node_pair_features = torch.cat([tcn_out_node[t][edge_src], tcn_out_node[t][edge_dst]], dim=1) 
            edge_features = torch.cat([node_pair_features, tcn_out_edge[t]], dim=1)  
            
            edge_preds = torch.relu(self.mlp_classifier_fc1(edge_features)) 
            edge_preds = self.classifier_dropout(edge_preds)
            edge_preds = self.mlp_classifier_fc2(edge_preds) 
            
            batch_edge_predictions.append(edge_preds)
        
        return batch_edge_predictions


class MultiScaleG_TCN(nn.Module):
    def __init__(
        self,
        node_in_channels,
        edge_in_channels,
        hidden_channels_node,
        hidden_channels_edge,
        mlp_hidden_channels,
        num_edge_classes,
        heads=4,
        tcn_kernel_sizes=(3, 5, 7),
    ):
        super(MultiScaleG_TCN, self).__init__()

        self.kernel_sizes = tcn_kernel_sizes
        num_scales = len(tcn_kernel_sizes)

        self.mlp_node_fc1 = nn.Linear(node_in_channels, hidden_channels_node)
        self.mlp_node_fc2 = nn.Linear(hidden_channels_node, hidden_channels_node)

        self.mlp_edge_fc1 = nn.Linear(edge_in_channels, hidden_channels_edge)
        self.mlp_edge_fc2 = nn.Linear(hidden_channels_edge, hidden_channels_edge)

        self.gat_node_layers1 = GATConv(
            hidden_channels_node,
            hidden_channels_node,
            heads=heads,
            concat=False,
            dropout=0.2,
            edge_dim=hidden_channels_edge,
        )
        self.bn_node1 = nn.BatchNorm1d(hidden_channels_node)

        self.gat_node_layers2 = GATConv(
            hidden_channels_node,
            hidden_channels_node,
            heads=heads,
            concat=False,
            dropout=0.2,
            edge_dim=hidden_channels_edge,
        )
        self.bn_node2 = nn.BatchNorm1d(hidden_channels_node)

        self.edge_fc1 = nn.Linear(hidden_channels_edge, hidden_channels_edge)
        self.edge_fc2 = nn.Linear(hidden_channels_edge, hidden_channels_edge)

        tcn_channels_node = [hidden_channels_node, hidden_channels_node]
        tcn_channels_edge = [hidden_channels_edge, hidden_channels_edge]

        self.tcn_node_branches = nn.ModuleList(
            [
                TemporalConvNet(
                    hidden_channels_node,
                    tcn_channels_node,
                    kernel_size=int(k),
                )
                for k in tcn_kernel_sizes
            ]
        )
        self.tcn_edge_branches = nn.ModuleList(
            [
                TemporalConvNet(
                    hidden_channels_edge,
                    tcn_channels_edge,
                    kernel_size=int(k),
                )
                for k in tcn_kernel_sizes
            ]
        )

        total_node_dim = hidden_channels_node * num_scales
        total_edge_dim = hidden_channels_edge * num_scales
        classifier_input_dim = 2 * total_node_dim + total_edge_dim

        self.mlp_classifier_fc1 = nn.Linear(classifier_input_dim, mlp_hidden_channels)
        self.mlp_classifier_fc2 = nn.Linear(mlp_hidden_channels, num_edge_classes)

    def forward(self, graphs, seq_len):
        batch_edge_predictions = []
        node_features_seq = []
        edge_features_seq = []

        for t in range(seq_len):
            x_t = graphs[t].x
            edge_index_t = graphs[t].edge_index
            edge_attr_t = graphs[t].edge_attr

            x_t = torch.relu(self.mlp_node_fc1(x_t))
            x_t = torch.relu(self.mlp_node_fc2(x_t))

            edge_attr_t = torch.relu(self.mlp_edge_fc1(edge_attr_t))
            edge_attr_t = torch.relu(self.mlp_edge_fc2(edge_attr_t))

            x_t = self.gat_node_layers1(x_t, edge_index_t, edge_attr=edge_attr_t)
            x_t = self.bn_node1(x_t)
            x_t = torch.relu(x_t)
            x_t = F.dropout(x_t, p=0.1, training=self.training)

            x_t = self.gat_node_layers2(x_t, edge_index_t, edge_attr=edge_attr_t)
            x_t = self.bn_node2(x_t)
            x_t = torch.relu(x_t)
            x_t = F.dropout(x_t, p=0.1, training=self.training)

            edge_attr_t = torch.relu(self.edge_fc1(edge_attr_t))
            edge_attr_t = torch.relu(self.edge_fc2(edge_attr_t))

            node_features_seq.append(x_t)
            edge_features_seq.append(edge_attr_t)

        node_features_seq = torch.stack(node_features_seq, dim=0)
        edge_features_seq = torch.stack(edge_features_seq, dim=0)

        node_seq_in = node_features_seq.permute(1, 2, 0)
        edge_seq_in = edge_features_seq.permute(1, 2, 0)

        node_branch_outs = [branch(node_seq_in) for branch in self.tcn_node_branches]
        edge_branch_outs = [branch(edge_seq_in) for branch in self.tcn_edge_branches]

        tcn_out_node = torch.cat(node_branch_outs, dim=1)
        tcn_out_edge = torch.cat(edge_branch_outs, dim=1)

        tcn_out_node = tcn_out_node.permute(2, 0, 1)
        tcn_out_edge = tcn_out_edge.permute(2, 0, 1)

        for t in range(seq_len):
            edge_index_t = graphs[t].edge_index
            edge_src = edge_index_t[0]
            edge_dst = edge_index_t[1]

            node_pair_features = torch.cat(
                [tcn_out_node[t][edge_src], tcn_out_node[t][edge_dst]],
                dim=1,
            )
            edge_features = torch.cat([node_pair_features, tcn_out_edge[t]], dim=1)

            edge_preds = torch.relu(self.mlp_classifier_fc1(edge_features))
            edge_preds = self.mlp_classifier_fc2(edge_preds)
            batch_edge_predictions.append(edge_preds)

        return batch_edge_predictions
