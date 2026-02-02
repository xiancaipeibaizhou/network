import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv
from gt import GraphTransformerLayer
# 替换LSTM的dilated_inception模块 - 采用MTGNN中的双膨胀Inception层结构
# 将gcn替换为graphtransformer
class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        # 滤波器层 - 后接tanh激活
        self.filter_tconv = nn.ModuleList()
        # 门控层 - 后接sigmoid激活
        self.gate_tconv = nn.ModuleList()
        
        self.kernel_set = [2, 3, 6, 7]
        cout_per_kernel = cout // len(self.kernel_set)
        
        # 为滤波器层和门控层创建相同的卷积核结构
        for kern in self.kernel_set:
            # 使用适当的padding保持时间维度不变
            padding = (kern - 1) * dilation_factor // 2
            
            # 滤波器层卷积
            self.filter_tconv.append(
                nn.Conv2d(cin, cout_per_kernel, (1, kern), 
                         dilation=(1, dilation_factor), 
                         padding=(0, padding))
            )
            
            # 门控层卷积
            self.gate_tconv.append(
                nn.Conv2d(cin, cout_per_kernel, (1, kern), 
                         dilation=(1, dilation_factor), 
                         padding=(0, padding))
            )

    def forward(self, input):
        # input shape: [batch_size, channels, time_steps, num_features]
        filter_outputs = []
        gate_outputs = []
        
        # 分别计算滤波器和门控的输出
        for i in range(len(self.kernel_set)):
            # 滤波器路径 - 后接tanh激活
            filter_x = torch.tanh(self.filter_tconv[i](input))
            # 门控路径 - 后接sigmoid激活
            gate_x = torch.sigmoid(self.gate_tconv[i](input))
            
            filter_outputs.append(filter_x)
            gate_outputs.append(gate_x)
        
        # 确保所有输出的时间维度相同
        min_time_dim = min([xx.size(3) for xx in filter_outputs])
        
        # 应用门控机制并连接所有路径
        x = []
        for i in range(len(self.kernel_set)):
            # 调整时间维度
            filter_x = filter_outputs[i][..., :min_time_dim]
            gate_x = gate_outputs[i][..., :min_time_dim]
            # 门控机制：filter * gate
            gated_x = filter_x * gate_x
            x.append(gated_x)
        
        # 在通道维度上连接所有路径的输出
        x = torch.cat(x, dim=1)
        return x
##################################################################################################
# 修改后的ROEN网络框架
class ROEN(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels_node, 
                 hidden_channels_edge, mlp_hidden_channels, num_edge_classes, 
                 seq_len=12):
        super(ROEN, self).__init__()
        
        self.seq_len = seq_len

        # MLP layers for processing node features (隐式)
        self.mlp_node_fc1 = nn.Linear(node_in_channels, hidden_channels_node)
        self.mlp_node_fc2 = nn.Linear(hidden_channels_node, hidden_channels_node)
        
        # MLP layers for processing edge features (显式)
        self.mlp_edge_fc1 = nn.Linear(edge_in_channels, hidden_channels_edge)
        self.mlp_edge_fc2 = nn.Linear(hidden_channels_edge, hidden_channels_edge)

        # Graph Transformer layers for processing node features
        self.gt_node_layers1 = GraphTransformerLayer(in_dim=hidden_channels_node, hidden_dim=hidden_channels_node, edge_dim=hidden_channels_edge, n_heads=4, dropout=0.1)
        self.gt_node_layers2 = GraphTransformerLayer(in_dim=hidden_channels_node, hidden_dim=hidden_channels_node, edge_dim=hidden_channels_edge, n_heads=4, dropout=0.1)
        
        # Fully connected layers for processing edge features
        self.edge_fc1 = nn.Linear(hidden_channels_edge, hidden_channels_edge)
        self.edge_fc2 = nn.Linear(hidden_channels_edge, hidden_channels_edge)

        # 替换LSTM为dilated_inception模块
        self.dilated_inception_node = dilated_inception(
            cin=1,  # 输入通道数，我们将调整维度
            cout=hidden_channels_node, 
            dilation_factor=2
        )
        
        self.dilated_inception_edge = dilated_inception(
            cin=1,  # 输入通道数，我们将调整维度
            cout=hidden_channels_edge,
            dilation_factor=2
        )

        # 用于调整维度的卷积层
        self.node_channel_adjust = nn.Conv2d(hidden_channels_node, 1, 1)
        self.edge_channel_adjust = nn.Conv2d(hidden_channels_edge, 1, 1)

        # Final MLP for edge classification
        self.mlp_classifier_fc1 = nn.Linear(2 * hidden_channels_node + hidden_channels_edge, mlp_hidden_channels)  
        self.mlp_classifier_fc2 = nn.Linear(mlp_hidden_channels, num_edge_classes)

    def forward(self, graphs, seq_len=None):
        if seq_len is None:
            seq_len = self.seq_len
            
        batch_edge_predictions = []
        
        node_features_seq = []
        edge_features_seq = []

        # 第一步：收集所有时间步的特征
        for t in range(seq_len):
            x_t = graphs[t].x  # Node features: [num_nodes, node_in_channels]
            edge_index_t = graphs[t].edge_index  # Edge index
            edge_attr_t = graphs[t].edge_attr  # Edge attributes: [num_edges, edge_in_channels]
            
            # Process node features through MLP
            x_t = torch.relu(self.mlp_node_fc1(x_t))
            x_t = torch.relu(self.mlp_node_fc2(x_t)) 
            
            # Process edge features through MLP
            edge_attr_t = torch.relu(self.mlp_edge_fc1(edge_attr_t))
            edge_attr_t = torch.relu(self.mlp_edge_fc2(edge_attr_t))  

            # Process node features through Graph Transformer
            x_t, edge_attr_t = self.gt_node_layers1(x_t, edge_index_t, edge_attr_t)
            x_t, edge_attr_t = self.gt_node_layers2(x_t, edge_index_t, edge_attr_t)
            # 确保edge_attr_t不为None
            if edge_attr_t is None:
                edge_attr_t = graphs[t].edge_attr  # 使用原始边属性
            
            # Process edge features through fully connected layers
            edge_attr_t = torch.relu(self.edge_fc1(edge_attr_t))
            edge_attr_t = torch.relu(self.edge_fc2(edge_attr_t))  
            
            # Save node and edge features for each time step
            node_features_seq.append(x_t.unsqueeze(0))  # [1, num_nodes, hidden_channels_node]
            edge_features_seq.append(edge_attr_t.unsqueeze(0))  # [1, num_edges, hidden_channels_edge]

        # 第二步：准备dilated_inception的输入
        # Node features: [batch_size, channels, time_steps, num_features]
        node_features_seq = torch.cat(node_features_seq, dim=0)  # [seq_len, num_nodes, hidden_channels_node]
        node_features_seq = node_features_seq.permute(1, 2, 0).unsqueeze(1)  # [num_nodes, 1, hidden_channels_node, seq_len]
        
        edge_features_seq = torch.cat(edge_features_seq, dim=0)  # [seq_len, num_edges, hidden_channels_edge]
        edge_features_seq = edge_features_seq.permute(1, 2, 0).unsqueeze(1)  # [num_edges, 1, hidden_channels_edge, seq_len]

        # 第三步：通过dilated_inception处理时间序列
        dilated_out_node = self.dilated_inception_node(node_features_seq)  # [num_nodes, hidden_channels_node, hidden_channels_node, seq_len]
        dilated_out_edge = self.dilated_inception_edge(edge_features_seq)  # [num_edges, hidden_channels_edge, hidden_channels_edge, seq_len]
        
        # 第四步：调整输出维度
        # 使用1x1卷积调整通道维度
        dilated_out_node = self.node_channel_adjust(dilated_out_node)  # [num_nodes, 1, hidden_channels_node, seq_len]
        dilated_out_edge = self.edge_channel_adjust(dilated_out_edge)  # [num_edges, 1, hidden_channels_edge, seq_len]
        
        # 恢复原始维度格式: [seq_len, num_nodes/edges, hidden_channels]
        dilated_out_node = dilated_out_node.squeeze(1).permute(2, 0, 1)  # [seq_len, num_nodes, hidden_channels_node]
        dilated_out_edge = dilated_out_edge.squeeze(1).permute(2, 0, 1)  # [seq_len, num_edges, hidden_channels_edge]
        
        # 第五步：分类边缘（与原始代码相同）
        for t in range(seq_len):
            edge_index_t = graphs[t].edge_index
            
            # Concatenate source node and target node features
            edge_src = edge_index_t[0]  # Source node index of edges
            edge_dst = edge_index_t[1]  # Target node index of edges
            node_pair_features = torch.cat([dilated_out_node[t][edge_src], dilated_out_node[t][edge_dst]], dim=1) 
            
            # Concatenate node pair features and dilated output edge features
            edge_features = torch.cat([node_pair_features, dilated_out_edge[t]], dim=1)  
            
            # Use the final MLP for edge classification
            edge_preds = torch.relu(self.mlp_classifier_fc1(edge_features)) 
            edge_preds = self.mlp_classifier_fc2(edge_preds) 
            
            batch_edge_predictions.append(edge_preds)
        
        return batch_edge_predictions