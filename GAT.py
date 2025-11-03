import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8):
        super(GATClassifier, self).__init__()

        # 图注意力网络层
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.2)

        # 全连接层，用于分类
        self.fc1 = nn.Linear(4096, 512)  # GAT输出后接全连接层
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, out_channels)  # 输出类别数，设定为out_channels

    def forward(self, data):
        # 从data中提取节点特征、边索引和边特征
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr  # 获取边权重
        batch = data.batch  # 获取每个节点属于哪个图的信息

        # 图注意力网络前向传播
        x = F.relu(self.gat1(x, edge_index, edge_attr))  # 第一层GAT
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gat2(x, edge_index, edge_attr))  # 第二层GAT
        x = x.view(-1, 8 * x.size(-1))

        # 全连接层分类
        x = F.relu(self.fc1(x))  # 第一层全连接
        x = F.relu(self.fc2(x))  # 第二层全连接
        x = self.fc3(x)  # 输出层

        # 直接返回类别得分（logits）
        return x