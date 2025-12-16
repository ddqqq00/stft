import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

# class GATClassifier(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8):
#         super(GATClassifier, self).__init__()

#         # 图注意力网络层
#         # gat1 输出维度: hidden_channels * num_heads (例如 512 * 8 = 4096)
#         self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
#         # gat2 输出维度: hidden_channels * 1 (例如 512)
#         self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.2)

#         # 全连接层
#         # 输入维度: 512 (来自 gat2 的输出)
#         self.fc2 = nn.Linear(hidden_channels, 128)
#         self.fc3 = nn.Linear(128, out_channels)  # 输出类别数

#     def forward(self, data, return_embeds=False):
#         x, edge_index = data.x, data.edge_index
#         edge_attr = data.edge_attr
#         batch = data.batch

#         # GAT 层
#         x = F.relu(self.gat1(x, edge_index, edge_attr))
#         x = F.dropout(x, p=0.2, training=self.training)
#         x = F.relu(self.gat2(x, edge_index, edge_attr))

#         # 全局池化
#         x = global_mean_pool(x, batch)

#         # --- 关键修改开始 ---
        
#         # 1. 先经过线性层 (不加 ReLU)
#         # x_linear: [Batch_Size, 128] - 包含正负值，信息最丰富
#         x_linear = self.fc2(x)
        
#         # 2. 如果需要可视化，保存这个 "线性特征"
#         if return_embeds:
#             features = x_linear 
#             # 可选：在这里做一下 L2 归一化，会让 t-SNE 的角度区分更明显
#             # features = F.normalize(features, p=2, dim=1) 

#         # 3. 再进行激活 (用于后续分类)
#         x_activated = F.relu(x_linear)
        
#         # 4. 最终分类
#         logits = self.fc3(x_activated)
        
#         # --- 关键修改结束 ---

#         if return_embeds:
#             return logits, features
#         else:
#             return logits
        

class GATClassifier(nn.Module):
    """
    修改版: 这是一个 '伪装' 成 GAT 的 MLP 模型。
    用于消融实验，证明图结构的重要性。
    它忽略 edge_index，直接拼接节点特征。
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8):
        super(GATClassifier, self).__init__()

        # --- MLP 特定配置 ---
        # 关键假设: 假设每个图固定有 8 个节点 (对应 Watkins 数据集)
        # 如果您的 ShipsEar 数据集节点数不同，请务必修改这里！
        self.num_nodes = 8 
        
        # MLP 输入维度: 所有节点特征拼接 (Num_Nodes * In_Channels)
        flat_input_dim = self.num_nodes * in_channels
        
        # GAT 中间层输出维度: hidden_channels * num_heads (例如 512 * 8 = 4096)
        # 我们让 MLP 的隐层也达到这个规模，以保证参数量在一个量级
        mlp_hidden_dim = hidden_channels * num_heads

        # --- 定义 MLP 层 ---
        # 替代 gat1 和 gat2
        self.mlp_layers = nn.Sequential(
            nn.Linear(flat_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(mlp_hidden_dim, hidden_channels), # 模拟全局池化后的维度 (512)
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # --- 后续分类层 (保持与 GAT 一致) ---
        # 输入维度: 512
        self.fc2 = nn.Linear(hidden_channels, 128)
        self.fc3 = nn.Linear(128, out_channels)

    def forward(self, data, return_embeds=False):
        x, batch = data.x, data.batch
        
        # 1. 获取 Batch Size
        # 通过 batch 向量的最大值来推断 batch_size
        batch_size = batch.max().item() + 1
        
        # 2. 展平 (模拟忽略图结构)
        # [Total_Nodes, C] -> [Batch_Size, Num_Nodes * C]
        try:
            # 强制将节点特征拉平。如果图节点数不固定，这里会报错
            x_flat = x.view(batch_size, -1)
        except RuntimeError:
             raise RuntimeError(f"MLP 消融实验报错: 无法将输入 Reshape 为 [B, -1]。"
                                f"当前 batch_size={batch_size}, x shape={x.shape}。"
                                f"请确保您的图数据中每个图的节点数是固定的 (代码假设为 {self.num_nodes})。")

        # 3. 通过 MLP 提取特征
        # 输出: [Batch_Size, 512]
        x = self.mlp_layers(x_flat)

        # --- 关键修改 (与 GAT 保持一致的特征提取逻辑) ---
        
        # 4. 先经过线性层 (不加 ReLU)
        # x_linear: [Batch_Size, 128]
        x_linear = self.fc2(x)
        
        # 5. 如果需要可视化，保存这个 "线性特征"
        if return_embeds:
            features = x_linear 

        # 6. 再进行激活 (用于后续分类)
        x_activated = F.relu(x_linear)
        
        # 7. 最终分类
        logits = self.fc3(x_activated)
        
        if return_embeds:
            return logits, features
        else:
            return logits