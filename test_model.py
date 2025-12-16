import torch
import torch.nn as nn
import torch.nn.functional as F
from complexNN.nn import cConv2d, cAvgPool2d, cBatchNorm2d, cLeakyRelu, cDropout, cLinear
# 确保导入 GATConv 和 global_mean_pool
from torch_geometric.nn import GATConv, global_mean_pool

# --- 辅助模块 (Complex Attention) 保持不变 ---
def complex_sigmoid(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z.real) + 1j * torch.sigmoid(z.imag)


# class ComplexChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=4):
#         super().__init__()
#         self.shared_mlp = nn.Sequential(
#             cLinear(in_channels, in_channels // reduction_ratio), cLeakyRelu(),
#             cLinear(in_channels // reduction_ratio, in_channels)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         avg_pool_out = F.adaptive_avg_pool2d(x, 1)
#         max_pool_out = torch.complex(F.adaptive_max_pool2d(x.real, 1), F.adaptive_max_pool2d(x.imag, 1))
#         avg_out_mlp = self.shared_mlp(avg_pool_out.squeeze(-1).squeeze(-1))
#         max_out_mlp = self.shared_mlp(max_pool_out.squeeze(-1).squeeze(-1))
#         channel_gate = complex_sigmoid(avg_out_mlp + max_out_mlp).unsqueeze(-1).unsqueeze(-1)
#         return x * channel_gate


# class ComplexSpatialAttention(nn.Module):
#     def __init__(self, kernel_size=3):
#         super().__init__()
#         self.conv = nn.Conv2d(4, 2, kernel_size, padding=kernel_size // 2, bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         avg_pool_real = torch.mean(x.real, dim=1, keepdim=True)
#         avg_pool_imag = torch.mean(x.imag, dim=1, keepdim=True)
#         max_pool_real, _ = torch.max(x.real, dim=1, keepdim=True)
#         max_pool_imag, _ = torch.max(x.imag, dim=1, keepdim=True)
#         pooled_features = torch.cat([avg_pool_real, avg_pool_imag, max_pool_real, max_pool_imag], dim=1)
#         gate_precursors = self.conv(pooled_features)
#         pre_gate_real, pre_gate_imag = gate_precursors.chunk(2, dim=1)
#         pre_gate_complex = torch.complex(pre_gate_real, pre_gate_imag)
#         spatial_gate = complex_sigmoid(pre_gate_complex)
#         return x * spatial_gate


# class ComplexCBAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
#         super().__init__()
#         self.channel_attention = ComplexChannelAttention(in_channels, reduction_ratio)
#         self.spatial_attention = ComplexSpatialAttention(spatial_kernel_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.channel_attention(x)
#         x = self.spatial_attention(x)
#         return x


# class ComplexCNNFeatureExtractor(nn.Module):
#     """
#     修改: 
#     - 移除了内部的 FC 降维层。
#     - Forward 方法返回 "中期" 和 "晚期" 的高维复数特征。
#     """
#     def __init__(self, dropout_rate=0.5):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             cConv2d(1, 4, kernel_size=3, stride=1, padding=(1, 1)), 
#             cBatchNorm2d(4), 
#             cLeakyRelu(),
#             # ComplexCBAM(in_channels=4), # 假设没有CBAM以简化
#             cAvgPool2d(kernel_size=(2, 2), stride=(2, 2))
#         )
#         self.conv2 = nn.Sequential(
#             cConv2d(4, 8, kernel_size=3, stride=1, padding=(1, 1)), 
#             cBatchNorm2d(8), 
#             cLeakyRelu(),
#             # ComplexCBAM(in_channels=8), # 假设没有CBAM以简化
#             cAvgPool2d(kernel_size=(2, 2), stride=(2, 2))
#         )

#     def forward(self, x):
#         cnn_mid_complex = self.conv1(x)
#         cnn_late_complex = self.conv2(cnn_mid_complex)
        
#         # 展平特征图
#         cnn_mid_flat = cnn_mid_complex.view(cnn_mid_complex.size(0), -1)
#         cnn_late_flat = cnn_late_complex.view(cnn_late_complex.size(0), -1)
        
#         # 返回: 中期(高维, complex), 晚期(高维, complex)
#         return cnn_mid_flat, cnn_late_flat

# # ===================================================================
# # 2. GAT 特征提取器
# # ===================================================================
# class GATFeatureExtractor(nn.Module):
#     """
#     修改:
#     - 移除了所有内部的 FC 降维层。
#     - Forward 方法改为使用 .view() (假设批次中节点数固定)。
#     - 返回 "中期" (gat1+pooled) 和 "晚期" (gat2+view)。
#     """
#     def __init__(self, in_channels, hidden_channels, num_heads=8, dropout_rate=0.5):
#         super().__init__()
#         self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
#         self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.2)
        
#         self.hidden_channels = hidden_channels
#         self.num_heads = num_heads

#     def forward(self, data):
#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

#         # --- 中期特征 ---
#         x_mid_nodes = F.relu(self.gat1(x, edge_index, edge_attr))
#         gat_mid_pooled = global_mean_pool(x_mid_nodes, batch) # [B, 4096]
        
#         # --- 晚期特征 ---
#         x_mid_dropout = F.dropout(x_mid_nodes, p=0.2, training=self.training)
#         x_late_nodes = self.gat2(x_mid_dropout, edge_index, edge_attr)
        
#         try:
#             gat_late_view = x_late_nodes.view(-1, self.num_heads * self.hidden_channels) # [B, 4096]
#         except RuntimeError as e:
#             print(f"GATFeatureExtractor .view() failed. Batch size > 1 or nodes per graph not fixed.")
#             raise e
        
#         return gat_mid_pooled, gat_late_view

# # ===================================================================
# # 3. MLP 特征提取器 (新增，用于消融实验)
# # ===================================================================
# class MLPFeatureExtractor(nn.Module):
#     """
#     GATFeatureExtractor 的对比实验版本。
#     它会忽略图结构，将所有节点特征拼接并通过一个MLP。
    
#     警告：这假设一个批次中所有图的节点数都是固定的。
#     """
#     def __init__(self, in_channels, num_nodes, hidden_dim, out_dim, dropout_rate=0.5):
#         super().__init__()
        
#         flat_dim = num_nodes * in_channels 
        
#         self.mlp = nn.Sequential(
#             nn.Linear(flat_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(hidden_dim, out_dim),
#         )
        
#         # 为了与GATFeatureExtractor的返回(mid, late)保持一致
#         self.mlp_mid = nn.Sequential(
#             nn.Linear(flat_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(hidden_dim, out_dim * 8) # 模拟 gat1 的 4096 维
#         )

#     def forward(self, data):
#         x, batch = data.x, data.batch
#         batch_size = batch.max().item() + 1
        
#         try:
#             flat_x = x.view(batch_size, -1) 
#         except RuntimeError as e:
#             print(f"MLPFeatureExtractor .view() failed. Nodes per graph not fixed.")
#             raise e
        
#         # 模拟 "中期" 和 "晚期"
#         mlp_mid_output = self.mlp_mid(flat_x)
#         mlp_late_output = self.mlp(flat_x)
        
#         return mlp_mid_output, mlp_late_output

# # ===================================================================
# # 4. 统一的融合模型 (已简化)
# # ===================================================================
# class DualPathFusionModel(nn.Module):
#     """
#     修改: 
#     - 移除了所有对比学习和多阶段逻辑。
#     - 强制 fusion_mode = 'none'。
#     - 允许 feature_extractor_b (原GAT) 是一个可替换的模块 (GAT 或 MLP)。
#     """

#     def __init__(self, cnn_extractor, gat_or_mlp_extractor, 
#                  cnn_late_dim, gat_or_mlp_late_dim, 
#                  num_classes, dropout_rate=0.2):
        
#         super().__init__()
        
#         self.contrastive_mode = 'none' # 强制为 'none'
        
#         self.cnn_feature_extractor = cnn_extractor
#         self.gat_or_mlp_extractor = gat_or_mlp_extractor

#         # (例如: 28672 + 4096 = 32768)
#         fusion_input_dim = cnn_late_dim + gat_or_mlp_late_dim
        
#         self.global_fusion_head = nn.Sequential(
#             nn.Linear(fusion_input_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, stft_data, graph_data):
        
#         # --- 1. 特征提取 ---
#         # (cnn_extractor 返回 mid, late)
#         _, cnn_late_flat_complex = self.cnn_feature_extractor(stft_data)
        
#         # (gat_or_mlp_extractor 返回 mid, late)
#         _, gat_or_mlp_late = self.gat_or_mlp_extractor(graph_data)
        
#         # --- 2. 特征处理 (拆分复数) ---
#         cnn_late_real = torch.cat([cnn_late_flat_complex.real, cnn_late_flat_complex.imag], dim=1)
        
#         # --- 3. 全局融合 ---
#         global_fused_features = torch.cat((
#             cnn_late_real,
#             gat_or_mlp_late,
#         ), dim=1)

#         # 4. 最终分类
#         logits = self.global_fusion_head(global_fused_features)
        
#         return logits

#以上是用于shipears数据集的GAT、MLP消融实验
#以下用于watkins1s数据集

class ComplexCNNFeatureExtractor(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            cConv2d(1, 8, kernel_size=3, stride=1, padding=(1, 1)),
            cBatchNorm2d(8),
            cLeakyRelu(),
            cAvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.conv2 = nn.Sequential(
            cConv2d(8, 16, kernel_size=3, stride=1, padding=(1, 1)),
            cBatchNorm2d(16),
            cLeakyRelu(),
            cAvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.cnn_fc1 = nn.Sequential(
            cLinear(39936, 512), 
            cLeakyRelu(),
            cDropout(p=dropout_rate)
        )
        self.cnn_fc2 = nn.Sequential(
            cLinear(512, 256), 
            cLeakyRelu(),
            cDropout(p=dropout_rate)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_flat = x.view(x.size(0), -1)
        
        # 降维
        x_fc = self.cnn_fc1(x_flat)
        x_fc = self.cnn_fc2(x_fc)
        
        # 计算幅度并归一化
        x_abs = torch.abs(x_fc)
        x_norm = F.normalize(x_abs, p=2, dim=1)
        
        return None, x_norm # 返回 (None, late_feature) 以兼容接口

# ===================================================================
# 2. GAT 特征提取器
# ===================================================================
class GATFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=8, dropout_rate=0.5):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.2)
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)
        
        # 使用 view 展平 (假设节点数固定)
        try:
            x_view = x.view(-1, self.num_heads * self.hidden_channels)
        except RuntimeError:
            # 如果 batch size 为 1 或节点数不匹配，尝试 global_mean_pool 作为 fallback
            # 但为了消融实验的公平性，最好保证节点数固定
            x_view = global_mean_pool(x, batch) 
            
        return None, x_view

# ===================================================================
# 3. MLP 特征提取器 (用于消融实验)
# ===================================================================
class MLPFeatureExtractor(nn.Module):
    """
    GAT 的 MLP 替代版。忽略图结构，直接拼接节点特征。
    """
    def __init__(self, in_channels, num_nodes, hidden_dim, out_dim, dropout_rate=0.5):
        super().__init__()
        self.flat_dim = num_nodes * in_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, data):
        x, batch = data.x, data.batch
        batch_size = batch.max().item() + 1
        
        # 强制展平 [B * N, C] -> [B, N * C]
        try:
            x_flat = x.view(batch_size, -1)
        except RuntimeError:
             raise RuntimeError(f"MLPFeatureExtractor: 无法将输入 reshape 为 [batch_size, -1]。"
                                f"请确保每个图的节点数是固定的。当前 shape: {x.shape}")

        out = self.mlp(x_flat)
        return None, out

# ===================================================================
# 4. 统一融合模型 (修改版 - 支持外部注入)
# ===================================================================
class DualPathFusionModel_watkins(nn.Module):
    """
    修改版: 
    - __init__ 接受已经实例化好的特征提取器对象。
    - 这样可以灵活地传入 GAT 或 MLP 提取器。
    """
    def __init__(self, cnn_extractor, gat_or_mlp_extractor, 
                 cnn_late_dim, gat_or_mlp_late_dim, 
                 num_classes, dropout_rate=0.5):
        super().__init__()
        
        self.cnn_feature_extractor = cnn_extractor
        self.gat_or_mlp_extractor = gat_or_mlp_extractor
        
        # 拼接后的维度
        fusion_input_dim = cnn_late_dim + gat_or_mlp_late_dim
        
        self.global_fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, stft_data, graph_data):
        # 提取特征
        _, cnn_out = self.cnn_feature_extractor(stft_data)
        _, gat_mlp_out = self.gat_or_mlp_extractor(graph_data)
        
        # 拼接
        fused = torch.cat([cnn_out, gat_mlp_out], dim=1)
        
        # 分类
        logits = self.global_fusion_head(fused)
        
        return logits