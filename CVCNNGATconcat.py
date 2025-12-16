import torch
import torch.nn as nn
import torch.nn.functional as F
from complexNN.nn import cConv2d, cAvgPool2d, cBatchNorm2d, cLeakyRelu, cDropout, cLinear
# 确保导入 GATConv 和 global_mean_pool
from torch_geometric.nn import GATConv, global_mean_pool


# --- 辅助模块 (Complex Attention) 保持不变 ---
def complex_sigmoid(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z.real) + 1j * torch.sigmoid(z.imag)


class ComplexChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            cLinear(in_channels, in_channels // reduction_ratio), cLeakyRelu(),
            cLinear(in_channels // reduction_ratio, in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool_out = F.adaptive_avg_pool2d(x, 1)
        max_pool_out = torch.complex(F.adaptive_max_pool2d(x.real, 1), F.adaptive_max_pool2d(x.imag, 1))
        avg_out_mlp = self.shared_mlp(avg_pool_out.squeeze(-1).squeeze(-1))
        max_out_mlp = self.shared_mlp(max_pool_out.squeeze(-1).squeeze(-1))
        channel_gate = complex_sigmoid(avg_out_mlp + max_out_mlp).unsqueeze(-1).unsqueeze(-1)
        return x * channel_gate


class ComplexSpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(4, 2, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool_real = torch.mean(x.real, dim=1, keepdim=True)
        avg_pool_imag = torch.mean(x.imag, dim=1, keepdim=True)
        max_pool_real, _ = torch.max(x.real, dim=1, keepdim=True)
        max_pool_imag, _ = torch.max(x.imag, dim=1, keepdim=True)
        pooled_features = torch.cat([avg_pool_real, avg_pool_imag, max_pool_real, max_pool_imag], dim=1)
        gate_precursors = self.conv(pooled_features)
        pre_gate_real, pre_gate_imag = gate_precursors.chunk(2, dim=1)
        pre_gate_complex = torch.complex(pre_gate_real, pre_gate_imag)
        spatial_gate = complex_sigmoid(pre_gate_complex)
        return x * spatial_gate


class ComplexCBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ComplexChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = ComplexSpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# =================================================================================
# --- 修改后的特征提取器 (MODIFIED Feature Extractors) ---
# =================================================================================

class ComplexCNNFeatureExtractor(nn.Module):
    """
    修改: 复值CNN，只进行卷积特征提取，不包含全连接层。
    """

    def __init__(self):  # 不再需要 dropout_rate 参数
        super().__init__()
        self.conv1 = nn.Sequential(
            cConv2d(1, 4, kernel_size=3, stride=1, padding=(1, 1)), cBatchNorm2d(4), cLeakyRelu(),
            ComplexCBAM(in_channels=4),
            cAvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.conv2 = nn.Sequential(
            cConv2d(4, 8, kernel_size=3, stride=1, padding=(1, 1)), cBatchNorm2d(8), cLeakyRelu(),
            ComplexCBAM(in_channels=8), 
            cAvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        # --- 移除了 self.fc1 和 self.fc2 ---

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 展平卷积特征图
        x = x.view(x.size(0), -1)
        # --- 移除了 fc1 和 fc2 的前向传播 ---
        # 直接返回高维复数特征
        return x



class GATFeatureExtractor(nn.Module):
    """
    修改: GAT，只进行图节点特征聚合，不包含全连接层。
    """

    def __init__(self, in_channels, hidden_channels, num_heads=8):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
        # 输出维度为 hidden_channels
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.2)
        # --- 移除了 self.fc1 和 self.fc2 ---

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index, edge_attr)

        # 关键修改: 使用全局平均池化得到图级别的表示
        x = global_mean_pool(x, batch) # 512
        #x = x.view(-1, x.size(-1)) # 4096

        # --- 移除了 fc1 和 fc2 的前向传播 ---
        return x


# ===============================================================================
# --- 修改后的最终融合模型 (MODIFIED Final Fusion Model) ---
# ===============================================================================

# class DualPathFusionModel(nn.Module):
#     """
#     新增: 融合 ComplexCNN 和 GAT 的双路模型。
#     新策略: 融合原始高维特征，然后通过共享分类头进行分类。
#     """

#     def __init__(self, gat_in_channels, gat_hidden_channels, num_classes, dropout_rate=0.2):
#         super().__init__()
#         self.cnn_feature_extractor = ComplexCNNFeatureExtractor()
#         self.gat_feature_extractor = GATFeatureExtractor(in_channels=gat_in_channels,
#                                                          hidden_channels=gat_hidden_channels)

#         # --- 重新计算融合输入的维度 ---
#         # 假设的CNN展平后维度 (需要根据您的数据确认，这里使用之前代码中的硬编码值)
#         cnn_flattened_dim = 14336
#         # 拆分实部和虚部后，维度翻倍
#         cnn_real_dim = cnn_flattened_dim * 2
#         # GAT输出维度
#         gat_output_dim = gat_hidden_channels  # e.g., 512


#         fusion_input_dim = cnn_real_dim + gat_output_dim  # 28672 + 512 = 29184

#         self.fusion_gate = nn.Linear(gat_output_dim, cnn_real_dim)

#         # 定义一个新的、更强大的分类头来处理高维融合特征
#         self.fusion_head = nn.Sequential(
#             nn.Linear(fusion_input_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, stft_data, graph_data):
#         # CNN通路: 得到高维复数特征
#         cnn_features_complex = self.cnn_feature_extractor(stft_data)

#         # GAT通路: 得到图嵌入特征
#         gat_features = self.gat_feature_extractor(graph_data)

#         # --- 新的融合策略 ---
#         # 1. 将复数特征拆分为实部和虚部
#         cnn_features_real = torch.cat([cnn_features_complex.real, cnn_features_complex.imag], dim=1)

#         # 2. 拼接两个通路的高维实数特征
#         fused_features = torch.cat((cnn_features_real, gat_features), dim=1)
#         #fused_features = torch.cat((gated_cnn_features, gat_features), dim=1)

#         # 3. 通过共享分类头进行分类
#         output = self.fusion_head(fused_features)
#         return output


# class DualPathFusionModel(nn.Module):
#     """
#     修改: 融合 ComplexCNN 和 GAT 的双路模型。
#     新策略: 
#     1. (主路径) 特征直接拼接 (Concat) 进行分类。
#     2. (辅助路径) 添加对比学习投影头，用于计算引导损失。
#     """

#     def __init__(self, gat_in_channels, gat_hidden_channels, num_classes, dropout_rate=0.2):
#         super().__init__()
#         self.cnn_feature_extractor = ComplexCNNFeatureExtractor()
#         self.gat_feature_extractor = GATFeatureExtractor(in_channels=gat_in_channels,
#                                                        hidden_channels=gat_hidden_channels)

#         # --- 1. 主分类路径 (Concat Baseline) ---
        
#         # CNN展平后维度
#         cnn_flattened_dim = 14336
#         cnn_real_dim = cnn_flattened_dim * 2  # 28672
#         # GAT输出维度
#         gat_output_dim = gat_hidden_channels  # e.g., 512
        
#         # Concat 路径的输入维度
#         fusion_input_dim = cnn_real_dim + gat_output_dim  # 28672 + 512 = 29184

#         # 定义分类头
#         self.fusion_head = nn.Sequential(
#             nn.Linear(fusion_input_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(256, num_classes)
#         )

#         # --- 2. 辅助对比路径 ---
#         # (这部分仅用于计算对比损失，不参与最终分类)
        
#         contrastive_dim = 256  # 定义一个合理的、低维的对比空间

#         # 新增: CNN的对比投影头
#         self.cnn_contrast_proj = nn.Sequential(
#             nn.Linear(cnn_real_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, contrastive_dim)
#         )
        
#         # 新增: GAT的对比投影头
#         self.gat_contrast_proj = nn.Sequential(
#             nn.Linear(gat_output_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, contrastive_dim)
#         )

#     def forward(self, stft_data, graph_data):
#         # --- 1. 特征提取 ---
#         # CNN通路: 得到高维复数特征
#         cnn_features_complex = self.cnn_feature_extractor(stft_data)
#         # GAT通路: 得到图嵌入特征
#         gat_features = self.gat_feature_extractor(graph_data)

#         # 将复数特征拆分为实部和虚部
#         cnn_features_real = torch.cat([cnn_features_complex.real, cnn_features_complex.imag], dim=1)

#         # --- 2. 主分类路径 (Concat) ---
#         # 拼接两个通路的高维实数特征
#         fused_features = torch.cat((cnn_features_real, gat_features), dim=1)
#         # 通过共享分类头进行分类
#         logits = self.fusion_head(fused_features)

#         # --- 3. 辅助对比路径 ---
#         # (我们使用 .detach() 来阻止对比损失的梯度流入分类头，
#         #  或者您可以移除 .detach() 来让对比损失也训练特征提取器)
        
#         # 投影到对比空间
#         cnn_contrast_vec = self.cnn_contrast_proj(cnn_features_real.detach())
#         gat_contrast_vec = self.gat_contrast_proj(gat_features.detach())

#         # --- 4. 返回所有输出 ---
#         return logits, cnn_contrast_vec, gat_contrast_vec


# class DualPathFusionModel(nn.Module):
#     """
#     修改: 实现了受 MSAFF 论文 [cite: 777] 启发的多阶段融合策略 。
    
#     该模型包含三条路径，并在 "Global Level" 进行最终融合:
#     1. 晚期 CVCNN 特征路径 (原始)
#     2. 晚期 GAT 特征路径 (原始)
#     3. 新增的中期融合路径 (Mid-Stage Fusion)
#     """

#     def __init__(self, gat_in_channels, gat_hidden_channels, num_classes, dropout_rate=0.2):
#         super().__init__()
        
#         # --- 1. 实例化原始的特征提取器 ---
#         self.cnn_feature_extractor = ComplexCNNFeatureExtractor()
#         self.gat_feature_extractor = GATFeatureExtractor(in_channels=gat_in_channels,
#                                                        hidden_channels=gat_hidden_channels)

#         # --- 2. 定义 "晚期" 特征维度 (来自原始模型) ---
#         cnn_late_dim = 14336 * 2  # (cnn_real_dim)
#         gat_late_dim = gat_hidden_channels  # e.g., 512

#         # --- 3. 定义 "中期" 特征维度 (新增) ---
        
#         # CVCNN Mid (conv1 输出: [B, 4, H/2, W/2])
#         # 根据之前的计算: 8 * (H/4 * W/4) = 14336 -> H*W = 28672
#         # 中期维度 = 4 * (H/2 * W/2) = H*W = 28672
#         # 拆分实部虚部后维度 * 2
#         cnn_mid_dim = 28672 * 2 

#         # GAT Mid (gat1 输出: [Nodes, 512 * 8 = 4096])
#         gat_mid_dim = gat_hidden_channels * 8  # 4096
        
#         # --- 4. 定义 "中期融合头" (新增) ---
#         # 类似论文中的 AFFM(Frame) [cite: 894] 之后的 MSSTFE
#         self.mid_fusion_head = nn.Sequential(
#             nn.Linear(cnn_mid_dim + gat_mid_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(1024, 512)  # 输出一个 512 维的中期融合特征
#         )

#         # --- 5. 定义 "全局融合头" (修改) ---
#         # 类似论文中的 "Global concatenate" [cite: 892]
#         # 输入 = 晚期CNN + 晚期GAT + 中期融合特征
#         fusion_input_dim = cnn_late_dim + gat_late_dim + 512 # 512 是 mid_fusion_head 的输出

#         # self.fusion_head 重命名为 self.global_fusion_head
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
        
#         # --- 中期特征提取 (Mid-Stage Feature Extraction) ---
        
#         # 1a. CVCNN 中期特征 (来自 conv1)
#         cnn_mid_complex = self.cnn_feature_extractor.conv1(stft_data)
        
#         # 1b. GAT 中期特征 (来自 gat1)
#         # (注意: 我们需要 GATFeatureExtractor 类中的 GATConv 层暴露为 self.gat1)
#         gat_mid_nodes = F.relu(self.gat_feature_extractor.gat1(
#             graph_data.x, 
#             graph_data.edge_index, 
#             graph_data.edge_attr
#         ))
#         gat_mid_nodes_dropout = F.dropout(gat_mid_nodes, p=0.2, training=self.training)
        
#         # --- 晚期特征提取 (Late-Stage Feature Extraction) ---
        
#         # 2a. CVCNN 晚期特征 (来自 conv2)
#         cnn_late_complex = self.cnn_feature_extractor.conv2(cnn_mid_complex)
#         cnn_late_flat_complex = cnn_late_complex.view(cnn_late_complex.size(0), -1)
#         # (路径 1) 最终 CVCNN 特征
#         cnn_late_real = torch.cat([cnn_late_flat_complex.real, cnn_late_flat_complex.imag], dim=1)
        
#         # 2b. GAT 晚期特征 (来自 gat2)
#         # (注意: 我们需要 GATFeatureExtractor 类中的 GATConv 层暴露为 self.gat2)
#         gat_late_nodes = self.gat_feature_extractor.gat2(
#             gat_mid_nodes_dropout, # 使用 gat1 的输出作为输入
#             graph_data.edge_index, 
#             graph_data.edge_attr
#         )
#         # (路径 2) 最终 GAT 特征
#         gat_late_pooled = global_mean_pool(gat_late_nodes, graph_data.batch)

#         # --- 中期融合路径 (Mid-Stage Fusion Path) ---
        
#         # 3a. 处理 CVCNN 中期特征
#         cnn_mid_flat_complex = cnn_mid_complex.view(cnn_mid_complex.size(0), -1)
#         cnn_mid_real = torch.cat([cnn_mid_flat_complex.real, cnn_mid_flat_complex.imag], dim=1)

#         # 3b. 处理 GAT 中期特征
#         gat_mid_pooled = global_mean_pool(gat_mid_nodes, graph_data.batch) # 使用 gat1 的原始输出
        
#         # 3c. 融合中期特征
#         fused_mid = torch.cat((cnn_mid_real, gat_mid_pooled), dim=1)
#         # (路径 3) 最终中期融合特征
#         mid_fusion_output = self.mid_fusion_head(fused_mid)

#         # --- 全局融合 (Global-Level Fusion) ---
#         # 
#         # 4. 拼接所有路径的特征
#         global_fused_features = torch.cat((
#             cnn_late_real,       # 路径 1
#             gat_late_pooled,     # 路径 2
#             mid_fusion_output    # 路径 3
#         ), dim=1)

#         # 5. 最终分类
#         output = self.global_fusion_head(global_fused_features)
        
#         return output
    
class DualPathFusionModel(nn.Module):
    """
    修改: 实现了多阶段融合 + (可配置的)多阶段对比学习引导。
    
    新增了一个 `contrastive_mode` 参数，用于控制辅助损失:
    - 'none': 只有MSAFF架构 (您的 95.56% 基线)
    - 'mid': MSAFF + 仅在中期施加对比损失
    - 'late': MSAFF + 仅在晚期施加对比损失
    - 'both': MSAFF + 在中期和晚期都施加对比损失
    """

    def __init__(self, gat_in_channels, gat_hidden_channels, num_classes, 
                 dropout_rate=0.2, 
                 contrastive_mode: str = 'both'): # <-- 新增的配置参数
        
        super().__init__()
        
        self.contrastive_mode = contrastive_mode
        
        # --- 1. 实例化原始的特征提取器 ---
        self.cnn_feature_extractor = ComplexCNNFeatureExtractor()
        self.gat_feature_extractor = GATFeatureExtractor(in_channels=gat_in_channels,
                                                       hidden_channels=gat_hidden_channels)

        # --- 2. 定义特征维度 ---
        cnn_late_dim = 14336 * 2  # 28672
        gat_late_dim = gat_hidden_channels  # e.g., 512
        cnn_mid_dim = 28672 * 2  # 57344
        gat_mid_dim = gat_hidden_channels * 8  # 4096
        
        # --- 3. 定义 "中期融合头" (用于主分类路径) ---
        mid_fusion_output_dim = 512
        self.mid_fusion_head = nn.Sequential(
            nn.Linear(cnn_mid_dim + gat_mid_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, mid_fusion_output_dim)
        )

        # --- 4. 定义 "全局融合头" (用于主分类路径) ---
        fusion_input_dim = cnn_late_dim + gat_late_dim + mid_fusion_output_dim
        self.global_fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # --- 5. 定义 "辅助对比路径" (根据 mode 条件化创建) ---
        contrastive_dim = 256  # 共同的对比空间维度

        # 5a. 中期对比投影头
        if self.contrastive_mode in ['mid', 'both']:
            self.mid_cnn_contrast_proj = nn.Sequential(
                nn.Linear(cnn_mid_dim, 512),
                nn.ReLU(),
                nn.Linear(512, contrastive_dim)
            )
            self.mid_gat_contrast_proj = nn.Sequential(
                nn.Linear(gat_mid_dim, 512),
                nn.ReLU(),
                nn.Linear(512, contrastive_dim)
            )

        # 5b. 晚期对比投影头
        if self.contrastive_mode in ['late', 'both']:
            self.late_cnn_contrast_proj = nn.Sequential(
                nn.Linear(cnn_late_dim, 512),
                nn.ReLU(),
                nn.Linear(512, contrastive_dim)
            )
            self.late_gat_contrast_proj = nn.Sequential(
                nn.Linear(gat_late_dim, 256),
                nn.ReLU(),
                nn.Linear(256, contrastive_dim)
            )


    def forward(self, stft_data, graph_data):
        
        # --- 中期特征提取 (Mid-Stage) ---
        cnn_mid_complex = self.cnn_feature_extractor.conv1(stft_data)
        gat_mid_nodes = F.relu(self.gat_feature_extractor.gat1(
            graph_data.x, graph_data.edge_index, graph_data.edge_attr
        ))
        gat_mid_nodes_dropout = F.dropout(gat_mid_nodes, p=0.2, training=self.training)
        
        # --- 晚期特征提取 (Late-Stage) ---
        cnn_late_complex = self.cnn_feature_extractor.conv2(cnn_mid_complex)
        gat_late_nodes = self.gat_feature_extractor.gat2(
            gat_mid_nodes_dropout, graph_data.edge_index, graph_data.edge_attr
        )
        
        # --- 特征处理 ---
        cnn_late_flat_complex = cnn_late_complex.view(cnn_late_complex.size(0), -1)
        cnn_late_real = torch.cat([cnn_late_flat_complex.real, cnn_late_flat_complex.imag], dim=1)
        gat_late_pooled = global_mean_pool(gat_late_nodes, graph_data.batch)

        cnn_mid_flat_complex = cnn_mid_complex.view(cnn_mid_complex.size(0), -1)
        cnn_mid_real = torch.cat([cnn_mid_flat_complex.real, cnn_mid_flat_complex.imag], dim=1)
        gat_mid_pooled = global_mean_pool(gat_mid_nodes, graph_data.batch)
        
        # --- 中期融合路径 (Mid-Stage Fusion Path 3) ---
        fused_mid = torch.cat((cnn_mid_real, gat_mid_pooled), dim=1)
        mid_fusion_output = self.mid_fusion_head(fused_mid)

        # --- 全局融合 (Global-Level Fusion) ---
        global_fused_features = torch.cat((
            cnn_late_real,       # 路径 1
            gat_late_pooled,     # 路径 2
            mid_fusion_output    # 路径 3
        ), dim=1)

        # (主输出) 最终分类
        logits = self.global_fusion_head(global_fused_features)
        
        # --- 辅助对比路径 (根据 mode 条件化返回) ---
        
        if self.contrastive_mode == 'mid':
            cnn_mid_vec = self.mid_cnn_contrast_proj(cnn_mid_real.detach())
            gat_mid_vec = self.mid_gat_contrast_proj(gat_mid_pooled.detach())
            return logits, cnn_mid_vec, gat_mid_vec  # 返回 3 个值
        
        elif self.contrastive_mode == 'late':
            cnn_late_vec = self.late_cnn_contrast_proj(cnn_late_real.detach())
            gat_late_vec = self.late_gat_contrast_proj(gat_late_pooled.detach())
            return logits, cnn_late_vec, gat_late_vec  # 返回 3 个值
        
        elif self.contrastive_mode == 'both':
            cnn_mid_vec = self.mid_cnn_contrast_proj(cnn_mid_real.detach())
            gat_mid_vec = self.mid_gat_contrast_proj(gat_mid_pooled.detach())
            cnn_late_vec = self.late_cnn_contrast_proj(cnn_late_real.detach())
            gat_late_vec = self.late_gat_contrast_proj(gat_late_pooled.detach())
            return logits, cnn_mid_vec, gat_mid_vec, cnn_late_vec, gat_late_vec  # 返回 5 个值
        
        else:  # self.contrastive_mode == 'none'
            return logits  # 只返回 1 个值
