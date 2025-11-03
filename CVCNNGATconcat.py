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
        
#         common_dim = 1024

#         self.cnn_proj = nn.Linear(cnn_real_dim, common_dim)
#         self.gat_proj = nn.Linear(gat_output_dim, common_dim)


#         fusion_input_dim = cnn_real_dim + gat_output_dim + common_dim # 28672 + 512 = 29184

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
#         #gate = torch.sigmoid(self.fusion_gate(gat_features))

#         # --- 新的融合策略 ---
#         # 1. 将复数特征拆分为实部和虚部
#         cnn_features_real = torch.cat([cnn_features_complex.real, cnn_features_complex.imag], dim=1)
#         #gated_cnn_features = cnn_features_real * gate

#         # cnn_proj = F.relu(self.cnn_proj(cnn_features_real))
#         # gat_proj = F.relu(self.gat_proj(gat_features))
#         # interaction_features = cnn_proj * gat_proj

#         # 2. 拼接两个通路的高维实数特征
#         fused_features = torch.cat((cnn_features_real, gat_features), dim=1)
#         #fused_features = torch.cat((gated_cnn_features, gat_features), dim=1)

#         # 3. 通过共享分类头进行分类
#         output = self.fusion_head(fused_features)
#         return output

# class DualPathFusionModel(nn.Module):
#     """
#     修改: 使用交叉注意力 (Cross-Attention) 融合模型。
#     """
#     def __init__(self, gat_in_channels, gat_hidden_channels, num_classes, dropout_rate=0.5):
#         super().__init__()
#         self.cnn_feature_extractor = ComplexCNNFeatureExtractor()
#         self.gat_feature_extractor = GATFeatureExtractor(in_channels=gat_in_channels,
#                                                        hidden_channels=gat_hidden_channels)

#         # --- 特征维度定义 ---
#         cnn_flattened_dim = 14336
#         cnn_real_dim = cnn_flattened_dim * 2  # 28672
#         gat_output_dim = gat_hidden_channels  # 512

#         # --- 注意力机制的新增模块 ---
        
#         # 1. 定义一个共同的注意力嵌入维度
#         # 我们将GAT的输出维度作为共同维度，这在概念上很清晰
#         self.attn_dim = gat_hidden_channels  # 512
#         num_heads = 8 # 注意力头数
        
#         # 2. 定义投影层，将两个模态投影到共同的 attn_dim
#         # GAT (Query): [B, 512] -> [B, 512]
#         self.query_proj = nn.Linear(gat_output_dim, self.attn_dim)
        
#         # CVCNN (Key): [B, 28672] -> [B, 512]
#         self.key_proj = nn.Linear(cnn_real_dim, self.attn_dim)
        
#         # CVCNN (Value): [B, 28672] -> [B, 512]
#         self.value_proj = nn.Linear(cnn_real_dim, self.attn_dim)

#         # 3. 定义多头注意力模块
#         self.attention = nn.MultiheadAttention(
#             embed_dim=self.attn_dim,
#             num_heads=num_heads,
#             dropout=dropout_rate,
#             batch_first=True  # 确保输入/输出的批次维在前面 (B, Seq, Dim)
#         )

#         # --- 重新计算融合输入的维度 ---
#         # 我们将拼接: 
#         # 1. 原始CVCNN特征 (28672)
#         # 2. 原始GAT特征 (512)
#         # 3. 新的注意力输出特征 (512)
#         fusion_input_dim = cnn_real_dim + gat_output_dim + self.attn_dim
        
#         # 4. 定义分类头 (fusion_head)
#         # (与之前相同，只是输入维度改变了)
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
#         # --- 1. 特征提取 ---
#         # CNN通路: [B, 14336] (complex)
#         cnn_features_complex = self.cnn_feature_extractor(stft_data)
#         # GAT通路: [B, 512] (real)
#         gat_features = self.gat_feature_extractor(graph_data)

#         # 将复数特征拆分为实部和虚部: [B, 28672] (real)
#         cnn_features_real = torch.cat([cnn_features_complex.real, cnn_features_complex.imag], dim=1)

#         # --- 2. 交叉注意力融合 ---
        
#         # 投影 Q, K, V
#         # Q: [B, 512] -> [B, 512]
#         q = self.query_proj(gat_features)
#         # K: [B, 28672] -> [B, 512]
#         k = self.key_proj(cnn_features_real)
#         # V: [B, 28672] -> [B, 512]
#         v = self.value_proj(cnn_features_real)

#         # 为 MultiheadAttention 增加一个 "序列长度" 维度 (Seq=1)
#         # Q, K, V: [B, 512] -> [B, 1, 512]
#         q = q.unsqueeze(1)
#         k = k.unsqueeze(1)
#         v = v.unsqueeze(1)
        
#         # 计算注意力
#         # attn_output: [B, 1, 512]
#         attn_output, _ = self.attention(q, k, v)
        
#         # 移除 "序列长度" 维度
#         # refined_features: [B, 512]
#         refined_features = attn_output.squeeze(1)

#         # --- 3. 最终拼接 ---
#         # 拼接所有信息：原始CVCNN + 原始GAT + 交互结果
#         fused_features = torch.cat((cnn_features_real, gat_features, refined_features), dim=1)

#         # --- 4. 分类 ---
#         output = self.fusion_head(fused_features)
#         return output

# class DualPathFusionModel(nn.Module):
#     """
#     修改: 采用 "复数域交互" 融合策略 (方向三)。
#     """

#     def __init__(self, gat_in_channels, gat_hidden_channels, num_classes, dropout_rate=0.5):
#         super().__init__()
#         self.cnn_feature_extractor = ComplexCNNFeatureExtractor()
#         self.gat_feature_extractor = GATFeatureExtractor(in_channels=gat_in_channels,
#                                                        hidden_channels=gat_hidden_channels)

#         # --- 原始特征维度 ---
#         cnn_flattened_dim = 14336                # CVCNN 展平后的复数维度
#         cnn_real_dim = cnn_flattened_dim * 2       # CVCNN 拆分实部虚部后的实数维度 (28672)
#         gat_output_dim = gat_hidden_channels       # GAT 输出的实数维度 (e.g., 512)
        
#         # --- 复数域交互模块 ---
        
#         # 1. 定义一个合理的共同复数交互维度 (超参数)
#         common_dim_complex = 512  

#         # 2. 投影层
#         #    复数 -> 复数
#         self.cnn_c_proj = cLinear(cnn_flattened_dim, common_dim_complex)
#         #    实数 -> 实数 (维度与复数交互维度匹配)
#         self.gat_r_proj = nn.Linear(gat_output_dim, common_dim_complex)

#         # --- 最终融合与分类 ---
        
#         # 3. 融合策略: 拼接 
#         #    1. 原始CNN (实数, 28672)
#         #    2. 原始GAT (实数, 512)
#         #    3. 复数交互特征 (拆分后为实数, 512 * 2 = 1024)
#         fusion_input_dim = cnn_real_dim + gat_output_dim + (common_dim_complex * 2)
#         # (总维度: 28672 + 512 + 1024 = 29696)

#         # 4. 定义分类头 (fusion_head)
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
#         # --- 1. 特征提取 ---
#         # CNN通路: [B, 14336] (complex)
#         cnn_features_complex = self.cnn_feature_extractor(stft_data)
#         # GAT通路: [B, 512] (real)
#         gat_features = self.gat_feature_extractor(graph_data)

#         # 准备好用于最终拼接的原始特征
#         # cnn_features_real: [B, 28672]
#         cnn_features_real = torch.cat([cnn_features_complex.real, cnn_features_complex.imag], dim=1)
#         # gat_features: [B, 512]

#         # --- 2. "复数域" 中的交互 ---
        
#         # 2a. 投影 (cLinear 支持复数输入)
#         proj_cnn_c = self.cnn_c_proj(cnn_features_complex) # [B, 512] (complex)
#         #    投影 (nn.Linear 支持实数输入)
#         proj_gat_r = self.gat_r_proj(gat_features)         # [B, 512] (real)

#         # 2b. 将GAT的实数投影 "提升" 为复数 (虚部为0)
#         proj_gat_c = torch.complex(proj_gat_r, torch.zeros_like(proj_gat_r)) # [B, 512] (complex)

#         # 2c. 复数域交互 (元素级乘法)
#         interaction_complex = proj_cnn_c * proj_gat_c # [B, 512] (complex)

#         # 2d. 将复数交互特征拆分为实部和虚部，以便后续拼接
#         interaction_features_real = torch.cat([interaction_complex.real, interaction_complex.imag], dim=1) # [B, 1024]

#         # --- 3. 最终融合 ---
#         # 拼接所有信息： 原始CNN(实) + 原始GAT(实) + 交互结果(实)
#         fused_features = torch.cat((cnn_features_real, gat_features, interaction_features_real), dim=1)

#         # --- 4. 分类 ---
#         output = self.fusion_head(fused_features)
#         return output

class DualPathFusionModel(nn.Module):
    """
    修改: 融合 ComplexCNN 和 GAT 的双路模型。
    新策略: 
    1. (主路径) 特征直接拼接 (Concat) 进行分类。
    2. (辅助路径) 添加对比学习投影头，用于计算引导损失。
    """

    def __init__(self, gat_in_channels, gat_hidden_channels, num_classes, dropout_rate=0.2):
        super().__init__()
        self.cnn_feature_extractor = ComplexCNNFeatureExtractor()
        self.gat_feature_extractor = GATFeatureExtractor(in_channels=gat_in_channels,
                                                       hidden_channels=gat_hidden_channels)

        # --- 1. 主分类路径 (Concat Baseline) ---
        
        # CNN展平后维度
        cnn_flattened_dim = 14336
        cnn_real_dim = cnn_flattened_dim * 2  # 28672
        # GAT输出维度
        gat_output_dim = gat_hidden_channels  # e.g., 512
        
        # Concat 路径的输入维度
        fusion_input_dim = cnn_real_dim + gat_output_dim  # 28672 + 512 = 29184

        # 定义分类头
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

        # --- 2. 辅助对比路径 ---
        # (这部分仅用于计算对比损失，不参与最终分类)
        
        contrastive_dim = 256  # 定义一个合理的、低维的对比空间

        # 新增: CNN的对比投影头
        self.cnn_contrast_proj = nn.Sequential(
            nn.Linear(cnn_real_dim, 512),
            nn.ReLU(),
            nn.Linear(512, contrastive_dim)
        )
        
        # 新增: GAT的对比投影头
        self.gat_contrast_proj = nn.Sequential(
            nn.Linear(gat_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, contrastive_dim)
        )

    def forward(self, stft_data, graph_data):
        # --- 1. 特征提取 ---
        # CNN通路: 得到高维复数特征
        cnn_features_complex = self.cnn_feature_extractor(stft_data)
        # GAT通路: 得到图嵌入特征
        gat_features = self.gat_feature_extractor(graph_data)

        # 将复数特征拆分为实部和虚部
        cnn_features_real = torch.cat([cnn_features_complex.real, cnn_features_complex.imag], dim=1)

        # --- 2. 主分类路径 (Concat) ---
        # 拼接两个通路的高维实数特征
        fused_features = torch.cat((cnn_features_real, gat_features), dim=1)
        # 通过共享分类头进行分类
        logits = self.fusion_head(fused_features)

        # --- 3. 辅助对比路径 ---
        # (我们使用 .detach() 来阻止对比损失的梯度流入分类头，
        #  或者您可以移除 .detach() 来让对比损失也训练特征提取器)
        
        # 投影到对比空间
        cnn_contrast_vec = self.cnn_contrast_proj(cnn_features_real.detach())
        gat_contrast_vec = self.gat_contrast_proj(gat_features.detach())

        # --- 4. 返回所有输出 ---
        return logits, cnn_contrast_vec, gat_contrast_vec