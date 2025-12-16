import torch
import torch.nn as nn
import torch.nn.functional as F
from complexNN.nn import  cConv2d, cAvgPool2d, cBatchNorm2d, cLeakyRelu, cDropout, cLinear
from torch_geometric.nn import GATConv, global_mean_pool


class ComplexCNNFeatureExtractor(nn.Module):
    """
    修改:
    - 重新加入了 STFT_complexnn 中的 cLinear FC 层来进行内部降维。
    - Forward 方法返回 "中期(高维, complex)" 和 "晚期(降维, real)"。
    """
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
        
        # 晚期降维FC
        self.cnn_fc1_late = nn.Sequential(
            cLinear(39936, 512), 
            cLeakyRelu(),
            cDropout(p=dropout_rate)
        )
        self.cnn_fc2_late = nn.Sequential(
            cLinear(512, 256), 
            cLeakyRelu(),
            cDropout(p=dropout_rate)
        )
        
        # 中期降维FC (用于 msaff 模式)
        self.cnn_fc1_mid = nn.Sequential(
            cLinear(79872, 1024),
            cLeakyRelu(),
            cDropout(p=dropout_rate)
        )
        self.cnn_fc2_mid = nn.Sequential(
            cLinear(1024, 512), 
            cLeakyRelu(),
            cDropout(p=dropout_rate)
        )

    def forward(self, x):
        # 1. 中期特征
        cnn_mid_complex = self.conv1(x)
        cnn_mid_flat = cnn_mid_complex.view(cnn_mid_complex.size(0), -1) # [B, 79872]c
        
        # 2. 晚期特征
        cnn_late_complex = self.conv2(cnn_mid_complex)
        cnn_late_flat = cnn_late_complex.view(cnn_late_complex.size(0), -1) # [B, 39936]c
        
        # 3. 晚期降维 & 幅度归一化
        cnn_late_fc = self.cnn_fc1_late(cnn_late_flat)
        cnn_late_fc = self.cnn_fc2_late(cnn_late_fc) # [B, 256]c
        cnn_late_output = F.normalize(torch.abs(cnn_late_fc), p=2, dim=1) # [B, 256]r
        
        # 4. 中期降维 & 幅度归一化
        cnn_mid_fc = self.cnn_fc1_mid(cnn_mid_flat)
        cnn_mid_fc = self.cnn_fc2_mid(cnn_mid_fc) # [B, 512]c
        cnn_mid_output = F.normalize(torch.abs(cnn_mid_fc), p=2, dim=1) # [B, 512]r
        
        return cnn_mid_output, cnn_late_output

# ===================================================================
# 2. GAT 特征提取器 (不对称架构：保持高维 + 可视化接口)
# ===================================================================
class GATFeatureExtractor(nn.Module):
    """
    修改:
    - 移除了所有内部 FC 降维层，保持高维输出 (4096)。
    - Forward 增加了 return_attention 参数，用于可视化 Heatmap。
    """
    def __init__(self, in_channels, hidden_channels, num_heads=8, dropout_rate=0.5):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.2)
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

    def forward(self, data, return_attention=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # --- 中期特征 (GAT1) ---
        # 如果需要可视化 Attention，我们需要在这里捕获
        edge_index_alpha = None
        alpha = None

        if return_attention:
            # 获取第一层的注意力权重用于可视化 (通常更有物理意义)
            x_mid_nodes, (edge_index_alpha, alpha) = self.gat1(x, edge_index, return_attention_weights=True)
            x_mid_nodes = F.relu(x_mid_nodes)
        else:
            x_mid_nodes = F.relu(self.gat1(x, edge_index, edge_attr))
        
        gat_mid_pooled = global_mean_pool(x_mid_nodes, batch) # [B, 4096]
        
        # --- 晚期特征 (GAT2) ---
        x_mid_dropout = F.dropout(x_mid_nodes, p=0.2, training=self.training)
        x_late_nodes = self.gat2(x_mid_dropout, edge_index, edge_attr)
        
        try:
            # [B, 4096] (假设 batch 中图节点数固定，否则这里会报错，需改用 global_mean_pool)
            gat_late_view = x_late_nodes.view(-1, self.num_heads * self.hidden_channels)
        except RuntimeError as e:
            raise e
        
        if return_attention:
            return gat_mid_pooled, gat_late_view, edge_index_alpha, alpha
        else:
            return gat_mid_pooled, gat_late_view

# ===================================================================
# 3. 统一融合模型 (支持三种模式 + 可视化接口)
# ===================================================================
class DualPathFusionModel_watkins(nn.Module):
    """
    修改: 
    - 融合了不对称架构 (CVCNN降维, GAT不降维)。
    - 支持 'visualize=True' 参数，返回特征和注意力权重。
    """

    def __init__(self, gat_in_channels, gat_hidden_channels, num_classes, 
                 dropout_rate=0.5, 
                 fusion_mode: str = 'concat'):
        
        super().__init__()
        
        self.fusion_mode = fusion_mode
        if self.fusion_mode not in ['concat', 'concat_contrastive', 'msaff_contrastive']:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        
        # --- 1. 实例化特征提取器 ---
        self.cnn_feature_extractor = ComplexCNNFeatureExtractor(dropout_rate=dropout_rate)
        self.gat_feature_extractor = GATFeatureExtractor(in_channels=gat_in_channels,
                                                       hidden_channels=gat_hidden_channels,
                                                       dropout_rate=dropout_rate)

        # --- 2. 定义特征维度 ---
        cnn_late_dim_real = 256           # CVCNN 晚期 (降维后)
        gat_late_dim = gat_hidden_channels * 8 # GAT 晚期 (4096)
        
        cnn_mid_dim_real = 512            # CVCNN 中期 (降维后)
        gat_mid_dim = gat_hidden_channels * 8 # GAT 中期 (4096)
        
        # --- 3. 中期融合头 (仅 msaff) ---
        mid_fusion_output_dim = 256
        if self.fusion_mode == 'msaff_contrastive':
            self.mid_fusion_head = nn.Sequential(
                nn.Linear(cnn_mid_dim_real + gat_mid_dim, 512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, mid_fusion_output_dim)
            )

        # --- 4. 辅助对比路径 ---
        contrastive_dim = 128
        if self.fusion_mode in ['concat_contrastive', 'msaff_contrastive']:
            self.late_cnn_contrast_proj = nn.Sequential(
                nn.Linear(cnn_late_dim_real, 128),
                nn.ReLU(),
                nn.Linear(128, contrastive_dim)
            )
            self.late_gat_contrast_proj = nn.Sequential(
                nn.Linear(gat_late_dim, 512), 
                nn.ReLU(),
                nn.Linear(512, contrastive_dim)
            )

        # --- 5. 全局融合头 ---
        fusion_input_dim = cnn_late_dim_real + gat_late_dim
        if self.fusion_mode == 'msaff_contrastive':
            fusion_input_dim += mid_fusion_output_dim

        self.global_fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, stft_data, graph_data, visualize=False):
        
        # --- 1. 特征提取 ---
        cnn_mid_output, cnn_late_output = self.cnn_feature_extractor(stft_data)
        
        # 处理 GAT 可视化逻辑
        if visualize:
            gat_mid_pooled, gat_late_output, edge_index, alpha = self.gat_feature_extractor(graph_data, return_attention=True)
        else:
            gat_mid_pooled, gat_late_output = self.gat_feature_extractor(graph_data)
            edge_index, alpha = None, None
        
        # --- 2. 主分类路径 ---
        features_to_concat = [cnn_late_output, gat_late_output]
        
        if self.fusion_mode == 'msaff_contrastive':
            # 中期融合 (都使用降维后的/池化后的特征)
            fused_mid_input = torch.cat((cnn_mid_output, gat_mid_pooled), dim=1)
            mid_fusion_output = self.mid_fusion_head(fused_mid_input)
            features_to_concat.append(mid_fusion_output)
            
        global_fused_features = torch.cat(features_to_concat, dim=1)
        logits = self.global_fusion_head(global_fused_features)
        
        # ==================================================
        # --- 可视化返回接口 ---
        # ==================================================
        if visualize:
            # 返回:
            # 1. logits: 预测结果
            # 2. global_fused_features: 最终融合特征 (用于 t-SNE)
            # 3. edge_index, alpha: 图结构和注意力权重 (用于 Heatmap)
            return logits, global_fused_features, edge_index, alpha

        # ==================================================
        # --- 训练返回接口 (与之前一致) ---
        # ==================================================
        if self.fusion_mode == 'concat':
            return logits 

        elif self.fusion_mode in ['concat_contrastive', 'msaff_contrastive']:
            cnn_late_vec = self.late_cnn_contrast_proj(cnn_late_output.detach())
            gat_late_vec = self.late_gat_contrast_proj(gat_late_output.detach())
            return logits, cnn_late_vec, gat_late_vec