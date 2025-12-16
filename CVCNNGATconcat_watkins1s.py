import torch
import torch.nn as nn
import torch.nn.functional as F
from complexNN.nn import  cConv2d, cAvgPool2d, cBatchNorm2d, cLeakyRelu, cDropout, cLinear
from torch_geometric.nn import GATConv, global_mean_pool


# class ComplexCNNFeatureExtractor(nn.Module):
#     """
#     修改:
#     - 重新加入了 STFT_complexnn 中的 cLinear FC 层来进行内部降维。
#     - Forward 方法在最后计算 "幅度(abs)" 并进行归一化。
#     """
#     def __init__(self, dropout_rate=0.5): # 传入 dropout_rate
#         super().__init__()
#         # 对应 STFT_complexnn
#         self.conv1 = nn.Sequential(
#             cConv2d(1, 8, kernel_size=3, stride=1, padding=(1, 1)),
#             cBatchNorm2d(8),
#             cLeakyRelu(),
#             cAvgPool2d(kernel_size=(2, 2), stride=(2, 2))
#         )
#         self.conv2 = nn.Sequential(
#             cConv2d(8, 16, kernel_size=3, stride=1, padding=(1, 1)),
#             cBatchNorm2d(16),
#             cLeakyRelu(),
#             cAvgPool2d(kernel_size=(2, 2), stride=(2, 2))
#         )
        
#         # --- 新增: 将降维FC层加回来 ---
#         # (来自 STFT_complexnn)
#         self.cnn_fc1 = nn.Sequential(
#             cLinear(39936, 512), 
#             cLeakyRelu(),
#             cDropout(p=dropout_rate) # 使用传入的 dropout
#         )
#         self.cnn_fc2 = nn.Sequential(
#             cLinear(512, 256), 
#             cLeakyRelu(),
#             cDropout(p=dropout_rate) # 使用传入的 dropout
#         )
#         # --- 结束新增 ---

#     def forward(self, x):
#         cnn_mid_complex = self.conv1(x)
#         cnn_late_complex = self.conv2(cnn_mid_complex)
        
#         # 展平特征图 [B, 39936] (complex)
#         cnn_late_flat = cnn_late_complex.view(cnn_late_complex.size(0), -1)

#         # --- 修改: 在复数域进行降维 ---
#         cnn_late_fc = self.cnn_fc1(cnn_late_flat)
#         cnn_late_fc = self.cnn_fc2(cnn_late_fc) # 最终输出: [B, 256] (complex)
        
#         # --- 修改: 计算幅度和归一化 ---
#         x_late_real_magnitude = torch.abs(cnn_late_fc)
#         x_late_normalized = F.normalize(x_late_real_magnitude, p=2, dim=1)
        
#         # 只返回降维后的晚期特征
#         return x_late_normalized


# # ===================================================================
# # 2. 重构的 GAT 特征提取器 (基于您的 GATClassifier)
# # ===================================================================
# class GATFeatureExtractor(nn.Module):
#     """
#     修改:
#     - 移除了所有内部的 FC 降维层。
#     - Forward 方法改为使用 .view() (假设批次中节点数固定)。
#     - 只返回晚期特征。
#     """
#     def __init__(self, in_channels, hidden_channels, num_heads=8, dropout_rate=0.5):
#         super().__init__()
#         self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
#         self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.2)
        
#         self.hidden_channels = hidden_channels
#         self.num_heads = num_heads
        
#         # --- 移除了 gat_mid_fc 和 gat_late_fc ---

#     def forward(self, data):
#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

#         x_mid_nodes = F.relu(self.gat1(x, edge_index, edge_attr))
#         x_mid_dropout = F.dropout(x_mid_nodes, p=0.2, training=self.training)
#         x_late_nodes = self.gat2(x_mid_dropout, edge_index, edge_attr) # [Nodes, hidden]
        
#         # --- 修改: 使用 .view() ---
#         # 警告: 这里的 8 必须与 GAT1 的 heads 数匹配
#         # 警告: 这假设一个批次中总是有固定数量的节点
#         try:
#             # 这里的 8 应该等于 self.num_heads
#             x_late_view = x_late_nodes.view(-1, self.num_heads * self.hidden_channels) # [B, 4096]
#         except RuntimeError as e:
#             print(f"GATFeatureExtractor .view() failed. "
#                   f"This usually means your batch size > 1 OR nodes per graph is not fixed.")
#             print(f"Input shape was: {x_late_nodes.shape}")
#             raise e
#         # --- 结束修改 ---
        
#         # 只返回晚期特征
#         return x_late_view


# # ===================================================================
# # 3. 统一的融合模型
# # ===================================================================
# class DualPathFusionModel_watkins(nn.Module):
#     """
#     修改: 
#     - 仅支持 'concat' 模式。
#     - CVCNN (ComplexCNNFeatureExtractor) 在内部降维并输出 [B, 256] (real)。
#     - GAT (GATFeatureExtractor) 不降维并输出 [B, 4096] (real)。
#     - 融合头 (fusion_head) 接收 (256 + 4096) = 4352 维的拼接向量。
#     - 融合头的参数量大大减少，以对抗过拟合。
#     """

#     def __init__(self, gat_in_channels, gat_hidden_channels, num_classes, 
#                  dropout_rate=0.5, 
#                  fusion_mode: str = 'concat'):
        
#         super().__init__()
        
#         # 只支持 'concat'
#         self.fusion_mode = 'concat' 
        
#         # --- 1. 实例化特征提取器 ---
#         self.cnn_feature_extractor = ComplexCNNFeatureExtractor(dropout_rate=dropout_rate)
#         self.gat_feature_extractor = GATFeatureExtractor(in_channels=gat_in_channels,
#                                                        hidden_channels=gat_hidden_channels,
#                                                        dropout_rate=dropout_rate)

#         # --- 2. 定义新特征维度 (不对称) ---
        
#         # CVCNN late (降维后 + 幅度): [B, 256] (real)
#         cnn_late_dim_real = 256
        
#         # GAT late (.view()): [B, 512 * 8 = 4096] (real)
#         gat_late_dim = gat_hidden_channels * 8
        
#         # --- 3. 定义 "全局融合头" ---
        
#         # (256 + 4096 = 4352)
#         fusion_input_dim = cnn_late_dim_real + gat_late_dim
        
#         # 这个融合头的参数量 (4352 * 1024) 远小于之前的 (44032 * 1024)
#         # 这正是解决过拟合的关键
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
#         # cnn_late_output: [B, 256] (real, normalized magnitude)
#         cnn_late_output = self.cnn_feature_extractor(stft_data)
#         # gat_late_output: [B, 4096] (real, from .view())
#         gat_late_output = self.gat_feature_extractor(graph_data)
        
#         # --- 2. 主分类路径 ---
#         features_to_concat = [cnn_late_output, gat_late_output]
#         global_fused_features = torch.cat(features_to_concat, dim=1)
#         logits = self.global_fusion_head(global_fused_features)
        
#         # --- 3. 返回 ---
#         return logits


class ComplexCNNFeatureExtractor(nn.Module):
    """
    修改:
    - 重新加入了 STFT_complexnn 中的 cLinear FC 层来进行内部降维。
    - Forward 方法返回 "中期(高维, complex)", "中期(降维, real)", 和 "晚期(降维, real)"。
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
        
        # 晚期降维FC (来自 STFT_complexnn)
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
        
        # --- 新增: 中期降维FC (用于 msaff 模式) ---
        self.cnn_fc1_mid = nn.Sequential(
            cLinear(79872, 1024), # 中期维度更大 [B, 79872]
            cLeakyRelu(),
            cDropout(p=dropout_rate)
        )
        self.cnn_fc2_mid = nn.Sequential(
            cLinear(1024, 512), # 降维到 512
            cLeakyRelu(),
            cDropout(p=dropout_rate)
        )
        # --- 结束新增 ---

    def forward(self, x):
        # 1. 中期特征
        cnn_mid_complex = self.conv1(x)
        cnn_mid_flat = cnn_mid_complex.view(cnn_mid_complex.size(0), -1) # [B, 79872] (complex)
        
        # 2. 晚期特征
        cnn_late_complex = self.conv2(cnn_mid_complex)
        cnn_late_flat = cnn_late_complex.view(cnn_late_complex.size(0), -1) # [B, 39936] (complex)
        
        # 3. 晚期降维 (用于主路径和对比)
        cnn_late_fc = self.cnn_fc1_late(cnn_late_flat)
        cnn_late_fc = self.cnn_fc2_late(cnn_late_fc) # [B, 256] (complex)
        cnn_late_abs = torch.abs(cnn_late_fc)
        cnn_late_output = F.normalize(cnn_late_abs, p=2, dim=1) # [B, 256] (real)
        
        # 4. 中期降维 (用于 msaff 模式)
        cnn_mid_fc = self.cnn_fc1_mid(cnn_mid_flat)
        cnn_mid_fc = self.cnn_fc2_mid(cnn_mid_fc) # [B, 512] (complex)
        cnn_mid_abs = torch.abs(cnn_mid_fc)
        cnn_mid_output = F.normalize(cnn_mid_abs, p=2, dim=1) # [B, 512] (real)
        
        # 返回: 中期(降维, real), 晚期(降维, real)
        return cnn_mid_output, cnn_late_output

class GATFeatureExtractor(nn.Module):
    """
    修改:
    - Forward 增加了 return_attention 参数。
    - 当 return_attention=True 时，返回 (mid_pooled, late_view, edge_index, alpha)。
    """
    def __init__(self, in_channels, hidden_channels, num_heads=8, dropout_rate=0.5):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.2)
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

    def forward(self, data, return_attention=False): # <--- 新增参数
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # --- 中期特征 (GAT1) ---
        edge_index_alpha = None
        alpha = None

        if return_attention:
            # 如果需要可视化，开启 return_attention_weights 获取第一层的注意力
            # GATConv 返回: (out, (edge_index, alpha))
            x_mid_nodes, (edge_index_alpha, alpha) = self.gat1(x, edge_index, return_attention_weights=True)
            x_mid_nodes = F.relu(x_mid_nodes)
        else:
            x_mid_nodes = F.relu(self.gat1(x, edge_index, edge_attr))

        gat_mid_pooled = global_mean_pool(x_mid_nodes, batch) # [B, 4096]
        
        # --- 晚期特征 (GAT2) ---
        x_mid_dropout = F.dropout(x_mid_nodes, p=0.2, training=self.training)
        x_late_nodes = self.gat2(x_mid_dropout, edge_index, edge_attr)
        
        try:
            gat_late_view = x_late_nodes.view(-1, self.num_heads * self.hidden_channels) # [B, 4096]
        except RuntimeError as e:
            raise e
        
        if return_attention:
            return gat_mid_pooled, gat_late_view, edge_index_alpha, alpha
        else:
            return gat_mid_pooled, gat_late_view

# class GATFeatureExtractor(nn.Module):
#     """
#     修改: 
#     - 这是用于消融实验的 MLP 版本 (伪装成 GATFeatureExtractor)。
#     - 接口(类名、参数)保持不变，但内部逻辑已替换为: 节点拼接 -> MLP。
#     - 仅用于证明 GAT 图结构有效性的消融实验。
#     """
#     def __init__(self, in_channels, hidden_channels, num_heads=8, dropout_rate=0.5):
#         super().__init__()
        
#         # --- 保持接口兼容 ---
#         self.hidden_channels = hidden_channels
#         self.num_heads = num_heads
        
#         # --- MLP 特定配置 ---
#         # 关键假设: 假设每个图固定有 8 个节点 
#         # (这是基于您之前 view(-1, 8*...) 的逻辑推断的，如果不是8请修改此处)
#         NUM_NODES = 8 
        
#         # 输入维度: 所有节点特征拼接 (Node_Num * In_Channels)
#         flat_input_dim = NUM_NODES * in_channels
        
#         # 输出维度: 保持与原 GAT 输出维度一致 (Hidden * Heads = 512 * 8 = 4096)
#         # 这样可以保证与 CVCNN 融合时的维度匹配不需要修改
#         output_dim = hidden_channels * num_heads 
        
#         # 定义 MLP
#         self.mlp = nn.Sequential(
#             nn.Linear(flat_input_dim, output_dim), # 投影到高维
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(output_dim, output_dim),     # 再加一层以模拟深度
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate)
#         )

#     def forward(self, data):
#         x, batch = data.x, data.batch
        
#         # 1. 获取 Batch Size
#         batch_size = batch.max().item() + 1
        
#         # 2. 将图节点拼接成一维长特征 (模拟忽略图结构)
#         # Shape: [Total_Nodes, C] -> [Batch_Size, Num_Nodes * C]
#         try:
#             x_flat = x.view(batch_size, -1)
#         except RuntimeError:
#              raise RuntimeError(f"消融实验报错: 无法将输入 Reshape 为 [B, -1]。"
#                                 f"请确保您的图数据中每个图的节点数是固定的 (当前代码假设为 8)。")

#         # 3. 经过 MLP 得到输出特征
#         out = self.mlp(x_flat) # [B, 4096]
        
#         # 4. 返回结果
#         # 为了兼容 DualPathFusionModel 的代码: mid_pooled, late_view = model(...)
#         # 我们返回两次 out，模型在 concat 模式下会使用第二个返回值
#         return out, out

class DualPathFusionModel_watkins(nn.Module):
    """
    修改: 
    - Forward 增加了 visualize=True 参数。
    - 可视化模式下返回: (logits, global_fused_features, edge_index, alpha)。
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
        
        # ... (以下所有 __init__ 代码保持完全不变) ...
        # ... (cnn_late_dim_real, gat_late_dim, mid_fusion_head, contrastive_proj, global_fusion_head 等) ...
        # 这里的代码与您提供的一模一样，无需改动，为了节省篇幅省略 ...
        # ------------------------------------------------------------------
        # 请务必保留您原来的 __init__ 代码内容！不要删除！
        # ------------------------------------------------------------------
        # (这里仅重复关键的定义以确保上下文正确)
        cnn_late_dim_real = 256
        gat_late_dim = gat_hidden_channels * 8
        cnn_mid_dim_real = 512
        gat_mid_dim = gat_hidden_channels * 8
        mid_fusion_output_dim = 256

        if self.fusion_mode == 'msaff_contrastive':
            self.mid_fusion_head = nn.Sequential(
                nn.Linear(cnn_mid_dim_real + gat_mid_dim, 512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, mid_fusion_output_dim)
            )

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


    def forward(self, stft_data, graph_data, visualize=False): # <--- 新增参数
        
        # --- 1. 特征提取 ---
        cnn_mid_output, cnn_late_output = self.cnn_feature_extractor(stft_data)
        
        # --- 处理 GAT/MLP 可视化逻辑 ---
        edge_index = None
        alpha = None
        
        # 检查是否开启可视化 且 提取器是否支持返回注意力 (即是 GATFeatureExtractor)
        if visualize and hasattr(self.gat_feature_extractor, 'forward') and \
           'return_attention' in self.gat_feature_extractor.forward.__code__.co_varnames:
            
            gat_mid_pooled, gat_late_output, edge_index, alpha = self.gat_feature_extractor(graph_data, return_attention=True)
            
        else:
            # 正常训练模式，或者提取器是 MLP (不支持 attention)
            gat_mid_pooled, gat_late_output = self.gat_feature_extractor(graph_data)
        
        # --- 2. 主分类路径 (保持不变) ---
        features_to_concat = [cnn_late_output, gat_late_output]
        
        if self.fusion_mode == 'msaff_contrastive':
            cnn_mid_real = cnn_mid_output
            gat_mid_real = gat_mid_pooled
            fused_mid_input = torch.cat((cnn_mid_real, gat_mid_real), dim=1)
            mid_fusion_output = self.mid_fusion_head(fused_mid_input)
            features_to_concat.append(mid_fusion_output)
            
        global_fused_features = torch.cat(features_to_concat, dim=1)
        logits = self.global_fusion_head(global_fused_features)
        
        # ==================================================
        # --- 可视化返回接口 ---
        # ==================================================
        if visualize:
            # 返回: logits, 融合特征(t-SNE用), 图结构, 注意力权重
            return logits, global_fused_features, edge_index, alpha

        # ==================================================
        # --- 训练返回接口 (保持不变) ---
        # ==================================================
        
        if self.fusion_mode == 'concat':
            return logits 

        elif self.fusion_mode in ['concat_contrastive', 'msaff_contrastive']:
            cnn_late_vec = self.late_cnn_contrast_proj(cnn_late_output.detach())
            gat_late_vec = self.late_gat_contrast_proj(gat_late_output.detach())
            return logits, cnn_late_vec, gat_late_vec
