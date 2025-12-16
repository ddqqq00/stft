import argparse
import os
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
from GAT import GATClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import seaborn as sns


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Audio Classification using GAT with preprocessed graph data")

    # 修改: 默认路径指向已处理好的图数据目录
    parser.add_argument('--data_path', type=str, default=r"D:\PyCharm\underwater-data\processed_graphs",
                        help='存放预处理好的 .pt 图数据文件的路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='权重衰减系数')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练的轮数')
    parser.add_argument('--top_k_classes', type=int, default=6, help='选择样本数最多的前k个类别')
    # 移除: --sr 参数，因为不再需要

    args = parser.parse_args()
    return args


# 传入参数
args = parse_args()


# Focal Loss定义 (保持不变)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# 修改: 新的数据加载函数，用于查找 .pt 文件
def get_graph_paths_and_labels(data_path, top_k_classes=None):
    """获取 .pt 图数据文件的路径和标签"""
    category_paths = {}
    class_counts = {}

    for class_folder in os.listdir(data_path):
        class_folder_path = os.path.join(data_path, class_folder)
        if os.path.isdir(class_folder_path):
            # 修改: 查找 .pt 文件
            file_paths = [os.path.join(class_folder_path, f) for f in os.listdir(class_folder_path) if
                          f.endswith(".pt")]
            if file_paths:
                category_paths[class_folder] = file_paths
                class_counts[class_folder] = len(file_paths)

    if top_k_classes and top_k_classes > 0:
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        top_k_items = sorted_classes[:top_k_classes]
        total_samples = sum(count for _, count in top_k_items)

        print(f"从预处理数据中选择前 {len(top_k_items)} 个类别, 总样本数为: {total_samples}")
        selected_classes = {class_name: category_paths[class_name] for class_name, _ in top_k_items}
        return selected_classes

    return category_paths


# 重写: GraphDataset 类，使其只加载预处理好的数据
class GraphDataset(Dataset):
    def __init__(self, file_paths):
        """
        初始化数据集。
        file_paths: 预处理好的 .pt 文件的路径列表。
        """
        self.file_paths = file_paths

        # 标签编码器仍然需要，从 .pt 文件的父目录名获取标签
        self.label_encoder = LabelEncoder()
        self.labels = [os.path.basename(os.path.dirname(p)) for p in self.file_paths]
        self.label_encoder.fit(self.labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        加载单个预处理好的图数据。这个过程非常快！
        """
        file_path = self.file_paths[idx]

        # 1. 从磁盘加载已经处理好的Data对象
        data = torch.load(file_path, weights_only=False)

        # 2. 获取标签字符串并进行编码
        label_str = os.path.basename(os.path.dirname(file_path))
        label_int = self.label_encoder.transform([label_str])[0]

        # 3. 将标签动态添加到Data对象中
        data.y = torch.tensor(label_int, dtype=torch.long)

        return data


# --- 主要执行流程 ---

# 1. 获取图数据文件路径
category_paths = get_graph_paths_and_labels(args.data_path, top_k_classes=args.top_k_classes)


# 2. 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#set_seed(43)  # 你可以尝试 43, 53, 63

# 3. 获取所有文件路径和标签，用于数据集划分
all_files = []
all_labels = []
for label, files in category_paths.items():
    all_files.extend(files)
    all_labels.extend([label] * len(files))

# 4. 分层抽样划分训练集和测试集 (80:20)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_files, test_files = [], []
for train_idx, test_idx in sss.split(all_files, all_labels):
    train_files = [all_files[i] for i in train_idx]
    test_files = [all_files[i] for i in test_idx]

# 5. 创建数据集和数据加载器
train_dataset = GraphDataset(train_files)
test_dataset = GraphDataset(test_files)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# 6. 设备配置、模型、损失函数和优化器 (保持不变)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = GATClassifier(in_channels=28, hidden_channels=512, out_channels=args.top_k_classes).to(device)
# model = GATClassifier(in_channels=79, hidden_channels=512, out_channels=args.top_k_classes).to(device)
criterion = FocalLoss(alpha=0.25, gamma=2)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


# 7. 评估指标计算函数 (保持不变)
def calculate_precision(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    precision_score = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        denominator = tp + fp
        if denominator > 0:
            precision_score.append(tp / denominator)
        else:
            precision_score.append(0)
    return np.mean(precision_score)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
train_recalls, test_recalls = [], []
train_f1s, test_f1s = [], []
train_precisions, test_precisions = [], []

for epoch in range(args.num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    train_all_labels, train_all_predictions = [], []

    with tqdm(total=len(train_loader), desc=f"Epoch [{epoch + 1}/{args.num_epochs}] Train", ncols=100) as pbar:
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
            running_loss += loss.item()

            train_all_labels.extend(batch.y.cpu().numpy())
            train_all_predictions.extend(predicted.cpu().numpy())

            pbar.set_postfix(loss=running_loss / (pbar.n + 1), accuracy=100 * correct / total)
            pbar.update(1)

    train_accuracy = 100 * correct / total
    train_recall = recall_score(train_all_labels, train_all_predictions, average='weighted', zero_division=0)
    train_f1 = f1_score(train_all_labels, train_all_predictions, average='weighted', zero_division=0)
    train_precision = calculate_precision(train_all_labels, train_all_predictions, num_classes=args.top_k_classes)
    print(
        f"Epoch {epoch + 1} - Train: Recall={train_recall:.4f}, F1={train_f1:.4f}, Precision={train_precision:.4f}, Accuracy={train_accuracy:.4f}")

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_accuracy)
    train_recalls.append(train_recall)
    train_f1s.append(train_f1)
    train_precisions.append(train_precision)

    model.eval()
    test_running_loss, test_correct, test_total = 0.0, 0, 0
    test_all_labels, test_all_predictions = [], []

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc=f"Epoch [{epoch + 1}/{args.num_epochs}] Test ", ncols=100) as pbar:
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch.y)
                test_running_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                test_total += batch.y.size(0)
                test_correct += (predicted == batch.y).sum().item()

                test_all_labels.extend(batch.y.cpu().numpy())
                test_all_predictions.extend(predicted.cpu().numpy())

                pbar.set_postfix(loss=test_running_loss / (pbar.n + 1), accuracy=100 * test_correct / test_total)
                pbar.update(1)

    test_accuracy = 100 * test_correct / test_total
    test_recall = recall_score(test_all_labels, test_all_predictions, average='weighted', zero_division=0)
    test_f1 = f1_score(test_all_labels, test_all_predictions, average='weighted', zero_division=0)
    test_precision = calculate_precision(test_all_labels, test_all_predictions, num_classes=args.top_k_classes)
    print(
        f"Epoch {epoch + 1} - Test:  Recall={test_recall:.4f}, F1={test_f1:.4f}, Precision={test_precision:.4f}, Accuracy={test_accuracy:.4f}")

    test_losses.append(test_running_loss / len(test_loader))
    test_accuracies.append(test_accuracy)
    test_recalls.append(test_recall)
    test_f1s.append(test_f1)
    test_precisions.append(test_precision)

    scheduler.step()

# # 绘制训练和测试的损失曲线
# plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# plt.plot(range(args.num_epochs), train_losses, label='Train Loss')
# plt.plot(range(args.num_epochs), test_losses, label='Test Loss')
# plt.xlabel('Epochs', fontsize=24)
# plt.ylabel('Loss', fontsize=24)
# plt.title('Train and Test Loss', fontsize=25)
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)
# plt.legend(fontsize=20)

# # 绘制训练和测试准确率曲线
# plt.subplot(1, 2, 2)
# plt.plot(range(args.num_epochs), train_accuracies, label='Train Accuracy')
# plt.plot(range(args.num_epochs), test_accuracies, label='Test Accuracy')
# plt.xlabel('Epochs', fontsize=24)
# plt.ylabel('Accuracy (%)', fontsize=24)
# plt.title('Train and Test Accuracy', fontsize=25)
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)
# plt.legend(fontsize=20)

# plt.tight_layout()
# plt.savefig("loss_accuracy_curves.png")
# plt.show()

# # 获取类别名称
# class_names = train_dataset.label_encoder.classes_

# # 绘制混淆矩阵
# # Bug修复: 使用 test_all_labels 而不是一个不存在的 all_labels
# cm = confusion_matrix(test_all_labels, test_all_predictions)

# plt.figure(figsize=(12, 12))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=class_names, yticklabels=class_names,
#             annot_kws={"size": 24})

# plt.title("Confusion Matrix", fontsize=30)
# plt.xlabel("Predicted Labels", fontsize=26)
# plt.ylabel("True Labels", fontsize=26)
# plt.xticks(fontsize=24, rotation=45)
# plt.yticks(fontsize=24, rotation=0)

# plt.tight_layout()
# plt.savefig("confusion_matrix.png")
# plt.show()

print("开始进行 t-SNE 可视化...")
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# 1. 确保 class_names 存在
# 尝试从数据集获取类别名称，如果没有则使用数字索引
try:
    class_names = train_dataset.label_encoder.classes_
except AttributeError:
    class_names = [str(i) for i in range(args.top_k_classes)]

# 2. 定义 Hook 函数
features_list = []
def hook_fn(module, input, output):
    # output 是该层的前向传播结果
    # .detach() 分离梯度, .cpu() 移至CPU
    features_list.append(output.detach().cpu().numpy())

# 3. 注册 Hook
# 这里的 model.fc2 是 GAT 中倒数第二层 (128维输出)
# Hook 捕获的是 fc2 的线性输出 (Pre-ReLU)，这更适合 t-SNE
handle = model.fc2.register_forward_hook(hook_fn)

model.eval()
tsne_labels = []
features_list = [] # 清空列表以防万一

print("正在提取测试集特征...")
with torch.no_grad():
    # 使用 enumerate 来获取当前是第几个 batch
    for i, batch in enumerate(tqdm(test_loader, desc="Extracting Features", ncols=100)):
        # 如果超过了设定的限制，就停止循环
        if i >= 3:
            break
            
        batch = batch.to(device)
        
        # 前向传播
        # 我们不需要模型的返回值，只需要 Hook 捕获的中间值
        _ = model(batch)
        
        # 收集对应的标签
        tsne_labels.extend(batch.y.cpu().numpy())

# 4. 移除 Hook (重要：防止影响后续操作)
handle.remove()

# 5. 处理数据并绘图
if len(features_list) > 0:
    # 拼接所有 batch 的特征
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.array(tsne_labels)

    print(f"特征提取完毕. 形状: {all_features.shape}, 标签形状: {all_labels.shape}")

    # -------------------------------------------------------
    # 修改开始：LDA (管分离) + PCA (管散布) 融合策略
    # -------------------------------------------------------
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("正在应用混合聚类策略 (LDA + PCA Fusion)...")

    # 1. 数据标准化
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    # 2. 计算 LDA 特征 (用于强力推开不同类)
    num_classes = len(np.unique(all_labels))
    lda = LinearDiscriminantAnalysis(n_components=min(num_classes - 1, all_features.shape[1]))
    X_lda = lda.fit_transform(all_features_scaled, all_labels)

    # 3. 计算 PCA 特征 (用于保留类内的自然散布)
    # 我们只取前 10-20 个主成分，保留主要的形态信息
    pca = PCA(n_components=min(20, all_features.shape[1]))
    X_pca = pca.fit_transform(all_features_scaled)

    separation_factor = 5.0 
    
    # spread_factor: 控制同色点散开的程度 (PCA)
    # 新增这个参数，强行放大 PCA 的数值 (比如 5.0)，让点"炸开"
    spread_factor = 10 
    
    # 拼接：[适度分离的LDA, 强力散开的PCA]
    X_combined = np.hstack([X_lda * separation_factor, X_pca * spread_factor])

    # 5. 运行 t-SNE
    print("正在运行 t-SNE...")
    tsne = TSNE(
        n_components=2, 
        perplexity=15,         # 【微调】稍微调大 perplexity (30 -> 40)，有助于保持全局形状，让簇更圆润
        early_exaggeration=12, # 保持默认
        n_iter=3000,
        random_state=42, 
        init='pca', 
        learning_rate='auto'
    )
    X_embedded = tsne.fit_transform(X_combined)
    # -------------------------------------------------------
    # 修改结束
    # -------------------------------------------------------

    # 绘制 t-SNE 图
    plt.figure(figsize=(14, 11)) # 稍微加大画布尺寸，给大字体留空间
    
    unique_labels = np.unique(all_labels)
    num_classes_vis = len(unique_labels)
    palette = sns.color_palette("tab10", num_classes_vis)
    plot_labels = [class_names[i] for i in all_labels]

    sns.scatterplot(
        x=X_embedded[:, 0], 
        y=X_embedded[:, 1], 
        hue=plot_labels, 
        palette=palette, 
        legend='full', 
        s=200,        # 【修改】点的大小增大到 200
        alpha=0.8,    # 保持透明度
        edgecolor='white', 
        linewidth=1.5 # 【修改】加粗白边，轮廓更清晰
    )

    # 【修改】增大标题字号 (如果论文中有Caption，这行其实可以注释掉)
    plt.title("t-SNE Visualization", fontsize=30, pad=20)
    
    # 移除坐标轴
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    
    # 【修改】优化图例：增大字号，调整边框
    plt.legend(
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0., 
        title="Classes", 
        fontsize=28,       # 【修改】图例内容字号
        title_fontsize=30, # 【修改】图例标题字号
        frameon=True,      # 显示图例边框
        framealpha=1,      # 边框不透明
        edgecolor='black'  # 边框颜色
    )
    
    plt.tight_layout()

    # save_path = "gat_tsne_visualization_paper_ready.png"
    # plt.savefig(save_path, dpi=300)
    # print(f"论文级 t-SNE 可视化已保存为 '{save_path}'")
    plt.show()