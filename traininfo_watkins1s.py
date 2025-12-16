import os
import librosa
import numpy as np
import torch.nn as nn
import torch
import seaborn as sns
import argparse
import random
from ssqueezepy import ssq_stft
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from CVCNNGATconcat_watkins1s import DualPathFusionModel_watkins
from sklearn.manifold import TSNE


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Path Audio Classification using ComplexCNN and GAT")
    parser.add_argument('--audio_data_path', type=str, default=r"D:\PyCharm\underwater-data\watkins_1s",
                        help='Path to the raw audio (.wav) dataset directory')
    parser.add_argument('--graph_data_path', type=str, default=r"D:\PyCharm\underwater-data\processed_graphs_watkins1s", help='Path to the preprocessed graph (.pt) data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--top_k_classes', type=int, default=6, help='Select top k classes with the most samples')
    parser.add_argument('--sr', type=int, default=40000, help='Audio sampling rate')
    parser.add_argument('--fusion_mode', type=str, default='msaff_contrastive', 
                        choices=['concat', 'concat_contrastive', 'msaff_contrastive'], 
                        help="Mode for fusion: 'concat' (baseline), "
                             "'concat_contrastive' (late fusion + late contrastive), "
                             "'msaff_contrastive' (msaff + late contrastive)")

    return parser.parse_args()

def padding1s(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 1:
        padding_length = int((1 - duration) * sr)
        y_padded = np.pad(y, (0, padding_length), mode='constant')
    elif duration > 1:
        y_padded = y[:int(1 * sr)]
    else:
        y_padded = y
    return y_padded

def features_extractor(filename, sr):
    audio, _ = librosa.load(filename, sr=sr)
    audio = (audio - np.mean(audio)) / np.std(audio)
    audio = padding1s(audio, sr=sr)
    n_fft = 512
    hop_length = 256
    stft_ssq, _, _, _ = ssq_stft(audio, fs=sr, n_fft=n_fft, hop_len=hop_length, window='hann')
    return stft_ssq

class DualPathDataset(Dataset):
    """
    新增: 为双路模型自定义的数据集。
    每次调用都同时返回 STFT 输入和 Graph 输入。
    """

    def __init__(self, wav_file_paths, graph_data_root, sr, transform=None, pre_transform=None):
        super().__init__(transform, pre_transform)
        self.wav_file_paths = wav_file_paths
        self.graph_data_root = graph_data_root
        self.sr = sr

        self.audio_data_root = os.path.dirname(os.path.dirname(wav_file_paths[0]))

        # 从文件路径创建并拟合标签编码器
        self.labels_str = [os.path.basename(os.path.dirname(p)) for p in self.wav_file_paths]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels_str)
        self.classes_ = self.label_encoder.classes_

    def len(self):
        return len(self.wav_file_paths)

    def get(self, idx):
        wav_path = self.wav_file_paths[idx]

        # 1. 处理音频通路 (Complex CNN)
        stft_complex = features_extractor(wav_path, self.sr)
        stft_tensor = torch.tensor(stft_complex, dtype=torch.complex64).unsqueeze(0)

        # 2. 处理图通路 (GAT)
        # 从 wav 路径推断出对应的 pt 文件路径
        relative_path = os.path.relpath(wav_path, self.audio_data_root)
        graph_filename = os.path.splitext(os.path.basename(relative_path))[0] + ".pt"
        class_folder = os.path.dirname(relative_path)
        graph_path = os.path.join(self.graph_data_root, class_folder, graph_filename)

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found for {wav_path} at expected location {graph_path}")

        graph_data = torch.load(graph_path, weights_only=False)

        # 3. 获取标签并添加到 graph_data 对象中
        label_str = os.path.basename(os.path.dirname(wav_path))
        label_int = self.label_encoder.transform([label_str])[0]
        graph_data.y = torch.tensor(label_int, dtype=torch.long)

        # 返回一个元组，DataLoader会分别对元组中的元素进行批处理
        return (stft_tensor, graph_data)

class SupervisedContrastiveLoss(nn.Module):
    """
    有监督对比损失 (SupCon) 的标准实现。
    https://arxiv.org/pdf/2004.11362.pdf
    """
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, cnn_features, gat_features, labels):
        """
        输入:
        - cnn_features: [batch_size, D]
        - gat_features: [batch_size, D]
        - labels: [batch_size]
        """
        
        # 1. 规范化和组合
        # [B, D]
        cnn_features = F.normalize(cnn_features, p=2, dim=1)
        # [B, D]
        gat_features = F.normalize(gat_features, p=2, dim=1)
        
        # 组合成一个大批次: [2 * batch_size, D]
        features = torch.cat([cnn_features, gat_features], dim=0)
        
        # 2. 计算相似度矩阵
        # [2B, 2B]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 3. 创建标签掩码 (Mask)
        # 组合标签: [2B]
        sup_labels = torch.cat([labels, labels], dim=0)
        batch_size_2B = sup_labels.shape[0]

        # [2B, 2B] 掩码， label_mask[i, j] = 1 if labels[i] == labels[j]
        label_mask = (sup_labels.unsqueeze(0) == sup_labels.unsqueeze(1)).to(features.device)

        # 4. 创建对角线掩码 (排除与自身的比较)
        # [2B, 2B]
        identity_mask = torch.eye(batch_size_2B, device=features.device, dtype=torch.bool)
        
        # --- 计算损失 ---
        
        # 5. 屏蔽掉对角线 (自-相似)
        # logits 现在是 [2B, 2B]，对角线为 -inf
        logits = similarity_matrix.masked_fill(identity_mask, -1e9)

        # 6. positives_mask: [2B, 2B]
        #    所有正样本对 (同类别, 非自身)
        positives_mask = label_mask & ~identity_mask
        
        # 7. 计算 Log-Softmax
        # log_probs: [2B, 2B]
        log_probs = F.log_softmax(logits, dim=1)

        # 8. 计算每个锚点(row)的损失
        #    我们想最大化所有正样本的 log-prob
        
        # (log_probs * positives_mask) -> 只保留正样本的 log-prob
        # .sum(dim=1) -> 对每个锚点，将其所有正样本的 log-prob 相加
        # .mean() -> 在整个批次(2B)上取平均
        
        # 计算每个锚点有多少个正样本
        num_positives_per_anchor = positives_mask.sum(dim=1)
        
        # 避免除以零 (如果一个锚点在批次中没有其他正样本)
        num_positives_per_anchor = torch.where(
            num_positives_per_anchor == 0, 
            torch.tensor(1.0, device=features.device), 
            num_positives_per_anchor
        )
        
        # 计算每个锚点的平均 log-prob
        # (sum of log-probs) / (number of positives)
        log_prob_pos = (log_probs * positives_mask).sum(dim=1) / num_positives_per_anchor
        
        # SupCon 损失是这个值的负数
        # 我们在整个批次(2B)上取平均
        loss = -log_prob_pos.mean()
        
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        #F_loss = BCE_loss + 2 * (1 - pt)
        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()

def get_audio_paths_and_labels(data_path, top_k_classes=None):
    category_paths = {}
    class_counts = {}
    for class_folder in os.listdir(data_path):
        class_folder_path = os.path.join(data_path, class_folder)
        if os.path.isdir(class_folder_path):
            file_paths = [os.path.join(class_folder_path, f) for f in os.listdir(class_folder_path) if
                          f.endswith(".wav")]
            if file_paths:
                category_paths[class_folder] = file_paths
                class_counts[class_folder] = len(file_paths)
    if top_k_classes:
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:top_k_classes]
        print(f"Selected Top {len(sorted_classes)} classes.")
        return {class_name: category_paths[class_name] for class_name, _ in sorted_classes}
    return category_paths


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_precision(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    precision_scores = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        precision_scores.append(tp / (tp + fp)) if (tp + fp) > 0 else precision_scores.append(0)
    return np.mean(precision_scores)


# ========================================================================================
# 3. 主执行流程 (Main Execution Flow)
# ========================================================================================
if __name__ == '__main__':
    args = parse_args()
    #set_seed(41)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 修改: 修正 AttributeError ---
    print(f"Running in fusion mode: {args.fusion_mode}") 
    # --- 结束修改 ---

    # 1. Get all audio file paths
    category_paths = get_audio_paths_and_labels(args.audio_data_path, top_k_classes=args.top_k_classes)
    all_files = [file for files in category_paths.values() for file in files]
    all_labels = [label for label, files in category_paths.items() for _ in files]

    # 2. Stratified split for train and test sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(all_files, all_labels))
    train_files = [all_files[i] for i in train_idx]
    test_files = [all_files[i] for i in test_idx]

    # 3. Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = DualPathDataset(train_files, args.graph_data_path, sr=args.sr)
    test_dataset = DualPathDataset(test_files, args.graph_data_path, sr=args.sr)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("Done.")

    # 4. Initialize model, loss function, and optimizer
    model = DualPathFusionModel_watkins(  # 确保这个名称与您的 import 一致
        gat_in_channels=79,
        gat_hidden_channels=512,
        num_classes=args.top_k_classes,
        dropout_rate=0.2,
        fusion_mode=args.fusion_mode  # <-- 修改: 将 args.fusion_mode 传入
    ).to(device)

    # --- 修改: 定义损失函数和超参数 ---
    criterion = FocalLoss(alpha=0.25, gamma=2).to(device)
    
    # 只有在需要时才定义对比损失
    if args.fusion_mode in ['concat_contrastive', 'msaff_contrastive']:
        contrast_criterion = SupervisedContrastiveLoss().to(device)
        # alpha 现在是唯一的对比损失权重 (用于晚期)
        alpha = 0.1 
    # --- 结束修改 ---

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # 5. Training and evaluation loop
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    train_recalls, test_recalls = [], []
    train_f1s, test_f1s = [], []
    train_precisions, test_precisions = [], []

    print("Starting training...")
    for epoch in range(args.num_epochs):
        # --- Training ---
        model.train()
        # --- 修改: 简化损失日志 ---
        running_loss_cls = 0.0
        running_loss_contrast = 0.0 # 只需要一个对比损失日志
        correct, total = 0, 0
        # --- 结束修改 ---
        all_train_labels, all_train_predictions = [], []

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Train]", ncols=100)
        
        # --- 修改: 训练循环 ---
        for stft_batch, graph_batch in pbar_train:
            stft_batch = stft_batch.to(device)
            graph_batch = graph_batch.to(device)
            labels = graph_batch.y

            optimizer.zero_grad()
            
            # --- 新增: 动态处理模型输出 ---
            model_outputs = model(stft_batch, graph_batch)

            # 初始化对比损失
            L_contrast = torch.tensor(0.0, device=device)

            if args.fusion_mode == 'concat':
                logits = model_outputs
                L_classify = criterion(logits, labels)
            
            # 模式 2 和 3 都返回 3 个值 (logits, cnn_late_vec, gat_late_vec)
            elif args.fusion_mode in ['concat_contrastive', 'msaff_contrastive']:
                logits, cnn_late_vec, gat_late_vec = model_outputs
                L_classify = criterion(logits, labels)
                L_contrast = contrast_criterion(cnn_late_vec, gat_late_vec, labels)
            
            # 总损失
            if args.fusion_mode in ['concat_contrastive', 'msaff_contrastive']:
                L_total = L_classify + alpha * L_contrast # (alpha 是唯一的权重)
            else: # 'concat'
                L_total = L_classify
            # --- 结束新增 ---
            
            L_total.backward()
            optimizer.step()

            # --- 修改: 更新损失日志 ---
            running_loss_cls += L_classify.item()
            if args.fusion_mode in ['concat_contrastive', 'msaff_contrastive']:
                running_loss_contrast += L_contrast.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_predictions.extend(predicted.cpu().numpy())
            
            # --- 修改: 动态更新进度条 ---
            postfix_dict = {
                'cls_loss': running_loss_cls / (pbar_train.n + 1),
                'acc': 100 * correct / total
            }
            if args.fusion_mode in ['concat_contrastive', 'msaff_contrastive']:
                postfix_dict['con_loss'] = running_loss_contrast / (pbar_train.n + 1)
            pbar_train.set_postfix(postfix_dict)
            # --- 结束修改 ---
        
        # --- 结束训练循环修改 ---

        # (训练指标计算和打印部分保持不变)
        train_accuracy = 100 * correct / total
        train_recall = recall_score(all_train_labels, all_train_predictions, average='weighted', zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_predictions, average='weighted', zero_division=0)
        train_precision = calculate_precision(all_train_labels, all_train_predictions, num_classes=args.top_k_classes)
        print(f"Epoch {epoch + 1} - Train: Recall={train_recall:.4f}, F1={train_f1:.4f}, Precision={train_precision:.4f}, Accuracy={train_accuracy:.2f}%")

        train_losses.append(running_loss_cls / len(train_loader)) # 只记录主分类损失
        train_accuracies.append(train_accuracy)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)
        train_precisions.append(train_precision)

        # --- Evaluation ---
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_test_labels, all_test_predictions = [], []

        pbar_test = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Test ]", ncols=100)
        with torch.no_grad():
            
            # --- 修改: 评估循环 ---
            for stft_batch, graph_batch in pbar_test:
                stft_batch = stft_batch.to(device)
                graph_batch = graph_batch.to(device)
                labels = graph_batch.y

                # --- 新增: 动态处理模型输出 ---
                model_outputs = model(stft_batch, graph_batch)
                
                # 不管返回几个值，logits 永远是第一个
                if isinstance(model_outputs, tuple):
                    logits = model_outputs[0]
                else:
                    logits = model_outputs
                
                # 只计算分类损失
                loss = criterion(logits, labels)
                # --- 结束修改 ---
                
                running_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_test_labels.extend(labels.cpu().numpy())
                all_test_predictions.extend(predicted.cpu().numpy())

                pbar_test.set_postfix(loss=running_loss / (pbar_test.n + 1), acc=100 * correct / total)

        # (评估指标计算、打印、scheduler.step() 和 绘图部分 保持不变)
        test_accuracy = 100 * correct / total
        test_recall = recall_score(all_test_labels, all_test_predictions, average='weighted', zero_division=0)
        test_f1 = f1_score(all_test_labels, all_test_predictions, average='weighted', zero_division=0)
        test_precision = calculate_precision(all_test_labels, all_test_predictions, num_classes=args.top_k_classes)
        print(f"Epoch {epoch + 1} - Test:  Recall={test_recall:.4f}, F1={test_f1:.4f}, Precision={test_precision:.4f}, Accuracy={test_accuracy:.2f}%")

        test_losses.append(running_loss / len(test_loader))
        test_accuracies.append(test_accuracy)
        test_recalls.append(test_recall)
        test_f1s.append(test_f1)
        test_precisions.append(test_precision)

        scheduler.step()

    print("Training finished.")

    # save_dir = "saved_models"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # # 2. 构建文件名 (包含融合模式和最终准确率，方便区分)
    # # 例如: saved_models/watkins_model_concat_acc94.52.pth
    # model_filename = f"watkins_model_MLP_{args.fusion_mode}_acc{test_accuracy:.2f}.pth"
    # save_path = os.path.join(save_dir, model_filename)

    # # 3. 保存模型参数
    # torch.save(model.state_dict(), save_path)
    # print(f"Model state_dict saved to: {save_path}")

    # MODIFICATION: Add plotting for loss and accuracy curves
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(range(args.num_epochs), train_losses, label='Train Loss')
    plt.plot(range(args.num_epochs), test_losses, label='Test Loss')
    plt.xlabel('Epochs', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.title('Train and Test Loss', fontsize=25)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=20)

    plt.subplot(1, 2, 2)
    plt.plot(range(args.num_epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(args.num_epochs), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs', fontsize=24)
    plt.ylabel('Accuracy (%)', fontsize=24)
    plt.title('Train and Test Accuracy', fontsize=25)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=20)
    plt.tight_layout()
    #plt.savefig("loss_accuracy_curves.png")
    plt.show()

    # MODIFICATION: Add plotting for the confusion matrix
    class_names = train_dataset.classes_
    cm = confusion_matrix(all_test_labels, all_test_predictions)

    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 24})
    plt.title("Confusion Matrix", fontsize=30)
    plt.xlabel("Predicted Labels", fontsize=26)
    plt.ylabel("True Labels", fontsize=26)
    plt.xticks(fontsize=24, rotation=45)
    plt.yticks(fontsize=24, rotation=0)
    plt.tight_layout()
    #plt.savefig("confusion_matrix.png")
    plt.show()

    print("Starting Visualization...")
    
    # ==========================================
    # --- 新增: t-SNE 和 Heatmap 可视化 ---
    # ==========================================
    
    model.eval() # 确保模型处于评估模式
    
    all_features = []
    all_labels_vis = []
    
    # 用于存储第一个样本的注意力信息 (用于 Heatmap)
    sample_edge_index = None
    sample_alpha = None
    heatmap_drawn = False
    
    print("Collecting features for visualization...")
    with torch.no_grad():
        for batch_idx, (stft_batch, graph_batch) in enumerate(tqdm(test_loader)):
            stft_batch = stft_batch.to(device)
            graph_batch = graph_batch.to(device)
            labels = graph_batch.y
            
            # --- 关键: 开启 visualize=True ---
            # 您的 DualPathFusionModel_watkins 需要支持这个参数
            # 如果您用的是我之前给您的 "支持可视化接口" 的模型版本，这将正常工作
            try:
                # 尝试调用可视化接口
                # 返回: logits, features, edge_index, alpha
                _, features, edge_index, alpha = model(stft_batch, graph_batch, visualize=True)
                
                # 收集 t-SNE 特征
                all_features.append(features.cpu().numpy())
                all_labels_vis.append(labels.cpu().numpy())
                
                # 捕获第一个 Batch 的注意力信息
                if not heatmap_drawn and alpha is not None:
                    # 提取第一个图的数据 (假设每个图固定 8 个节点)
                    nodes_per_graph = 8 
                    # 筛选出源节点和目标节点都在 0-7 之间的边
                    mask = (edge_index[0] < nodes_per_graph) & (edge_index[1] < nodes_per_graph)
                    
                    sample_edge_index = edge_index[:, mask]
                    sample_alpha = alpha[mask]
                    heatmap_drawn = True
                    
            except TypeError:
                print("警告: 当前模型 forward() 不支持 visualize=True 参数。跳过可视化。")
                break

    # --- 1. 绘制 t-SNE ---
    if len(all_features) > 0:
        print("Plotting t-SNE...")
        all_features = np.concatenate(all_features, axis=0)
        all_labels_vis = np.concatenate(all_labels_vis, axis=0)
        
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(all_features)
        
        plt.figure(figsize=(10, 8))
        # 获取类别数量
        num_classes_vis = len(np.unique(all_labels_vis))
        palette = sns.color_palette("tab10", num_classes_vis)
        
        sns.scatterplot(
            x=X_embedded[:, 0], 
            y=X_embedded[:, 1], 
            hue=all_labels_vis, 
            palette=palette, 
            legend='full', 
            s=60, 
            alpha=0.8
        )
        plt.title(f"t-SNE Visualization ({args.fusion_mode})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        # 保存 t-SNE
        plt.savefig(f"tsne_{args.fusion_mode}.png", dpi=300)
        plt.show()

    # --- 2. 绘制 Attention Heatmap ---
    if sample_edge_index is not None:
        print("Plotting Attention Heatmap...")
        # alpha shape: [num_edges, num_heads] -> 平均 -> [num_edges]
        weights = sample_alpha.mean(dim=1).detach().cpu()
        
        # 构建邻接矩阵 (8x8)
        num_nodes = 8
        attn_matrix = torch.zeros((num_nodes, num_nodes))
        sources = sample_edge_index[0].cpu()
        targets = sample_edge_index[1].cpu()
        
        for s, t, w in zip(sources, targets, weights):
            attn_matrix[s, t] = w

        plt.figure(figsize=(10, 8))
        # 自定义特征名 (请修改为您真实的特征名)
        feature_names = [f"Feat {i+1}" for i in range(num_nodes)]
        
        sns.heatmap(
            attn_matrix.numpy(), 
            xticklabels=feature_names, 
            yticklabels=feature_names, 
            cmap="viridis", 
            annot=False
        )
        plt.title("GAT Attention Weights")
        plt.xlabel("Source")
        plt.ylabel("Target")
        # 保存 Heatmap
        plt.savefig(f"heatmap_{args.fusion_mode}.png", dpi=300)
        plt.show()
    else:
        print("未检测到注意力权重 (可能是 MLP 模式或模型不支持)，跳过 Heatmap。")