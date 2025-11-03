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
from CVCNNGATconcat import DualPathFusionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Path Audio Classification using ComplexCNN and GAT")
    parser.add_argument('--audio_data_path', type=str, default=r"D:\PyCharm\underwater-data\shipsEar2s",
                        help='Path to the raw audio (.wav) dataset directory')
    parser.add_argument('--graph_data_path', type = str, default = r"D:\PyCharm\underwater-data\processed_graphs", help = 'Path to the preprocessed graph (.pt) data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--top_k_classes', type=int, default=5, help='Select top k classes with the most samples')
    parser.add_argument('--sr', type=int, default=7000, help='Audio sampling rate')
    return parser.parse_args()


# --- 2.2 特征提取器 (SSQ-STFT) ---
def features_extractor(filename, sr):
    audio, _ = librosa.load(filename, sr=sr)
    audio = (audio - np.mean(audio)) / np.std(audio)
    n_fft = 2048
    hop_length = 512
    stft_ssq, _, _, _ = ssq_stft(audio, fs=sr, n_fft=n_fft, hop_len=hop_length, window='boxcar')
    return stft_ssq


# --- 2.3 自定义双路数据集 ---
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


# --- 2.4 其他辅助函数 ---
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
    # set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    model = DualPathFusionModel(
        gat_in_channels=28,
        gat_hidden_channels=512,
        num_classes=args.top_k_classes,
        dropout_rate=0.2
    ).to(device)

    criterion = FocalLoss(alpha=0.25, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # 5. Training and evaluation loop
    # MODIFICATION: Initialize lists to store all metrics
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    train_recalls, test_recalls = [], []
    train_f1s, test_f1s = [], []
    train_precisions, test_precisions = [], []

    print("Starting training...")
    for epoch in range(args.num_epochs):
        # --- Training ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_train_labels, all_train_predictions = [], []

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Train]", ncols=100)
        for stft_batch, graph_batch in pbar_train:
            stft_batch = stft_batch.to(device)
            graph_batch = graph_batch.to(device)
            labels = graph_batch.y

            optimizer.zero_grad()
            outputs = model(stft_batch, graph_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # MODIFICATION: Collect all labels and predictions for metric calculation
            all_train_labels.extend(labels.cpu().numpy())
            all_train_predictions.extend(predicted.cpu().numpy())
            
            pbar_train.set_postfix(loss=running_loss / (pbar_train.n + 1), acc=100 * correct / total)

        # MODIFICATION: Calculate and store all training metrics for the epoch
        train_accuracy = 100 * correct / total
        train_recall = recall_score(all_train_labels, all_train_predictions, average='weighted', zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_predictions, average='weighted', zero_division=0)
        train_precision = calculate_precision(all_train_labels, all_train_predictions, num_classes=args.top_k_classes)
        print(f"Epoch {epoch + 1} - Train: Recall={train_recall:.4f}, F1={train_f1:.4f}, Precision={train_precision:.4f}, Accuracy={train_accuracy:.2f}%")

        train_losses.append(running_loss / len(train_loader))
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
            for stft_batch, graph_batch in pbar_test:
                stft_batch = stft_batch.to(device)
                graph_batch = graph_batch.to(device)
                labels = graph_batch.y

                outputs = model(stft_batch, graph_batch)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # MODIFICATION: Collect all labels and predictions for metric calculation
                all_test_labels.extend(labels.cpu().numpy())
                all_test_predictions.extend(predicted.cpu().numpy())

                pbar_test.set_postfix(loss=running_loss / (pbar_test.n + 1), acc=100 * correct / total)

        # MODIFICATION: Calculate and store all test metrics for the epoch
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