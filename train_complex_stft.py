import os
import random

import librosa
import numpy as np
import torch.nn as nn
import torch
import seaborn as sns
import argparse
from sklearn.preprocessing import LabelEncoder
from ssqueezepy import ssq_stft
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from torch.optim.lr_scheduler import StepLR
from Complex_CNN import STFT_complex
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from CNN2d import CNN2d
from complexNNstft import STFT_complexnn


# 设置ArgumentParser以支持命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Audio Classification using STFT features")

    # 添加需要的参数
    parser.add_argument('--data_path', type=str, default=r"D:\PyCharm\underwater-data\watkins_1s", help='路径至数据集')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='权重衰减系数')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练的轮数')
    parser.add_argument('--model', type=str, choices=['CNN2d', 'STFT_complexnn'], default='CNN2d', help='选择模型')
    parser.add_argument('--segment_duration', type=int, default=1, help='音频片段时长（秒）')
    parser.add_argument('--n_segments', type=int, default=4, help='分段数')
    parser.add_argument('--sr', type=int, default=40000, help='音频采样率')
    parser.add_argument('--top_k_classes', type=int, default=6, help='选择样本数最多的前k个类别')

    args = parser.parse_args()
    return args


# 传入参数
args = parse_args()


# Focal Loss定义
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


# Padding方法
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


# 获取音频路径和标签
def get_audio_paths_and_labels(data_path, top_k_classes=None):
    category_paths = {}
    class_counts = {}

    # 统计每个类别的样本数
    for class_folder in os.listdir(data_path):
        class_folder_path = os.path.join(data_path, class_folder)
        if os.path.isdir(class_folder_path):
            file_paths = [os.path.join(class_folder_path, f) for f in os.listdir(class_folder_path) if
                          f.endswith(".wav")]
            category_paths[class_folder] = file_paths
            class_counts[class_folder] = len(file_paths)

    # 如果需要选择top_k_classes个类别
    if top_k_classes:
        # 根据样本数降序排序，选择前k个类别
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        top_k_classes = sorted_classes[:top_k_classes]
        # 计算前k类的总样本数
        total_samples = sum(count for _, count in top_k_classes)

        # 打印前k类的总样本数
        print(f"前{top_k_classes}类的总样本数为: {total_samples}")
        # 更新category_paths，只保留前k个类别的数据
        selected_classes = {class_name: category_paths[class_name] for class_name, _ in top_k_classes}
        return selected_classes
    return category_paths


# # 特征提取
# def features_extractor(filename):
#     audio, sr = librosa.load(filename, sr=40000)
#     audio = padding1s(y=audio, sr=sr)
#     audio = (audio - np.mean(audio)) / np.std(audio)
#     n_fft = 512
#     hop_length = 256
#     stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
#     magnitude, _ = librosa.magphase(stft_result)
#     return magnitude

def features_extractor(filename):
    audio, sr = librosa.load(filename, sr=args.sr)
    audio = padding1s(y=audio, sr=args.sr)
    audio = (audio - np.mean(audio)) / np.std(audio)
    # audio = audio / np.max(np.abs(audio))
    n_fft = 512
    hop_length = 256

    # 计算STFT
    stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    # # 最大幅度归一化
    # max_magnitude = np.max(np.abs(stft_result))
    # stft_normalized = stft_result / max_magnitude

    # # 计算同步压缩STFT
    stft_ssq, _, _, _ = ssq_stft(audio, fs=sr, n_fft=n_fft, hop_len=hop_length, window='hann')
    #
    # # 分离幅度和相位
    magnitude = np.abs(stft_ssq)
    # phase = np.angle(stft_ssq)
    #
    # # 对幅度进行归一化
    # magnitude_normalized = magnitude / np.max(magnitude)  # 归一化到[0, 1]区间
    #
    # # 重构STFT：归一化后的幅度与原相位相乘
    # stft_reconstructed = magnitude_normalized * np.exp(1j * phase)

    return magnitude

# # 特征提取
# def features_extractor(filename):
#     # 读取音频文件
#     audio, sr = librosa.load(filename, sr=40000)
#
#     # 对音频信号进行填充处理
#     audio = padding1s(y=audio, sr=sr)
#
#     # 设置梅尔谱图的参数
#     n_fft = 512  # FFT窗口大小
#     hop_length = 256  # 每次跳跃的样本数
#     n_mels = 64  # 梅尔频带的数量
#
#     # 计算梅尔谱图
#     mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#     # print(mel_spec)
#
#     # 对梅尔谱图进行归一化处理
#     mel_spec = librosa.power_to_db(mel_spec)  # 将功率谱转换为对数谱
#
#     # 归一化到[0, 1]的范围
#     mel_spec = (mel_spec - np.min(mel_spec)) / (np.max(mel_spec) - np.min(mel_spec)) # (64, 157)
#     # print(mel_spec.shape)
#
#     return mel_spec


# 数据集类
class complexstftDataset(Dataset):
    def __init__(self, file_paths, segment_duration=1, n_segments=4, sr=args.sr):
        self.file_paths = file_paths
        self.segment_duration = segment_duration
        self.n_segments = n_segments
        self.sr = sr
        self.label_encoder = LabelEncoder()
        self.labels = []
        for file_path in self.file_paths:
            label = os.path.basename(os.path.dirname(file_path))
            self.labels.append(label)
        self.label_encoder.fit(self.labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = os.path.basename(os.path.dirname(file_path))
        label = self.label_encoder.transform([label])[0]
        label = torch.tensor(label, dtype=torch.long)
        stft = features_extractor(file_path)
        stft_tensor = torch.tensor(stft, dtype=torch.float32)
        # stft_tensor = torch.tensor(stft, dtype=torch.complex64)
        stft_tensor = stft_tensor.unsqueeze(0)
        return stft_tensor, label


# 获取音频文件路径
category_paths = get_audio_paths_and_labels(args.data_path, top_k_classes=args.top_k_classes)

# 固定随机种子
def set_seed(seed):
    random.seed(seed)  # Python内置的random模块
    np.random.seed(seed)  # NumPy随机模块
    torch.manual_seed(seed)  # PyTorch CPU随机数种子
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU随机数种子
    torch.backends.cudnn.deterministic = True  # 设置为True以确保可重复性
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化

# 设定随机种子
set_seed(41)

# 获取所有文件路径和标签
all_files = []
all_labels = []
for label, files in category_paths.items():
    all_files.extend(files)
    all_labels.extend([label] * len(files))

# 分层抽样划分训练集和测试集 (80:20)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(all_files, all_labels):
    train_files = [all_files[i] for i in train_idx]
    test_files = [all_files[i] for i in test_idx]

# 创建训练集和测试集的数据集对象
train_dataset = complexstftDataset(train_files, segment_duration=args.segment_duration, n_segments=args.n_segments,
                                   sr=args.sr)
test_dataset = complexstftDataset(test_files, segment_duration=args.segment_duration, n_segments=args.n_segments,
                                  sr=args.sr)

# 使用DataLoader加载数据
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 根据args选择模型
if args.model == 'CNN2d':
    model = CNN2d().to(device)  # 请确保CNN2d是已定义的模型
elif args.model == 'STFT_complexnn':
    model = STFT_complexnn().to(device)  # 请确保STFT_complex是已定义的模型

# 初始化损失函数和优化器
criterion = FocalLoss(alpha=0.25, gamma=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


# 计算 Specificity
def calculate_specificity(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificity = []
    for i in range(num_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    return np.mean(specificity)


# 训练过程
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
train_recalls = []
test_recalls = []
train_f1s = []
test_f1s = []
test_specificities = []

for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with tqdm(total=len(train_loader), desc=f"Epoch [{epoch + 1}/{args.num_epochs}]", ncols=100) as pbar:
        for stft, data in train_loader:
            stft_spectrogram = stft.to(device)
            data = data.to(device)

            optimizer.zero_grad()

            output = model(stft_spectrogram)
            loss = criterion(output, data)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total += data.size(0)
            correct += (predicted == data).sum().item()

            running_loss += loss.item()

            all_labels.extend(data.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            avg_loss = running_loss / (pbar.n + 1)
            accuracy = (100 * correct / total)
            pbar.set_postfix(loss=avg_loss, accuracy=accuracy)
            pbar.update(1)

    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    specificity = calculate_specificity(all_labels, all_predictions, num_classes=output.size(1))

    print(
        f"Epoch {epoch + 1} - Train: Recall={recall:.4f}, F1={f1:.4f}, Specificity={specificity:.4f}, Accuracy={100 * correct / total:.4f}")

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)
    train_recalls.append(recall)
    train_f1s.append(f1)

    # 验证过程
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc=f"Testing Epoch [{epoch + 1}/{args.num_epochs}]", ncols=100) as pbar:
            for stft, data in test_loader:
                stft_spectrogram = stft.to(device)
                data = data.to(device)

                output = model(stft_spectrogram)
                loss = criterion(output, data)
                running_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += data.size(0)
                correct += (predicted == data).sum().item()

                all_labels.extend(data.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                avg_loss = running_loss / (pbar.n + 1)
                accuracy = 100 * correct / total
                pbar.set_postfix(loss=avg_loss, accuracy=accuracy)
                pbar.update(1)

    test_recall = recall_score(all_labels, all_predictions, average='weighted')
    test_f1 = f1_score(all_labels, all_predictions, average='weighted')
    test_specificity = calculate_specificity(all_labels, all_predictions, num_classes=output.size(1))

    print(
        f"Epoch {epoch + 1} - Test: Recall={test_recall:.4f}, F1={test_f1:.4f}, Specificity={test_specificity:.4f}, Accuracy={(100 * correct / total):.4f}")
    test_losses.append(running_loss / len(test_loader))
    test_accuracies.append(100 * correct / total)
    test_recalls.append(test_recall)
    test_f1s.append(test_f1)
    test_specificities.append(test_specificity)
    # 更新学习率
    scheduler.step()


# 绘制训练和测试的损失曲线
plt.figure(figsize=(20, 10))

# 绘制训练和测试损失
plt.subplot(1, 2, 1)
plt.plot(range(args.num_epochs), train_losses, label='Train Loss', color='blue')
plt.plot(range(args.num_epochs), test_losses, label='Test Loss', color='red')
plt.xlabel('Epochs', fontsize= 24)
plt.ylabel('Loss', fontsize= 24)
plt.title('Train and Test Loss', fontsize= 25)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=20)

# 绘制训练和测试准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(args.num_epochs), train_accuracies, label='Train Accuracy', color='blue')
plt.plot(range(args.num_epochs), test_accuracies, label='Test Accuracy', color='red')
plt.xlabel('Epochs', fontsize= 24)
plt.ylabel('Accuracy', fontsize= 24)
plt.title('Train and Test Accuracy', fontsize= 25)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=20)

# 显示图像
plt.tight_layout()
plt.show()

# 训练和测试过程之前，获取类别名称
class_names = train_dataset.label_encoder.classes_

# 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_predictions)

# 使用Seaborn绘制热图
plt.figure(figsize=(12, 12))

# 设置字体大小参数
title_fontsize = 30
label_fontsize = 26
tick_fontsize = 24
annot_fontsize = 24
cbar_fontsize = 24

# 绘制热图
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": annot_fontsize},  # 设置数字字体大小
            cbar_kws={"shrink": 0.8})  # 调整颜色条大小

# 设置标题和标签
plt.title("Confusion Matrix", fontsize=title_fontsize)
plt.xlabel("Predicted Labels", fontsize=label_fontsize)
plt.ylabel("True Labels", fontsize=label_fontsize)

# 设置坐标轴刻度标签字体大小
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# 设置颜色条刻度字体大小
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=cbar_fontsize)

plt.tight_layout()  # 自动调整布局
plt.show()

