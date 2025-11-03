import numpy as np
import librosa
import pywt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

def padding2s(y, sr, s):
    # 获取音频信号的时长（秒）
    duration = librosa.get_duration(y=y, sr=sr)

    # 判断音频时长并处理
    if duration < s:
        # 若音频时长小于4秒，进行零填充
        padding_length = int((s - duration) * sr)  # 计算需要填充的样本数
        y_padded = np.pad(y, (0, padding_length), mode='constant')  # 进行零填充
    elif duration > s:
        # 若音频时长超过4秒，截取前4秒
        y_padded = y[:int(s * sr)]  # 仅保留前5秒
    else:
        # 若音频时长正好为4秒，无需处理
        y_padded = y

    return y_padded

def extract_mfcc(y, sr, s, n_mfcc=13):
    # 1. 将音频切分为两个1秒的片段
    # 计算切分点，即1秒对应的采样点数
    split_point = int(sr * (s / 2))

    # 创建两个片段
    segment1 = y[:split_point]
    segment2 = y[split_point: split_point * 2]  # 确保第二个片段也是1秒长

    segments = [segment1, segment2]
    processed_vectors = []

    # 2. 循环处理每个片段
    for segment in segments:
        # 计算MFCC
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)

        # 在时间轴上取平均，得到 (13,) 的向量
        mfcc_mean = np.mean(mfcc, axis=1)

        # 3. 在末尾补一个0，得到 (14,) 的向量
        mfcc_padded = np.pad(mfcc_mean, (0, 1), mode='constant')
        processed_vectors.append(mfcc_padded)

    # 4. 拼接两个处理后的 (14,) 向量，得到 (28,) 的最终向量
    final_vector = np.concatenate(processed_vectors)

    return final_vector


# 计算谱滚降点的函数
def extract_spectral_rolloff(y, sr, target_length):
    # 计算谱滚降点
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # 获取谱滚降点的值（谱滚降点的第一个维度是特征维度，第二个维度是时间帧）
    spectral_rolloff_values = spectral_rolloff[0]

    # 对谱滚降点做归一化
    spectral_rolloff_normalized = (spectral_rolloff_values - np.min(spectral_rolloff_values)) / (
            np.max(spectral_rolloff_values) - np.min(spectral_rolloff_values))

    # 获取当前谱滚降点的长度
    current_length = len(spectral_rolloff_normalized)


    # 如果长度大于126，取前126个元素
    if current_length > target_length:
        spectral_rolloff_normalized = spectral_rolloff_normalized[:target_length]
    # 如果长度小于126，补零到126维
    elif current_length < target_length:
        spectral_rolloff_normalized = np.pad(spectral_rolloff_normalized, (0, target_length - current_length), 'constant')

    # 返回处理后的数据
    return spectral_rolloff_normalized

# 计算过零率的函数
def extract_zero_crosings(y, target_length):
    # 计算过零率
    zero_crossings = librosa.feature.zero_crossing_rate(y)[0]

    # 对过零率进行归一化
    zero_crossings_normalized = (zero_crossings - np.min(zero_crossings)) / \
                                (np.max(zero_crossings) - np.min(zero_crossings))

    # 获取当前过零率的长度
    current_length = len(zero_crossings_normalized)


    # 如果长度大于126，取前126个元素
    if current_length > target_length:
        zero_crossings_normalized = zero_crossings_normalized[:target_length]
    # 如果长度小于126，补零到126维
    elif current_length < target_length:
        zero_crossings_normalized = np.pad(zero_crossings_normalized, (0, target_length - current_length), 'constant')

    # 返回处理后的数据
    return zero_crossings_normalized

def extract_spectral_centroid(y, sr, target_length):
    # 计算谱质心
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # 对谱质心进行归一化
    spectral_centroid_normalized = (spectral_centroid - np.min(spectral_centroid)) / \
                                   (np.max(spectral_centroid) - np.min(spectral_centroid))

    # 获取当前谱质心的长度
    current_length = len(spectral_centroid_normalized)


    if current_length > target_length:
        spectral_centroid_normalized = spectral_centroid_normalized[:target_length]
    elif current_length < target_length:
        spectral_centroid_normalized = np.pad(spectral_centroid_normalized, (0, target_length - current_length), 'constant')

    # 返回处理后的数据
    return spectral_centroid_normalized

def extract_RMS(y, target_length):
    # 计算短时能量（Root Mean Square, RMS）
    rms = librosa.feature.rms(y=y)[0]  # 获取短时能量的第一维（每帧的能量值）

    # 对短时能量进行归一化
    rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))

    # 获取当前短时能量的长度
    current_length = len(rms_normalized)

    if current_length > target_length:
        rms_normalized = rms_normalized[:target_length]
    elif current_length < target_length:
        rms_normalized = np.pad(rms_normalized, (0, target_length - current_length), 'constant')

    # 返回处理后的数据
    return rms_normalized

def extract_entropy(y):
    # 计算短时傅里叶变换（STFT）
    D = librosa.stft(y)

    # 计算功率谱（Power Spectrum）
    S = np.abs(D) ** 2

    # 归一化功率谱
    S_norm = librosa.util.normalize(S)

    # 计算每一帧的功率谱熵
    # 对谱图进行归一化处理
    S_norm = S / np.sum(S, axis=0, keepdims=True)  # 归一化谱图
    S_norm = np.clip(S_norm, 1e-10, 1.0)  # 避免log(0)的情况

    # 计算谱熵
    entropy = -np.sum(S_norm * np.log(S_norm), axis=0)

    # 获取降维目标大小
    target_size = 32  # 降到125维

    # 原始谱熵的长度
    original_size = len(entropy)

    # 创建线性插值函数
    x = np.arange(original_size)  # 原始数据点
    x_new = np.linspace(0, original_size - 1, target_size)  # 目标数据点
    f = interp1d(x, entropy, kind='linear')

    # 进行插值，得到降维后的谱熵数据
    entropy_downsampled = f(x_new)

    return entropy_downsampled


def extract_autocorr(y, sr, target_length, n_fft=2048, hop_length=512):
    # 使用 librosa 的标准分帧功能
    # librosa.util.frame 会自动处理边界填充，确保信号被完整分析
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)

    # 结果的 shape 是 (n_fft, num_frames)
    # 我们需要转置为 (num_frames, n_fft) 以便逐帧处理
    frames = frames.T

    autocorr_values = []

    # 对每一帧计算自相关
    for frame in frames:
        # 计算当前帧的自相关系数
        auto_corr = np.correlate(frame, frame, mode='full')

        # 提取零延迟位置的值
        zero_lag_corr = auto_corr[auto_corr.size // 2]
        autocorr_values.append(zero_lag_corr)

    autocorr_vector = np.array(autocorr_values)

    # 归一化
    min_value = np.min(autocorr_vector)
    max_value = np.max(autocorr_vector)
    if max_value == min_value:
        return np.zeros_like(autocorr_vector)  # 返回与输入形状相同的零向量

    normalized_autocorr = (autocorr_vector - min_value) / (max_value - min_value)

    if len(normalized_autocorr) < target_length:
        normalized_autocorr = np.pad(normalized_autocorr, (0, target_length - len(normalized_autocorr)), 'constant')
    elif len(normalized_autocorr) > target_length:
        normalized_autocorr = normalized_autocorr[:target_length]

    return normalized_autocorr

def extract_stpsd(y, n_components):
    # 1. 计算短时傅里叶变换 (STFT)
    D = librosa.stft(y)
    S = np.abs(D) ** 2
    log_S = librosa.amplitude_to_db(S, ref=np.max)

    # 2. 转置数据，使每一行代表一个时间帧
    # 形状变为 (n_frames, n_freqs)，例如 (28, 1025)
    log_S_transposed = log_S.T

    # 3. 检查并处理 NaN/Inf 值
    if np.any(np.isnan(log_S_transposed)) or np.any(np.isinf(log_S_transposed)):
        log_S_transposed = np.nan_to_num(log_S_transposed, nan=0.0, posinf=0.0, neginf=0.0)

    # 如果帧数少于目标维度，无法进行PCA，直接返回零向量
    if log_S_transposed.shape[0] < n_components:
        return np.zeros(n_components)

    # 4. 标准化和PCA降维
    scaler = StandardScaler()
    log_S_scaled = scaler.fit_transform(log_S_transposed)

    kernel_pca = KernelPCA(n_components=n_components, kernel='rbf', gamma=None)
    log_S_pca = kernel_pca.fit_transform(log_S_scaled)

    # 5. 聚合：将降维后的所有时间帧向量取平均
    final_vector = np.mean(log_S_pca, axis=0)

    # 6. (新增步骤) 对最终的聚合向量进行归一化
    min_val = np.min(final_vector)
    max_val = np.max(final_vector)

    # 安全检查：如果所有值都相同，无法归一化，直接返回零向量
    if max_val == min_val:
        return np.zeros_like(final_vector)

    # 应用最小-最大归一化公式
    normalized_vector = (final_vector - min_val) / (max_val - min_val)

    return normalized_vector


def extract_Wavelet(y, target_length):
    # 设置小波包变换的参数
    wavelet = 'db4'  # 使用 db4 小波（你可以选择其他小波基，如 'haar', 'db1', 'sym2' 等）
    level = 5  # 小波包变换的分解层数

    # 执行小波包变换
    wp = pywt.WaveletPacket(data=y, wavelet=wavelet, mode='symmetric')

    # 获取所有的小波包系数
    # 这里我们获取分解到指定层数后的所有叶节点
    nodes = wp.get_level(level, order='freq')  # 获取第5层的小波包叶节点，按频率顺序排序

    # 提取每个节点（频带）的能量作为特征
    features = []
    for node in nodes:
        energy = np.sum(node.data ** 2)  # 计算能量（平方和）
        features.append(energy)

    features = np.array(features)

    # 补零使特征向量长度为125
    if len(features) < target_length:
        # 如果 features 的长度小于 126，使用零填充
        features = np.pad(features, (0, target_length - len(features)), 'constant')
    elif len(features) > target_length:
        # 如果 features 的长度大于 126，截断至 126
        features = features[:target_length]

    return features









