import os
import argparse
import librosa
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm


try:
    from utilities import (
        padding2s, extract_RMS, extract_mfcc, extract_Wavelet,
        extract_stpsd, extract_autocorr, extract_zero_crosings,
        extract_spectral_rolloff, extract_spectral_centroid
    )
except ImportError:
    print("错误: 无法导入 'utilities.py'。")
    print("请确保 'utilities.py' 文件与 'preprocess.py' 在同一个文件夹下。")
    exit()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Preprocess audio files into graph data for GAT training.")
    parser.add_argument('--source_dir', type=str, required=True, help='存放原始 .wav 文件的根目录')
    parser.add_argument('--dest_dir', type=str, required=True, help='用于保存处理后的 .pt 图数据文件的新目录')
    parser.add_argument('--sr', type=int, default=7000, help='音频采样率')
    parser.add_argument('--top_k_classes', type=int, default=5, help='选择样本数最多的前k个类别进行处理')
    args = parser.parse_args()
    return args


def get_audio_paths_and_labels(data_path, top_k_classes=None):
    """
    获取音频文件路径和标签。
    (此函数与您主训练脚本中的版本完全相同)
    """
    category_paths = {}
    class_counts = {}

    for class_folder in os.listdir(data_path):
        class_folder_path = os.path.join(data_path, class_folder)
        if os.path.isdir(class_folder_path):
            file_paths = [os.path.join(class_folder_path, f) for f in os.listdir(class_folder_path) if
                          f.endswith(".wav")]
            if file_paths:  # 确保文件夹不为空
                category_paths[class_folder] = file_paths
                class_counts[class_folder] = len(file_paths)

    if top_k_classes and top_k_classes > 0:
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        top_k_classes_items = sorted_classes[:top_k_classes]
        total_samples = sum(count for _, count in top_k_classes_items)
        print(f"选择样本数最多的前 {len(top_k_classes_items)} 个类别, 总样本数为: {total_samples}")
        selected_classes = {class_name: category_paths[class_name] for class_name, _ in top_k_classes_items}
        return selected_classes

    return category_paths


def process_audio_file(file_path, sr):
    """
    将单个音频文件转换为 PyG 的 Data 对象。
    此函数的核心逻辑完全来自于您原来的 GraphDataset 的 __getitem__ 方法。
    """
    # 1. 加载和预处理音频
    audio, _ = librosa.load(file_path, sr=sr)
    audio = padding2s(audio, sr=sr, s=2)

    # 2. 特征提取
    spectral_rolloff = extract_spectral_rolloff(audio, sr=sr, target_length=28)
    zero_crossings = extract_zero_crosings(audio, target_length=28)
    spectral_centroid = extract_spectral_centroid(audio, sr=sr, target_length=28)
    RMS = extract_RMS(audio, target_length=28)
    autocorr = extract_autocorr(audio, sr=sr, target_length=28)
    stpsd = extract_stpsd(audio, n_components=28)
    mfcc = extract_mfcc(audio, sr=sr, s=2)
    wavelet = extract_Wavelet(audio, target_length=28)

    # 3. 构建图 (使用 NetworkX)
    G = nx.Graph()
    feature_names = ["mfcc", "spectral_rolloff", "zero_crossings", "wavelet", "spectral_centroid", "RMS", "stpsd",
                     "autocorr"]
    feature_vectors = [mfcc, spectral_rolloff, zero_crossings, wavelet, spectral_centroid, RMS, stpsd, autocorr]

    # 将节点名映射到整数索引
    node_map = {name: i for i, name in enumerate(feature_names)}

    for name, feature_vector in zip(feature_names, feature_vectors):
        G.add_node(node_map[name], feature=feature_vector)  # 使用整数作为节点ID

    # 定义基元分组
    timbre_features = {"zero_crossings", "wavelet", "spectral_centroid"}
    periodic_features = {"autocorr"}
    loudness_features = {"RMS", "stpsd"}
    pitch_features = {"spectral_rolloff", "mfcc"}  # 注意：原代码有"mel"，但实际特征里没有，已移除

    group_map = {}
    for f in timbre_features: group_map[f] = 'timbre'
    for f in periodic_features: group_map[f] = 'periodic'
    for f in loudness_features: group_map[f] = 'loudness'
    for f in pitch_features: group_map[f] = 'pitch'

    # 添加边和权重
    for i, name1 in enumerate(feature_names):
        for j, name2 in enumerate(feature_names):
            if i >= j: continue  # 避免重复和自环
            weight = 0.7 if group_map.get(name1) == group_map.get(name2) else 0.3
            G.add_edge(node_map[name1], node_map[name2], weight=weight)

    # 4. 转换为 PyG 的 Data 对象
    # 节点特征
    node_features_np = np.array([G.nodes[i]["feature"] for i in sorted(G.nodes())])
    x = torch.tensor(node_features_np, dtype=torch.float32)

    # 边索引和边权重 (更稳健的提取方式)
    edge_index_list = []
    edge_attr_list = []
    for u, v, data in G.edges(data=True):
        edge_index_list.append([u, v])
        edge_index_list.append([v, u])  # GAT通常需要双向边
        edge_attr_list.append(data['weight'])
        edge_attr_list.append(data['weight'])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    # 注意：标签(y)不在这里添加，训练时再动态添加
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def main():
    """主执行函数"""
    source_dir = r"D:\PyCharm\underwater-data\shipsEar2s"
    dest_dir = r"D:\PyCharm\underwater-data\processed_graphs"

    print("=" * 50)
    print("开始进行音频数据预处理")
    print(f"音频源目录: {source_dir}")
    print(f"图数据保存目录: {dest_dir}")
    print("=" * 50)

    # 确保保存目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 获取所有需要处理的音频文件路径
    category_paths = get_audio_paths_and_labels(source_dir)
    all_files = []
    for files in category_paths.values():
        all_files.extend(files)

    print(f"总共找到 {len(all_files)} 个音频文件进行处理...")

    # 使用tqdm显示进度条
    for file_path in tqdm(all_files, desc="Processing files"):
        try:
            # 处理单个文件
            graph_data = process_audio_file(file_path, sr=7000)

            # 构建保存路径，保持原始的类别文件夹结构
            class_name = os.path.basename(os.path.dirname(file_path))
            save_class_dir = os.path.join(dest_dir, class_name)
            os.makedirs(save_class_dir, exist_ok=True)

            # 定义最终的.pt文件名
            base_filename = os.path.basename(file_path).replace('.wav', '.pt')
            save_path = os.path.join(save_class_dir, base_filename)

            # 保存处理好的图数据对象
            torch.save(graph_data, save_path)

        except Exception as e:
            print(f"\n[警告] 处理文件 {file_path} 时发生错误: {e}")
            print("该文件将被跳过。")

    print("\n预处理完成！")
    print(f"所有图数据已保存至: {dest_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()