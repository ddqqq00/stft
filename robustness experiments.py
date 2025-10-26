import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 您的数据 ---
SNR_labels = ['-5', '0', '5', '10', '20', 'No Noise']
accuracies = np.array([83.36, 90.56, 92.27, 93.10, 93.55, 94.11])

# --- 假设的标准差数据（请替换为您的实际数据）---
# 这些值代表了在每个SNR下，模型准确率的波动范围
std_devs = np.array([1.15, 1.28, 0.49, 0.27, 0.23, 0.21])



# --- 图表美化设置 ---
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.facecolor'] = 'whitesmoke'
plt.rcParams['grid.color'] = '#CCCCCC'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5

# 设置字体大小
title_fontsize = 25
xlabel_fontsize = 20
ylabel_fontsize = 20
tick_label_fontsize = 20
bar_label_fontsize = 20

# 创建figure和axes
fig, ax = plt.subplots(figsize=(10, 6))

# 为x轴刻度创建一个数值序列，方便定位柱子
x_pos = np.arange(len(SNR_labels))

# --- 创建带有误差棒的柱状图 ---
# yerr 参数用于指定误差棒的长度
bars = ax.bar(x_pos, accuracies, yerr=std_devs, capsize=5, # capsize控制误差棒顶端横线的长度
              color='skyblue', width=0.7, edgecolor='black', linewidth=1.2,
              error_kw={'elinewidth':1.5, 'ecolor':'darkred'}) # error_kw用于设置误差棒的样式

# 设置标题和标签
ax.set_title('Robustness Experiment Results', fontsize=title_fontsize, pad=20)
ax.set_xlabel('SNR (dB)', fontsize=xlabel_fontsize)
ax.set_ylabel('Accuracy (%)', fontsize=ylabel_fontsize)

# 设置y轴的格式和范围
min_acc = np.min(accuracies - std_devs) # 确保y轴能包含误差棒的最低点
max_acc = np.max(accuracies + std_devs) # 确保y轴能包含误差棒的最高点
ax.set_ylim(min_acc * 0.95, max_acc * 1.05) # 留出适当的上下空间
ax.set_yticks(np.arange(int(min_acc/5)*5, int(max_acc/5)*5 + 6, 5)) # 动态设置y轴刻度
ax.set_yticklabels([f'{i}%' for i in np.arange(int(min_acc/5)*5, int(max_acc/5)*5 + 6, 5)], fontsize=tick_label_fontsize)

# 设置x轴刻度标签
ax.set_xticks(x_pos)
ax.set_xticklabels(SNR_labels, fontsize=tick_label_fontsize)

# 可选：添加准确率值作为每个柱子的标签
for i, bar in enumerate(bars):
    yval = bar.get_height()
    # 标签位置稍微高于柱子顶端，同时考虑误差棒的长度
    ax.text(bar.get_x() + bar.get_width()/2, yval + std_devs[i] + 0.5, # 向上偏移，考虑误差棒
            f'{yval:.2f}%', ha='center', va='bottom', fontsize=bar_label_fontsize, color='darkslategrey')

# 移除顶部和右侧的边框
sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
ax.tick_params(axis='x', length=0)

plt.tight_layout()
plt.show()