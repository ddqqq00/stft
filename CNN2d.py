import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2d(nn.Module):
    def __init__(self, out_channel=5, dropout_prob=0.2):
        super(CNN2d, self).__init__()

        # 第一层卷积，实值卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(4),  # 使用2D BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 使用2D池化
        )

        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(8),  # 使用2D BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 使用2D池化
        )

        # 全连接层，输入维度应基于前面的池化尺寸
        self.fc1 = nn.Sequential(
            nn.Linear(9728, 256), # 39936 是池化层后的输出大小
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)  # Dropout层，防止过拟合
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)  # Dropout层
        )

        self.fc3 = nn.Linear(128, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # 展平层: 展平卷积层输出为一个一维向量
        x = x.view(x.size(0), -1)  # 适应输出维度
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
