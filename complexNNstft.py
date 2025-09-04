import torch
import torch.nn as nn
from complexNN.nn import  cConv2d, cAvgPool2d, cBatchNorm2d, cLeakyRelu, cDropout, cLinear

import inspect
from ssqueezepy import ssq_stft, ssq_cwt

class STFT_complexnn(nn.Module):
    def __init__(self, out_channel=5, dropout_rate=0.2):
        super(STFT_complexnn, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Sequential(
            cConv2d(2, 4, kernel_size=3, stride=1, padding=(1, 1)),
            cBatchNorm2d(4),
            cLeakyRelu(),
            cAvgPool2d(kernel_size=(2, 2), stride=(2, 2))  # 使用自定义池化
        )

        self.conv2 = nn.Sequential(
            cConv2d(4, 8, kernel_size=3, stride=1, padding=(1, 1)),
            cBatchNorm2d(8),
            cLeakyRelu(),
            cAvgPool2d(kernel_size=(2, 2), stride=(2, 2))  # 使用自定义池化
        )

        # 全连接层，添加 Dropout 层
        self.fc1 = nn.Sequential(
            cLinear(16384, 512),
            cLeakyRelu(),
            cDropout(p=self.dropout_rate)
        )

        self.fc2 = nn.Sequential(
            cLinear(512, 256),
            cLeakyRelu(),
            cDropout(p=self.dropout_rate)
        )

        self.fc3 = cLinear(256, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 展平层
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # 取复值的模作为分类结果
        x = torch.abs(x)

        return x


# print(inspect.getsource(ssq_stft))