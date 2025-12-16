import torch
import torch.nn as nn
import torch.nn.functional as F
from complexNN.nn import  cConv2d, cAvgPool2d, cBatchNorm2d, cLeakyRelu, cDropout, cLinear


def complex_sigmoid(z: torch.Tensor) -> torch.Tensor:
    """对复数张量的实部和虚部分别应用Sigmoid函数。"""
    return torch.sigmoid(z.real) + 1j * torch.sigmoid(z.imag)


class ComplexChannelAttention(nn.Module):
    """复值通道注意力模块。"""

    def __init__(self, in_channels, reduction_ratio= 4):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            cLinear(in_channels, in_channels // reduction_ratio),
            cLeakyRelu(),
            cLinear(in_channels // reduction_ratio, in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 全局平均池化
        avg_pool_out = F.adaptive_avg_pool2d(x, 1)

        # 全局最大池化 (分别作用于实部和虚部)
        max_pool_out = torch.complex(
            F.adaptive_max_pool2d(x.real, 1),
            F.adaptive_max_pool2d(x.imag, 1)
        )

        # 通过共享MLP
        avg_out_mlp = self.shared_mlp(avg_pool_out.squeeze(-1).squeeze(-1))
        max_out_mlp = self.shared_mlp(max_pool_out.squeeze(-1).squeeze(-1))

        # 生成通道注意力权重并应用
        channel_gate = complex_sigmoid(
            avg_out_mlp + max_out_mlp
        ).unsqueeze(-1).unsqueeze(-1)

        return x * channel_gate


class ComplexSpatialAttention(nn.Module):
    """复值空间注意力模块。"""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=4,
            out_channels=2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿通道维度池化
        avg_pool_real = torch.mean(x.real, dim=1, keepdim=True)
        avg_pool_imag = torch.mean(x.imag, dim=1, keepdim=True)
        max_pool_real, _ = torch.max(x.real, dim=1, keepdim=True)
        max_pool_imag, _ = torch.max(x.imag, dim=1, keepdim=True)

        # 拼接为4通道实值张量
        pooled_features = torch.cat(
            [avg_pool_real, avg_pool_imag, max_pool_real, max_pool_imag],
            dim=1
        )

        # 通过卷积层生成权重
        gate_precursors = self.conv(pooled_features)

        # 拆分通道，组合成复数值
        pre_gate_real, pre_gate_imag = gate_precursors.chunk(2, dim=1)
        pre_gate_complex = torch.complex(pre_gate_real, pre_gate_imag)

        # 生成空间注意力权重并应用
        spatial_gate = complex_sigmoid(pre_gate_complex)

        return x * spatial_gate

class ComplexCBAM(nn.Module):
    def __init__(
        self,
        in_channels,
        reduction_ratio= 16,
        spatial_kernel_size= 7
    ):
        super().__init__()
        self.channel_attention = ComplexChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = ComplexSpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class STFT_complexnn(nn.Module):
    def __init__(self, out_channel=5, dropout_rate=0.2):
        super(STFT_complexnn, self).__init__()
        self.dropout_rate = dropout_rate
        self.ComplexChannelAttention1 = ComplexChannelAttention(in_channels=4)
        self.ComplexChannelAttention2 = ComplexChannelAttention(in_channels=8)
        self.ComplexSpatialAttention1 = ComplexSpatialAttention()
        self.ComplexSpatialAttention2 = ComplexSpatialAttention()
        self.CBMA1 = ComplexCBAM(in_channels=4)
        self.CBMA2 = ComplexCBAM(in_channels=8)
        self.conv1 = nn.Sequential(
            cConv2d(1, 4, kernel_size=3, stride=1, padding=(1, 1)),
            cBatchNorm2d(4),
            cLeakyRelu(),
            #ComplexCBAM(in_channels=4),
            #ComplexSpatialAttention(),
            #ComplexChannelAttention(in_channels=4),
            cAvgPool2d(kernel_size=(2, 2), stride=(2, 2))  # 使用自定义池化
        )

        self.conv2 = nn.Sequential(
            cConv2d(4, 8, kernel_size=3, stride=1, padding=(1, 1)),
            cBatchNorm2d(8),
            cLeakyRelu(),
            #ComplexCBAM(in_channels=8),
            #ComplexSpatialAttention(),
            #ComplexChannelAttention(in_channels=8),
            cAvgPool2d(kernel_size=(2, 2), stride=(2, 2))  # 使用自定义池化
        )

        # 全连接层，添加 Dropout 层
        self.fc1 = nn.Sequential(
            cLinear(14336, 512),
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


