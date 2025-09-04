import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexReLU, ComplexMaxPool2d
from complexNN.nn import cConv1d, cConv2d


class ComplexAbsMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(ComplexAbsMaxPool2d, self).__init__()
        # 将kernel_size转换为元组（如果输入是整数）
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        # 处理步长参数
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        # 处理填充参数
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

    def forward(self, x):
        """
        x: complex tensor of shape (batch, channels, height, width)
           where each element is a complex number
        """
        # 分离实部和虚部
        real = x.real
        imag = x.imag

        # 计算每个位置的实部与虚部乘积的绝对值
        abs_product = torch.abs(real * imag)

        # 使用unfold操作提取滑动窗口
        abs_unfolded = F.unfold(
            abs_product,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        # 获取输入形状
        batch, channels, height, width = x.shape

        # 计算输出尺寸（分别处理高度和宽度）
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding

        h_out = (height + 2 * padding_h - kernel_h) // stride_h + 1
        w_out = (width + 2 * padding_w - kernel_w) // stride_w + 1
        num_windows = h_out * w_out

        # 重塑abs_unfolded以便按通道处理
        abs_unfolded = abs_unfolded.view(
            batch, channels, kernel_h * kernel_w, num_windows
        )

        # 找到每个窗口中最大绝对值的索引
        _, max_indices = torch.max(abs_unfolded, dim=2)

        # 对实部和虚部分别进行相同的unfold操作
        real_unfolded = F.unfold(
            real,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        ).view(batch, channels, kernel_h * kernel_w, num_windows)

        imag_unfolded = F.unfold(
            imag,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        ).view(batch, channels, kernel_h * kernel_w, num_windows)

        # 根据索引选择最大值
        selected_real = torch.gather(
            real_unfolded,
            dim=2,
            index=max_indices.unsqueeze(2)
        ).squeeze(2)

        selected_imag = torch.gather(
            imag_unfolded,
            dim=2,
            index=max_indices.unsqueeze(2)
        ).squeeze(2)

        # 重塑为 (batch, channels, h_out, w_out)
        real_out = selected_real.view(batch, channels, h_out, w_out)
        imag_out = selected_imag.view(batch, channels, h_out, w_out)

        # 合并实部和虚部为复数张量
        return torch.complex(real_out, imag_out)


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # 对实部和虚部应用相同的Dropout模式
        mask = torch.ones_like(x.real)
        mask = self.dropout(mask)
        return x * mask


class ComplexDropout2d(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout2d, self).__init__()
        self.p = p
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # 对实部和虚部应用相同的Dropout模式
        mask = torch.ones_like(x.real)
        mask = self.dropout(mask)
        return x * mask

class STFT_complex(nn.Module):
    def __init__(self, out_channel=6, dropout_rate=0.2):
        super(STFT_complex, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Sequential(
            ComplexConv2d(1, 4, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1)),
            ComplexBatchNorm2d(4),
            ComplexReLU(),
            ComplexAbsMaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 使用自定义池化
        )

        # 在卷积层后添加空间Dropout
        self.conv1_dropout = ComplexDropout2d(p=self.dropout_rate)

        self.conv2 = nn.Sequential(
            ComplexConv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ComplexBatchNorm2d(8),
            ComplexReLU(),
            ComplexAbsMaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 使用自定义池化
        )

        # 在卷积层后添加空间Dropout
        self.conv2_dropout = ComplexDropout2d(p=self.dropout_rate)

        # 全连接层，添加 Dropout 层
        self.fc1 = nn.Sequential(
            ComplexLinear(9576, 512),
            ComplexReLU(),
            ComplexDropout(p=self.dropout_rate)
        )

        self.fc2 = nn.Sequential(
            ComplexLinear(512, 256),
            ComplexReLU(),
            # ComplexDropout(p=self.dropout_rate)
        )

        self.fc3 = ComplexLinear(256, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv1_dropout(x)  # 应用空间Dropout

        x = self.conv2(x)
        # x = self.conv2_dropout(x)  # 应用空间Dropout

        # 展平层
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # 取复值的模作为分类结果
        x = torch.abs(x)

        return x
