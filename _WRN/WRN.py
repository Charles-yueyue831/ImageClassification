# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : WRN.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        """
        fan_in: 权重矩阵的输入单元数
            var(W)=\frac{2}{fan_in}
        fan_out: 权重矩阵的输出单元数
            var(W)=\frac{2}{fan_out}
        """
        nn.init.kaiming_normal_(module.weight.data, mode="fan_in")
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.uniform_()
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class WRNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout):
        super(WRNBlock, self).__init__()

        self.dropout = dropout
        self._preactivate_both = (in_channels != out_channels)

        # 两层3*3的卷积的感受野等价于5*5的卷积
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if self._preactivate_both:
            self.shortcut.add_module(
                "conv",
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            )

    def forward(self, x):
        y = F.relu(self.bn1(x), inplace=True)
        y = self.conv1(y)

        if self.dropout > 0:
            F.dropout(y, p=self.dropout, training=self.training, inplace=False)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)

        if self._preactivate_both:
            y += self.shortcut(x)
        else:
            y += x

        return y


class WRN(nn.Module):
    def __init__(self, config):
        super(WRN, self).__init__()

        self.config = config

        input_shape = config["input_shape"]  # 输入图像大小
        n_classes = config["n_classes"]  # 分类类别数

        base_channels = config["base_channels"]  # 通道数
        widen_factor = config["widen_factor"]  # WRN_k
        dropout = config["dropout"]  # Dropout
        depth = config["depth"]  # 神经网络的深度

        block = WRNBlock

        # 神经网络深度和shortcut中块数之间的关系
        n_blocks_per_stage = (depth - 4) // 6
        assert n_blocks_per_stage * 6 + 4 == depth

        n_channels = [base_channels, base_channels * widen_factor, base_channels * widen_factor * 2,
                      base_channels * widen_factor * 4]

        self.conv = nn.Conv2d(input_shape[1], n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.stage1 = self._make_stage(n_channels[0], n_channels[1], n_blocks_per_stage, block, 1, dropout)

        self.stage2 = self._make_stage(n_channels[1], n_channels[2], n_blocks_per_stage, block, 2, dropout)

        self.stage3 = self._make_stage(n_channels[2], n_channels[3], n_blocks_per_stage, block, 2, dropout)

        self.bn = nn.BatchNorm2d(n_channels[3])

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(init_weights)

    # 残差连接
    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride, dropout):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f"WRNBlock_{index + 1}"
            if index == 0:
                stage.add_module(block_name,
                                 block(in_channels, out_channels, stride, dropout))
            else:
                stage.add_module(block_name,
                                 block(out_channels, out_channels, 1, dropout))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)

        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
