import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet的基本残差块"""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Args:
            in_planes: 输入通道数
            planes: 输出通道数
            stride: 卷积步长
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Conv(nn.Module):
    """封装的卷积层，包含BN和激活函数"""

    def __init__(
        self,
        c1: int,  # 输入通道
        c2: int,  # 输出通道
        k: int,  # kernel size
        s: int = 1,  # stride
        p: int = 0,  # padding
        d: int = 1,  # dilation
        g: int = 1,  # groups
        act: bool = True,  # 是否使用激活函数
    ) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class SPP(nn.Module):
    """空间金字塔池化层"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """使用不同大小的maxpool并拼接结果"""
        x_1 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = F.max_pool2d(x, 13, stride=1, padding=6)
        return torch.cat([x, x_1, x_2, x_3], dim=1)
