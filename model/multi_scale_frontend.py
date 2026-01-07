"""
多尺度特征提取前端
包含三个并行分支：小卷积核、大卷积核、深度可分离卷积
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallKernelBranch(nn.Module):
    """分支1：小卷积核---提取局部精细特征"""
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3):
        super(SmallKernelBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: [B, 1, T]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class LargeKernelBranch(nn.Module):
    """分支2：大卷积核---提取全局轮廓特征"""
    def __init__(self, in_channels=1, out_channels=64, kernel_size=15):
        super(LargeKernelBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: [B, 1, T]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DepthwiseSeparableBranch(nn.Module):
    """分支3：深度可分离卷积---轻量化设计"""
    def __init__(self, in_channels=1, out_channels=64, kernel_size=7):
        super(DepthwiseSeparableBranch, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=kernel_size//2, groups=in_channels)
        self.bn1 = nn.BatchNorm1d(in_channels)
        # 点卷积
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: [B, 1, T]
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class MultiScaleFeatureExtraction(nn.Module):
    """多尺度特征提取前端"""
    def __init__(self, in_channels=1, out_channels=64):
        super(MultiScaleFeatureExtraction, self).__init__()
        # 三个并行分支
        self.branch1 = SmallKernelBranch(in_channels, out_channels, kernel_size=3)
        self.branch2 = LargeKernelBranch(in_channels, out_channels, kernel_size=15)
        self.branch3 = DepthwiseSeparableBranch(in_channels, out_channels, kernel_size=7)
        
    def forward(self, x):
        """
        Args:
            x: [B, 1, T] 输入语音波形
        Returns:
            features: [B, 3*C, T] 三个分支的特征拼接
        """
        feat1 = self.branch1(x)  # [B, C, T]
        feat2 = self.branch2(x)  # [B, C, T]
        feat3 = self.branch3(x)  # [B, C, T]
        
        # 拼接三个分支的特征
        features = torch.cat([feat1, feat2, feat3], dim=1)  # [B, 3*C, T]
        return features





