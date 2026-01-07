"""
多层次动态注意力融合模块
包含：局部帧级注意力、全局时序注意力、特征通道注意力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LocalFrameAttention(nn.Module):
    """局部帧级注意力---聚焦关键帧"""
    def __init__(self, in_channels, reduction=4):
        super(LocalFrameAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # 使用1D卷积进行局部特征提取和注意力计算
        self.conv1 = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels // reduction, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T] 卷积层后的特征
        Returns:
            attended_x: [B, C, T] 经过注意力加权的特征
        """
        # 计算注意力权重
        attention = self.conv1(x)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        # 应用注意力权重
        attended_x = x * attention
        return attended_x


class GlobalTemporalAttention(nn.Module):
    """全局时序注意力---建模长程依赖"""
    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super(GlobalTemporalAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels必须能被num_heads整除"
        
        # 多头自注意力
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.layer_norm = nn.LayerNorm(in_channels)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T] 输入特征
        Returns:
            attended_x: [B, C, T] 经过全局时序注意力加权的特征
        """
        B, C, T = x.size()
        
        # 转换为 [B, T, C] 用于自注意力
        x = x.transpose(1, 2)  # [B, T, C]
        
        # 保存残差连接
        residual = x
        
        # 计算Q, K, V
        Q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        K = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, T, D]
        V = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attended = torch.matmul(attention_weights, V)  # [B, H, T, D]
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        
        # 输出投影和残差连接
        output = self.out_proj(attended)
        output = self.layer_norm(output + residual)
        
        # 转换回 [B, C, T]
        output = output.transpose(1, 2)
        
        return output


class ChannelAttention(nn.Module):
    """特征通道注意力---校准通道权重"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T] 输入特征
        Returns:
            attended_x: [B, C, T] 经过通道注意力加权的特征
        """
        B, C, T = x.size()
        
        # 全局平均池化和最大池化
        avg_out = self.avg_pool(x).view(B, C)  # [B, C]
        max_out = self.max_pool(x).view(B, C)  # [B, C]
        
        # 通过共享的MLP
        avg_out = self.fc(avg_out)  # [B, C]
        max_out = self.fc(max_out)  # [B, C]
        
        # 融合并生成通道权重
        channel_weights = self.sigmoid(avg_out + max_out)  # [B, C]
        channel_weights = channel_weights.unsqueeze(2)  # [B, C, 1]
        
        # 应用通道权重
        attended_x = x * channel_weights
        
        return attended_x


class MultiLevelDynamicAttentionFusion(nn.Module):
    """多层次动态注意力融合模块"""
    def __init__(self, in_channels, num_heads=8, reduction=16, dropout=0.1):
        super(MultiLevelDynamicAttentionFusion, self).__init__()
        self.in_channels = in_channels
        
        # 三个注意力模块
        self.local_attention = LocalFrameAttention(in_channels, reduction=reduction)
        self.global_attention = GlobalTemporalAttention(in_channels, num_heads=num_heads, dropout=dropout)
        self.channel_attention = ChannelAttention(in_channels, reduction=reduction)
        
        # 自适应融合权重（可学习的）
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T] 输入特征
        Returns:
            fused_features: [B, C, T] 融合后的特征
        """
        # 应用三种注意力机制
        local_feat = self.local_attention(x)      # [B, C, T]
        global_feat = self.global_attention(x)   # [B, C, T]
        channel_feat = self.channel_attention(x) # [B, C, T]
        
        # 归一化融合权重
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # 加权融合
        fused_features = (weights[0] * local_feat + 
                         weights[1] * global_feat + 
                         weights[2] * channel_feat)
        
        return fused_features





