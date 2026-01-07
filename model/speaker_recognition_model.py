"""
基于注意力机制的短语音说话人识别模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multi_scale_frontend import MultiScaleFeatureExtraction
from .attention_modules import MultiLevelDynamicAttentionFusion


class SpeakerRecognitionModel(nn.Module):
    """
    说话人识别模型主类
    架构：
    1. 多尺度特征提取前端
    2. 多层次动态注意力融合模块
    3. 特征拼接与融合
    4. 嵌入向量映射层
    """
    def __init__(self, 
                 in_channels=1,
                 frontend_channels=64,
                 attention_channels=192,  # 3 * frontend_channels
                 embedding_dim=256,
                 num_classes=100,
                 num_heads=8,
                 dropout=0.1):
        super(SpeakerRecognitionModel, self).__init__()
        
        # 1. 多尺度特征提取前端
        self.multi_scale_frontend = MultiScaleFeatureExtraction(
            in_channels=in_channels,
            out_channels=frontend_channels
        )
        
        # 2. 额外的卷积层用于特征提取
        self.conv_layers = nn.Sequential(
            nn.Conv1d(attention_channels, attention_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(attention_channels),
            nn.ReLU(),
            nn.Conv1d(attention_channels, attention_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(attention_channels),
            nn.ReLU(),
        )
        
        # 3. 多层次动态注意力融合模块
        self.attention_fusion = MultiLevelDynamicAttentionFusion(
            in_channels=attention_channels,
            num_heads=num_heads,
            reduction=16,
            dropout=dropout
        )
        
        # 4. 特征拼接与融合（在注意力融合后）
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(attention_channels, attention_channels, kernel_size=1),
            nn.BatchNorm1d(attention_channels),
            nn.ReLU(),
        )
        
        # 5. 全局池化（在嵌入映射前）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 6. 嵌入向量映射层（全连接层）
        self.embedding_layers = nn.Sequential(
            nn.Linear(attention_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
    def forward(self, x, return_embedding=True):
        """
        Args:
            x: [B, 1, T] 输入短语音波形
            return_embedding: 是否返回嵌入向量（True）或分类logits（False）
        Returns:
            embedding: [B, embedding_dim] 判别性说话人嵌入向量
            或
            logits: [B, num_classes] 分类logits（如果return_embedding=False）
        """
        # 1. 多尺度特征提取
        features = self.multi_scale_frontend(x)  # [B, 3*C, T]
        
        # 2. 卷积特征提取
        features = self.conv_layers(features)  # [B, 3*C, T]
        
        # 3. 多层次动态注意力融合
        attended_features = self.attention_fusion(features)  # [B, 3*C, T]
        
        # 4. 特征拼接与融合
        fused_features = self.feature_fusion(attended_features)  # [B, 3*C, T]
        
        # 5. 全局池化
        pooled_features = self.global_pool(fused_features)  # [B, 3*C, 1]
        pooled_features = pooled_features.squeeze(-1)  # [B, 3*C]
        
        # 6. 嵌入向量映射
        # 处理batch_size=1的情况（BatchNorm需要至少2个样本）
        batch_size = pooled_features.size(0)
        if batch_size == 1 and self.training:
            # 在训练模式下，如果batch_size=1，临时切换到eval模式
            was_training = self.training
            self.eval()
            with torch.no_grad():
                embedding = self.embedding_layers(pooled_features)  # [B, embedding_dim]
            # 恢复训练模式
            if was_training:
                self.train()
        else:
            embedding = self.embedding_layers(pooled_features)  # [B, embedding_dim]
        
        # 归一化嵌入向量（用于余弦相似度计算）
        embedding = F.normalize(embedding, p=2, dim=1)
        
        if return_embedding:
            return embedding
        else:
            # 如果需要分类logits，需要额外的分类层
            # 这里返回嵌入向量，分类层在损失函数中实现
            return embedding

