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
                 dropout=0.1,
                 use_video=False,
                 video_in_channels=3,
                 video_channels=32):
        super(SpeakerRecognitionModel, self).__init__()
        self.use_video = use_video
        self.embedding_dim = embedding_dim
        
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
        
        # 5. 统计池化（ASP/SAP简化版）：拼接均值和标准差
        self.stats_dim = attention_channels * 2
        
        # 6. 嵌入向量映射层（全连接层）
        self.embedding_layers = nn.Sequential(
            nn.Linear(self.stats_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        if self.use_video:
            self.video_encoder = nn.Sequential(
                nn.Conv3d(video_in_channels, video_channels, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
                nn.BatchNorm3d(video_channels),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2)),
                nn.Conv3d(video_channels, video_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm3d(video_channels * 2),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 2)),
                nn.Conv3d(video_channels * 2, video_channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm3d(video_channels * 4),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
            )
            self.video_projection = nn.Sequential(
                nn.Linear(video_channels * 4, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.modal_fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
        
    def _apply_embedding_layers(self, pooled_features):
        batch_size = pooled_features.size(0)
        if batch_size == 1 and self.training:
            was_training = self.training
            self.eval()
            with torch.no_grad():
                embedding = self.embedding_layers(pooled_features)
            if was_training:
                self.train()
            return embedding
        return self.embedding_layers(pooled_features)

    def encode_audio(self, x):
        """
        提取音频说话人嵌入

        Args:
            x: [B, 1, T] 输入短语音波形
        Returns:
            [B, embedding_dim] 判别性说话人嵌入向量
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # 1. 多尺度特征提取
        features = self.multi_scale_frontend(x)  # [B, 3*C, T]
        
        # 2. 卷积特征提取
        features = self.conv_layers(features)  # [B, 3*C, T]
        
        # 3. 多层次动态注意力融合
        attended_features = self.attention_fusion(features)  # [B, 3*C, T]
        
        # 4. 特征拼接与融合
        fused_features = self.feature_fusion(attended_features)  # [B, 3*C, T]
        
        # 5. 统计池化（均值+标准差）
        mean = fused_features.mean(dim=2)
        std = fused_features.std(dim=2)
        pooled_features = torch.cat([mean, std], dim=1)  # [B, 2*3*C]
        embedding = self._apply_embedding_layers(pooled_features)
        return F.normalize(embedding, p=2, dim=1)

    def encode_video(self, video):
        """提取视频说话人嵌入，输入 [B, F, C, H, W]"""
        if video.dim() != 5:
            raise ValueError("视频输入应为 [B, F, C, H, W]")

        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        features = self.video_encoder(video).flatten(1)
        embedding = self.video_projection(features)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x, video=None, return_embedding=True):
        audio_embedding = self.encode_audio(x)

        if self.use_video and video is not None:
            video_embedding = self.encode_video(video)
            fused_embedding = self.modal_fusion(torch.cat([audio_embedding, video_embedding], dim=1))
            fused_embedding = F.normalize(fused_embedding, p=2, dim=1)
        else:
            fused_embedding = audio_embedding

        if return_embedding:
            return fused_embedding
        return fused_embedding

