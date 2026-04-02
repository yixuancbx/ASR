"""
基于注意力机制的短语音说话人识别模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from .multi_scale_frontend import MultiScaleFeatureExtraction
from .attention_modules import MultiLevelDynamicAttentionFusion


class SqueezeExcite2d(nn.Module):
    """轻量 Squeeze-Excitation，用于增强通道选择性。"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class DepthwiseSeparableConv2d(nn.Module):
    """带扩展与 SE 的轻量倒残差块。"""

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2, use_se=True):
        super().__init__()
        hidden_channels = max(in_channels, in_channels * int(expand_ratio))
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        if hidden_channels != in_channels:
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        ])
        if use_se:
            layers.append(SqueezeExcite2d(hidden_channels, reduction=4))
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.block = nn.Sequential(*layers)
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return self.out_act(out)


class LightweightVisualEncoder(nn.Module):
    """
    轻量视觉编码器：
    1) 帧级深度可分离卷积提取空间特征
    2) 时序卷积混合帧间信息
    3) 注意力池化得到视频级说话人表征
    """

    def __init__(self, in_channels=3, base_channels=24, embedding_dim=256, dropout=0.1):
        super().__init__()
        base_channels = max(8, int(base_channels))
        final_channels = base_channels * 4
        attn_hidden = max(32, final_channels // 2)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv2d(base_channels, base_channels, stride=1, expand_ratio=2, use_se=True),
            DepthwiseSeparableConv2d(base_channels, base_channels * 2, stride=2, expand_ratio=2, use_se=True),
            DepthwiseSeparableConv2d(base_channels * 2, base_channels * 2, stride=1, expand_ratio=2, use_se=True),
            DepthwiseSeparableConv2d(base_channels * 2, final_channels, stride=2, expand_ratio=2, use_se=True),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        self.temporal_mixer = nn.Sequential(
            nn.Conv1d(
                final_channels,
                final_channels,
                kernel_size=3,
                padding=1,
                groups=final_channels,
                bias=False,
            ),
            nn.BatchNorm1d(final_channels),
            nn.SiLU(),
            nn.Conv1d(final_channels, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(final_channels),
            nn.SiLU(),
        )
        self.temporal_norm = nn.LayerNorm(final_channels)

        self.temporal_attention = nn.Sequential(
            nn.Linear(final_channels, attn_hidden),
            nn.GELU(),
            nn.Linear(attn_hidden, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(final_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, video):
        """
        Args:
            video: [B, F, C, H, W]
        Returns:
            [B, embedding_dim] 归一化视频说话人嵌入
        """
        if video.dim() != 5:
            raise ValueError("视频输入应为 [B, F, C, H, W]")

        batch_size, num_frames, channels, height, width = video.shape
        x = video.reshape(batch_size * num_frames, channels, height, width)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.spatial_pool(x).flatten(1)  # [B*F, C]

        frame_features = x.view(batch_size, num_frames, -1)  # [B, F, C]
        temporal_features = self.temporal_mixer(frame_features.transpose(1, 2)).transpose(1, 2)
        temporal_features = self.temporal_norm(temporal_features)
        temporal_logits = self.temporal_attention(temporal_features).squeeze(-1)  # [B, F]
        temporal_weights = F.softmax(temporal_logits, dim=1).unsqueeze(-1)  # [B, F, 1]
        pooled_features = (temporal_features * temporal_weights).sum(dim=1)
        embedding = self.projection(pooled_features)
        return F.normalize(embedding, p=2, dim=1)


class CrossModalAdaptiveFusion(nn.Module):
    """
    自适应跨模态融合：
    1) 质量评估分支估计音频/视频可靠性
    2) 特征门控分支对两种模态逐维加权
    3) 融合后再映射到统一说话人嵌入空间
    4) 增加音频残差旁路，降低训练初期随机融合带来的破坏
    """

    def __init__(self, embedding_dim=256, dropout=0.1, modality_dropout=0.0, num_heads=4):
        super().__init__()
        hidden_dim = max(64, embedding_dim // 2)
        self.modality_dropout = max(0.0, float(modality_dropout))
        self.num_heads = self._resolve_num_heads(embedding_dim, num_heads)

        self.audio_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.video_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        self.modality_tokens = nn.Parameter(torch.zeros(1, 2, embedding_dim))
        nn.init.normal_(self.modality_tokens, mean=0.0, std=0.02)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embedding_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.cross_ffn_norm = nn.LayerNorm(embedding_dim)

        self.quality_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Sigmoid(),
        )
        self.output_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.audio_bypass_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # 偏置初始化为正值，使训练初期更偏向音频直通，避免随机融合扰动。
        self.audio_bypass_bias = nn.Parameter(torch.tensor(2.0))
        self.min_audio_bypass = 0.25
        self.max_audio_bypass = 0.98

    @staticmethod
    def _resolve_num_heads(embedding_dim, num_heads):
        heads = max(1, int(num_heads))
        while heads > 1 and embedding_dim % heads != 0:
            heads -= 1
        return heads

    def _maybe_apply_modality_dropout(self, audio_embedding, video_embedding):
        if not self.training or self.modality_dropout <= 0:
            return audio_embedding, video_embedding
        if torch.rand(1, device=audio_embedding.device).item() >= self.modality_dropout:
            return audio_embedding, video_embedding
        if torch.rand(1, device=audio_embedding.device).item() < 0.5:
            return torch.zeros_like(audio_embedding), video_embedding
        return audio_embedding, torch.zeros_like(video_embedding)

    def forward(self, audio_embedding, video_embedding):
        audio_proj = self.audio_proj(audio_embedding)
        video_proj = self.video_proj(video_embedding)
        audio_proj, video_proj = self._maybe_apply_modality_dropout(audio_proj, video_proj)

        tokens = torch.stack([audio_proj, video_proj], dim=1) + self.modality_tokens
        attn_tokens, _ = self.cross_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.cross_norm(tokens + attn_tokens)
        tokens = self.cross_ffn_norm(tokens + self.cross_ffn(tokens))
        audio_ctx = tokens[:, 0]
        video_ctx = tokens[:, 1]

        quality_logits = self.quality_head(torch.cat([audio_ctx, video_ctx], dim=1))
        quality_weights = F.softmax(quality_logits, dim=1)
        quality_fused = quality_weights[:, :1] * audio_ctx + quality_weights[:, 1:] * video_ctx

        gate_inputs = torch.cat(
            [audio_ctx, video_ctx, torch.abs(audio_ctx - video_ctx), audio_ctx * video_ctx],
            dim=1,
        )
        gates = self.gate_head(gate_inputs)
        gated_fused = gates * audio_ctx + (1.0 - gates) * video_ctx

        fusion_core = self.output_head(quality_fused + gated_fused + 0.5 * (audio_ctx + video_ctx))

        # 音频残差旁路：训练初期默认更信任音频，随着视频质量提升再放开融合占比。
        bypass_inputs = torch.cat([audio_embedding, audio_ctx, video_ctx], dim=1)
        dynamic_bypass = torch.sigmoid(self.audio_bypass_head(bypass_inputs) + self.audio_bypass_bias)
        quality_bypass = quality_weights[:, :1]
        bypass_ratio = 0.7 * dynamic_bypass + 0.3 * quality_bypass
        bypass_ratio = torch.clamp(
            bypass_ratio,
            min=self.min_audio_bypass,
            max=self.max_audio_bypass
        )
        fused = bypass_ratio * audio_embedding + (1.0 - bypass_ratio) * fusion_core
        return F.normalize(fused, p=2, dim=1)


class SpeakerRecognitionModel(nn.Module):
    """
    说话人识别模型主类
    架构：
    1. 多尺度音频特征提取前端
    2. 多层次动态注意力融合模块
    3. 统计池化 + 嵌入映射
    4. (可选) 轻量视觉编码 + 自适应跨模态融合
    """

    def __init__(
        self,
        in_channels=1,
        frontend_channels=64,
        attention_channels=192,  # 3 * frontend_channels
        embedding_dim=256,
        num_classes=100,
        num_heads=8,
        dropout=0.1,
        temporal_pool_stride=1,
        max_attention_frames=0,
        temporal_pool_type='avg',
        use_attention_checkpoint=False,
        use_frequency_transform=True,
        freq_n_fft=512,
        freq_hop_length=160,
        freq_win_length=400,
        freq_projection_channels=64,
        freq_fusion_scale=0.4,
        enable_local_attention=True,
        enable_global_attention=True,
        enable_channel_attention=True,
        use_video=False,
        video_in_channels=3,
        video_channels=24,
        visual_encoder_dropout=0.1,
        fusion_dropout=0.1,
        modality_dropout=0.0,
        fusion_num_heads=4,
    ):
        super(SpeakerRecognitionModel, self).__init__()
        self.use_video = use_video
        self.embedding_dim = embedding_dim

        expected_attention_channels = frontend_channels * 3
        if attention_channels != expected_attention_channels:
            print(
                f"警告: attention_channels={attention_channels} 与 3*frontend_channels={expected_attention_channels} 不一致，"
                "已自动调整为匹配值"
            )
            attention_channels = expected_attention_channels

        self.temporal_pool_stride = max(1, int(temporal_pool_stride))
        self.max_attention_frames = max(0, int(max_attention_frames))
        self.temporal_pool_type = str(temporal_pool_type).lower()
        if self.temporal_pool_type not in {'avg', 'max'}:
            print(f"警告: temporal_pool_type={temporal_pool_type} 无效，已回退到 avg")
            self.temporal_pool_type = 'avg'
        self.use_attention_checkpoint = bool(use_attention_checkpoint)

        # 1. 多尺度特征提取前端
        self.multi_scale_frontend = MultiScaleFeatureExtraction(
            in_channels=in_channels,
            out_channels=frontend_channels,
            use_frequency_transform=use_frequency_transform,
            freq_n_fft=freq_n_fft,
            freq_hop_length=freq_hop_length,
            freq_win_length=freq_win_length,
            freq_projection_channels=freq_projection_channels,
            freq_fusion_scale=freq_fusion_scale,
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
            dropout=dropout,
            enable_local_attention=enable_local_attention,
            enable_global_attention=enable_global_attention,
            enable_channel_attention=enable_channel_attention,
        )

        # 4. 特征拼接与融合（在注意力融合后）
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(attention_channels, attention_channels, kernel_size=1),
            nn.BatchNorm1d(attention_channels),
            nn.ReLU(),
        )

        # 5. 统计池化（均值 + 标准差）
        self.stats_dim = attention_channels * 2

        # 6. 嵌入向量映射层
        self.embedding_layers = nn.Sequential(
            nn.Linear(self.stats_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        if self.use_video:
            self.video_encoder = LightweightVisualEncoder(
                in_channels=video_in_channels,
                base_channels=video_channels,
                embedding_dim=embedding_dim,
                dropout=visual_encoder_dropout,
            )
            self.modal_fusion = CrossModalAdaptiveFusion(
                embedding_dim=embedding_dim,
                dropout=fusion_dropout,
                modality_dropout=modality_dropout,
                num_heads=fusion_num_heads,
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

    def _downsample_temporal(self, features):
        """在进入全局注意力前进行时序降采样，降低显存占用。"""
        if self.temporal_pool_stride > 1:
            if self.temporal_pool_type == 'max':
                features = F.max_pool1d(
                    features,
                    kernel_size=self.temporal_pool_stride,
                    stride=self.temporal_pool_stride,
                    ceil_mode=True,
                )
            else:
                features = F.avg_pool1d(
                    features,
                    kernel_size=self.temporal_pool_stride,
                    stride=self.temporal_pool_stride,
                    ceil_mode=True,
                )

        if self.max_attention_frames > 0 and features.size(-1) > self.max_attention_frames:
            features = F.adaptive_avg_pool1d(features, self.max_attention_frames)
        return features

    def _apply_attention_fusion(self, features):
        """可选启用梯度检查点，用计算换显存。"""
        if not (self.use_attention_checkpoint and self.training):
            return self.attention_fusion(features)
        try:
            return torch_checkpoint(self.attention_fusion, features, use_reentrant=False)
        except TypeError:
            return torch_checkpoint(self.attention_fusion, features)

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
        features = self.multi_scale_frontend(x)  # [B, 3*C, T]
        features = self.conv_layers(features)  # [B, 3*C, T]
        features = self._downsample_temporal(features)
        attended_features = self._apply_attention_fusion(features)  # [B, 3*C, T]
        fused_features = self.feature_fusion(attended_features)  # [B, 3*C, T]

        mean = fused_features.mean(dim=2)
        std = fused_features.std(dim=2)
        pooled_features = torch.cat([mean, std], dim=1)  # [B, 2*3*C]
        embedding = self._apply_embedding_layers(pooled_features)
        return F.normalize(embedding, p=2, dim=1)

    def encode_video(self, video):
        """提取视频说话人嵌入，输入 [B, F, C, H, W]"""
        if not self.use_video:
            raise RuntimeError("当前模型未启用视频分支，请设置 use_video=True")
        return self.video_encoder(video)

    def forward(self, x, video=None, return_embedding=True, return_modal_embeddings=False):
        audio_embedding = self.encode_audio(x)
        video_embedding = None

        if self.use_video and video is not None:
            video_embedding = self.encode_video(video)
            fused_embedding = self.modal_fusion(audio_embedding, video_embedding)
        else:
            fused_embedding = audio_embedding

        if return_modal_embeddings:
            return {
                'embedding': fused_embedding,
                'audio_embedding': audio_embedding,
                'video_embedding': video_embedding,
            }
        if return_embedding:
            return fused_embedding
        return fused_embedding

