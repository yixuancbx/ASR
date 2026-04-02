"""
多尺度特征提取前端
包含三个并行分支：小卷积核、大卷积核、深度可分离卷积
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDomainTransform(nn.Module):
    """将时域波形转换为与原序列等长的频域描述。"""

    def __init__(
        self,
        n_fft=512,
        hop_length=160,
        win_length=400,
        projection_channels=64,
        eps=1e-6,
    ):
        super().__init__()
        self.n_fft = max(16, int(n_fft))
        self.hop_length = max(1, int(hop_length))
        self.win_length = max(16, int(win_length))
        if self.win_length > self.n_fft:
            self.win_length = self.n_fft
        self.freq_bins = self.n_fft // 2 + 1
        hidden_channels = max(16, int(projection_channels))
        self.eps = float(eps)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        # 用可学习投影替代简单频率均值，保留更丰富的频域判别信息。
        self.spectral_projector = nn.Sequential(
            nn.Conv1d(self.freq_bins * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def _resolve_window(self, device, dtype):
        window = self.window
        if window.device != device:
            window = window.to(device=device)
        if window.dtype != dtype:
            window = window.to(dtype=dtype)
        return window

    def forward(self, x):
        """
        Args:
            x: [B, C, T] 输入波形（通常 C=1）
        Returns:
            [B, 1, T] 归一化频域包络
        """
        if x.dim() != 3:
            raise ValueError("FrequencyDomainTransform 输入应为 [B, C, T]")

        # 多通道音频时先做单声道聚合，保持主流程鲁棒性。
        waveform = x.mean(dim=1)  # [B, T]
        orig_dtype = waveform.dtype
        stft_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
        waveform = waveform.to(dtype=stft_dtype)

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._resolve_window(waveform.device, waveform.dtype),
            return_complex=True,
            center=True,
            pad_mode='reflect',
        )  # [B, F, N]

        power = stft.abs().pow(2.0)
        log_power = torch.log1p(power)  # [B, F, N]

        # 引入一阶时间差分（频域动态信息），增强短语音中的过渡特征建模。
        delta = torch.zeros_like(log_power)
        delta[:, :, 1:] = log_power[:, :, 1:] - log_power[:, :, :-1]
        spectral_features = torch.cat([log_power, delta], dim=1)  # [B, 2F, N]
        projector_dtype = self.spectral_projector[0].weight.dtype
        if spectral_features.dtype != projector_dtype:
            spectral_features = spectral_features.to(dtype=projector_dtype)
        spectral_envelope = self.spectral_projector(spectral_features)  # [B, 1, N]
        spectral_envelope = F.interpolate(
            spectral_envelope,
            size=x.size(-1),
            mode='linear',
            align_corners=False
        )  # [B, 1, T]

        # 每条样本做标准化，稳定不同录音条件下的动态范围。
        spectral_envelope = spectral_envelope - spectral_envelope.mean(dim=-1, keepdim=True)
        spectral_std = spectral_envelope.std(dim=-1, keepdim=True).clamp_min(self.eps)
        spectral_envelope = spectral_envelope / spectral_std
        return spectral_envelope.to(dtype=orig_dtype)


class TimeFrequencyFusion(nn.Module):
    """时域-频域门控融合。"""

    def __init__(self, freq_fusion_scale=0.4):
        super().__init__()
        self.freq_fusion_scale = max(0.0, float(freq_fusion_scale))
        self.gate = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, time_signal, freq_signal):
        if time_signal.shape != freq_signal.shape:
            raise ValueError("TimeFrequencyFusion 需要时域与频域特征形状一致")
        gate = self.gate(torch.cat([time_signal, freq_signal], dim=1))
        return time_signal + self.freq_fusion_scale * gate * freq_signal


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
    def __init__(
        self,
        in_channels=1,
        out_channels=64,
        use_frequency_transform=True,
        freq_n_fft=512,
        freq_hop_length=160,
        freq_win_length=400,
        freq_projection_channels=64,
        freq_fusion_scale=0.4,
    ):
        super(MultiScaleFeatureExtraction, self).__init__()
        self.use_frequency_transform = bool(use_frequency_transform)

        if self.use_frequency_transform:
            self.frequency_transform = FrequencyDomainTransform(
                n_fft=freq_n_fft,
                hop_length=freq_hop_length,
                win_length=freq_win_length,
                projection_channels=freq_projection_channels,
            )
            self.time_frequency_fusion = TimeFrequencyFusion(
                freq_fusion_scale=freq_fusion_scale
            )

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
        if self.use_frequency_transform:
            freq_features = self.frequency_transform(x)  # [B, 1, T]
            x = self.time_frequency_fusion(x, freq_features)

        feat1 = self.branch1(x)  # [B, C, T]
        feat2 = self.branch2(x)  # [B, C, T]
        feat3 = self.branch3(x)  # [B, C, T]
        
        # 拼接三个分支的特征
        features = torch.cat([feat1, feat2, feat3], dim=1)  # [B, 3*C, T]
        return features





