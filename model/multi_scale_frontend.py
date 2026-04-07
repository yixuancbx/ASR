"""
多尺度特征提取前端
包含三个时域并行分支，可选频域并行分支。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDomainTransform(nn.Module):
    """将时域波形转换为与原序列等长的频域时序特征。"""

    def __init__(
        self,
        n_fft=512,
        hop_length=160,
        win_length=400,
        projection_channels=64,
        output_channels=1,
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
        self.output_channels = max(1, int(output_channels))
        self.eps = float(eps)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        # 用可学习投影提取频域特征，支持多通道输出。
        self.spectral_projector = nn.Sequential(
            nn.Conv1d(self.freq_bins * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_channels, self.output_channels, kernel_size=1, bias=True),
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
            [B, C_f, T] 归一化频域特征
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
        spectral_embedding = self.spectral_projector(spectral_features)  # [B, C_f, N]
        spectral_embedding = F.interpolate(
            spectral_embedding,
            size=x.size(-1),
            mode='linear',
            align_corners=False
        )  # [B, C_f, T]

        # 每条样本的每个通道做标准化，稳定不同录音条件下的动态范围。
        spectral_embedding = spectral_embedding - spectral_embedding.mean(dim=-1, keepdim=True)
        spectral_std = spectral_embedding.std(dim=-1, keepdim=True).clamp_min(self.eps)
        spectral_embedding = spectral_embedding / spectral_std
        return spectral_embedding.to(dtype=orig_dtype)


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


class FrequencyParallelBranch(nn.Module):
    """频域并行分支：提取频域特征并做局部时序建模。"""

    def __init__(
        self,
        out_channels=64,
        n_fft=512,
        hop_length=160,
        win_length=400,
        projection_channels=64,
    ):
        super().__init__()
        self.out_channels = max(1, int(out_channels))
        projector_hidden = max(int(projection_channels), self.out_channels)
        self.frequency_transform = FrequencyDomainTransform(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            projection_channels=projector_hidden,
            output_channels=self.out_channels,
        )
        self.refine = nn.Sequential(
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        freq_features = self.frequency_transform(x)  # [B, C_f, T]
        return self.refine(freq_features)


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
        freq_integration_mode='parallel',
        freq_n_fft=512,
        freq_hop_length=160,
        freq_win_length=400,
        freq_projection_channels=64,
        freq_fusion_scale=0.4,
    ):
        super(MultiScaleFeatureExtraction, self).__init__()
        self.use_frequency_transform = bool(use_frequency_transform)
        self.freq_branch_channels = max(1, int(freq_projection_channels))
        self.freq_integration_mode = str(freq_integration_mode).lower()
        if self.freq_integration_mode not in {'parallel', 'gated'}:
            print(
                f"警告: freq_integration_mode={freq_integration_mode} 无效，"
                "已回退到 parallel"
            )
            self.freq_integration_mode = 'parallel'

        if self.use_frequency_transform:
            if self.freq_integration_mode == 'gated':
                self.frequency_transform = FrequencyDomainTransform(
                    n_fft=freq_n_fft,
                    hop_length=freq_hop_length,
                    win_length=freq_win_length,
                    projection_channels=freq_projection_channels,
                    output_channels=1,
                )
                self.time_frequency_fusion = TimeFrequencyFusion(
                    freq_fusion_scale=freq_fusion_scale
                )
            else:
                self.frequency_branch = FrequencyParallelBranch(
                    out_channels=self.freq_branch_channels,
                    n_fft=freq_n_fft,
                    hop_length=freq_hop_length,
                    win_length=freq_win_length,
                    projection_channels=freq_projection_channels,
                )

        # 三个时域并行分支
        self.branch1 = SmallKernelBranch(in_channels, out_channels, kernel_size=3)
        self.branch2 = LargeKernelBranch(in_channels, out_channels, kernel_size=15)
        self.branch3 = DepthwiseSeparableBranch(in_channels, out_channels, kernel_size=7)
        self.output_channels = out_channels * 3
        if self.use_frequency_transform and self.freq_integration_mode == 'parallel':
            self.output_channels += self.freq_branch_channels
        
    def forward(self, x):
        """
        Args:
            x: [B, 1, T] 输入语音波形
        Returns:
            features: [B, C_out, T] 多分支特征拼接
        """
        time_input = x
        if self.use_frequency_transform and self.freq_integration_mode == 'gated':
            freq_features = self.frequency_transform(x)  # [B, 1, T]
            time_input = self.time_frequency_fusion(x, freq_features)

        feat1 = self.branch1(time_input)  # [B, C, T]
        feat2 = self.branch2(time_input)  # [B, C, T]
        feat3 = self.branch3(time_input)  # [B, C, T]

        features = [feat1, feat2, feat3]
        if self.use_frequency_transform and self.freq_integration_mode == 'parallel':
            freq_branch_features = self.frequency_branch(x)  # [B, C_f, T]
            features.append(freq_branch_features)

        # 拼接多分支特征
        features = torch.cat(features, dim=1)
        return features





