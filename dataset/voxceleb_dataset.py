"""
VoxCeleb数据集加载器
支持VoxCeleb1和VoxCeleb2数据集
"""
import os
import glob
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import numpy as np


class VoxCelebDataset(Dataset):
    """
    VoxCeleb数据集类
    数据集结构应该是：
    data_root/
        wav/
            id00001/
                video_id/
                    *.wav
            id00002/
                ...
    """
    def __init__(self, 
                 data_root,
                 sample_rate=16000,
                 segment_length=16000,  # 1秒，16kHz
                 train=True,
                 augmentation=False):
        """
        Args:
            data_root: 数据集根目录（包含wav文件夹）
            sample_rate: 目标采样率
            segment_length: 音频片段长度（采样点数）
            train: 是否为训练集（决定是否使用数据增强）
            augmentation: 是否使用数据增强
        """
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.train = train
        self.augmentation = augmentation
        
        # 查找所有音频文件
        self.audio_files = []
        self.speaker_ids = []
        self.speaker_to_id = {}
        
        # 缓存重采样器（避免每次创建）
        self.resampler = None
        self.last_sr = None
        
        self._load_audio_files()
        
        print(f"加载了 {len(self.audio_files)} 个音频文件")
        print(f"共 {len(self.speaker_to_id)} 个说话人")
    
    def _load_audio_files(self):
        """加载所有音频文件路径和对应的说话人标签"""
        wav_dir = os.path.join(self.data_root, 'wav')
        if not os.path.exists(wav_dir):
            # 尝试直接使用data_root
            wav_dir = self.data_root
        
        # 查找所有说话人目录（id开头）
        speaker_dirs = sorted([d for d in os.listdir(wav_dir) 
                              if os.path.isdir(os.path.join(wav_dir, d)) and d.startswith('id')])
        
        # 为每个说话人分配ID
        for speaker_id, speaker_dir in enumerate(speaker_dirs):
            self.speaker_to_id[speaker_dir] = speaker_id
        
        # 遍历所有说话人目录，收集音频文件
        for speaker_dir in speaker_dirs:
            speaker_path = os.path.join(wav_dir, speaker_dir)
            speaker_label = self.speaker_to_id[speaker_dir]
            
            # 查找所有.wav文件（包括子目录）
            wav_files = glob.glob(os.path.join(speaker_path, '**', '*.wav'), recursive=True)
            
            for wav_file in wav_files:
                self.audio_files.append(wav_file)
                self.speaker_ids.append(speaker_label)
    
    def _load_audio(self, filepath):
        """加载音频文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(filepath):
                return torch.zeros(self.segment_length)
            
            # 使用torchaudio加载音频
            waveform, sr = torchaudio.load(filepath)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 重采样到目标采样率（复用resampler以避免内存问题）
            if sr != self.sample_rate:
                # 只在采样率改变时创建新的resampler
                if self.resampler is None or self.last_sr != sr:
                    self.resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    self.last_sr = sr
                waveform = self.resampler(waveform)
            
            return waveform.squeeze(0)  # [T]
        except Exception as e:
            # 静默失败，只返回零向量（避免输出过多错误信息）
            # 如果需要调试，可以取消下面的注释
            # print(f"加载音频失败 {filepath}: {e}")
            return torch.zeros(self.segment_length)
    
    def _augment_audio(self, audio):
        """数据增强"""
        if not self.augmentation or not self.train:
            return audio
        
        # 随机时间平移（在segment_length内）
        if len(audio) > self.segment_length:
            max_shift = len(audio) - self.segment_length
            shift = random.randint(0, max_shift)
            audio = audio[shift:shift + self.segment_length]
        
        # 随机音量缩放
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            audio = audio * scale
        
        # 随机添加噪声
        if random.random() > 0.7:
            noise = torch.randn_like(audio) * 0.01
            audio = audio + noise
        
        return audio
    
    def _pad_or_truncate(self, audio):
        """填充或截断音频到固定长度"""
        if len(audio) > self.segment_length:
            # 训练时随机裁剪，验证时取中心
            if self.train:
                start = random.randint(0, len(audio) - self.segment_length)
                audio = audio[start:start + self.segment_length]
            else:
                start = (len(audio) - self.segment_length) // 2
                audio = audio[start:start + self.segment_length]
        else:
            # 填充到固定长度
            padding = self.segment_length - len(audio)
            audio = F.pad(audio, (0, padding))
        
        return audio
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # 加载音频
        audio = self._load_audio(self.audio_files[idx])
        label = self.speaker_ids[idx]
        
        # 数据增强
        audio = self._augment_audio(audio)
        
        # 填充或截断
        audio = self._pad_or_truncate(audio)
        
        # 添加通道维度: [T] -> [1, T]
        audio = audio.unsqueeze(0)
        
        return audio, label
    
    def get_num_speakers(self):
        """获取说话人数量"""
        return len(self.speaker_to_id)


class VoxCelebDatasetFromList(Dataset):
    """
    从列表文件加载VoxCeleb数据集
    列表文件格式（每行）：
    wav_file_path speaker_id
    或
    speaker_id/video_id/file.wav speaker_id
    """
    def __init__(self,
                 list_file,
                 data_root=None,
                 sample_rate=16000,
                 segment_length=16000,
                 train=True,
                 augmentation=False):
        """
        Args:
            list_file: 列表文件路径
            data_root: 数据集根目录（如果列表中是相对路径）
            sample_rate: 目标采样率
            segment_length: 音频片段长度
            train: 是否为训练集
            augmentation: 是否使用数据增强
        """
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.train = train
        self.augmentation = augmentation
        
        # 加载文件列表
        self.audio_files = []
        self.speaker_ids = []
        self.speaker_to_id = {}
        
        # 缓存重采样器（避免每次创建）
        self.resampler = None
        self.last_sr = None
        
        self._load_list_file(list_file)
        
        print(f"从列表文件加载了 {len(self.audio_files)} 个音频文件")
        print(f"共 {len(self.speaker_to_id)} 个说话人")
    
    def _load_list_file(self, list_file):
        """从列表文件加载数据"""
        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                audio_file = parts[0]
                speaker_name = parts[1]
                
                # 构建完整路径
                if self.data_root and not os.path.isabs(audio_file):
                    audio_file = os.path.join(self.data_root, audio_file)
                
                # 分配说话人ID
                if speaker_name not in self.speaker_to_id:
                    self.speaker_to_id[speaker_name] = len(self.speaker_to_id)
                
                speaker_id = self.speaker_to_id[speaker_name]
                
                if os.path.exists(audio_file):
                    self.audio_files.append(audio_file)
                    self.speaker_ids.append(speaker_id)
    
    def _load_audio(self, filepath):
        """加载音频文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(filepath):
                return torch.zeros(self.segment_length)
            
            waveform, sr = torchaudio.load(filepath)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 重采样（复用resampler以避免内存问题）
            if sr != self.sample_rate:
                # 只在采样率改变时创建新的resampler
                if self.resampler is None or self.last_sr != sr:
                    self.resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    self.last_sr = sr
                waveform = self.resampler(waveform)
            
            return waveform.squeeze(0)
        except Exception as e:
            # 静默失败，只返回零向量（避免输出过多错误信息）
            # 如果需要调试，可以取消下面的注释
            # print(f"加载音频失败 {filepath}: {e}")
            return torch.zeros(self.segment_length)
    
    def _augment_audio(self, audio):
        """数据增强"""
        if not self.augmentation or not self.train:
            return audio
        
        if len(audio) > self.segment_length:
            max_shift = len(audio) - self.segment_length
            shift = random.randint(0, max_shift)
            audio = audio[shift:shift + self.segment_length]
        
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            audio = audio * scale
        
        if random.random() > 0.7:
            noise = torch.randn_like(audio) * 0.01
            audio = audio + noise
        
        return audio
    
    def _pad_or_truncate(self, audio):
        """填充或截断音频到固定长度"""
        if len(audio) > self.segment_length:
            if self.train:
                start = random.randint(0, len(audio) - self.segment_length)
                audio = audio[start:start + self.segment_length]
            else:
                start = (len(audio) - self.segment_length) // 2
                audio = audio[start:start + self.segment_length]
        else:
            padding = self.segment_length - len(audio)
            audio = F.pad(audio, (0, padding))
        
        return audio
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio = self._load_audio(self.audio_files[idx])
        label = self.speaker_ids[idx]
        
        audio = self._augment_audio(audio)
        audio = self._pad_or_truncate(audio)
        audio = audio.unsqueeze(0)
        
        return audio, label
    
    def get_num_speakers(self):
        """获取说话人数量"""
        return len(self.speaker_to_id)

