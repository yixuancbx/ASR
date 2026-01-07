"""
优化版本的训练脚本
包含混合精度训练、优化数据加载等性能优化
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import json
import gc
from datetime import datetime

from model.speaker_recognition_model import SpeakerRecognitionModel
from model.loss_functions import MixedLossFunction


class OptimizedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = SpeakerRecognitionModel(
            in_channels=config['in_channels'],
            frontend_channels=config['frontend_channels'],
            attention_channels=config['attention_channels'],
            embedding_dim=config['embedding_dim'],
            num_classes=config['num_classes'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(self.device)
        
        # 创建损失函数并移动到设备
        self.criterion = MixedLossFunction(
            embedding_dim=config['embedding_dim'],
            num_classes=config['num_classes'],
            am_margin=config['am_margin'],
            am_scale=config['am_scale'],
            intra_margin=config['intra_margin'],
            lambda_intra=config['lambda_intra']
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("启用混合精度训练 (AMP)")
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'am_loss': [],
            'intra_loss': []
        }
    
    def train_epoch(self, dataloader):
        """训练一个epoch（优化版本）"""
        self.model.train()
        total_loss = 0.0
        total_am_loss = 0.0
        total_intra_loss = 0.0
        
        # 减少进度条更新频率
        pbar = tqdm(dataloader, desc='训练中', mininterval=1.0)
        for batch_idx, (audio, labels) in enumerate(pbar):
            # 减少GPU缓存清理频率（从每20个batch改为每100个batch）
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            audio = audio.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # 使用混合精度训练
            if self.use_amp:
                with autocast():
                    embeddings = self.model(audio)
                    loss, loss_dict = self.criterion(embeddings, labels)
                
                # 反向传播（混合精度）
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准训练
                self.optimizer.zero_grad()
                embeddings = self.model(audio)
                loss, loss_dict = self.criterion(embeddings, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # 记录损失
            total_loss += loss_dict['total_loss']
            total_am_loss += loss_dict['am_loss']
            total_intra_loss += loss_dict['intra_loss']
            
            # 更新进度条（减少更新频率）
            if batch_idx % 10 == 0:
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    pbar.set_postfix({
                        'loss': f"{loss_dict['total_loss']:.4f}",
                        'mem': f"{memory_used:.2f}GB"
                    })
                else:
                    pbar.set_postfix({
                        'loss': f"{loss_dict['total_loss']:.4f}"
                    })
            
            # 减少垃圾回收频率
            if batch_idx % 200 == 0:
                gc.collect()
        
        avg_loss = total_loss / len(dataloader)
        avg_am_loss = total_am_loss / len(dataloader)
        avg_intra_loss = total_intra_loss / len(dataloader)
        
        return avg_loss, avg_am_loss, avg_intra_loss
    
    def validate(self, dataloader):
        """验证（优化版本）"""
        self.model.eval()
        total_loss = 0.0
        total_am_loss = 0.0
        total_intra_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (audio, labels) in enumerate(tqdm(dataloader, desc='验证中', mininterval=1.0)):
                # 减少GPU缓存清理频率
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                audio = audio.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # 使用混合精度（推理时也可以加速）
                if self.use_amp:
                    with autocast():
                        embeddings = self.model(audio)
                        loss, loss_dict = self.criterion(embeddings, labels)
                else:
                    embeddings = self.model(audio)
                    loss, loss_dict = self.criterion(embeddings, labels)
                
                total_loss += loss_dict['total_loss']
                total_am_loss += loss_dict['am_loss']
                total_intra_loss += loss_dict['intra_loss']
        
        avg_loss = total_loss / len(dataloader)
        avg_am_loss = total_am_loss / len(dataloader)
        avg_intra_loss = total_intra_loss / len(dataloader)
        
        return avg_loss, avg_am_loss, avg_intra_loss


