"""
内存优化版本的训练脚本
添加了GPU缓存清理和更激进的内存优化
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import gc

from model.speaker_recognition_model import SpeakerRecognitionModel
from model.loss_functions import MixedLossFunction


class MemoryOptimizedTrainer:
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
        
        # 计算模型大小
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")
        
        if torch.cuda.is_available():
            model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32，4字节
            print(f"模型大小（估算）: {model_size_mb:.2f} MB")
            print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 创建损失函数
        self.criterion = MixedLossFunction(
            embedding_dim=config['embedding_dim'],
            num_classes=config['num_classes'],
            am_margin=config['am_margin'],
            am_scale=config['am_scale'],
            intra_margin=config['intra_margin'],
            lambda_intra=config['lambda_intra']
        )
        
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
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'am_loss': [],
            'intra_loss': []
        }
    
    def train_epoch(self, dataloader):
        """训练一个epoch（内存优化版本）"""
        self.model.train()
        total_loss = 0.0
        total_am_loss = 0.0
        total_intra_loss = 0.0
        
        pbar = tqdm(dataloader, desc='训练中')
        for batch_idx, (audio, labels) in enumerate(pbar):
            # 清理GPU缓存（每10个batch）
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            audio = audio.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # 前向传播
            self.optimizer.zero_grad()
            embeddings = self.model(audio)
            
            # 计算损失
            loss, loss_dict = self.criterion(embeddings, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss_dict['total_loss']
            total_am_loss += loss_dict['am_loss']
            total_intra_loss += loss_dict['intra_loss']
            
            # 更新进度条
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'mem': f"{memory_used:.2f}GB"
                })
            else:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'am_loss': f"{loss_dict['am_loss']:.4f}",
                    'intra_loss': f"{loss_dict['intra_loss']:.4f}"
                })
            
            # 清理中间变量
            del audio, labels, embeddings, loss
            if batch_idx % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(dataloader)
        avg_am_loss = total_am_loss / len(dataloader)
        avg_intra_loss = total_intra_loss / len(dataloader)
        
        return avg_loss, avg_am_loss, avg_intra_loss
    
    def validate(self, dataloader):
        """验证（内存优化版本）"""
        self.model.eval()
        total_loss = 0.0
        total_am_loss = 0.0
        total_intra_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (audio, labels) in enumerate(tqdm(dataloader, desc='验证中')):
                # 清理GPU缓存
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                audio = audio.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                embeddings = self.model(audio)
                loss, loss_dict = self.criterion(embeddings, labels)
                
                total_loss += loss_dict['total_loss']
                total_am_loss += loss_dict['am_loss']
                total_intra_loss += loss_dict['intra_loss']
                
                # 清理中间变量
                del audio, labels, embeddings, loss
                if batch_idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(dataloader)
        avg_am_loss = total_am_loss / len(dataloader)
        avg_intra_loss = total_intra_loss / len(dataloader)
        
        return avg_loss, avg_am_loss, avg_intra_loss





