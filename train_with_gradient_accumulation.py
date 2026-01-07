"""
带梯度累积的训练脚本（用于小batch_size时保持训练效果）
如果batch_size太小，可以使用梯度累积来模拟更大的batch size
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model.speaker_recognition_model import SpeakerRecognitionModel
from model.loss_functions import MixedLossFunction


def train_with_gradient_accumulation(model, criterion, optimizer, dataloader, device, 
                                     accumulation_steps=4):
    """
    使用梯度累积的训练函数
    
    Args:
        model: 模型
        criterion: 损失函数
        optimizer: 优化器
        dataloader: 数据加载器
        device: 设备
        accumulation_steps: 梯度累积步数（实际batch_size = batch_size * accumulation_steps）
    """
    model.train()
    total_loss = 0.0
    total_am_loss = 0.0
    total_intra_loss = 0.0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc='训练中')
    for batch_idx, (audio, labels) in enumerate(pbar):
        audio = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 前向传播
        embeddings = model(audio)
        loss, loss_dict = criterion(embeddings, labels)
        
        # 归一化损失（除以累积步数）
        loss = loss / accumulation_steps
        
        # 反向传播（累积梯度）
        loss.backward()
        
        # 每accumulation_steps步更新一次参数
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # 记录损失（需要乘以accumulation_steps以恢复原始值）
        total_loss += loss_dict['total_loss']
        total_am_loss += loss_dict['am_loss']
        total_intra_loss += loss_dict['intra_loss']
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'am_loss': f"{loss_dict['am_loss']:.4f}",
            'intra_loss': f"{loss_dict['intra_loss']:.4f}"
        })
    
    # 处理剩余的梯度（如果batch数不能被accumulation_steps整除）
    if len(dataloader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    avg_am_loss = total_am_loss / len(dataloader)
    avg_intra_loss = total_intra_loss / len(dataloader)
    
    return avg_loss, avg_am_loss, avg_intra_loss


# 使用示例：
# 如果batch_size=8，accumulation_steps=4，则有效batch_size = 8 * 4 = 32
# 在train_epoch函数中可以这样调用：
# loss = train_with_gradient_accumulation(
#     self.model, self.criterion, self.optimizer, dataloader, 
#     self.device, accumulation_steps=4
# )





