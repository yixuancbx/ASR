"""
训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import os
import json
import gc
from datetime import datetime

from model.speaker_recognition_model import SpeakerRecognitionModel
from model.loss_functions import MixedLossFunction

# 为固定长度卷积启用最优算法搜索，加速卷积
torch.backends.cudnn.benchmark = True


class Trainer:
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
        # 可选编译加速（需要 PyTorch 2.0+）
        if self.config.get('compile', False) and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                print("已启用 torch.compile 以加速训练/推理")
            except Exception as e:
                print(f"torch.compile 启用失败，继续使用原模型: {e}")

        # 创建损失函数并移动到设备
        self.criterion = MixedLossFunction(
            embedding_dim=config['embedding_dim'],
            num_classes=config['num_classes'],
            am_margin=config['am_margin'],
            am_scale=config['am_scale'],
            intra_margin=config['intra_margin'],
            lambda_intra=config['lambda_intra']
        ).to(self.device)
        
        # 创建优化器（包含模型参数 + 损失函数参数，例如 AAM-Softmax 的可学习权重）
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器（默认余弦退火重启，可回退到StepLR）
        if self.config.get('scheduler', 'cosine') == 'cosine':
            T_0 = self.config.get('cosine_T0', 10)
            T_mult = self.config.get('cosine_Tmult', 2)
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=T_0, T_mult=T_mult
            )
            print(f"使用 CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult})")
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['lr_step_size'],
                gamma=config['lr_gamma']
            )
            print("使用 StepLR 调度器")
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("启用混合精度训练 (AMP) - 可提升训练速度约1.5-2倍")
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'am_loss': [],
            'intra_loss': []
        }

    def _load_state_dict_flexible(self, state_dict):
        """
        为了兼容架构升级（例如池化维度变化），对shape不匹配的参数跳过加载。
        返回：missing, unexpected, skipped
        """
        current = self.model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k in current and current[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"跳过因shape不匹配的参数: {skipped}")
        return missing, unexpected, skipped
        
    def train_epoch(self, dataloader):
        """训练一个epoch，支持梯度累积"""
        self.model.train()
        total_loss = 0.0
        total_am_loss = 0.0
        total_intra_loss = 0.0
        accumulation_steps = max(1, self.config.get('accumulation_steps', 1))
        max_batches = self.config.get('max_train_batches', None)
        self.optimizer.zero_grad()
        
        # 减少进度条更新频率以提升性能
        pbar = tqdm(dataloader, desc='训练中', mininterval=1.0)
        pending_grad = False
        for batch_idx, (audio, labels) in enumerate(pbar):
            # 提前截断本epoch的批次数，直接降低单epoch耗时
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            audio = audio.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # 使用混合精度训练
            if self.use_amp:
                with autocast():
                    embeddings = self.model(audio)
                    loss, loss_dict = self.criterion(embeddings, labels)
                loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                # 每 accumulation_steps 次才更新参数
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    pending_grad = False
                else:
                    pending_grad = True
            else:
                # 标准训练
                embeddings = self.model(audio)
                loss, loss_dict = self.criterion(embeddings, labels)
                loss = loss / accumulation_steps
                loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pending_grad = False
                else:
                    pending_grad = True
            
            # 记录损失
            total_loss += loss_dict['total_loss']
            total_am_loss += loss_dict['am_loss']
            total_intra_loss += loss_dict['intra_loss']
            
            # 更新进度条（减少更新频率以提升性能）
            if batch_idx % 10 == 0:
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
            
            # 减少垃圾回收频率（从每50个batch改为每200个batch）
            if batch_idx % 200 == 0:
                gc.collect()
        
        # 处理剩余未更新的梯度
        if pending_grad:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(dataloader)
        avg_am_loss = total_am_loss / len(dataloader)
        avg_intra_loss = total_intra_loss / len(dataloader)
        
        return avg_loss, avg_am_loss, avg_intra_loss
    
    def validate(self, dataloader):
        """验证，增加 EER 度量"""
        self.model.eval()
        total_loss = 0.0
        total_am_loss = 0.0
        total_intra_loss = 0.0
        max_batches = self.config.get('max_val_batches', None)
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (audio, labels) in enumerate(tqdm(dataloader, desc='验证中', mininterval=1.0)):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
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
                all_embeddings.append(embeddings.detach().cpu())
                all_labels.append(labels.detach().cpu())
                
                # 清理中间变量
                del audio, labels, embeddings, loss
                if batch_idx % 50 == 0:
                    gc.collect()
        
        avg_loss = total_loss / len(dataloader)
        avg_am_loss = total_am_loss / len(dataloader)
        avg_intra_loss = total_intra_loss / len(dataloader)
        eer = self.compute_eer(all_embeddings, all_labels)
        
        return avg_loss, avg_am_loss, avg_intra_loss, eer

    def compute_eer(self, embedding_chunks, label_chunks):
        """根据验证集嵌入计算 EER（Equal Error Rate）"""
        if len(embedding_chunks) == 0:
            return None
        embeddings = torch.cat(embedding_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)
        if embeddings.size(0) < 2:
            return None
        # 归一化
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # 余弦相似度矩阵
        sim = torch.matmul(embeddings, embeddings.t())
        n = sim.size(0)
        triu_mask = torch.triu(torch.ones(n, n, device=sim.device, dtype=torch.bool), diagonal=1)
        scores = sim[triu_mask]
        pair_labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(sim.device)
        targets = pair_labels[triu_mask].float()
        pos_total = targets.sum()
        neg_total = targets.numel() - pos_total
        if pos_total == 0 or neg_total == 0:
            return None
        # 按得分降序排序，累积TP/FP
        scores_sorted, idx = torch.sort(scores, descending=True)
        targets_sorted = targets[idx]
        tp = targets_sorted.cumsum(0)
        fp = torch.arange(1, targets_sorted.numel() + 1, device=scores.device) - tp
        tpr = tp / pos_total
        fpr = fp / neg_total
        fnr = 1 - tpr
        diff = torch.abs(fpr - fnr)
        min_idx = diff.argmin()
        eer = (fpr[min_idx] + fnr[min_idx]) / 2
        return eer.item()
    
    def train(self, train_loader, val_loader=None, num_epochs=100, resume_from=None):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 总训练轮数
            resume_from: 从指定checkpoint恢复（如果为None，会自动查找latest_checkpoint.pth）
        """
        # 尝试恢复训练
        start_epoch = 0
        best_val_loss = float('inf')
        
        if resume_from is None:
            # 自动查找最新的checkpoint
            checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
            latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            if os.path.exists(latest_checkpoint):
                resume_from = latest_checkpoint
        
        if resume_from and os.path.exists(resume_from):
            print(f"\n发现检查点: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            # 兼容shape变化的加载
            missing, unexpected, skipped = self._load_state_dict_flexible(checkpoint['model_state_dict'])
            if missing:
                print(f"加载检查点: 缺少参数 {missing}")
            if unexpected:
                print(f"加载检查点: 多余参数 {unexpected}")
            if skipped:
                print(f"加载检查点: 跳过shape不匹配参数 {skipped}")

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_history = checkpoint.get('train_history', {'loss': [], 'am_loss': [], 'intra_loss': []})
            print(f"从第 {start_epoch} 轮继续训练（已完成 {checkpoint['epoch']} 轮）")
            if 'best_val_loss' in checkpoint:
                print(f"当前最佳验证损失: {best_val_loss:.4f}")
        
        # 获取保存间隔
        save_interval = self.config.get('save_interval', 10)
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_am_loss, train_intra_loss = self.train_epoch(train_loader)
            
            # 记录训练历史
            self.train_history['loss'].append(train_loss)
            self.train_history['am_loss'].append(train_am_loss)
            self.train_history['intra_loss'].append(train_intra_loss)
            
            print(f"训练损失: {train_loss:.4f} (AM: {train_am_loss:.4f}, Intra: {train_intra_loss:.4f})")
            
            # 验证
            if val_loader is not None:
                val_loss, val_am_loss, val_intra_loss, val_eer = self.validate(val_loader)
                if val_eer is not None:
                    print(f"验证损失: {val_loss:.4f} (AM: {val_am_loss:.4f}, Intra: {val_intra_loss:.4f}), EER: {val_eer:.4f}")
                else:
                    print(f"验证损失: {val_loss:.4f} (AM: {val_am_loss:.4f}, Intra: {val_intra_loss:.4f}), EER: N/A（正负样本不足）")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True, best_val_loss=best_val_loss)
                    print(f"保存最佳模型 (验证损失: {val_loss:.4f})")
            
            # 定期保存检查点（每个save_interval epoch）
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, is_best=False, best_val_loss=best_val_loss)
            
            # 更新学习率
            if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.scheduler.step(epoch)
            else:
                self.scheduler.step()
            print(f"当前学习率: {self.scheduler.get_last_lr()[0]:.6f}")
        
        # 训练结束前保存最后一次checkpoint
        self.save_checkpoint(num_epochs - 1, is_best=False, best_val_loss=best_val_loss)
        print("\n训练完成！")
    
    def save_checkpoint(self, epoch, is_best=False, best_val_loss=float('inf')):
        """
        保存检查点
        
        Args:
            epoch: 当前epoch
            is_best: 是否为最佳模型
            best_val_loss: 最佳验证损失
        """
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'config': self.config,
            'best_val_loss': best_val_loss
        }
        
        # 保存最新检查点（每次保存都更新latest_checkpoint）
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        if not is_best:  # 如果是保存最佳模型，已经在下面打印了
            print(f"检查点已保存: {checkpoint_path} (Epoch {epoch+1})")
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存: {best_path} (验证损失: {best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        Returns:
            start_epoch: 开始训练的epoch（checkpoint的epoch + 1）
            best_val_loss: 最佳验证损失
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 允许因架构升级（例如统计池化由 C->2C）导致的 shape 变化
        missing, unexpected, skipped = self._load_state_dict_flexible(checkpoint['model_state_dict'])
        if missing:
            print(f"加载检查点: 缺少参数 {missing}")
        if unexpected:
            print(f"加载检查点: 多余参数 {unexpected}")
        if skipped:
            print(f"加载检查点: 跳过shape不匹配参数 {skipped}")

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint.get('train_history', {'loss': [], 'am_loss': [], 'intra_loss': []})
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"加载检查点: {checkpoint_path} (Epoch {checkpoint['epoch']})")
        return start_epoch, best_val_loss


def create_dummy_dataset(batch_size=32, num_samples=1000, num_classes=100, seq_length=16000):
    """创建虚拟数据集用于测试"""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, num_classes, seq_length):
            self.num_samples = num_samples
            self.num_classes = num_classes
            self.seq_length = seq_length
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # 生成随机语音波形
            audio = torch.randn(1, self.seq_length)
            # 随机标签
            label = torch.randint(0, self.num_classes, (1,)).item()
            return audio, label
    
    train_dataset = DummyDataset(num_samples, num_classes, seq_length)
    val_dataset = DummyDataset(num_samples // 5, num_classes, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, num_classes


def create_voxceleb_dataset(data_root, 
                            batch_size=32,
                            sample_rate=16000,
                            segment_length=16000,
                            train_split=0.9,
                            augmentation=True,
                            num_workers=4):
    """
    创建VoxCeleb数据集加载器
    
    Args:
        data_root: VoxCeleb数据集根目录（包含wav文件夹）
        batch_size: 批次大小
        sample_rate: 采样率
        segment_length: 音频片段长度（采样点数）
        train_split: 训练集比例
        augmentation: 是否使用数据增强
        num_workers: 数据加载器工作线程数
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_classes: 说话人数量
    """
    from dataset import VoxCelebDataset
    
    # 创建完整数据集（用于获取说话人数量和划分）
    full_dataset = VoxCelebDataset(
        data_root=data_root,
        sample_rate=sample_rate,
        segment_length=segment_length,
        train=True,
        augmentation=False
    )
    
    num_classes = full_dataset.get_num_speakers()
    
    # 使用random_split划分训练集和验证集
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子保证可复现
    )
    
    # 为训练集启用数据增强（需要重新创建数据集实例）
    # 注意：这里简化处理，实际建议使用列表文件方式以获得更好的控制
    if augmentation:
        # 获取训练集的索引
        train_indices = train_dataset.indices
        # 重新创建训练数据集（带增强）
        train_dataset_full = VoxCelebDataset(
            data_root=data_root,
            sample_rate=sample_rate,
            segment_length=segment_length,
            train=True,
            augmentation=True
        )
        train_dataset = Subset(train_dataset_full, train_indices)
    
    # 如果使用GPU则开启pin_memory以提升主机到GPU的数据传输吞吐
    use_pin_memory = torch.cuda.is_available()
    # 对于多进程加载，持续worker可避免频繁重启带来的开销
    enable_persistent = num_workers > 0
    # 适度预取，提升吞吐
    prefetch = 2 if num_workers > 0 else None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=enable_persistent,
        prefetch_factor=prefetch,
        drop_last=True  # 丢弃最后一个不完整的batch，避免batch_size=1的问题
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=enable_persistent,
        prefetch_factor=prefetch,
        drop_last=False  # 验证时可以保留最后一个batch
    )
    
    return train_loader, val_loader, num_classes


def create_voxceleb_dataset_from_list(train_list,
                                     val_list,
                                     data_root=None,
                                     batch_size=32,
                                     sample_rate=16000,
                                     segment_length=16000,
                                     augmentation=True,
                                     num_workers=4):
    """
    从列表文件创建VoxCeleb数据集加载器
    
    Args:
        train_list: 训练集列表文件路径
        val_list: 验证集列表文件路径
        data_root: 数据集根目录（如果列表中是相对路径）
        batch_size: 批次大小
        sample_rate: 采样率
        segment_length: 音频片段长度
        augmentation: 是否使用数据增强
        num_workers: 数据加载器工作线程数
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_classes: 说话人数量
    """
    from dataset import VoxCelebDatasetFromList
    
    # 创建训练集
    train_dataset = VoxCelebDatasetFromList(
        list_file=train_list,
        data_root=data_root,
        sample_rate=sample_rate,
        segment_length=segment_length,
        train=True,
        augmentation=augmentation
    )
    
    # 创建验证集（需要共享说话人映射）
    val_dataset = VoxCelebDatasetFromList(
        list_file=val_list,
        data_root=data_root,
        sample_rate=sample_rate,
        segment_length=segment_length,
        train=False,
        augmentation=False
    )
    
    # 获取说话人数量（使用训练集的说话人数量）
    num_classes = train_dataset.get_num_speakers()
    
    # 如果使用GPU则开启pin_memory以提升主机到GPU的数据传输吞吐
    use_pin_memory = torch.cuda.is_available()
    enable_persistent = num_workers > 0
    prefetch = 2 if num_workers > 0 else None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=enable_persistent,  # 保持worker常驻，减少启动开销
        prefetch_factor=prefetch,  # 适度预取
        drop_last=True  # 丢弃最后一个不完整的batch，避免batch_size=1的问题
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=enable_persistent,
        prefetch_factor=prefetch,
        drop_last=False  # 验证时可以保留最后一个batch
    )
    
    return train_loader, val_loader, num_classes


if __name__ == '__main__':
    # 配置参数
    config = {
        'in_channels': 1,
        'frontend_channels': 64,
        'attention_channels': 192,  # 3 * frontend_channels
        'embedding_dim': 256,
        'num_classes': 100,
        'num_heads': 8,
        'dropout': 0.1,
        'am_margin': 0.3,
        'am_scale': 30.0,
        'intra_margin': 0.5,
        'lambda_intra': 0.1,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'lr_step_size': 20,
        'lr_gamma': 0.5,
        'checkpoint_dir': 'checkpoints',
        'num_epochs': 100
    }
    
    # 创建数据加载器（这里使用虚拟数据，实际使用时需要替换为真实数据集）
    print("创建数据加载器...")
    train_loader, val_loader = create_dummy_dataset(
        batch_size=32,
        num_samples=1000,
        num_classes=config['num_classes'],
        seq_length=16000
    )
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader, num_epochs=config['num_epochs'])

