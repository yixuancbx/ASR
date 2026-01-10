"""
基于注意力机制的短语音说话人识别模型 - 主入口
"""
import torch
import json
import os
import gc
from model import SpeakerRecognitionModel
from train import Trainer, create_dummy_dataset, create_voxceleb_dataset, create_voxceleb_dataset_from_list

# 清理GPU缓存（在导入后立即执行）
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def main():
    """主函数"""
    print("=" * 60)
    print("基于注意力机制的短语音说话人识别模型")
    print("=" * 60)
    
    # 加载配置
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    except FileNotFoundError:
        print("未找到config.json，使用默认配置")
        config_dict = {
            'model': {
                'in_channels': 1,
                'frontend_channels': 64,
                'attention_channels': 192,
                'embedding_dim': 256,
                'num_classes': 100,
                'num_heads': 8,
                'dropout': 0.1
            },
            'loss': {
                'am_margin': 0.3,
                'am_scale': 30.0,
                'intra_margin': 0.5,
                'lambda_intra': 0.1
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'num_epochs': 100,
                'lr_step_size': 20,
                'lr_gamma': 0.5,
                'checkpoint_dir': 'checkpoints'
            },
            'data': {
                'seq_length': 16000
            }
        }
    
    # 合并配置：优先使用 config.json 的全部字段，缺失时再补默认值
    config = {}
    for section in ['model', 'loss', 'training']:
        config.update(config_dict.get(section, {}))
    # 补充默认值（仅当缺失时）
    defaults = {
        'in_channels': 1,
        'frontend_channels': 64,
        'attention_channels': 192,
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
        'num_epochs': 100,
        # 新增：训练与调度相关的可选项，保证从 config.json 读取时不丢失
        'scheduler': 'cosine',
        'cosine_T0': 10,
        'cosine_Tmult': 2,
        'compile': False,
        'use_amp': True,
        'accumulation_steps': 1,
        'max_train_batches': None,
        'max_val_batches': None,
        'save_interval': 10,
    }
    for k, v in defaults.items():
        config.setdefault(k, v)
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    
    # 检查是否指定了VoxCeleb数据集路径
    data_config = config_dict.get('data', {})
    data_root = data_config.get('data_root', None)
    train_list = data_config.get('train_list', None)
    val_list = data_config.get('val_list', None)
    
    if train_list and val_list:
        # 使用列表文件方式
        print(f"从列表文件加载数据集...")
        print(f"  训练集列表: {train_list}")
        print(f"  验证集列表: {val_list}")
        train_loader, val_loader, num_classes = create_voxceleb_dataset_from_list(
            train_list=train_list,
            val_list=val_list,
            data_root=data_root,
            batch_size=config_dict['training']['batch_size'],
            sample_rate=data_config.get('sample_rate', 16000),
            segment_length=data_config.get('seq_length', 16000),
            augmentation=data_config.get('augmentation', True),
            num_workers=data_config.get('num_workers', 4)
        )
        # 更新配置中的说话人数量
        config['num_classes'] = num_classes
        config_dict['model']['num_classes'] = num_classes
        print(f"检测到 {num_classes} 个说话人")
    elif data_root and os.path.exists(data_root):
        # 使用目录方式
        print(f"从目录加载数据集: {data_root}")
        train_loader, val_loader, num_classes = create_voxceleb_dataset(
            data_root=data_root,
            batch_size=config_dict['training']['batch_size'],
            sample_rate=data_config.get('sample_rate', 16000),
            segment_length=data_config.get('seq_length', 16000),
            train_split=data_config.get('train_split', 0.9),
            augmentation=data_config.get('augmentation', True),
            num_workers=data_config.get('num_workers', 4)
        )
        # 更新配置中的说话人数量
        config['num_classes'] = num_classes
        config_dict['model']['num_classes'] = num_classes
        print(f"检测到 {num_classes} 个说话人")
    else:
        # 使用虚拟数据（用于测试）
        print("未找到VoxCeleb数据集路径，使用虚拟数据进行测试")
        print("提示：在config.json中设置data_root或train_list/val_list以使用真实数据集")
        train_loader, val_loader, _ = create_dummy_dataset(
            batch_size=config_dict['training']['batch_size'],
            num_samples=1000,
            num_classes=config['num_classes'],
            seq_length=config_dict['data']['seq_length']
        )
    
    # 创建训练器（使用更新后的配置）
    print("初始化模型和训练器...")
    trainer = Trainer(config)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 检查是否要从checkpoint恢复
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    resume_checkpoint = config_dict['training'].get('resume_from', None)
    
    if resume_checkpoint:
        checkpoint_path = resume_checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
        if os.path.exists(checkpoint_path):
            print(f"\n从指定checkpoint恢复: {checkpoint_path}")
        else:
            print(f"\n警告: 指定的checkpoint不存在: {checkpoint_path}")
            print("将自动查找最新的checkpoint或从头开始训练")
            resume_checkpoint = None
    else:
        # 检查是否存在最新的checkpoint
        latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        if os.path.exists(latest_checkpoint):
            print(f"\n发现之前的训练记录: {latest_checkpoint}")
            response = input("是否从中断的地方继续训练？(y/n，默认y): ").strip().lower()
            if response in ['', 'y', 'yes']:
                resume_checkpoint = latest_checkpoint
                print("将从检查点继续训练...")
            else:
                print("将从头开始训练...")
                resume_checkpoint = None
    
    # 开始训练
    print("\n开始训练...")
    trainer.train(train_loader, val_loader, num_epochs=config['num_epochs'], resume_from=resume_checkpoint)
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()
