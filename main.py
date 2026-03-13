"""
基于注意力机制的短语音说话人识别模型 - 主入口
"""
import torch
import json
import os
import gc
import random
from model import SpeakerRecognitionModel
from train import (
    Trainer,
    create_dummy_dataset,
    create_lrs_dataset_from_list,
    create_voxceleb_dataset,
    create_voxceleb_dataset_from_list,
    create_vox2video_dataset_from_list,
)

# 清理GPU缓存（在导入后立即执行）
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def infer_dataset_type(train_list):
    """根据列表文件内容推断数据集类型"""
    try:
        with open(train_list, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                file_path = line.split()[0].lower()
                if file_path.endswith('.wav'):
                    return 'voxceleb'
                if file_path.endswith(('.mp4', '.avi', '.mov')):
                    if 'vox' in file_path:
                        return 'vox2video'
                    return 'lrs'
    except OSError:
        pass
    return None


def _load_list_entries_with_speaker(list_file):
    """读取列表文件，提取原始行和 speaker_id。"""
    entries = []
    skipped_invalid = 0

    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 2:
                skipped_invalid += 1
                continue

            entries.append({
                'line': stripped,
                'speaker_id': parts[1],
            })

    return entries, skipped_invalid


def _count_utterances_by_speaker(entries):
    """统计每个说话人的样本条数。"""
    speaker_counts = {}
    for item in entries:
        speaker_id = item['speaker_id']
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
    return speaker_counts


def _format_count_stats(values):
    """格式化 min/avg/max 统计信息。"""
    if not values:
        return "min=0, avg=0.00, max=0"
    return f"min={min(values)}, avg={sum(values) / len(values):.2f}, max={max(values)}"


def create_speaker_limited_subset_lists(train_list,
                                        val_list,
                                        subset_num_speakers=500,
                                        min_utterances_per_speaker=80,
                                        subset_output_dir='subset_lists',
                                        seed=42):
    """
    先随机采样说话人，再保留这些说话人的全部语音。
    说话人从训练列表中抽样，验证列表按同一说话人集合过滤，确保标签空间一致。
    """
    subset_num_speakers = int(subset_num_speakers)
    min_utterances_per_speaker = int(min_utterances_per_speaker)
    if subset_num_speakers <= 0:
        print("警告: subset_num_speakers <= 0，使用完整列表")
        return train_list, val_list

    train_entries, train_invalid = _load_list_entries_with_speaker(train_list)
    val_entries, val_invalid = _load_list_entries_with_speaker(val_list)

    if train_invalid > 0:
        print(f"训练列表跳过无效行: {train_invalid} 条")
    if val_invalid > 0:
        print(f"验证列表跳过无效行: {val_invalid} 条")

    if not train_entries:
        raise ValueError("训练列表中没有可用样本，无法按说话人采样")

    train_counts = _count_utterances_by_speaker(train_entries)
    val_counts = _count_utterances_by_speaker(val_entries)

    all_train_speakers = sorted(train_counts.keys())
    total_counts = {
        speaker_id: train_counts.get(speaker_id, 0) + val_counts.get(speaker_id, 0)
        for speaker_id in all_train_speakers
    }

    if min_utterances_per_speaker > 0:
        candidate_speakers = [
            speaker_id for speaker_id in all_train_speakers
            if total_counts[speaker_id] >= min_utterances_per_speaker
        ]
    else:
        candidate_speakers = list(all_train_speakers)

    if not candidate_speakers:
        raise ValueError(
            f"没有说话人满足最小语音条数阈值: min_utterances_per_speaker={min_utterances_per_speaker}"
        )

    rng = random.Random(seed)
    if len(candidate_speakers) > subset_num_speakers:
        selected_speakers = set(rng.sample(candidate_speakers, subset_num_speakers))
    else:
        selected_speakers = set(candidate_speakers)
        if len(candidate_speakers) < subset_num_speakers:
            print(
                f"警告: 满足阈值的说话人仅有 {len(candidate_speakers)} 个，"
                f"小于目标 {subset_num_speakers} 个，将全部保留"
            )

    selected_train_entries = [item for item in train_entries if item['speaker_id'] in selected_speakers]
    selected_val_entries = [item for item in val_entries if item['speaker_id'] in selected_speakers]

    os.makedirs(subset_output_dir, exist_ok=True)
    train_subset_list = os.path.join(
        subset_output_dir,
        f"train_subset_spk{subset_num_speakers}_min{max(min_utterances_per_speaker, 0)}_seed{seed}.txt"
    )
    val_subset_list = os.path.join(
        subset_output_dir,
        f"val_subset_spk{subset_num_speakers}_min{max(min_utterances_per_speaker, 0)}_seed{seed}.txt"
    )

    with open(train_subset_list, 'w', encoding='utf-8') as f:
        for item in selected_train_entries:
            f.write(item['line'] + '\n')

    with open(val_subset_list, 'w', encoding='utf-8') as f:
        for item in selected_val_entries:
            f.write(item['line'] + '\n')

    per_speaker_train_counts = [train_counts.get(speaker_id, 0) for speaker_id in selected_speakers]
    per_speaker_val_counts = [val_counts.get(speaker_id, 0) for speaker_id in selected_speakers]
    per_speaker_total_counts = [
        train_counts.get(speaker_id, 0) + val_counts.get(speaker_id, 0)
        for speaker_id in selected_speakers
    ]

    print(
        f"说话人采样完成: 目标说话人={subset_num_speakers}, 实际保留={len(selected_speakers)}, "
        f"min_utterances_per_speaker={min_utterances_per_speaker}"
    )
    print(
        f"采样后样本数: train={len(selected_train_entries)}, "
        f"val={len(selected_val_entries)}"
    )
    print(f"每说话人样本统计(train): {_format_count_stats(per_speaker_train_counts)}")
    print(f"每说话人样本统计(val): {_format_count_stats(per_speaker_val_counts)}")
    print(f"每说话人样本统计(total): {_format_count_stats(per_speaker_total_counts)}")
    print(f"子集列表已生成: {train_subset_list}, {val_subset_list}")

    return train_subset_list, val_subset_list


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
        'val_interval': 1,
        'compute_eer': False,
        'eer_max_samples': 2048,
        'empty_cache_each_epoch': True,
        'use_video': False,
        'video_in_channels': 3,
        'video_channels': 32,
    }
    for k, v in defaults.items():
        config.setdefault(k, v)
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    
    # 检查是否指定了LRS数据集路径
    data_config = config_dict.get('data', {})
    data_root = data_config.get('data_root', None)
    train_list = data_config.get('train_list', None)
    val_list = data_config.get('val_list', None)
    dataset_type = data_config.get('dataset_type')
    
    if train_list and val_list:
        inferred_dataset_type = infer_dataset_type(train_list)
        if dataset_type and inferred_dataset_type and dataset_type != inferred_dataset_type:
            print(f"警告: config.json 中的 dataset_type={dataset_type}，但列表文件看起来更像 {inferred_dataset_type}")
        dataset_type = dataset_type or inferred_dataset_type

        if dataset_type == 'voxceleb':
            dataset_loader = create_voxceleb_dataset_from_list
            dataset_name = 'VoxCeleb'
        elif dataset_type == 'vox2video':
            dataset_loader = create_vox2video_dataset_from_list
            dataset_name = 'Vox2Video'
        elif dataset_type == 'lrs':
            dataset_loader = create_lrs_dataset_from_list
            dataset_name = 'LRS'
        else:
            raise ValueError('无法根据列表文件推断数据集类型，请在config.json中设置 data.dataset_type')

        subset_num_speakers = data_config.get('subset_num_speakers', None)
        if subset_num_speakers is None and data_config.get('max_dataset_size_gb', None) is not None:
            # 兼容旧配置：此前通过 max_dataset_size_gb 触发随机容量抽样，
            # 现在改为优先保证说话人完整性，默认采样 500 个说话人。
            subset_num_speakers = 500
            print(
                f"检测到旧配置 max_dataset_size_gb={data_config.get('max_dataset_size_gb')}，"
                "已切换为按说话人采样（默认 500 人）"
            )

        if subset_num_speakers is not None:
            try:
                subset_num_speakers = int(subset_num_speakers)
            except (TypeError, ValueError):
                print(f"警告: subset_num_speakers={subset_num_speakers} 无效，忽略说话人采样")
                subset_num_speakers = None

        if subset_num_speakers is not None and subset_num_speakers > 0:
            subset_seed_raw = data_config.get('subset_seed', 42)
            subset_min_utterances_raw = data_config.get('subset_min_utterances', 80)
            subset_output_dir = data_config.get('subset_list_dir', 'subset_lists')

            try:
                subset_seed = int(subset_seed_raw)
            except (TypeError, ValueError):
                print(f"警告: subset_seed={subset_seed_raw} 无效，使用默认值 42")
                subset_seed = 42

            try:
                subset_min_utterances = int(subset_min_utterances_raw)
            except (TypeError, ValueError):
                print(f"警告: subset_min_utterances={subset_min_utterances_raw} 无效，使用默认值 80")
                subset_min_utterances = 80

            print(
                f"启用说话人采样，目标保留 {subset_num_speakers} 个说话人，"
                f"每个说话人至少 {subset_min_utterances} 条语音"
            )
            train_list, val_list = create_speaker_limited_subset_lists(
                train_list=train_list,
                val_list=val_list,
                subset_num_speakers=subset_num_speakers,
                min_utterances_per_speaker=subset_min_utterances,
                subset_output_dir=subset_output_dir,
                seed=subset_seed
            )
        elif subset_num_speakers is not None and subset_num_speakers <= 0:
            print("subset_num_speakers <= 0，忽略说话人采样，使用完整列表")

        print(f"从列表文件加载数据集...")
        print(f"  数据集类型: {dataset_name}")
        print(f"  训练集列表: {train_list}")
        print(f"  验证集列表: {val_list}")
        loader_kwargs = {
            'train_list': train_list,
            'val_list': val_list,
            'data_root': data_root,
            'batch_size': config_dict['training']['batch_size'],
            'sample_rate': data_config.get('sample_rate', 16000),
            'segment_length': data_config.get('seq_length', 16000),
            'augmentation': data_config.get('augmentation', True),
            'num_workers': data_config.get('num_workers', 4),
            'pin_memory': data_config.get('pin_memory', None),
            'persistent_workers': data_config.get('persistent_workers', False),
            'prefetch_factor': data_config.get('prefetch_factor', 1),
        }
        if dataset_type == 'vox2video':
            loader_kwargs['num_frames'] = data_config.get('video_num_frames', 8)
            loader_kwargs['frame_size'] = data_config.get('video_frame_size', 112)

        train_loader, val_loader, num_classes = dataset_loader(**loader_kwargs)
        # 更新配置中的说话人数量
        config['num_classes'] = num_classes
        config['use_video'] = (dataset_type == 'vox2video')
        config_dict['model']['num_classes'] = num_classes
        print(f"检测到 {num_classes} 个说话人")
    elif data_root and dataset_type == 'voxceleb':
        print("从VoxCeleb目录直接扫描并划分数据集...")
        train_loader, val_loader, num_classes = create_voxceleb_dataset(
            data_root=data_root,
            batch_size=config_dict['training']['batch_size'],
            sample_rate=data_config.get('sample_rate', 16000),
            segment_length=data_config.get('seq_length', 16000),
            train_split=data_config.get('train_split', 0.8),
            augmentation=data_config.get('augmentation', True),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', None),
            persistent_workers=data_config.get('persistent_workers', False),
            prefetch_factor=data_config.get('prefetch_factor', 1)
        )
        config['num_classes'] = num_classes
        config_dict['model']['num_classes'] = num_classes
        print(f"检测到 {num_classes} 个说话人")
    else:
        # 使用虚拟数据（用于测试）
        print("未找到可用的数据集配置，使用虚拟数据进行测试")
        print("提示：在config.json中设置 train_list/val_list 或 VoxCeleb 的 data_root")
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
            # response = input("是否从中断的地方继续训练？(y/n，默认y): ").strip().lower()
            response = 'y'  # 强制自动从中断处继续训练
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
