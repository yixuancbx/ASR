"""
基于注意力机制的短语音说话人识别模型 - 主入口
"""
import torch
import json
import os
import gc
import random
from collections import defaultdict
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


def _format_size(num_bytes):
    """将字节数格式化为易读单位。"""
    size = float(num_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0 or unit == 'TB':
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}TB"


def _resolve_media_path(media_path, data_root):
    """将列表中的相对路径解析为绝对路径。"""
    media_path = os.path.normpath(media_path)
    if data_root and not os.path.isabs(media_path):
        return os.path.join(data_root, media_path)
    return media_path


def _load_list_entries_with_size(list_file, data_root=None):
    """读取列表文件并统计每个样本文件大小。"""
    entries = []
    skipped_invalid = 0
    skipped_missing = 0

    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 2:
                skipped_invalid += 1
                continue

            media_path = _resolve_media_path(parts[0], data_root=data_root)
            if not os.path.exists(media_path):
                skipped_missing += 1
                continue

            try:
                size_bytes = os.path.getsize(media_path)
            except OSError:
                skipped_missing += 1
                continue

            entries.append({
                'line': stripped,
                'size_bytes': size_bytes,
                'speaker_id': parts[1],
            })

    return entries, skipped_invalid, skipped_missing


def _group_entries_by_speaker(entries):
    """按说话人分组。"""
    grouped = defaultdict(list)
    for item in entries:
        grouped[item['speaker_id']].append(item)
    return grouped


def create_speaker_limited_subset_lists(train_list,
                                        val_list,
                                        data_root=None,
                                        target_num_speakers=500,
                                        min_utterances_per_speaker=80,
                                        subset_output_dir='subset_lists',
                                        seed=42):
    """
    随机抽取指定数量的说话人，并保留其全部语音样本（train/val 同步过滤）。
    可选地仅保留语音条数达到阈值的说话人，避免样本过少类别。
    """
    if target_num_speakers is None:
        return train_list, val_list

    target_num_speakers = int(target_num_speakers)
    if target_num_speakers <= 0:
        print("警告: target_num_speakers <= 0，使用完整列表")
        return train_list, val_list

    train_entries, train_invalid, train_missing = _load_list_entries_with_size(train_list, data_root=data_root)
    val_entries, val_invalid, val_missing = _load_list_entries_with_size(val_list, data_root=data_root)

    train_total_bytes = sum(item['size_bytes'] for item in train_entries)
    val_total_bytes = sum(item['size_bytes'] for item in val_entries)
    available_total_bytes = train_total_bytes + val_total_bytes

    print(
        f"原始列表可用容量: train={_format_size(train_total_bytes)}, "
        f"val={_format_size(val_total_bytes)}, total={_format_size(available_total_bytes)}"
    )
    if train_invalid + train_missing > 0:
        print(f"训练列表跳过: 无效行 {train_invalid} 条, 缺失文件 {train_missing} 条")
    if val_invalid + val_missing > 0:
        print(f"验证列表跳过: 无效行 {val_invalid} 条, 缺失文件 {val_missing} 条")

    if available_total_bytes == 0:
        raise ValueError("列表中没有可用文件，无法按说话人采样")

    all_entries = train_entries + val_entries
    grouped = _group_entries_by_speaker(all_entries)
    all_speakers = sorted(grouped.keys())

    if min_utterances_per_speaker is not None:
        min_utterances_per_speaker = int(min_utterances_per_speaker)
        if min_utterances_per_speaker <= 1:
            min_utterances_per_speaker = None

    if min_utterances_per_speaker is None:
        eligible_speakers = all_speakers
    else:
        eligible_speakers = sorted([
            speaker_id
            for speaker_id, items in grouped.items()
            if len(items) >= min_utterances_per_speaker
        ])

    if not eligible_speakers:
        raise ValueError(
            f"没有说话人满足最小语音条数要求: min_utterances_per_speaker={min_utterances_per_speaker}"
        )

    if target_num_speakers >= len(eligible_speakers):
        selected_speakers = set(eligible_speakers)
        print(
            f"目标说话人数 {target_num_speakers} >= 可选说话人数 {len(eligible_speakers)}，"
            "将使用全部可选说话人"
        )
    else:
        selected_speakers = set(
            random.Random(seed).sample(eligible_speakers, target_num_speakers)
        )

    selected_train_entries = [item for item in train_entries if item['speaker_id'] in selected_speakers]
    selected_val_entries = [item for item in val_entries if item['speaker_id'] in selected_speakers]

    selected_train_bytes = sum(item['size_bytes'] for item in selected_train_entries)
    selected_val_bytes = sum(item['size_bytes'] for item in selected_val_entries)
    selected_total_bytes = selected_train_bytes + selected_val_bytes

    selected_utterance_counts = [
        len(grouped[speaker_id]) for speaker_id in sorted(selected_speakers)
    ]
    min_utt = min(selected_utterance_counts) if selected_utterance_counts else 0
    max_utt = max(selected_utterance_counts) if selected_utterance_counts else 0
    avg_utt = (
        sum(selected_utterance_counts) / len(selected_utterance_counts)
        if selected_utterance_counts else 0.0
    )

    os.makedirs(subset_output_dir, exist_ok=True)
    utt_tag = f"_minutt{min_utterances_per_speaker}" if min_utterances_per_speaker else ""
    speaker_tag = f"spk{len(selected_speakers)}{utt_tag}"
    train_subset_list = os.path.join(subset_output_dir, f"train_subset_{speaker_tag}_seed{seed}.txt")
    val_subset_list = os.path.join(subset_output_dir, f"val_subset_{speaker_tag}_seed{seed}.txt")

    with open(train_subset_list, 'w', encoding='utf-8') as f:
        for item in selected_train_entries:
            f.write(item['line'] + '\n')

    with open(val_subset_list, 'w', encoding='utf-8') as f:
        for item in selected_val_entries:
            f.write(item['line'] + '\n')

    print(
        f"按说话人采样完成: 保留 {len(selected_speakers)} 人, "
        f"总样本={len(selected_train_entries) + len(selected_val_entries)}"
    )
    print(
        f"保留后容量: train={_format_size(selected_train_bytes)}, "
        f"val={_format_size(selected_val_bytes)}, total={_format_size(selected_total_bytes)}"
    )
    print(
        f"每说话人语音条数统计: min={min_utt}, avg={avg_utt:.1f}, max={max_utt}"
    )
    print(
        f"采样后样本数: train={len(selected_train_entries)}, "
        f"val={len(selected_val_entries)}"
    )
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

        max_dataset_size_gb = data_config.get('max_dataset_size_gb', None)
        if max_dataset_size_gb is not None:
            try:
                max_dataset_size_gb = float(max_dataset_size_gb)
            except (TypeError, ValueError):
                print(f"警告: max_dataset_size_gb={max_dataset_size_gb} 无效，已忽略容量限制")
                max_dataset_size_gb = None

        if max_dataset_size_gb is not None and max_dataset_size_gb > 0:
            subset_seed = int(data_config.get('subset_seed', 42))
            subset_output_dir = data_config.get('subset_list_dir', 'subset_lists')
            print(f"启用容量限制采样，目标总容量约 {max_dataset_size_gb:.2f}GB")
            train_list, val_list = create_size_limited_subset_lists(
                train_list=train_list,
                val_list=val_list,
                data_root=data_root,
                target_size_gb=max_dataset_size_gb,
                subset_output_dir=subset_output_dir,
                seed=subset_seed
            )
        elif max_dataset_size_gb is not None and max_dataset_size_gb <= 0:
            print("max_dataset_size_gb <= 0，忽略容量限制，使用完整列表")

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
