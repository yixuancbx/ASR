"""
生成VoxCeleb数据集列表文件的工具脚本
用于创建train_list.txt和val_list.txt
"""
import os
import glob
import random
from collections import defaultdict


def collect_voxceleb_entries(data_root):
    """扫描目录并返回 [(相对路径, speaker_id), ...]"""
    wav_dir = os.path.join(data_root, 'wav')
    if not os.path.exists(wav_dir):
        wav_dir = data_root

    speaker_dirs = sorted([d for d in os.listdir(wav_dir)
                          if os.path.isdir(os.path.join(wav_dir, d)) and d.startswith('id')])

    all_files = []
    for speaker_dir in speaker_dirs:
        speaker_path = os.path.join(wav_dir, speaker_dir)
        wav_files = sorted(glob.glob(os.path.join(speaker_path, '**', '*.wav'), recursive=True))

        for wav_file in wav_files:
            rel_path = os.path.relpath(wav_file, data_root)
            all_files.append((rel_path, speaker_dir))

    return all_files


def generate_list_from_directory(data_root, output_file):
    """
    从目录结构生成列表文件
    
    Args:
        data_root: VoxCeleb数据集根目录（包含wav文件夹）
        output_file: 输出的列表文件路径
    """
    all_files = collect_voxceleb_entries(data_root)
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path, speaker_id in all_files:
            f.write(f"{file_path} {speaker_id}\n")
    
    print(f"生成了 {len(all_files)} 条记录到 {output_file}")


def split_entries_by_speaker(entries, train_ratio=0.8, seed=42):
    """
    按说话人分组后再划分，保证训练/验证集的标签空间一致。
    对于仅有1条样本的说话人，保留在训练集，避免验证集出现未训练类别。
    """
    rng = random.Random(seed)
    grouped_entries = defaultdict(list)

    for rel_path, speaker_id in entries:
        grouped_entries[speaker_id].append((rel_path, speaker_id))

    train_entries = []
    val_entries = []

    for speaker_id in sorted(grouped_entries):
        speaker_entries = grouped_entries[speaker_id]
        rng.shuffle(speaker_entries)

        if len(speaker_entries) == 1:
            train_entries.extend(speaker_entries)
            continue

        train_size = int(len(speaker_entries) * train_ratio)
        train_size = max(1, min(train_size, len(speaker_entries) - 1))

        train_entries.extend(speaker_entries[:train_size])
        val_entries.extend(speaker_entries[train_size:])

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    return train_entries, val_entries


def split_list_file(list_file, train_list_file, val_list_file, train_ratio=0.8, seed=42):
    """
    将一个列表文件划分为训练集和验证集
    
    Args:
        list_file: 输入的完整列表文件
        train_list_file: 训练集列表文件输出路径
        val_list_file: 验证集列表文件输出路径
        train_ratio: 训练集比例
        seed: 随机种子
    """
    entries = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            entries.append((parts[0], parts[1]))

    train_entries, val_entries = split_entries_by_speaker(entries, train_ratio=train_ratio, seed=seed)
    train_lines = [f"{file_path} {speaker_id}\n" for file_path, speaker_id in train_entries]
    val_lines = [f"{file_path} {speaker_id}\n" for file_path, speaker_id in val_entries]
    
    # 写入训练集
    with open(train_list_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # 写入验证集
    with open(val_list_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print(f"训练集: {len(train_lines)} 条记录 -> {train_list_file}")
    print(f"验证集: {len(val_lines)} 条记录 -> {val_list_file}")


def generate_and_split_from_directory(data_root, train_list_file, val_list_file, train_ratio=0.8, seed=42):
    """直接从目录扫描并生成按说话人划分后的训练/验证列表"""
    entries = collect_voxceleb_entries(data_root)
    train_entries, val_entries = split_entries_by_speaker(entries, train_ratio=train_ratio, seed=seed)

    with open(train_list_file, 'w', encoding='utf-8') as f:
        for file_path, speaker_id in train_entries:
            f.write(f"{file_path} {speaker_id}\n")

    with open(val_list_file, 'w', encoding='utf-8') as f:
        for file_path, speaker_id in val_entries:
            f.write(f"{file_path} {speaker_id}\n")

    print(f"训练集: {len(train_entries)} 条记录 -> {train_list_file}")
    print(f"验证集: {len(val_entries)} 条记录 -> {val_list_file}")
    print(f"说话人数: {len({speaker_id for _, speaker_id in entries})}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成VoxCeleb数据集列表文件')
    parser.add_argument('--data_root', type=str, required=True,
                       help='VoxCeleb数据集根目录')
    parser.add_argument('--output', type=str, default='all_list.txt',
                       help='输出列表文件路径')
    parser.add_argument('--train_list', type=str, default='train_list.txt',
                       help='训练集列表文件路径')
    parser.add_argument('--val_list', type=str, default='val_list.txt',
                       help='验证集列表文件路径')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--split', action='store_true',
                       help='是否划分训练集和验证集')
    
    args = parser.parse_args()
    
    # 生成完整列表
    generate_list_from_directory(args.data_root, args.output)
    
    # 如果需要划分
    if args.split:
        generate_and_split_from_directory(
            args.data_root,
            args.train_list,
            args.val_list,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        print(f"\n请在config.json中设置:")
        print(f'  "train_list": "{args.train_list}",')
        print(f'  "val_list": "{args.val_list}",')
        print(f'  "data_root": "{args.data_root}"')





