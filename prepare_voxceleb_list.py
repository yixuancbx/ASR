"""
生成VoxCeleb数据集列表文件的工具脚本
用于创建train_list.txt和val_list.txt
"""
import os
import glob
import random


def generate_list_from_directory(data_root, output_file, train_ratio=0.9):
    """
    从目录结构生成列表文件
    
    Args:
        data_root: VoxCeleb数据集根目录（包含wav文件夹）
        output_file: 输出的列表文件路径
        train_ratio: 训练集比例（用于生成训练列表时使用）
    """
    wav_dir = os.path.join(data_root, 'wav')
    if not os.path.exists(wav_dir):
        wav_dir = data_root
    
    # 查找所有说话人目录
    speaker_dirs = sorted([d for d in os.listdir(wav_dir) 
                          if os.path.isdir(os.path.join(wav_dir, d)) and d.startswith('id')])
    
    all_files = []
    
    # 收集所有音频文件
    for speaker_dir in speaker_dirs:
        speaker_path = os.path.join(wav_dir, speaker_dir)
        wav_files = glob.glob(os.path.join(speaker_path, '**', '*.wav'), recursive=True)
        
        for wav_file in wav_files:
            # 使用相对路径
            rel_path = os.path.relpath(wav_file, data_root)
            all_files.append((rel_path, speaker_dir))
    
    # 随机打乱
    random.shuffle(all_files)
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path, speaker_id in all_files:
            f.write(f"{file_path} {speaker_id}\n")
    
    print(f"生成了 {len(all_files)} 条记录到 {output_file}")


def split_list_file(list_file, train_list_file, val_list_file, train_ratio=0.9):
    """
    将一个列表文件划分为训练集和验证集
    
    Args:
        list_file: 输入的完整列表文件
        train_list_file: 训练集列表文件输出路径
        val_list_file: 验证集列表文件输出路径
        train_ratio: 训练集比例
    """
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 随机打乱
    random.shuffle(lines)
    
    # 划分
    train_size = int(len(lines) * train_ratio)
    train_lines = lines[:train_size]
    val_lines = lines[train_size:]
    
    # 写入训练集
    with open(train_list_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # 写入验证集
    with open(val_list_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print(f"训练集: {len(train_lines)} 条记录 -> {train_list_file}")
    print(f"验证集: {len(val_lines)} 条记录 -> {val_list_file}")


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
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='训练集比例')
    parser.add_argument('--split', action='store_true',
                       help='是否划分训练集和验证集')
    
    args = parser.parse_args()
    
    # 生成完整列表
    generate_list_from_directory(args.data_root, args.output, args.train_ratio)
    
    # 如果需要划分
    if args.split:
        split_list_file(args.output, args.train_list, args.val_list, args.train_ratio)
        print(f"\n请在config.json中设置:")
        print(f'  "train_list": "{args.train_list}",')
        print(f'  "val_list": "{args.val_list}",')
        print(f'  "data_root": "{args.data_root}"')





