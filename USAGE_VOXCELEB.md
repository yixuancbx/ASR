# VoxCeleb数据集快速使用指南

## 快速开始

### 方法1：使用列表文件（推荐，最稳定）

1. **生成列表文件**：
```bash
python prepare_voxceleb_list.py --data_root D:/voxceleb --split --train_list train_list.txt --val_list val_list.txt
```

2. **修改config.json**：
```json
{
  "data": {
    "data_root": "D:/voxceleb",
    "train_list": "train_list.txt",
    "val_list": "val_list.txt",
    "sample_rate": 16000,
    "seq_length": 16000,
    "augmentation": true,
    "num_workers": 4
  }
}
```

3. **开始训练**：
```bash
python main.py
```

### 方法2：直接使用目录（简单但不推荐用于大型数据集）

1. **修改config.json**：
```json
{
  "data": {
    "data_root": "D:/voxceleb",
    "sample_rate": 16000,
    "seq_length": 16000,
    "augmentation": true,
    "train_split": 0.9,
    "num_workers": 4
  }
}
```

2. **开始训练**：
```bash
python main.py
```

## 数据集目录结构

VoxCeleb数据集应该具有以下结构：

```
voxceleb/
    wav/
        id00001/
            video_id1/
                file1.wav
                file2.wav
            video_id2/
                file3.wav
        id00002/
            ...
```

或者已解压的格式：
```
voxceleb/
    id00001/
        video_id/
            file.wav
    id00002/
        ...
```

## 配置文件说明

在 `config.json` 的 `data` 部分：

- `data_root`: VoxCeleb数据集根目录路径
- `train_list`: 训练集列表文件路径（可选，推荐）
- `val_list`: 验证集列表文件路径（可选，推荐）
- `sample_rate`: 目标采样率（默认16000）
- `seq_length`: 音频片段长度，采样点数（默认16000，即1秒）
- `augmentation`: 是否使用数据增强（默认true）
- `train_split`: 训练集比例，仅在未使用列表文件时生效（默认0.9）
- `num_workers`: 数据加载线程数（默认4）

## 注意事项

1. **列表文件方式更推荐**：可以精确控制训练/验证集划分，避免每次重新扫描目录
2. **说话人数量自动检测**：程序会自动检测数据集中的说话人数量并更新模型配置
3. **数据增强**：训练时自动启用，验证时自动关闭
4. **内存使用**：如果内存不足，可以减少 `batch_size` 或 `num_workers`

## 示例配置

### 完整配置示例（使用列表文件）

```json
{
  "model": {
    "num_classes": 1211
  },
  "data": {
    "data_root": "D:/voxceleb",
    "train_list": "train_list.txt",
    "val_list": "val_list.txt",
    "sample_rate": 16000,
    "seq_length": 16000,
    "augmentation": true,
    "num_workers": 4
  },
  "training": {
    "batch_size": 32,
    "num_epochs": 100
  }
}
```

**注意**：`model.num_classes` 会在运行时自动更新，初始值可以设置为任意值。





