# VoxCeleb数据集使用指南

## 数据集准备

### 方式1：使用列表文件（推荐）

这是最推荐的方式，可以精确控制训练集和验证集的划分。

#### 步骤1：生成列表文件

```bash
python prepare_voxceleb_list.py \
    --data_root /path/to/voxceleb \
    --output all_list.txt \
    --split \
    --train_list train_list.txt \
    --val_list val_list.txt \
    --train_ratio 0.9
```

这将生成：
- `all_list.txt`: 所有音频文件的列表
- `train_list.txt`: 训练集列表（90%）
- `val_list.txt`: 验证集列表（10%）

#### 步骤2：配置config.json

在 `config.json` 的 `data` 部分添加：

```json
{
  "data": {
    "data_root": "/path/to/voxceleb",
    "train_list": "train_list.txt",
    "val_list": "val_list.txt",
    "sample_rate": 16000,
    "seq_length": 16000,
    "augmentation": true,
    "num_workers": 4
  }
}
```

#### 步骤3：开始训练

```bash
python main.py
```

### 方式2：直接使用目录结构

如果数据集目录结构如下：

```
voxceleb/
    wav/
        id00001/
            video_id/
                file.wav
        id00002/
            ...
```

只需要在 `config.json` 中设置：

```json
{
  "data": {
    "data_root": "/path/to/voxceleb",
    "sample_rate": 16000,
    "seq_length": 16000,
    "augmentation": true,
    "train_split": 0.9,
    "num_workers": 4
  }
}
```

程序会自动扫描目录并划分训练/验证集。

## 列表文件格式

列表文件每行格式为：
```
wav/id00001/video_id/file.wav id00001
wav/id00002/video_id/file.wav id00002
```

即：`音频文件路径（相对于data_root） 说话人ID`

## 数据集要求

- 音频格式：WAV文件
- 采样率：任意（会自动重采样到16kHz）
- 声道：支持单声道和立体声（会自动转换为单声道）
- 长度：任意长度（会自动截断或填充到指定长度）

## 数据增强

训练时默认启用以下数据增强：
- 随机时间平移（随机裁剪）
- 随机音量缩放（0.8-1.2倍）
- 随机添加噪声（30%概率）

可在 `config.json` 中通过 `data.augmentation` 控制。

## 常见问题

### Q: 数据集太大，加载很慢怎么办？

A: 建议使用列表文件方式，并且：
- 减少 `num_workers`（如果内存不足）
- 使用SSD存储数据集
- 预先处理音频文件（重采样、转换为单声道）

### Q: 内存不足怎么办？

A: 
- 减少 `batch_size`
- 减少 `num_workers`
- 使用更短的 `seq_length`

### Q: 如何自定义数据增强？

A: 修改 `dataset/voxceleb_dataset.py` 中的 `_augment_audio` 方法。

### Q: 说话人数量不对？

A: 程序会自动检测说话人数量并更新模型配置。如果发现问题，检查：
- 数据集目录结构是否正确
- 列表文件格式是否正确
- 说话人ID是否一致

## 性能建议

- **batch_size**: 根据GPU内存调整，推荐16-64
- **num_workers**: CPU核心数，推荐4-8
- **seq_length**: 短语音推荐16000（1秒@16kHz），可根据需要调整
- **augmentation**: 训练时开启，验证时关闭





