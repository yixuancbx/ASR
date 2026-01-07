# 基于注意力机制的短语音说话人识别模型

## 项目简介

本项目实现了一个基于多层次动态注意力机制的短语音说话人识别模型，针对短语音特征稀疏且关键信息分布不均的问题，设计了层次化的注意力网络和多尺度特征融合机制。

## 模型架构

### 1. 多尺度特征提取前端

包含三个并行分支，同步提取不同尺度的特征：

- **分支1：小卷积核** (kernel_size=3) - 提取局部精细特征（音素细节）
- **分支2：大卷积核** (kernel_size=15) - 提取全局轮廓特征（语调轮廓）
- **分支3：深度可分离卷积** (kernel_size=7) - 轻量化设计，平衡效率和性能

### 2. 多层次动态注意力融合模块

包含三种注意力机制，动态融合特征：

- **局部帧级注意力** - 在卷积层后，聚焦于短时谱中判别性最强的局部特征
- **全局时序注意力** - 使用多头自注意力机制，建模整个短语音片段内帧与帧之间的长程依赖关系，捕捉说话人的韵律和节奏模式
- **特征通道注意力** - 自适应地重新校准不同特征通道的权重，增强重要特征维度的贡献

### 3. 特征融合与嵌入映射

- 特征拼接与融合模块
- 全局平均池化
- 全连接层映射到嵌入向量空间

### 4. 混合损失函数

- **AM-Softmax损失** - 引入角度间隔，最大化类间距离
- **类内聚合损失** - 最小化类内距离，使同一说话人的嵌入向量在超球面上更加紧凑

## 项目结构

```
ASR/
├── model/
│   ├── __init__.py
│   ├── multi_scale_frontend.py      # 多尺度特征提取前端
│   ├── attention_modules.py         # 多层次动态注意力模块
│   ├── loss_functions.py            # 混合损失函数
│   └── speaker_recognition_model.py # 主模型
├── dataset/
│   ├── __init__.py
│   └── voxceleb_dataset.py          # VoxCeleb数据集加载器
├── train.py                          # 训练脚本
├── main.py                           # 主入口
├── config.json                       # 配置文件
├── prepare_voxceleb_list.py          # 生成VoxCeleb列表文件工具
├── test_model.py                     # 模型测试脚本
├── inference.py                      # 推理脚本
├── README.md                         # 说明文档
├── README_VOXCELEB.md                # VoxCeleb详细使用指南
└── USAGE_VOXCELEB.md                 # VoxCeleb快速使用指南
```

## 安装依赖

```bash
pip install torch torchvision torchaudio
pip install numpy tqdm
```

## 使用方法

### 1. 准备VoxCeleb数据集

#### 方式1：使用目录结构（自动扫描）

将VoxCeleb数据集放在以下结构中：
```
data_root/
    wav/
        id00001/
            video_id/
                *.wav
        id00002/
            ...
```

然后在 `config.json` 中设置：
```json
{
  "data": {
    "data_root": "path/to/voxceleb",
    ...
  }
}
```

#### 方式2：使用列表文件（推荐）

1. 生成列表文件：
```bash
python prepare_voxceleb_list.py --data_root path/to/voxceleb --output all_list.txt --split --train_list train_list.txt --val_list val_list.txt
```

2. 在 `config.json` 中设置：
```json
{
  "data": {
    "data_root": "path/to/voxceleb",
    "train_list": "train_list.txt",
    "val_list": "val_list.txt",
    ...
  }
}
```

列表文件格式（每行）：
```
wav/id00001/video_id/file.wav id00001
wav/id00002/video_id/file.wav id00002
```

### 2. 训练模型

```bash
python main.py
```

或者直接使用训练脚本：

```python
from train import Trainer, create_voxceleb_dataset
import json

# 加载配置
with open('config.json', 'r') as f:
    config_dict = json.load(f)

# 创建VoxCeleb数据加载器
train_loader, val_loader, num_classes = create_voxceleb_dataset(
    data_root=config_dict['data']['data_root'],
    batch_size=config_dict['training']['batch_size'],
    sample_rate=config_dict['data']['sample_rate'],
    segment_length=config_dict['data']['seq_length']
)

# 更新说话人数量
config_dict['model']['num_classes'] = num_classes

# 创建训练器并开始训练
trainer = Trainer(config_dict)
trainer.train(train_loader, val_loader, num_epochs=config_dict['training']['num_epochs'])
```

### 2. 使用模型进行推理

```python
import torch
from model import SpeakerRecognitionModel

# 加载模型
model = SpeakerRecognitionModel(
    in_channels=1,
    frontend_channels=64,
    attention_channels=192,
    embedding_dim=256,
    num_classes=100
)

# 加载权重
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    audio = torch.randn(1, 1, 16000)  # [B, 1, T]
    embedding = model(audio)  # [B, embedding_dim]
    print(f"嵌入向量形状: {embedding.shape}")
```

## 配置说明

主要配置参数在 `config.json` 中：

- **model**: 模型架构参数
  - `embedding_dim`: 嵌入向量维度
  - `num_classes`: 说话人数量
  - `num_heads`: 多头注意力的头数

- **loss**: 损失函数参数
  - `am_margin`: AM-Softmax的角度间隔
  - `lambda_intra`: 类内聚合损失的权重

- **training**: 训练参数
  - `batch_size`: 批次大小
  - `learning_rate`: 学习率
  - `num_epochs`: 训练轮数

## 模型特点

1. **多尺度特征提取** - 同时捕获细粒度和粗粒度特征
2. **层次化注意力机制** - 从局部到全局，从帧级到通道级的多层次注意力
3. **自适应特征融合** - 可学习的融合权重，动态调整不同特征的贡献
4. **混合损失优化** - 同时优化类间分离和类内聚合

## 数据集支持

- **VoxCeleb1/VoxCeleb2**: 完整支持，自动检测说话人数量
- **自定义数据集**: 使用列表文件格式，每行格式为 `音频路径 说话人ID`

## 注意事项

- 建议使用GPU进行训练，模型计算量较大
- VoxCeleb数据集较大，首次加载可能需要一些时间
- 可以根据实际数据集调整超参数，特别是损失函数中的 `lambda_intra` 参数
- 如果内存不足，可以减少 `batch_size` 或 `num_workers`
- 数据增强默认开启，可在 `config.json` 中通过 `data.augmentation` 控制

## 参考文献

- AM-Softmax: Additive Margin Softmax for Face Verification
- Attention Is All You Need (Transformer)
- Squeeze-and-Excitation Networks (Channel Attention)

