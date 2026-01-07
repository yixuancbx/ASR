# 紧急内存修复指南

## 当前问题
即使batch_size=4, seq_length=8000，仍然出现CUDA内存不足错误。

## 可能的原因
1. **GPU内存已被其他进程占用**
2. **模型本身占用内存较大**
3. **内存碎片化**
4. **PyTorch缓存未清理**

## 立即执行的步骤

### 步骤1：检查GPU内存使用
```bash
python check_gpu_memory.py
```

或者在Python中：
```python
import torch
print(f"已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"已保留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

### 步骤2：清理GPU缓存
在训练前运行：
```python
import torch
import gc

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
gc.collect()
```

或者在命令行：
```bash
python -c "import torch; torch.cuda.empty_cache(); print('GPU缓存已清理')"
```

### 步骤3：检查是否有其他进程占用GPU
```bash
# Windows
nvidia-smi

# 如果有其他进程，关闭它们或使用
# set CUDA_VISIBLE_DEVICES=0
```

### 步骤4：进一步减少配置
如果仍然不行，尝试：

```json
{
  "training": {
    "batch_size": 2  // 进一步减少
  },
  "data": {
    "seq_length": 4000,  // 0.25秒
    "num_workers": 0  // 禁用多进程
  }
}
```

### 步骤5：使用环境变量限制内存分配
在训练前设置：

**Windows (PowerShell)**:
```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
python main.py
```

**Windows (CMD)**:
```cmd
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python main.py
```

**Linux/Mac**:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python main.py
```

### 步骤6：使用代码中的内存优化
代码已经添加了：
- 定期清理GPU缓存（每20个batch）
- 清理中间变量
- 垃圾回收

## 最终方案：使用CPU训练
如果GPU内存实在不足，可以强制使用CPU（会很慢）：
```python
# 在main.py或train.py开始处添加
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

## 推荐的最小配置（8GB GPU）

```json
{
  "training": {
    "batch_size": 2
  },
  "data": {
    "seq_length": 4000,
    "num_workers": 0
  }
}
```

## 检查模型大小
运行以下代码查看模型占用的内存：
```python
from model import SpeakerRecognitionModel

model = SpeakerRecognitionModel(
    in_channels=1,
    frontend_channels=64,
    attention_channels=192,
    embedding_dim=256,
    num_classes=1211,  # 你的说话人数量
    num_heads=8,
    dropout=0.1
)

total_params = sum(p.numel() for p in model.parameters())
model_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4字节
print(f"模型参数: {total_params:,}")
print(f"模型大小: {model_size_mb:.2f} MB")
```

如果模型本身很大（>1GB），可能需要：
1. 减少embedding_dim
2. 减少attention_channels
3. 使用模型量化





