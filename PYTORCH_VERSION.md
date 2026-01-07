# PyTorch版本要求

## 推荐版本

基于代码中使用的特性，推荐使用以下版本：

### 最低要求
- **PyTorch >= 1.9.0**
- **torchaudio >= 0.9.0**
- **torchvision >= 0.10.0**

### 推荐版本（稳定且性能好）
- **PyTorch 1.12.0 - 1.13.1** （稳定，兼容性好）
- **PyTorch 2.0.0+** （最新特性，性能更好）

## 版本说明

### 代码中使用的关键特性

1. **DataLoader参数**:
   - `persistent_workers`: PyTorch 1.7.0+
   - `prefetch_factor`: PyTorch 1.7.0+
   - 这两个参数在旧版本中不存在但不会报错（会被忽略）

2. **torchaudio.transforms.Resample**:
   - 在torchaudio 0.8.0+中可用
   - 推荐0.9.0+以获得更好的性能

3. **其他特性**:
   - `non_blocking`参数: PyTorch 1.0.0+
   - 所有其他特性都在PyTorch 1.0.0+中可用

## 安装命令

### CUDA版本（如果有GPU）

**PyTorch 1.12.0 (推荐稳定版)**:
```bash
# CUDA 11.3
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# CUDA 11.6
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

**PyTorch 2.0.0+ (最新版)**:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CPU版本

```bash
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0
```

## 版本兼容性说明

- **PyTorch 1.9.0 - 1.11.0**: 完全兼容，所有特性可用
- **PyTorch 1.12.0 - 1.13.1**: 完全兼容，推荐使用
- **PyTorch 2.0.0+**: 完全兼容，性能更好，但可能需要CUDA 11.8+

## 检查当前版本

```python
import torch
import torchaudio

print(f"PyTorch版本: {torch.__version__}")
print(f"torchaudio版本: {torchaudio.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
```

## 注意事项

1. **CUDA版本匹配**: PyTorch的CUDA版本需要与系统的CUDA驱动版本兼容
2. **torchaudio依赖**: torchaudio需要与PyTorch版本匹配
3. **Windows用户**: 某些版本在Windows上可能有兼容性问题，建议使用PyTorch 1.12.0或2.0.0+





