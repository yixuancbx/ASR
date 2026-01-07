# 内存问题修复说明

## 问题描述

遇到 `RuntimeError: DefaultCPUAllocator: not enough memory` 错误，尝试分配了约244GB内存。

## 原因分析

1. **重采样器重复创建**：每次加载音频时都创建新的Resampler对象，导致内存泄漏
2. **pin_memory在CPU模式下使用**：在CPU模式下使用pin_memory会导致额外内存开销
3. **num_workers过多**：多个worker进程同时预加载数据导致内存占用过高

## 已修复的问题

### 1. 重采样器缓存
- 将Resampler作为实例变量缓存
- 只在采样率改变时创建新的resampler
- 避免每次`__getitem__`调用时创建新对象

### 2. pin_memory优化
- 只在GPU可用时启用pin_memory
- CPU模式下自动禁用

### 3. DataLoader优化
- 添加`persistent_workers=False`：避免worker进程保持数据在内存中
- 设置`prefetch_factor=2`：减少预取数据量

## 如果仍然遇到内存问题

### 方法1：减少num_workers

在`config.json`中设置：
```json
{
  "data": {
    "num_workers": 0  // 或1、2
  }
}
```

### 方法2：减少batch_size

在`config.json`中设置：
```json
{
  "training": {
    "batch_size": 16  // 或8、4
  }
}
```

### 方法3：减少seq_length（音频长度）

在`config.json`中设置：
```json
{
  "data": {
    "seq_length": 8000  // 0.5秒，而不是1秒
  }
}
```

### 方法4：使用更小的数据集子集

可以修改列表文件，只使用部分数据进行训练。

## 推荐配置（内存受限时）

```json
{
  "training": {
    "batch_size": 16
  },
  "data": {
    "num_workers": 2,
    "seq_length": 16000
  }
}
```

## 检查内存使用

在Python中检查内存使用：
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"内存使用: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
```





