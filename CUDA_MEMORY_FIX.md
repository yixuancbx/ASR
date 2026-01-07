# CUDA内存优化指南

## 问题分析

GPU只有8GB显存，但尝试分配244GB，这通常是由于：
1. **batch_size太大** (32对于音频数据在8GB GPU上过大)
2. **pin_memory和prefetch机制**预分配了过多内存
3. **数据在加载时被pin到GPU内存**

## 立即解决方案

### 方案1：减少batch_size（推荐）

修改 `config.json`:
```json
{
  "training": {
    "batch_size": 8  // 从32减少到8
  }
}
```

### 方案2：禁用pin_memory

如果方案1不够，可以临时禁用pin_memory：
- 在代码中设置 `pin_memory=False`

### 方案3：使用梯度累积

如果batch_size=8太小影响训练效果，可以使用梯度累积来模拟更大的batch size。

## 推荐配置（8GB GPU）

```json
{
  "training": {
    "batch_size": 8,
    "learning_rate": 0.001
  },
  "data": {
    "num_workers": 2,
    "seq_length": 16000
  }
}
```

## 进一步优化

如果仍然内存不足：
1. 减少batch_size到4
2. 减少seq_length到8000（0.5秒）
3. 使用梯度累积（见代码修改）





