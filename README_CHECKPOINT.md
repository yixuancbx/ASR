# 断点续训功能说明

## 功能概述

代码已实现完整的断点续训功能，支持训练中途退出后从上次的进度继续训练。

## 使用方法

### 自动恢复（推荐）

1. **正常训练**：
```bash
python main.py
```

2. **如果中途退出**（Ctrl+C、断电等）：
再次运行 `python main.py`，程序会：
- 自动检测是否存在 `checkpoints/latest_checkpoint.pth`
- 询问是否继续训练
- 如果选择继续，从上次中断的epoch继续训练

### 手动指定恢复点

在 `config.json` 中设置：
```json
{
  "training": {
    "resume_from": "checkpoints/best_model.pth"
  }
}
```

### 从头开始训练

如果要从头开始，在询问时输入 `n`，或删除 `checkpoints/` 目录。

## 保存机制

- **定期保存**：每 `save_interval` 个epoch保存一次（默认10个epoch）
- **最佳模型**：验证损失最低时自动保存
- **最后检查点**：训练结束时保存

## 检查点文件

- `checkpoints/latest_checkpoint.pth` - 最新的检查点
- `checkpoints/best_model.pth` - 最佳模型

## 配置参数

在 `config.json` 中：
- `checkpoint_dir`: 检查点目录（默认: "checkpoints"）
- `save_interval`: 保存间隔（默认: 10）
- `resume_from`: 指定恢复点（默认: null，自动检测）



