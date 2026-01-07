# 检查点（Checkpoint）使用指南

## 功能说明

代码已实现完整的断点续训功能，支持：
1. **自动保存检查点**：定期保存训练状态（模型、优化器、学习率调度器、训练历史等）
2. **自动恢复训练**：程序启动时自动检测并询问是否从上次中断的地方继续
3. **手动指定恢复点**：可以在配置文件中指定要恢复的checkpoint

## 检查点文件

训练过程中会生成以下文件：

- `checkpoints/latest_checkpoint.pth` - 最新的检查点（每个save_interval epoch保存一次）
- `checkpoints/best_model.pth` - 最佳模型（验证损失最低时保存）

## 使用方法

### 方法1：自动恢复（推荐）

1. **第一次训练**：
```bash
python main.py
```
程序会从头开始训练。

2. **如果中途退出**（Ctrl+C或其他原因）：
再次运行：
```bash
python main.py
```
程序会自动检测到 `checkpoints/latest_checkpoint.pth`，并询问：
```
发现之前的训练记录: checkpoints/latest_checkpoint.pth
是否从中断的地方继续训练？(y/n，默认y):
```
输入 `y` 或直接回车即可继续训练。

### 方法2：手动指定恢复点

在 `config.json` 中设置：
```json
{
  "training": {
    "resume_from": "checkpoints/latest_checkpoint.pth",
    ...
  }
}
```

或者指定其他checkpoint文件：
```json
{
  "training": {
    "resume_from": "checkpoints/best_model.pth",
    ...
  }
}
```

### 方法3：从头开始训练

如果要从头开始训练（忽略之前的checkpoint），在询问时输入 `n`：
```
是否从中断的地方继续训练？(y/n，默认y): n
```

或者删除checkpoint目录：
```bash
rm -rf checkpoints/  # Linux/Mac
rmdir /s checkpoints  # Windows
```

## 配置参数

在 `config.json` 的 `training` 部分：

- `checkpoint_dir`: 检查点保存目录（默认: "checkpoints"）
- `save_interval`: 每隔多少个epoch保存一次检查点（默认: 10）
- `resume_from`: 指定要恢复的checkpoint路径（默认: null，自动检测）

## 检查点内容

每个checkpoint包含：
- `epoch`: 训练轮数
- `model_state_dict`: 模型参数
- `optimizer_state_dict`: 优化器状态
- `scheduler_state_dict`: 学习率调度器状态
- `train_history`: 训练历史（损失值等）
- `config`: 训练配置
- `best_val_loss`: 最佳验证损失

## 注意事项

1. **配置文件一致性**：恢复训练时，建议使用相同的配置文件，特别是模型架构参数
2. **数据集一致性**：确保使用相同的数据集和划分方式
3. **说话人数量**：如果数据集发生变化，需要更新 `num_classes`
4. **保存频率**：`save_interval` 越小，恢复的粒度越细，但会增加磁盘I/O

## 示例场景

### 场景1：训练中途断电

1. 训练到第50轮时断电
2. 重新启动后运行 `python main.py`
3. 程序检测到checkpoint，询问是否继续
4. 选择继续，从第51轮开始训练

### 场景2：想从最佳模型继续训练

1. 训练完成后，想从最佳模型继续训练（可能是微调）
2. 在 `config.json` 中设置：
```json
{
  "training": {
    "resume_from": "checkpoints/best_model.pth",
    "num_epochs": 150  // 继续训练到150轮
  }
}
```

### 场景3：定期检查训练进度

设置较小的 `save_interval`：
```json
{
  "training": {
    "save_interval": 5  // 每5个epoch保存一次
  }
}
```

这样可以更频繁地保存进度，避免丢失太多训练进度。

## 故障排除

### 问题1：无法加载checkpoint

**错误信息**：`KeyError: 'xxx'` 或 `RuntimeError: ...`

**原因**：模型架构或配置发生变化

**解决**：
- 确保使用相同的模型架构配置
- 如果模型结构改变，需要从头开始训练

### 问题2：恢复后损失值异常

**原因**：优化器状态可能与当前配置不匹配

**解决**：
- 检查学习率等超参数是否一致
- 如果调整了超参数，建议从头开始训练

### 问题3：想重新开始训练

**解决**：
- 删除 `checkpoints/` 目录
- 或在询问时选择 `n`



