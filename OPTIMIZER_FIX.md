# 优化器参数说明

## 当前实现
优化器只包含模型的参数：
```python
self.optimizer = optim.Adam(self.model.parameters(), ...)
```

## 损失函数的参数
AMSoftmaxLoss中包含一个可训练的参数：
- `self.weight`: 权重矩阵 [num_classes, embedding_dim]

## 是否需要添加到优化器？
**是的！** 损失函数中的权重矩阵是可训练参数，应该包含在优化器中。

## 建议的修复（如果需要训练权重矩阵）

如果需要训练损失函数的权重矩阵，应该这样修改：

```python
# 将模型和损失函数的参数都添加到优化器
model_params = list(self.model.parameters())
criterion_params = list(self.criterion.parameters())
all_params = model_params + criterion_params

self.optimizer = optim.Adam(all_params, ...)
```

或者更简洁的方式：
```python
self.optimizer = optim.Adam(
    list(self.model.parameters()) + list(self.criterion.parameters()),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)
```

## 当前状态
当前代码中，损失函数的权重矩阵会被创建在GPU上（通过.to(device)），但如果不在优化器中，它不会被更新。

**注意**：对于AM-Softmax，权重矩阵通常需要被训练，所以建议将其添加到优化器中。





