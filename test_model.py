"""
测试模型架构是否正确
"""
import torch
from model import SpeakerRecognitionModel

def test_model():
    """测试模型前向传播"""
    print("测试模型架构...")
    
    # 创建模型
    model = SpeakerRecognitionModel(
        in_channels=1,
        frontend_channels=64,
        attention_channels=192,
        embedding_dim=256,
        num_classes=100,
        num_heads=8,
        dropout=0.1
    )
    
    # 创建随机输入
    batch_size = 4
    seq_length = 16000
    x = torch.randn(batch_size, 1, seq_length)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        embedding = model(x)
        print(f"输出嵌入向量形状: {embedding.shape}")
        print(f"嵌入向量范数: {embedding.norm(dim=1)}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print("\n模型测试通过！")


if __name__ == '__main__':
    test_model()





