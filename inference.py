"""
推理脚本：使用训练好的模型进行说话人识别
"""
import torch
import torch.nn.functional as F
from model import SpeakerRecognitionModel
import numpy as np


class SpeakerIdentifier:
    """说话人识别器"""
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: 模型检查点路径
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # 创建模型
        self.model = SpeakerRecognitionModel(
            in_channels=config['in_channels'],
            frontend_channels=config['frontend_channels'],
            attention_channels=config['attention_channels'],
            embedding_dim=config['embedding_dim'],
            num_classes=config['num_classes'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"模型已加载: {checkpoint_path}")
        print(f"使用设备: {self.device}")
    
    def extract_embedding(self, audio):
        """
        提取说话人嵌入向量
        
        Args:
            audio: numpy array 或 torch.Tensor, shape [T] 或 [1, T] 或 [B, 1, T]
        Returns:
            embedding: torch.Tensor, shape [embedding_dim] 或 [B, embedding_dim]
        """
        # 转换为torch.Tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # 确保是3D tensor [B, 1, T]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        elif audio.dim() == 2:
            if audio.shape[0] == 1:
                audio = audio.unsqueeze(0)  # [1, 1, T]
            else:
                audio = audio.unsqueeze(1)  # [B, 1, T]
        
        audio = audio.to(self.device)
        
        with torch.no_grad():
            embedding = self.model(audio)
        
        # 如果是单样本，去掉batch维度
        if embedding.shape[0] == 1 and audio.shape[0] == 1:
            embedding = embedding.squeeze(0)
        
        return embedding
    
    def compute_similarity(self, embedding1, embedding2):
        """
        计算两个嵌入向量之间的余弦相似度
        
        Args:
            embedding1: torch.Tensor, shape [embedding_dim] 或 [B, embedding_dim]
            embedding2: torch.Tensor, shape [embedding_dim] 或 [B, embedding_dim]
        Returns:
            similarity: float 或 torch.Tensor, 余弦相似度值
        """
        # 确保是2D tensor
        if embedding1.dim() == 1:
            embedding1 = embedding1.unsqueeze(0)
        if embedding2.dim() == 1:
            embedding2 = embedding2.unsqueeze(0)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        
        # 如果是单样本，返回标量
        if similarity.shape[0] == 1:
            return similarity.item()
        
        return similarity
    
    def identify_speaker(self, audio, reference_embeddings, threshold=0.7):
        """
        识别说话人
        
        Args:
            audio: 输入音频
            reference_embeddings: dict, {speaker_id: embedding_tensor}
            threshold: 相似度阈值
        Returns:
            speaker_id: 识别出的说话人ID，如果相似度低于阈值则返回None
            similarity: 最大相似度值
        """
        # 提取输入音频的嵌入向量
        query_embedding = self.extract_embedding(audio)
        
        # 计算与所有参考说话人的相似度
        best_similarity = -1.0
        best_speaker = None
        
        for speaker_id, ref_embedding in reference_embeddings.items():
            similarity = self.compute_similarity(query_embedding, ref_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_id
        
        # 判断是否超过阈值
        if best_similarity >= threshold:
            return best_speaker, best_similarity
        else:
            return None, best_similarity


def example_usage():
    """使用示例"""
    print("说话人识别推理示例")
    print("=" * 60)
    
    # 注意：这里需要先训练模型或使用已有的检查点
    checkpoint_path = 'checkpoints/best_model.pth'
    
    try:
        # 创建识别器
        identifier = SpeakerIdentifier(checkpoint_path)
        
        # 示例：创建一些虚拟音频数据
        print("\n示例：提取嵌入向量")
        audio1 = torch.randn(16000)  # 1秒音频，16kHz采样率
        embedding1 = identifier.extract_embedding(audio1)
        print(f"音频1嵌入向量形状: {embedding1.shape}")
        
        audio2 = torch.randn(16000)
        embedding2 = identifier.extract_embedding(audio2)
        print(f"音频2嵌入向量形状: {embedding2.shape}")
        
        # 计算相似度
        similarity = identifier.compute_similarity(embedding1, embedding2)
        print(f"两个音频的相似度: {similarity:.4f}")
        
        # 说话人识别示例
        print("\n示例：说话人识别")
        reference_embeddings = {
            'speaker_001': embedding1,
            'speaker_002': embedding2
        }
        
        # 使用与speaker_001相似的音频
        test_audio = audio1 + 0.1 * torch.randn(16000)  # 添加少量噪声
        speaker_id, sim = identifier.identify_speaker(test_audio, reference_embeddings)
        print(f"识别结果: {speaker_id}, 相似度: {sim:.4f}")
        
    except FileNotFoundError:
        print(f"错误：未找到模型检查点文件 {checkpoint_path}")
        print("请先训练模型或指定正确的检查点路径")


if __name__ == '__main__':
    example_usage()





