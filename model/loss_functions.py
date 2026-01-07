"""
混合损失函数：AM-Softmax + 类内聚合损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmaxLoss(nn.Module):
    """
    AM-Softmax损失：引入角度间隔，强制类间分离
    """
    def __init__(self, embedding_dim, num_classes, margin=0.3, scale=30.0):
        super(AMSoftmaxLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # 权重矩阵 W: [num_classes, embedding_dim]
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [B, embedding_dim] 说话人嵌入向量
            labels: [B] 说话人标签
        Returns:
            loss: 标量损失值
        """
        # 归一化嵌入向量和权重
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # 计算余弦相似度
        cosine = F.linear(embeddings, weight)  # [B, num_classes]
        
        # 添加角度间隔
        target_cosine = cosine.gather(1, labels.unsqueeze(1))  # [B, 1]
        target_cosine_margin = target_cosine - self.margin
        
        # 构建one-hot标签
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        
        # 应用角度间隔
        output = (one_hot * target_cosine_margin + (1.0 - one_hot) * cosine) * self.scale
        
        # 计算交叉熵损失
        loss = F.cross_entropy(output, labels)
        
        return loss


class IntraClassAggregationLoss(nn.Module):
    """
    类内聚合损失：最小化类内距离，使同一说话人的嵌入向量在超球面上更加紧凑
    """
    def __init__(self, margin=0.5):
        super(IntraClassAggregationLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [B, embedding_dim] 归一化的说话人嵌入向量
            labels: [B] 说话人标签
        Returns:
            loss: 标量损失值
        """
        # 确保嵌入向量已归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 优化：限制每个说话人的样本数量，避免计算量过大
        max_samples_per_speaker = 10  # 每个说话人最多使用10个样本计算类内距离
        
        batch_size = embeddings.size(0)
        loss = 0.0
        count = 0
        
        # 按说话人分组
        unique_labels = torch.unique(labels)
        
        for label in unique_labels:
            # 找到属于当前说话人的所有样本
            mask = (labels == label)
            if mask.sum() < 2:  # 至少需要2个样本才能计算类内距离
                continue
            
            speaker_embeddings = embeddings[mask]  # [N, embedding_dim]
            n = speaker_embeddings.size(0)
            
            # 如果样本太多，随机采样
            if n > max_samples_per_speaker:
                indices = torch.randperm(n, device=embeddings.device)[:max_samples_per_speaker]
                speaker_embeddings = speaker_embeddings[indices]
                n = max_samples_per_speaker
            
            # 计算所有样本对之间的余弦距离
            # 使用矩阵运算提高效率
            cosine_sim = torch.matmul(speaker_embeddings, speaker_embeddings.t())  # [N, N]
            cosine_dist = 1 - cosine_sim  # 余弦距离
            
            # 只取上三角矩阵（避免重复计算和自身距离）
            triu_mask = torch.triu(torch.ones(n, n, device=embeddings.device), diagonal=1).bool()
            pairwise_dist = cosine_dist[triu_mask]  # [N*(N-1)/2]
            
            # 计算平均类内距离
            if pairwise_dist.numel() > 0:
                mean_intra_dist = pairwise_dist.mean()
                # 鼓励类内距离小于margin
                loss += F.relu(mean_intra_dist - self.margin)
                count += 1
        
        if count > 0:
            loss = loss / count
        else:
            loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return loss


class MixedLossFunction(nn.Module):
    """
    混合损失函数：AM-Softmax + 类内聚合损失
    """
    def __init__(self, embedding_dim, num_classes, 
                 am_margin=0.3, am_scale=30.0, 
                 intra_margin=0.5, 
                 lambda_intra=0.1):
        super(MixedLossFunction, self).__init__()
        self.am_loss = AMSoftmaxLoss(embedding_dim, num_classes, am_margin, am_scale)
        self.intra_loss = IntraClassAggregationLoss(intra_margin)
        self.lambda_intra = lambda_intra
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [B, embedding_dim] 说话人嵌入向量
            labels: [B] 说话人标签
        Returns:
            total_loss: 总损失
            loss_dict: 包含各项损失的字典
        """
        # AM-Softmax损失（最大化类间距离）
        am_loss = self.am_loss(embeddings, labels)
        
        # 类内聚合损失（最小化类内距离）
        intra_loss = self.intra_loss(embeddings, labels)
        
        # 总损失
        total_loss = am_loss + self.lambda_intra * intra_loss
        
        loss_dict = {
            'am_loss': am_loss.item(),
            'intra_loss': intra_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict




