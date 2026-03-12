"""
混合损失函数：AM-Softmax + 类内聚合损失
"""
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast as cuda_autocast


def _full_precision_context(tensor):
    """在 CUDA AMP 下强制损失头使用 FP32，避免 ArcFace 数值不稳定"""
    if tensor.is_cuda:
        return cuda_autocast(enabled=False)
    return nullcontext()


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
        with _full_precision_context(embeddings):
            labels = labels.long()
            embeddings = F.normalize(embeddings.float(), p=2, dim=1)
            weight = F.normalize(self.weight.float(), p=2, dim=1)

            cosine = F.linear(embeddings, weight)
            target_cosine = cosine.gather(1, labels.unsqueeze(1))
            target_cosine_margin = target_cosine - self.margin

            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

            output = (one_hot * target_cosine_margin + (1.0 - one_hot) * cosine) * self.scale
            loss = F.cross_entropy(output, labels)

        return loss


class AAMSoftmaxLoss(nn.Module):
    """
    Additive Angular Margin Softmax (ArcFace)
    相比 AM-Softmax 在角度空间上施加 margin，通常收敛更稳定、性能更好
    """
    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30.0, eps=1e-4, easy_margin=False):
        super(AAMSoftmaxLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.eps = eps
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        with _full_precision_context(embeddings):
            labels = labels.long()
            embeddings = F.normalize(embeddings.float(), p=2, dim=1)
            weight = F.normalize(self.weight.float(), p=2, dim=1)

            cosine = F.linear(embeddings, weight)
            cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)

            # 使用标准 ArcFace 的 trig identity，避免 FP16 下 acos/cos 数值不稳
            sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=self.eps))
            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

            output = self.scale * (one_hot * phi + (1.0 - one_hot) * cosine)
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
        with _full_precision_context(embeddings):
            embeddings = F.normalize(embeddings.float(), p=2, dim=1)

            max_samples_per_speaker = 10
            loss = 0.0
            count = 0
            unique_labels = torch.unique(labels)

            for label in unique_labels:
                mask = (labels == label)
                if mask.sum() < 2:
                    continue

                speaker_embeddings = embeddings[mask]
                n = speaker_embeddings.size(0)

                if n > max_samples_per_speaker:
                    indices = torch.randperm(n, device=embeddings.device)[:max_samples_per_speaker]
                    speaker_embeddings = speaker_embeddings[indices]
                    n = max_samples_per_speaker

                cosine_sim = torch.matmul(speaker_embeddings, speaker_embeddings.t())
                cosine_dist = 1 - cosine_sim
                triu_mask = torch.triu(torch.ones(n, n, device=embeddings.device), diagonal=1).bool()
                pairwise_dist = cosine_dist[triu_mask]

                if pairwise_dist.numel() > 0:
                    mean_intra_dist = pairwise_dist.mean()
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
        # 使用 AAM-Softmax 作为主分类损失
        self.am_loss = AAMSoftmaxLoss(embedding_dim, num_classes, margin=am_margin, scale=am_scale)
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




