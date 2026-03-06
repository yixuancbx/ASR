"""
数据集模块
"""
from .voxceleb_dataset import VoxCelebDataset, VoxCelebDatasetFromList
from .lrs_dataset import LRSDatasetFromList

__all__ = ['VoxCelebDataset', 'VoxCelebDatasetFromList', 'LRSDatasetFromList']





