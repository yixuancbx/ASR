"""
数据集模块
"""
from .voxceleb_dataset import VoxCelebDataset, VoxCelebDatasetFromList
from .lrs_dataset import LRSDatasetFromList
from .vox2video_dataset import Vox2VideoDatasetFromList

__all__ = ['VoxCelebDataset', 'VoxCelebDatasetFromList', 'LRSDatasetFromList', 'Vox2VideoDatasetFromList']





