"""
说话人识别模型包
"""
from .speaker_recognition_model import SpeakerRecognitionModel
from .multi_scale_frontend import MultiScaleFeatureExtraction
from .attention_modules import (
    LocalFrameAttention,
    GlobalTemporalAttention,
    ChannelAttention,
    MultiLevelDynamicAttentionFusion
)
from .loss_functions import (
    AMSoftmaxLoss,
    IntraClassAggregationLoss,
    MixedLossFunction
)

__all__ = [
    'SpeakerRecognitionModel',
    'MultiScaleFeatureExtraction',
    'LocalFrameAttention',
    'GlobalTemporalAttention',
    'ChannelAttention',
    'MultiLevelDynamicAttentionFusion',
    'AMSoftmaxLoss',
    'IntraClassAggregationLoss',
    'MixedLossFunction'
]





