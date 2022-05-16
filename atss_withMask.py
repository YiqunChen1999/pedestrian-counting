# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
# from .single_stage import SingleStageDetector
from .single_stage_instance_seg import SingleStageInstanceSegmentor
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class ATSS_withMask(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ATSS_withMask, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
