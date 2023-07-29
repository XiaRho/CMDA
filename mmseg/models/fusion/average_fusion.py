from ..builder import FUSION
from mmcv.runner import BaseModule


@FUSION.register_module()
class AverageFusion(BaseModule):
    def __init__(self,
                 init_cfg=None):
        super().__init__(init_cfg)

    def forward(self, image_features, events_features):
        fusion_features = []
        for i in range(len(image_features)):
            fusion_features.append((image_features[i] + events_features[i]) / 2)
        return fusion_features
