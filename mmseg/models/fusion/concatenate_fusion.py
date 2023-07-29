import torch
import torch.nn as nn
from ..builder import FUSION
from mmcv.runner import BaseModule


@FUSION.register_module()
class ConcatenateFusion(BaseModule):
    def __init__(self,
                 in_channels=[64 * 2, 128 * 2, 320 * 2, 512 * 2],
                 out_channels=[64, 128, 320, 512],
                 init_cfg=None):
        super().__init__(init_cfg)

        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=in_channels[3], out_channels=out_channels[3], kernel_size=3, padding=1)])

    def forward(self, image_features, events_features):
        fusion_features = []
        for i in range(len(image_features)):
            fusion_features.append(self.conv[i](torch.cat((image_features[i], events_features[i]), dim=1)))
        return fusion_features
