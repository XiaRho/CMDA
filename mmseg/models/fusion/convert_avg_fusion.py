import torch
import torch.nn as nn
from ..builder import FUSION
from mmcv.runner import BaseModule
from mmseg.models.backbones.resnet import BasicBlock


@FUSION.register_module()
class ConvertAvgFusion(BaseModule):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 out_channels=[64, 128, 320, 512],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.basic_block = nn.ModuleList([
            BasicBlock(inplanes=in_channels[0], planes=out_channels[0]),
            BasicBlock(inplanes=in_channels[0], planes=out_channels[0]),
            BasicBlock(inplanes=in_channels[1], planes=out_channels[1]),
            BasicBlock(inplanes=in_channels[1], planes=out_channels[1]),
            BasicBlock(inplanes=in_channels[2], planes=out_channels[2]),
            BasicBlock(inplanes=in_channels[2], planes=out_channels[2]),
            BasicBlock(inplanes=in_channels[3], planes=out_channels[3]),
            BasicBlock(inplanes=in_channels[3], planes=out_channels[3])])

    def forward(self, image_features, events_features):
        fusion_features = []
        for i in range(len(image_features)):  # [0, 1, 2, 3]
            cache = (self.basic_block[2*i](image_features[i]) + self.basic_block[2*i+1](events_features[i])) / 2
            fusion_features.append(cache)
        return fusion_features


if __name__ == '__main__':
    features = [torch.rand(2, 64, 32, 64).cuda(),
                torch.rand(2, 128, 16, 32).cuda(),
                torch.rand(2, 320, 8, 16).cuda(),
                torch.rand(2, 512, 4, 8).cuda()]
    fusion = ConvertAvgFusion().cuda()
    output = fusion(features, features)
    torch.save(fusion, r'D:\研究生\Python\Events_DAFormer\work_dirs\ConvertAvgFusion.pth')
