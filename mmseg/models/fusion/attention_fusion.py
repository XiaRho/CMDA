import torch
import torch.nn as nn
from ..builder import FUSION
from mmcv.runner import BaseModule
from mmseg.models.backbones.mix_transformer import Block, Mlp
from functools import partial


@FUSION.register_module()
class AttentionFusion(BaseModule):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_heads=1,
                 mlp_ratios=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.05,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 sr_ratios=[8, 4, 2, 1],
                 act_layer=nn.GELU,
                 init_cfg=None):
        super().__init__(init_cfg)
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.basic_block = nn.ModuleList([
            Block(
                dim=in_channels[i] * 2,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                # drop_path=dpr[cur + i],
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[i]) for i in range(4)
        ])
        self.linear_block = nn.ModuleList([
            Mlp(
                in_features=in_channels[i] * 2,
                hidden_features=in_channels[i],
                act_layer=act_layer,
                drop=drop_rate,
                out_features=in_channels[i]) for i in range(4)
        ])

    def forward(self, image_features, events_features):
        fusion_features = []
        for i in range(len(image_features)):  # [0, 1, 2, 3]
            x = torch.cat((image_features[i], events_features[i]), dim=1)
            B, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.basic_block[i](x, H, W)
            x = self.linear_block[i](x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            fusion_features.append(x)
        return fusion_features


if __name__ == '__main__':
    features = [torch.rand(2, 64, 32, 64).cuda(),
                torch.rand(2, 128, 16, 32).cuda(),
                torch.rand(2, 320, 8, 16).cuda(),
                torch.rand(2, 512, 4, 8).cuda()]
    fusion = AttentionFusion().cuda()
    torch.save(fusion, r'D:\研究生\Python\Events_DAFormer\work_dirs\AttentionFusion.pth')
    output = fusion(features, features)
