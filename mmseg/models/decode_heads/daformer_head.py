import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead, BaseDecodeHeadFusion
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule


class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()
        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


@HEADS.register_module()
class DAFormerHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(DAFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)

    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        x = self.cls_seg(x)

        return x


@HEADS.register_module()
class DAFormerHeadFusion(BaseDecodeHeadFusion):

    def __init__(self, **kwargs):
        super(DAFormerHeadFusion, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        # image decoder
        self.embed_layers_image = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels, embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers_image[str(i)] = build_layer(in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers_image[str(i)] = build_layer(in_channels, embed_dim, **embed_cfg)
        self.embed_layers_image = nn.ModuleDict(self.embed_layers_image)
        self.fuse_layer_image = build_layer(sum(embed_dims), self.channels, **fusion_cfg)

        # events decoder
        self.embed_layers_events = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels, embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers_events[str(i)] = build_layer(in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers_events[str(i)] = build_layer(in_channels, embed_dim, **embed_cfg)
        self.embed_layers_events = nn.ModuleDict(self.embed_layers_events)
        self.fuse_layer_events = build_layer(sum(embed_dims), self.channels, **fusion_cfg)

        # fusion decoder
        self.embed_layers_fusion = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels, embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers_fusion[str(i)] = build_layer(in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers_fusion[str(i)] = build_layer(in_channels, embed_dim, **embed_cfg)
        self.embed_layers_fusion = nn.ModuleDict(self.embed_layers_fusion)
        self.fuse_layer_fusion = build_layer(sum(embed_dims), self.channels, **fusion_cfg)

        if self.half_share_decoder:
            self.fuse_layer_events = self.fuse_layer_image
            self.fuse_layer_fusion = self.fuse_layer_image
        elif self.share_decoder:
            self.embed_layers_events = self.embed_layers_image
            self.fuse_layer_events = self.fuse_layer_image
            self.embed_layers_fusion = self.embed_layers_image
            self.fuse_layer_fusion = self.fuse_layer_image

        # projection head
        # self.projection_head = ProjectionHead(self.channels)

    def forward_image(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            _c[i] = self.embed_layers_image[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous().reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if _c[i].size()[2:] != os_size:
                _c[i] = resize(_c[i], size=os_size, mode='bilinear', align_corners=self.align_corners)
        x = self.fuse_layer_image(torch.cat(list(_c.values()), dim=1))
        return x

    def forward_events(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            _c[i] = self.embed_layers_events[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous().reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if _c[i].size()[2:] != os_size:
                _c[i] = resize(_c[i], size=os_size, mode='bilinear', align_corners=self.align_corners)
        x = self.fuse_layer_events(torch.cat(list(_c.values()), dim=1))
        return x

    def forward_fusion(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            _c[i] = self.embed_layers_fusion[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous().reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if _c[i].size()[2:] != os_size:
                _c[i] = resize(_c[i], size=os_size, mode='bilinear', align_corners=self.align_corners)
        x = self.fuse_layer_fusion(torch.cat(list(_c.values()), dim=1))
        return x

    def forward(self, inputs, cfg=None):
        events_output, fusion_output, img_self_res_output = None, None, None
        image_feat = self.forward_image(inputs['f_image'])
        image_output = self.cls_seg(image_feat)
        if 'f_events' in inputs.keys() and inputs['f_events'] is not None:
            events_feat = self.forward_events(inputs['f_events'])
            events_output = self.cls_seg_events(events_feat)

        if 'f_fusion' in inputs.keys() and inputs['f_fusion'] is not None:
            fusion_feat = self.forward_fusion(inputs['f_fusion'])
            fusion_output = self.cls_seg_fusion(fusion_feat)

        if 'f_img_self_res' in inputs.keys() and inputs['f_img_self_res'] is not None:
            img_self_res_feat = self.forward_events(inputs['f_img_self_res'])
            img_self_res_output = self.cls_seg_events(img_self_res_feat)

        return {'image_output': image_output, 'events_output': events_output, 'fusion_output': fusion_output,
                'img_self_res_output': img_self_res_output}

