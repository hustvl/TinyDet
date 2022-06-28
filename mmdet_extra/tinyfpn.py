import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from mmdet.models import NECKS
from mmdet.models.utils import ConvModule

from mmdet_extra.utils_extra import AttentionModule
import torch

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

@NECKS.register_module
class TinyFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 
                 fpn_conv_groups=1,
                 with_shuffle=False,
                 with_attention=False,
                 attention_conv_groups=1,
                 lateral_conv_with_bn_relu=False,
                 lateral_conv_without_bn_with_relu=False,
                 fpn_conv_with_bn_relu=False,
                 fpn_conv_without_bn_with_relu=False):
        super(TinyFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        self.fpn_conv_groups = fpn_conv_groups####
        self.with_shuffle = with_shuffle####
        self.with_attention = with_attention####
        self.attention_conv_groups = attention_conv_groups####
        self.lateral_conv_with_bn_relu = lateral_conv_with_bn_relu####
        self.lateral_conv_without_bn_with_relu = lateral_conv_without_bn_with_relu####
        self.fpn_conv_with_bn_relu = fpn_conv_with_bn_relu####
        self.fpn_conv_without_bn_with_relu = fpn_conv_without_bn_with_relu####

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.attention_convs = nn.ModuleList()####
        self.fpn_convs = nn.ModuleList()

 
        for i in range(self.start_level, self.backbone_end_level):
            if self.lateral_conv_with_bn_relu:
                l_conv = nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=False)   
                )
            
            elif self.lateral_conv_without_bn_with_relu:
                l_conv = nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels, 1, bias=False),
                    nn.ReLU(inplace=False)   
                )

            else:
                l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    activation=self.activation,
                    inplace=False)

            if self.with_attention:
                attention_conv = AttentionModule(
                    in_channels=out_channels,
                    groups=self.attention_conv_groups,
                    )
            else:
                attention_conv = nn.Sequential()



            if self.fpn_conv_with_bn_relu:
                raise Exception
            elif self.fpn_conv_without_bn_with_relu:
                raise Exception
            else:
                fpn_conv = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, stride=1,
                        padding=1, dilation=1, groups=out_channels, bias=True),
                    nn.Conv2d(out_channels, out_channels, 1, stride=1,
                        padding=0, dilation=1, groups=self.fpn_conv_groups[i], bias=True),
                    )              





            self.lateral_convs.append(l_conv)
            self.attention_convs.append(attention_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
 
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')


        attention_outs = [
            attension_conv(laterals[i + self.start_level])
            for i, attension_conv in enumerate(self.attention_convs)
            ]



        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](attention_outs[i]) for i in range(used_backbone_levels)
        ]






        if self.with_shuffle == True:
            outs = [channel_shuffle(out, max(self.fpn_conv_groups)) for out in outs]
            



        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
