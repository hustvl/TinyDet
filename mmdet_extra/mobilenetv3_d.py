'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm


import logging

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.models import BACKBONES####

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
   
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.in_size = in_size
        self.out_size = out_size
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))

        if self.stride == 2:
            out = torch.nn.functional.avg_pool2d(out, 2, 1, 0, False, True)####

        # out = out
        out = self.nolinear2(self.bn2(self.conv2(out)))
        if self.se != None:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
 
        out = out + self.shortcut(x) if (self.in_size == self.out_size and self.stride == 1) else out
        return out


class BlockE1(nn.Module):
    '''depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(BlockE1, self).__init__()
        self.stride = stride
        self.in_size = in_size
        self.out_size = out_size
        self.se = semodule

        self.conv2 = nn.Conv2d(in_size, in_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(in_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.nolinear2(self.bn2(self.conv2(x)))
        if self.se != None:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
 
        out = out + self.shortcut(x) if self.in_size == self.out_size else out
        return out

@BACKBONES.register_module
class MobileNetV3D(nn.Module):
    def __init__(
        self, 
        with_fc=False,
        out_indices=[12, 15, 16],
        norm_eval=False,
        norm_eps=1e-5,
        num_classes=1000,
        
        ####for lightweight fpn
        add_extra_stages=False):
        super(MobileNetV3D, self).__init__()


        _conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),####224
            nn.BatchNorm2d(16),
            hswish())

        _block1 = nn.Sequential()
        self.blocks = nn.ModuleList((
            _conv1,
            BlockE1(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),####112
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),####112
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),####56

            ####added
            # Block(5, 24, 72, 24, nn.ReLU(inplace=True), None, 1),####56
            # Block(5, 24, 72, 24, nn.ReLU(inplace=True), None, 1),####56
            # Block(5, 24, 72, 24, nn.ReLU(inplace=True), None, 1),####56
            # Block(5, 24, 72, 24, nn.ReLU(inplace=True), None, 1),####56


            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(72), 2),####56
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(120), 1),####28
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(120), 1),####28
            Block(3, 40, 240, 80, hswish(), None, 2),####28
            Block(3, 80, 240, 80, hswish(), None, 1),####14
            Block(5, 80, 240, 80, hswish(), None, 1),####14
            Block(5, 80, 240, 80, hswish(), None, 1),####14
            Block(3, 80, 240, 112, hswish(), SeModule(240), 1),####14
            Block(3, 112, 336, 112, hswish(), SeModule(336), 1),####14
            Block(5, 112, 336, 160, hswish(), SeModule(336), 2),####14
            Block(5, 160, 480, 160, hswish(), SeModule(480), 1),####7
            Block(5, 160, 480, 160, hswish(), SeModule(480), 1)####7
        ))

        if with_fc:
            self.conv2 = nn.Sequential(
                nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(960),
                hswish())

            self.linear3 = nn.Sequential(
                nn.Linear(960, 1280),
                # self.bn3 = nn.BatchNorm1d(1280)
                hswish())

            self.linear4 = nn.Linear(1280, num_classes)
        else:

            self.add_extra_stages = add_extra_stages
            if self.add_extra_stages == True:
                self.blocks.append(Block(5, 160, 960, 160, hswish(), SeModule(960), 2))####fifth stage for fpn

            else:
                self.blocks.append(nn.AdaptiveAvgPool2d(1))

        # self.init_params()
        




        self.with_fc = with_fc
        self.out_indices = out_indices
        self.norm_eval = norm_eval
        self.init_weights()
        self._freeze_stages()
        self._set_bn_param(0.1, norm_eps)



    def forward(self, x):


        if self.with_fc:
            for block_ind, block in enumerate(self.blocks):
                x = block(x)
            x = self.conv2(x)
            x = F.avg_pool2d(x, 7)
            x = x.view(x.size(0), -1)
            x = self.linear3(x)
            x = self.linear4(x)
            return x
        else:
            outs = []
            for block_ind, block in enumerate(self.blocks):
                x = block(x)
                if block_ind in self.out_indices:
                    outs.append(x)
                    if len(outs) == len(self.out_indices):####no neeed to forward all blocks, avoid generating redudant flops when calculating flops 

                        break
            return tuple(outs)



    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            model_dict = self.state_dict()

            load_checkpoint(self, pretrained, strict=False, logger=logger)####False

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        super(MobileNetV3D, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


    def _freeze_stages(self):
        pass

    def _set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return






# test()


if __name__ == '__main__':
    # net = MobileNetV3D(with_fc=True)
    # net.eval()
    # print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    # input_size=(1, 3, 224, 224)
    # input = torch.randn(1, 3, 224, 224)
    # from thop import profile
    # flops, params = profile(net, inputs=(input,))

    # print('Total params: %.2fM' % (params/1000000.0))
    # print('Total flops: %.2fM' % (flops/1000000.0))







    print('-----------------')
    input_size=(1, 3, 960, 960)

    x = torch.randn(input_size).cuda()
    net = MobileNetV3D(with_fc=True).cuda()

    out = net(x)


    from tools.multadds_count import comp_multadds_fw
    mult_adds, output_data = comp_multadds_fw(net, x)

    print('mult_adds:{:.2f}, output_data:{}'.format(mult_adds, output_data.shape))

    
    
    print('-----------------')
    input_size=(1, 3, 960, 960)
    x = torch.randn(input_size).cuda()
    net = MobileNetV3D().cuda()
    out = net(x)








