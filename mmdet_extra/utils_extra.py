import torch
import torch.nn as nn
from tools.multadds_count import comp_multadds_fw
from tools.utils import count_parameters_in_MB

def convert_sync_batchnorm(module, process_group=None):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.SyncBatchNorm(module.num_features,
            module.eps, module.momentum,
            module.affine,
            module.track_running_stats,
            process_group)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep reuqires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output._specify_ddp_gpu_num(1)
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm(child, process_group))
    del module
    return module_output


def replace_relu6_with_relu(module, process_group=None):
    module_output = module
    if isinstance(module, torch.nn.ReLU6):
        module_output = torch.nn.ReLU()
      

    for name, child in module.named_children():
        module_output.add_module(name, replace_relu6_with_relu(child, process_group))
    del module
    return module_output

def remove_relu_series(module, process_group=None):
    module_output = module
    relu_series = (nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)
    if isinstance(module, relu_series):
        module_output = torch.nn.Sequential()
      

    for name, child in module.named_children():
        module_output.add_module(name, remove_relu_series(child, process_group))
    del module
    return module_output

def get_backbone_madds(backbone, input_size, logger):

    input_data = torch.randn((1,3,)+input_size).cuda()
    backbone_madds, backbone_data = comp_multadds_fw(backbone, input_data)
    backbone_params = count_parameters_in_MB(backbone)


    logger.info("Derived Mult-Adds: [Backbone] %.2fGB", backbone_madds/1e3)
    logger.info("Derived Num Params: [Backbone] %.2fMB", backbone_params)



class AttentionModule(nn.Module):
    def __init__(self, in_channels, groups=1):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
            )
    def forward(self, x):
        return self.attention(x).mul(x)  
        
