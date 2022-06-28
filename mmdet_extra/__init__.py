from .shared_fc_bbox_head_modified import SharedFCBBoxHeadModified
from .single_level_modified import SingleRoIExtractorModified
from .flops_counter_modified import get_model_complexity_info

from .ps_roi_align_ori.ps_roi_align import PSROIAlign_ori
from mmdet import ops
ops.PSROIAlign_ori = PSROIAlign_ori


from .utils_extra import convert_sync_batchnorm, replace_relu6_with_relu, \
    remove_relu_series, get_backbone_madds, AttentionModule


from .tinyhrdet import TinyHRDet
from .mobilenetv3_bc import MobileNetV3BC
from .mobilenetv3_d import MobileNetV3D
from .tinyfpn import TinyFPN
from .tinyrpn import TinyRPN


