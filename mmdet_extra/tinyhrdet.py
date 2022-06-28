from mmdet.models import DETECTORS####
from mmdet.models import TwoStageDetector####

import torch
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models import builder####
from mmdet.models import BaseDetector####
from mmdet.models.detectors.test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin####


from mmdet_extra.utils_extra import convert_sync_batchnorm

@DETECTORS.register_module
class TinyHRDet(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 norm_cfg = dict(type='BN', requires_grad='not used')):
                
        super(TinyHRDet, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

 
        if norm_cfg['type'] == 'BN':
            pass
        elif norm_cfg['type'] == 'SyncBN':
            model = convert_sync_batchnorm(self)



    def extract_feat(self, img):


        x = self.backbone(img)

        if self.with_neck:
            x = self.neck(x)
            return x
        else:
            assert Exception('should be with neck')



    def forward_dummy(self, img):
        
        outs = ()
        # backbone
        backbone_outs = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(backbone_outs)

            feature_for_psroialign = backbone_outs
            outs = outs + (rpn_outs, )
        proposals = torch.randn(200, 4).cuda()
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
 
            bbox_feats = self.bbox_roi_extractor(
                feature_for_psroialign[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)

        return outs


    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):


        backbone_outs = self.extract_feat(img)

        losses = dict()


        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(backbone_outs)

  
            feature_for_psroialign = backbone_outs####

            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)

        
 

            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in feature_for_psroialign])####
                sampling_results.append(sampling_result)

  
        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
      
            bbox_feats = self.bbox_roi_extractor(
                feature_for_psroialign[:self.bbox_roi_extractor.num_inputs], rois)
      

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            raise Exception('without mask')

        return losses


    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."



        backbone_outs = self.extract_feat(img)
    


        if proposals is None:
            rpn_outs = self.rpn_head(backbone_outs)
            feature_for_psroialign = backbone_outs
            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals


  


        det_bboxes, det_labels = self.simple_test_bboxes(
            feature_for_psroialign, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)


        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)


        if not self.with_mask:
            return bbox_results
        else:
            raise Exception('without mask')
    





    def aug_test(self, imgs, img_metas, **kwargs):
        pass