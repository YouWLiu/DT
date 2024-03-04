# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

# 导入所需的模块和类
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean

# 导入其他模块中的函数和类
from ..utils import multi_apply, unpack_gt_instances
from .crosskd_single_stage import CrossKDSingleStageDetector


# 注册当前模型为MMDetection的模型组件
@MODELS.register_module()
class CrossKDGFL(CrossKDSingleStageDetector):

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.
        计算一批输入和数据样本的损失。

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
                输入图像，形状为(N, C, H, W)，通常需要进行均值中心化和标准化。
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                数据样本列表，通常包括`gt_instance`等信息。

        Returns:
            dict: A dictionary of loss components.
            损失组件的字典。
        """
        # 提取教师模型的特征和预测
        tea_x = self.teacher.extract_feat(batch_inputs)
        tea_cls_scores, tea_bbox_preds, tea_cls_hold, tea_reg_hold = \
            multi_apply(self.forward_crosskd_single, tea_x,
                        self.teacher.bbox_head.scales, module=self.teacher)

        # 提取学生模型的特征和预测
        stu_x = self.extract_feat(batch_inputs)
        stu_cls_scores, stu_bbox_preds, stu_cls_hold, stu_reg_hold = \
            multi_apply(self.forward_crosskd_single, stu_x,
                        self.bbox_head.scales, module=self)

        # 重用教师模型的头部进行预测
        reused_cls_scores, reused_bbox_preds = multi_apply(
            self.reuse_teacher_head, tea_cls_hold, tea_reg_hold, stu_cls_hold,
            stu_reg_hold, self.teacher.bbox_head.scales)

        # 解包批量数据样本
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        # 计算损失
        losses = self.loss_by_feat(tea_cls_scores, tea_bbox_preds, tea_x,
                                   stu_cls_scores, stu_bbox_preds, stu_x,
                                   reused_cls_scores, reused_bbox_preds,
                                   batch_gt_instances, batch_img_metas,
                                   batch_gt_instances_ignore)
        return losses

    def forward_crosskd_single(self, x, scale, module):
        # 提取分类和回归特征
        cls_feat, reg_feat = x, x
        cls_feat_hold, reg_feat_hold = x, x
        for i, cls_conv in enumerate(module.bbox_head.cls_convs):
            cls_feat = cls_conv(cls_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                cls_feat_hold = cls_feat
            cls_feat = cls_conv.activate(cls_feat)
        for i, reg_conv in enumerate(module.bbox_head.reg_convs):
            reg_feat = reg_conv(reg_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                reg_feat_hold = reg_feat
            reg_feat = reg_conv.activate(reg_feat)
        cls_score = module.bbox_head.gfl_cls(cls_feat)
        bbox_pred = scale(module.bbox_head.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred, cls_feat_hold, reg_feat_hold

    def reuse_teacher_head(self, tea_cls_feat, tea_reg_feat, stu_cls_feat,
                           stu_reg_feat, scale):
        # 对齐特征
        reused_cls_feat = self.align_scale(stu_cls_feat, tea_cls_feat)
        reused_reg_feat = self.align_scale(stu_reg_feat, tea_reg_feat)
        if self.reused_teacher_head_idx != 0:
            reused_cls_feat = F.relu(reused_cls_feat)
            reused_reg_feat = F.relu(reused_reg_feat)

        # 重复构建教师模型的头部
        module = self.teacher.bbox_head
        for i in range(self.reused_teacher_head_idx, module.stacked_convs):
            reused_cls_feat = module.cls_convs[i](reused_cls_feat)
            reused_reg_feat = module.reg_convs[i](reused_reg_feat)
        reused_cls_score = module.gfl_cls(reused_cls_feat)
        reused_bbox_pred = scale(module.gfl_reg(reused_reg_feat)).float()
        return reused_cls_score, reused_bbox_pred

    def align_scale(self, stu_feat, tea_feat):
        # 归一化学生特征
        N, C, H, W = stu_feat.size()
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdim=True)
        stu_std = stu_feat.std(dim=-1, keepdim=True)
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)

        # 对齐尺度并反归一化
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdim=True)
        tea_std = tea_feat.std(dim=-1, keepdim=True)
        stu_feat = stu_feat * tea_std + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def loss_by_feat(
            self,
            tea_cls_scores: List[Tensor],
            tea_bbox_preds: List[Tensor],
            tea_feats: List[Tensor],
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            feats: List[Tensor],
            reused_cls_scores: List[Tensor],
            reused_bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.
        基于检测头部提取的特征计算损失。

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
                每个尺度级别的Cls和质量分数，形状为(N, num_classes, H, W)。
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
                每个尺度级别的盒子分布对数，形状为(N, 4*(n+1), H, W)，n是整数集的最大值。
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
                批量的gt_instance。通常包括``bboxes``和``labels``属性。
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
                每张图像的元信息，例如图像大小、缩放因子等。
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
                忽略训练和测试期间的数据。包含``bboxes``属性数据。
                默认为None。

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
            损失组件的字典。
        """

        # 计算锚点和有效标志
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.bbox_head.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        # 获取分类和回归的目标
        cls_reg_targets = self.bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        # 计算分类和回归的损失
        losses_cls, losses_bbox, losses_dfl, \
            new_avg_factor = multi_apply(
            self.bbox_head.loss_by_feat_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.bbox_head.prior_generator.strides,
            avg_factor=avg_factor)

        new_avg_factor = sum(new_avg_factor)
        new_avg_factor = reduce_mean(new_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / new_avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / new_avg_factor, losses_dfl))
        losses = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

        # 计算分类和回归蒸馏的损失
        losses_cls_kd, losses_reg_kd, kd_avg_factor = multi_apply(
            self.pred_mimicking_loss_single,
            tea_cls_scores,
            tea_bbox_preds,
            reused_cls_scores,
            reused_bbox_preds,
            label_weights_list,
            avg_factor=avg_factor)
        kd_avg_factor = sum(kd_avg_factor)
        losses_reg_kd = list(map(lambda x: x / kd_avg_factor, losses_reg_kd))
        losses.update(
            dict(loss_cls_kd=losses_cls_kd, loss_reg_kd=losses_reg_kd))

        # 如果使用了特征蒸馏，计算特征蒸馏的损失
        if self.with_feat_distill:
            losses_feat_kd = [
                self.loss_feat_kd(feat, tea_feat)
                for feat, tea_feat in zip(feats, tea_feats)
            ]
            losses.update(loss_feat_kd=losses_feat_kd)
        return losses

    def pred_mimicking_loss_single(self, tea_cls_score, tea_bbox_pred,
                                   reused_cls_score, reused_bbox_pred,
                                   label_weights, avg_factor):
        # 对教师模型和学生模型的分类预测进行蒸馏损失计算
        tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.bbox_head.cls_out_channels)
        reused_cls_score = reused_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.bbox_head.cls_out_channels)
        label_weights = label_weights.reshape(-1)
        loss_cls_kd = self.loss_cls_kd(
            reused_cls_score,
            tea_cls_score,
            label_weights,
            avg_factor=avg_factor)

        # 对教师模型和学生模型的回归预测进行蒸馏损失计算
        reg_max = self.bbox_head.reg_max
        tea_bbox_pred = tea_bbox_pred.permute(0, 2, 3,
                                              1).reshape(-1, reg_max + 1)
        reused_bbox_pred = reused_bbox_pred.permute(0, 2, 3, 1).reshape(
            -1, reg_max + 1)
        reg_weights = tea_cls_score.max(dim=1)[0].sigmoid()
        reg_weights[label_weights == 0] = 0
        loss_reg_kd = self.loss_reg_kd(
            reused_bbox_pred,
            tea_bbox_pred,
            weight=reg_weights[:, None].expand(-1, 4).reshape(-1),
            avg_factor=4.0)

        return loss_cls_kd, loss_reg_kd, reg_weights.sum()