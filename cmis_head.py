#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolox.utils import bboxes_iou
from .losses import IOUloss
from .dice_loss import DiceLoss
from .network_blocks import BaseConv, DWConv
import time
from torchvision.ops import nms, roi_align, roi_pool
from yolox.layers.conv_with_kaiming_uniform import conv_with_kaiming_uniform1
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import torch.nn.modules.conv as conv
import random

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)
            zz_channel = zz_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
                zz_channel = zz_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CMISHead(nn.Module):
    def __init__(
            self,
            num_classes,
            width=1.0,
            strides=[8, 16, 32],
            in_channels=[128, 256, 512],
            act="silu",
            depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        self.epoch = None
        self.mask_flag = True
        super().__init__()
        self.mask_dim = 16
        self.top_interp = 'bilinear'
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False
        self.mask_size = 80
        self.mask_size_up = 160

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        self.class_atten = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv


        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
        self.class_atten.append(
            nn.Sequential(
                *[
                    nn.Conv2d(
                        in_channels=int(256 * width),
                        out_channels=self.mask_dim * self.num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )])
        )

        conv_block = conv_with_kaiming_uniform1('GN', True, True)  # conv relu bn

        planes = int(256 * width)
        num_convs = 3

        self.refine = nn.ModuleList()
        x_input_size = in_channels  # 128,256,512 #[256,512,1024]
        for in_feature in range(len(in_channels)):
            self.refine.append(conv_block(
                int(x_input_size[in_feature] * width), planes, 3, 1, 1))
        tower = []
        tower.append(CoordConv2d(
            2*planes, planes, with_r=False, kernel_size=3, stride=1, padding=1))
        tower.append(
            conv_block(planes, planes, 3, 1, 1))
        tower.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        tower.append(
            nn.Conv2d(planes, self.mask_dim, 3, 1, 1))
        self.add_module('tower', nn.Sequential(*tower))
        self.use_l1 = False
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(logits=True)
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none", loss_type="iou")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.confthre = 0.3
        self.nmsthre = 0.35

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def merge_bases(self, rois, coeffs, location_to_inds=None):
        # merge predictions
        N = coeffs.size(0)
        if location_to_inds is not None:
            rois = rois[location_to_inds]
        N, B, H, W = rois.size()

        one_stage = False
        if one_stage:
            rois = F.interpolate(rois, (self.mask_size, self.mask_size),
                                 mode=self.top_interp)
            coeffs = F.interpolate(coeffs, (self.mask_size, self.mask_size),
                                   mode=self.top_interp).softmax(dim=1)

        else:
            rois = F.interpolate(rois, (self.mask_size, self.mask_size),
                                 mode=self.top_interp)
            coeffs = F.interpolate(coeffs, (H, W),
                                   mode=self.top_interp).softmax(dim=1)

        masks_preds = (rois * coeffs).sum(dim=1)

        return masks_preds.view(N, -1)

    def forward(self, xin, labels=None, imgs=None, masklabels=None, epoch=None):
        self.epoch = epoch
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        attnss = []
        bases = []
        pred_mask_logits = []
        xin0 = xin[3:]
        xin = xin[:3]

        image_size = imgs.shape[-1]
        self.strides = [8, 16, 32]
        device = xin[0].device
        for k, (cls_conv, reg_conv, stride_this_level, x, x_backbone) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, xin, xin0)):

            # Handle x_seg initialization and accumulation
            x_seg = self.refine[k](x) if k == 0 else x_seg + F.interpolate(self.refine[k](x), x_seg.size()[2:],
                                                                           mode="bilinear", align_corners=False)

            # Process x_seg_add and bases when k == 2
            if k == 2:
                x_seg_add = torch.cat([x_seg, xin0[0]], 1)
                bases = F.interpolate(self.tower(x_seg_add).to(device), (self.mask_size_up, self.mask_size_up),
                                      mode=self.top_interp)

            x = self.stems[k](x)
            cls_feat = cls_conv(x)
            reg_feat = reg_conv(x)

            cls_output = self.cls_preds[k](cls_feat)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if k == 0:
                attnss = self.class_atten[0](reg_feat)

            # Training mode specific operations
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())
            else:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

            outputs.append(output)

        if self.training:
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg, ce_loss, dist_loss = self.get_losses(
                imgs, x_shifts, y_shifts, expanded_strides, labels,
                torch.cat(outputs, 1), origin_preds, dtype=xin[0].dtype, masklabels=masklabels,
                 bases=bases, attnss=attnss
            )
            return loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg, ce_loss, dist_loss
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)

            if self.decode_in_inference:
                outputs = self.decode_outputs(outputs, dtype=xin[0].type())
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=False
                )

                pred_mask_logits_list = []
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                if outputs[0] is not None:
                    for batch_idx, output in enumerate(outputs):
                        if output is None:
                            pred_mask_logits_list.append(None)
                            continue

                        bbox_preds = output[:, :4].clone().to(device)
                        p = torch.full((bbox_preds.shape[0], 1), 0, device=device)
                        boxes = torch.cat((p, bbox_preds), 1)
                        attn_list = [attnss[batch_idx][i:i + self.mask_dim, :, :] for i in range(self.num_classes)]
                        attn = torch.stack(attn_list, 0)[output[:, 6].long()].to(device)

                        one_stage = False
                        if one_stage:
                            rois = bases[batch_idx].unsqueeze(0).repeat(boxes.shape[0], 1, 1, 1)
                        else:
                            rois = roi_align(bases[batch_idx].unsqueeze(0), boxes, [self.mask_size, self.mask_size],
                                             spatial_scale=bases.shape[-1] / image_size, sampling_ratio=1).to(device)

                        pred_mask_logits = self.merge_bases(rois, attn)
                        pred_mask_logits = pred_mask_logits.sigmoid().view(-1, self.mask_size, self.mask_size)
                        pred_mask_logits_list.append(pred_mask_logits)
                return outputs, pred_mask_logits_list
            else:
                return outputs, pred_mask_logits

    def get_output_and_grid(self, output, k, stride, dtype):
        # output torch.Size([4,  36, 20, 20])
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)  # torch.Size([4, 1, 36, 20, 20])
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )  # torch.Size([4, 400, 36])
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
            masklabels,
            bases,
            attnss):

        image_size = imgs.shape[-1]
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        num_fg = 0.0
        num_gts = 0.0
        loss_mask = 0

        for batch_idx in range(outputs.shape[0]):  # ([4, 8400, 27])
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt

            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]  # (4, 4)
                gt_classes = labels[batch_idx, :num_gt, 0]  # (4)
                bboxes_preds_per_image = bbox_preds[batch_idx]  # (8400, 4)

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )

            if self.mask_flag:
                romdom_num = 0.2
                gt_mask_per_image = masklabels[batch_idx, :num_gt, :, :]
                perturbations = [random.uniform(-romdom_num, romdom_num) for _ in range(4)]
                l, r, t, b = perturbations


                gt_bboxes_per_image1 = torch.stack([
                    gt_bboxes_per_image[:, 0] - l * gt_bboxes_per_image[:, 2] - 0.5 * gt_bboxes_per_image[:, 2],
                    gt_bboxes_per_image[:, 1] - r * gt_bboxes_per_image[:, 3] - 0.5 * gt_bboxes_per_image[:, 3],
                    gt_bboxes_per_image[:, 0] + t * gt_bboxes_per_image[:, 2] + 0.5 * gt_bboxes_per_image[:, 2],
                    gt_bboxes_per_image[:, 1] + b * gt_bboxes_per_image[:, 3] + 0.5 * gt_bboxes_per_image[:, 3]
                ], 1).cuda()


                struct_idx = torch.arange(gt_bboxes_per_image.shape[0], device=gt_bboxes_per_image.device).view(-1, 1)
                boxes = torch.cat((struct_idx, gt_bboxes_per_image1), 1)


                gt_mask_per_image = gt_mask_per_image.view(-1, 1, gt_mask_per_image.shape[-2],
                                                           gt_mask_per_image.shape[-1])
                gt_mask_per_image = roi_align(gt_mask_per_image, boxes, [self.mask_size, self.mask_size]).view(-1,
                                                                                                               self.mask_size,
                                                                                                               self.mask_size)


                p = torch.full([gt_bboxes_per_image.shape[0], 1], 0, device=gt_bboxes_per_image.device)
                boxes = torch.cat((p, gt_bboxes_per_image1), 1)
                rois = roi_align(bases[batch_idx].unsqueeze(0), boxes, [self.mask_size, self.mask_size],
                                 spatial_scale=bases.shape[-1] / image_size, sampling_ratio=1).cuda()
                rois = rois[matched_gt_inds]


                gt_mask_per_image = gt_mask_per_image[matched_gt_inds].view(1, -1, self.mask_size, self.mask_size)
                attn_list = [attnss[batch_idx][i:i + self.mask_dim, :, :] for i in range(self.num_classes)]
                attn = torch.stack(attn_list, 0)
                attn = attn[gt_matched_classes.long()]


                pred_mask_logits = self.merge_bases(rois, attn).view(1, -1, self.mask_size, self.mask_size).cuda()
                loss_list = ['ab', 'dice']
                loss_mask += self.seg_loss(loss_list, pred_mask_logits, gt_mask_per_image, gt_matched_classes)

            num_fg += num_fg_img

            cls_target = F.one_hot(gt_matched_classes.to(torch.int64),
                                       self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
            obj_target = fg_mask.unsqueeze(-1)
            reg_target = gt_bboxes_per_image[matched_gt_inds]

            if self.use_l1:
                l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                    gt_bboxes_per_image[matched_gt_inds],
                    expanded_strides[0][fg_mask],
                    x_shifts=x_shifts[0][fg_mask],
                    y_shifts=y_shifts[0][fg_mask],
                )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
            torch.cuda.empty_cache()

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)

        loss_l1 = 0
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                   ).sum() / num_fg
        loss_cls = (
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg

        loss_mask = loss_mask / num_fg
        reg_weight = 5.0
        loss_ce = torch.tensor(0.0, requires_grad=True)
        loss_dist = loss_mask * 10

        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 + loss_ce + loss_dist

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
            loss_ce,
            loss_dist
        )

    def seg_loss(self, loss_list, pred_mask_logits, gt_mask_per_image, gt_matched_classes):
        loss_mask = 0
        if 'focal' in loss_list:
            focal = self.focal_loss(pred_mask_logits, gt_mask_per_image)
            loss_mask += focal
        if 'bce' in loss_list:
            ce = torch.nn.functional.binary_cross_entropy_with_logits(pred_mask_logits, gt_mask_per_image)
            loss_mask += ce
        if 'ab' in loss_list:
            input_tensor = gt_mask_per_image.to("cuda")
            input_tensor = input_tensor.permute(1, 0, 2, 3)


            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to("cuda") * 2.0
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to("cuda") * 2.0


            edge_x = torch.conv2d(input_tensor, sobel_x.view(1, 1, 3, 3), padding=1)
            edge_y = torch.conv2d(input_tensor, sobel_y.view(1, 1, 3, 3), padding=1)


            edge_weight = torch.sqrt(edge_x.pow(2) + edge_y.pow(2)).sigmoid().squeeze().squeeze()
            edge_weight = torch.where(edge_weight > 0.5, 10, 1)
            class_weights = torch.ones([gt_mask_per_image.shape[0], gt_mask_per_image.shape[1], gt_mask_per_image.shape[2],
                                  gt_mask_per_image.shape[3]])
            for j in range(class_weights.shape[1]):
                class_weight = gt_mask_per_image[0][j].sum() / (self.mask_size * self.mask_size)
                class_weights[0][j] = torch.where(gt_mask_per_image[0][j] > 0.5, (1.0 - class_weight) * 10,
                                            class_weight * 10)  # + edge_weight
            class_weights = class_weights.cuda()
            ce = torch.nn.functional.binary_cross_entropy_with_logits(pred_mask_logits, gt_mask_per_image,
                                                                      weight=edge_weight+class_weights)
            loss_mask += ce
        if 'dice' in loss_list:
            dice = self.dice_loss(pred_mask_logits, gt_mask_per_image).sum()
            loss_mask += dice
        return loss_mask

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            obj_preds,
            mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()
        # is_in_boxes_and_center torch.Size([26, 878])
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
        )
        # pair_wise_ious torch.Size([26, 878])
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


if __name__ == "__main__":
    num_classes = 80
    yolox_head = CMISHead(num_classes)


    batch_size = 4
    channels = 128
    height = 64
    width = 64
    input_data = torch.randn(batch_size, channels, height, width)


    output = yolox_head(input_data)


    print(output)