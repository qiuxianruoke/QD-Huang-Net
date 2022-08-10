import torch
from distillation_tools import bmask, smask, plane_mask, channel_mask, gcblock
import torch.nn as nn


def distillation_loss(target, \
        head_distillation=True, \
        neck_distillation=True, \
        backbone_distillation=True, \
        student_head_feature=[], \
        student_neck_feature=[], \
        student_backbone_feature=[], \
        teacher_head_feature=[], \
        teacher_neck_feature=[], \
        teacher_backbone_feature=[],\
        head_weight=1.0, \
        neck_weight=1.0, \
        backbone_weight=1.0, device='cpu', \
        target_weight=0.5, \
        background_weight=0.5, \
        attention_weight=0.5, \
        global_weight=0.5):
    
    # head loss
    # focal loss
    h_local_loss = torch.zeros(1).to(device)
    # global loss
    h_global_loss = torch.zeros(1).to(device)
    # attention loss
    h_attention_loss = torch.zeros(1).to(device)

    # neck loss
    n_local_loss = torch.zeros(1).to(device)
    # global loss
    n_global_loss = torch.zeros(1).to(device)
    # attention loss
    n_attention_loss = torch.zeros(1).to(device)
 
    # backbone loss
    b_local_loss = torch.zeros(1).to(device)
    # global loss
    b_global_loss = torch.zeros(1).to(device)
    # attention loss
    b_attention_loss = torch.zeros(1).to(device)

    # gcblock
    head_gcbolck = [gcblock(f.shape(1)) for f in teacher_head_feature]
    neck_gcbolck = [gcblock(f.shape(1)) for f in teacher_neck_feature]
    backbone_gcbolck = [gcblock(f.shape(1)) for f in teacher_backbone_feature]

    l1 = nn.L1Loss()

    for i, (tbf, sbf, tnf, snf, thf, shf) in enumerate(zip(\
            teacher_backbone_feature, student_backbone_feature,\
            teacher_neck_feature, student_neck_feature, \
            teacher_head_feature, student_head_feature)):
        # head
        if head_distillation:
            tacmask = channel_mask(thf)  # channel attention mask
            sacmask = channel_mask(shf)
            tapmask = plane_mask(thf)   # plane attention mask
            sapmask = plane_mask(shf)
            tsmask = smask(thf) # scale mask
            # ssmask = smask(shf)
            tbmask = bmask(thf) # binary mask
            # sbmask = bmask(shf)
            tg_feature = head_gcbolck[i](thf)  # global loss
            sg_feature = head_gcbolck[i](shf)
            h_local_loss += (torch.sum(\
                tbmask * tsmask * tacmask * tapmask * (thf - shf)**2) * target_weight +\
                torch.sum((1-tbmask) * tsmask * tacmask * tapmask * (thf - shf)**2) * backbone_weight)
            h_attention_loss += (l1(tapmask, sapmask) + l1(tacmask, sacmask)) * attention_weight
            h_global_loss += torch.sum((tg_feature - sg_feature)**2) * global_weight

        if neck_distillation:
            tacmask = channel_mask(tnf)  # channel attention mask
            sacmask = channel_mask(snf)
            tapmask = plane_mask(tnf)   # plane attention mask
            sapmask = plane_mask(snf)
            tsmask = smask(tnf) # scale mask
            # ssmask = smask(shf)
            tbmask = bmask(tnf) # binary mask
            # sbmask = bmask(shf)
            tg_feature = head_gcbolck[i](tnf)  # global loss
            sg_feature = head_gcbolck[i](snf)
            n_local_loss += (torch.sum(\
                tbmask * tsmask * tacmask * tapmask * (tnf - snf)**2) * target_weight +\
                torch.sum((1-tbmask) * tsmask * tacmask * tapmask * (tnf - snf)**2) * backbone_weight)
            n_attention_loss += (l1(tapmask, sapmask) + l1(tacmask, sacmask)) * attention_weight
            n_global_loss += torch.sum((tg_feature - sg_feature)**2) * global_weight
        
        if backbone_distillation:
            tacmask = channel_mask(tbf)  # channel attention mask
            sacmask = channel_mask(sbf)
            tapmask = plane_mask(tbf)   # plane attention mask
            sapmask = plane_mask(sbf)
            tsmask = smask(tbf) # scale mask
            # ssmask = smask(shf)
            tbmask = bmask(tbf) # binary mask
            # sbmask = bmask(shf)
            tg_feature = head_gcbolck[i](tbf)  # global loss
            sg_feature = head_gcbolck[i](sbf)
            b_local_loss += (torch.sum(\
                tbmask * tsmask * tacmask * tapmask * (tbf - sbf)**2) * target_weight +\
                torch.sum((1-tbmask) * tsmask * tacmask * tapmask * (tbf - sbf)**2) * backbone_weight)
            b_attention_loss += (l1(tapmask, sapmask) + l1(tacmask, sacmask)) * attention_weight
            b_global_loss += torch.sum((tg_feature - sg_feature)**2) * global_weight
    return (h_local_loss + h_attention_loss + h_global_loss)*head_weight + \
                (n_local_loss + n_attention_loss + n_global_loss)*neck_weight + \
                    (b_local_loss + b_attention_loss + b_global_loss)*backbone_weight
