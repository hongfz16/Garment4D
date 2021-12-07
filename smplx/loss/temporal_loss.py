import torch
from chamferdist import ChamferDistance, knn_points
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from utils import mesh_utils
from .laplacian import OnetimeLaplacianLoss
from utils.config import args, cfg
from utils import dataloader

def calc_shape_l2_loss(real_shape, fake_shape):
    diff = (real_shape - fake_shape) ** 2
    # return diff.sum() / real_shape.shape[0]
    return diff.mean()

vf_fid = None
vf_vid = None

def calc_interpenetration_loss(body_model, so, garment_v, reduce_fn='sum', to_root_joint=False):
    global vf_fid
    global vf_vid
    if vf_vid is None or vf_fid is None:
        vf_fid, vf_vid = mesh_utils.calc_body_mesh_info(body_model)
        vf_fid = vf_fid.cuda()
        vf_vid = vf_vid.cuda()
    if len(garment_v.shape) == 4:
        garment_v = garment_v.reshape(garment_v.shape[0] * garment_v.shape[1], garment_v.shape[2], 3)
    vn = mesh_utils.compute_vnorms(so['vertices'], torch.from_numpy(body_model.faces.astype(np.int64)).cuda(), vf_vid, vf_fid)
    # print(garment_v.shape)
    # print(so['vertices'].shape)
    if to_root_joint:
        garment_v_root_joint = garment_v + so['joints'][:, 0, :].unsqueeze(1)
    else:
        garment_v_root_joint = garment_v
    nn = knn_points(garment_v_root_joint, so['vertices'])
    expand_idx = nn.idx.expand(nn.idx.size(0), nn.idx.size(1), vn.size(2))
    vn_indexed = torch.gather(vn, 1, expand_idx)
    pred_body_v_indexed = torch.gather(so['vertices'], 1, expand_idx)
    if reduce_fn == 'sum':
        loss = F.relu(-torch.mul(vn_indexed, garment_v_root_joint - pred_body_v_indexed).sum(-1)).sum(-1).mean() # add another sum(-1) to increase the interpenetration term
    elif reduce_fn == 'mean':
        loss = F.relu(-torch.mul(vn_indexed, garment_v_root_joint - pred_body_v_indexed).sum(-1)).mean()
    else:
        raise NotImplementedError
    return loss

def calc_garment_onetime_laplacian_loss(gt_garment_v, pred_garment_v, garment_f, b):
    garment_f_t = torch.from_numpy(garment_f).cuda().unsqueeze(0).repeat(gt_garment_v.shape[0], 1, 1)
    if gt_garment_v.shape[0] != b:
        gt_garment_v = torch.cat([gt_garment_v, gt_garment_v[0].reshape(1, gt_garment_v.shape[1], 3).repeat(b - gt_garment_v.shape[0], 1, 1)], 0)
        pred_garment_v = torch.cat([pred_garment_v, pred_garment_v[0].reshape(1, pred_garment_v.shape[1], 3).repeat(b - pred_garment_v.shape[0], 1, 1)], 0)
        garment_f_t = torch.cat([garment_f_t, garment_f_t[0].reshape(1, garment_f_t.shape[1], 3).repeat(b - garment_f_t.shape[0], 1, 1)], 0)
        assert gt_garment_v.shape[0] == b
        assert pred_garment_v.shape[0] == b
        assert garment_f_t.shape[0] == b
    LapLoss = OnetimeLaplacianLoss(garment_f_t, gt_garment_v)
    return LapLoss(pred_garment_v)

def temporal_loss_PCA(output_dict, inputs, body_model, args):
    seq_batch_size = inputs['pose_torch'].shape[0]
    seq_length = inputs['pose_torch'].shape[1]

    total_loss = 0
    loss_dict = {}
    pred_so = {
        'vertices': inputs['smpl_vertices_torch'].cuda().reshape(seq_batch_size * seq_length, -1, 3),
        'joints': inputs['smpl_root_joints_torch'].cuda().reshape(seq_batch_size * seq_length, 1, 3)
    }
    T_pred_shape_so = {
        'vertices': inputs['Tpose_smpl_vertices_torch'].cuda().reshape(seq_batch_size, 6890, 3),
        'joints': inputs['Tpose_smpl_root_joints_torch'].cuda().reshape(seq_batch_size, 1, 3),
    }

    # sem_seg_loss
    pred_logits = output_dict['sem_logits'].reshape(seq_batch_size * seq_length * cfg.NETWORK.NPOINTS, -1)
    gt_labels = inputs['pcd_label_torch'].cuda().reshape(seq_batch_size * seq_length * cfg.NETWORK.NPOINTS).long()
    loss_func = torch.nn.CrossEntropyLoss()
    sem_seg_loss = loss_func(pred_logits, gt_labels)
    total_loss += sem_seg_loss * cfg.LOSS.SEM_SEG_LOSS_LAMBDA
    loss_dict['sem_seg_loss'] = sem_seg_loss
    
    if args.only_seg:
        loss_dict['total_loss'] = total_loss
        return loss_dict

    # garment_pca_coeff_l2:
    pred_pca_coeff = output_dict['garment_PCA_coeff'].reshape(seq_batch_size, cfg.GARMENT.PCADIM)
    gt_pca_coeff = inputs['PCACoeff'].reshape(seq_batch_size, cfg.GARMENT.PCADIM).cuda()
    garment_pca_coeff_l2 = calc_shape_l2_loss(pred_pca_coeff, gt_pca_coeff)
    total_loss += garment_pca_coeff_l2 * cfg.LOSS.GARMENT_PCA_COEFF_L2_LAMBDA
    loss_dict['garment_pca_coeff_l2'] = garment_pca_coeff_l2

    # garment_l2_loss:
    gt_garment_v = inputs['garment_template_vertices'].cuda().reshape(seq_batch_size, -1, 3)
    regressed_garment_v = output_dict['tpose_garment']
    garment_l2_loss = ((regressed_garment_v.reshape(seq_batch_size, -1, 3) - gt_garment_v) ** 2).sum(-1).mean()
    garment_msre = torch.sqrt(((regressed_garment_v.reshape(seq_batch_size, -1, 3) - gt_garment_v) ** 2).sum(-1)).mean()
    total_loss += garment_l2_loss * cfg.LOSS.GARMENT_L2_LOSS_LAMBDA
    loss_dict['garment_l2_loss'] = garment_l2_loss
    loss_dict['garment_msre'] = garment_msre

    # interpenetration_loss:
    regressed_garment_v = output_dict['tpose_garment']
    interpenetration_loss = calc_interpenetration_loss(body_model, T_pred_shape_so, regressed_garment_v, reduce_fn='mean', to_root_joint=True)
    total_loss += interpenetration_loss * cfg.LOSS.INTERPENETRATION_LOSS_LAMBDA
    loss_dict['interpenetration_loss'] = interpenetration_loss

    # garment_lap_loss:
    regressed_garment_v = output_dict['tpose_garment']
    last_regressed_garment_v = inputs['garment_template_vertices'].cuda().reshape(seq_batch_size, -1, 3)
    garment_lap_loss = calc_garment_onetime_laplacian_loss(last_regressed_garment_v,
                                                            regressed_garment_v.reshape(seq_batch_size, -1, 3),
                                                            output_dict['garment_f_3'], args.batch_size)
    total_loss += garment_lap_loss * cfg.LOSS.GARMENT_LAP_LOSS_LAMBDA
    loss_dict['garment_lap_loss'] = garment_lap_loss

    loss_dict['total_loss'] = total_loss
    return loss_dict

def calc_temporal_constraint_loss(pred_garment_v, nbatch, T):
    pred_garment_v = pred_garment_v.reshape(nbatch, T, -1, 3)
    diff = pred_garment_v[:, :-1, :, :] - pred_garment_v[:, 1:, :, :]
    diff = (diff ** 2).sum(-1).sqrt().mean()
    return diff

def calc_simple_self_laplacian_regularization(pred_garment_v, lap_adj, nbatch, T):
    pred_lap = torch.spmm(lap_adj, pred_garment_v.reshape(nbatch * T, -1, 3).transpose(0, 1).reshape(-1, nbatch * T * 3))
    pred_lap = pred_lap.reshape(-1, nbatch * T, 3).transpose(0, 1)
    pred_lap_norm = torch.norm(pred_lap, p=2, dim=-1)
    return pred_lap_norm.mean()

def calc_acceleration(pred_garment_v, nbatch, T):
    pred_garment_v = pred_garment_v.reshape(nbatch, T, -1, 3)
    deltat = 1/30
    v = (pred_garment_v[:, 1:] - pred_garment_v[:, :-1]) / deltat
    v = v.reshape(nbatch, T-1, pred_garment_v.shape[-2], 3)
    accel = (v[:, 1:] - v[:, :-1]) / deltat
    return accel.reshape(nbatch, T-2, pred_garment_v.shape[-2], 3)

def calc_acceleration_error(pred_garment_v, gt_garment_v, nbatch, T):
    pred_accel = calc_acceleration(pred_garment_v, nbatch, T)
    gt_accel = calc_acceleration(gt_garment_v, nbatch, T)
    err = ((pred_accel - gt_accel) ** 2).sum(-1).sqrt().reshape(nbatch, T-2, pred_garment_v.shape[-2])
    return err.mean()

def temporal_loss_PCA_LBS(output_dict, inputs, body_model, args):
    loss_dict = {}
    total_loss = 0
    seq_batch_size = inputs['pose_torch'].shape[0]
    seq_length = inputs['pose_torch'].shape[1]
    pred_so = {
        'vertices': inputs['smpl_vertices_torch'].cuda().reshape(seq_batch_size * seq_length, -1, 3),
        'joints': inputs['smpl_root_joints_torch'].cuda().reshape(seq_batch_size * seq_length, 1, 3)
    }

    # lbs_garment_l2_loss:
    lbs_garment_l2_loss_acc = 0
    gt_garment_v = inputs['garment_torch'].cuda().reshape(seq_batch_size * seq_length, -1, 3) + pred_so['joints'][:, 0, :].unsqueeze(1)
    for i, regressed_garment_v in enumerate(output_dict['iter_regressed_lbs_garment_v']):
        lbs_garment_l2_loss_acc += ((regressed_garment_v - gt_garment_v) ** 2).sum(-1).mean()
        if i == len(output_dict['iter_regressed_lbs_garment_v']) - 1:
            lbs_garment_msre = torch.sqrt(((regressed_garment_v - gt_garment_v) ** 2).sum(-1)).mean(-1)
            loss_dict['lbs_garment_msre'] = lbs_garment_msre.mean()
            loss_dict['lbs_garment_msre_list'] = lbs_garment_msre.reshape(seq_batch_size, seq_length)
    loss_dict['only_lbs_garment_msre'] = ((output_dict['lbs_pred_garment_v'].reshape(seq_batch_size * seq_length, -1, 3) - gt_garment_v) ** 2).sum(-1).sqrt().mean()
    total_loss += lbs_garment_l2_loss_acc * cfg.LOSS.LBS_GARMENT_L2_LOSS_LAMBDA
    loss_dict['lbs_garment_l2_loss'] = lbs_garment_l2_loss_acc

    # lbs_garment_lap_loss:
    lbs_garment_lap_loss_acc = 0
    T_template_v = inputs['garment_template_vertices'].reshape(seq_batch_size, 1, -1, 3).repeat(1, seq_length, 1, 1).reshape(seq_batch_size * seq_length, -1, 3).cuda()
    for regressed_garment_v in output_dict['iter_regressed_lbs_garment_v']:
        lbs_garment_lap_loss_acc += calc_simple_self_laplacian_regularization(regressed_garment_v, output_dict['lap_adj'], seq_batch_size, seq_length)
    total_loss += lbs_garment_lap_loss_acc * cfg.LOSS.LBS_GARMENT_LAP_LOSS_LAMBDA
    loss_dict['lbs_garment_lap_loss'] = lbs_garment_lap_loss_acc

    # lbs_interpenetration_loss:
    lbs_interpenetration_loss_acc = 0
    for regressed_garment_v in output_dict['iter_regressed_lbs_garment_v']:
        lbs_interpenetration_loss_acc += calc_interpenetration_loss(body_model, pred_so, regressed_garment_v, reduce_fn='mean', to_root_joint=False)
    total_loss += lbs_interpenetration_loss_acc * cfg.LOSS.LBS_INTERPENETRATION_LOSS_LAMBDA
    loss_dict['lbs_interpenetration_loss'] = lbs_interpenetration_loss_acc

    # temporal_constraint_loss:
    temporal_constraint_loss_acc = 0
    regressed_garment_v = output_dict['iter_regressed_lbs_garment_v'][-1]
    temporal_constraint_loss_acc += calc_temporal_constraint_loss(regressed_garment_v, seq_batch_size, seq_length)
    total_loss += temporal_constraint_loss_acc * cfg.LOSS.TEMPORAL_CONSTRAINT_LOSS_LAMBDA
    loss_dict['temporal_constraint_loss'] = temporal_constraint_loss_acc

    gt_garment_v = inputs['garment_torch'].cuda().reshape(seq_batch_size * seq_length, -1, 3) + pred_so['joints'][:, 0, :].unsqueeze(1)
    acceleration_error = calc_acceleration_error(output_dict['iter_regressed_lbs_garment_v'][-1], gt_garment_v, seq_batch_size, seq_length)
    loss_dict['acceleration_error'] = acceleration_error

    only_lbs_acceleration_error = calc_acceleration_error(output_dict['lbs_pred_garment_v'], gt_garment_v, seq_batch_size, seq_length)
    loss_dict['only_lbs_acceleration_error'] = only_lbs_acceleration_error

    loss_dict['total_loss'] = total_loss

    return loss_dict
