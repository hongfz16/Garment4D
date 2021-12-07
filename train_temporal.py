import os
import torch
import torch.distributed as dist
import torch.nn as nn
import pickle
from tqdm import tqdm
import numpy as np
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from torch.nn import SyncBatchNorm

from modules.mesh_encoder import PCAGarmentEncoderSeg, PCALBSGarmentUseSegEncoderSeg, PCALBSGarmentUseSegEncoderSegMGN

from utils.dataloader import SeqPointSMPLDataset, SeqPointSMPL_collate_fn
from utils.config import args, cfg
from utils import train_utils
from utils.train_utils import merge_results, collect_decisions

from smplx import build_layer
from smplx import parse_args, batch_rodrigues
from smplx import temporal_loss_PCA, temporal_loss_PCA_LBS

def build(log_to_file=True, dont_load_train=False):
    #-------------------------------- INIT --------------------------------#
    if args.launcher == None:
        args.dist_train = False
    else:
        args.batch_size, args.local_rank = getattr(train_utils, 'init_dist_%s' % args.launcher)(
            args.batch_size, args.tcp_port, args.local_rank, backend='nccl'
        )
        args.dist_train = True
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok = True)
    tmp_dir = os.path.join(args.output_dir, 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok = True)
    if args.local_rank == 0 and log_to_file:
        logger.add(os.path.join(args.output_dir, 'log.txt'))
    ckpt_dir = os.path.join(args.output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok = True)
    vis_dir = os.path.join(args.output_dir, 'vis')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir, exist_ok = True)

    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    if args.local_rank == 0:
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if args.dist_train and args.local_rank == 0:
        total_gpus = dist.get_world_size()
        logger.info('Total Batch Size: {} x {} = {}'.format(total_gpus, args.batch_size, total_gpus * args.batch_size))
    if args.local_rank == 0:
        for key, val in vars(args).items():
            logger.info('{:16} {}'.format(key, val))

    #-------------------------------- BUILDING BODY MODEL --------------------------------#
    if args.local_rank == 0:
        logger.info("Building Body Model...")
    body_exp_cfg = parse_args()
    model_path = body_exp_cfg.body_model.folder
    body_model = build_layer(model_path, **body_exp_cfg.body_model)
    body_model = body_model.cuda()

    body_exp_cfg.body_model.gender = 'male'
    body_model_male = build_layer(model_path, **body_exp_cfg.body_model)
    body_exp_cfg.body_model.gender = 'female'
    body_model_female = build_layer(model_path, **body_exp_cfg.body_model)
    body_exp_cfg.body_model.gender = 'neutral'
    body_model_neutral = build_layer(model_path, **body_exp_cfg.body_model)

    #-------------------------------- BUILDING DATALOADER --------------------------------#
    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    cur_dataset = SeqPointSMPLDataset

    if not dont_load_train:
        if args.local_rank == 0:
            logger.info("Building Train DataLoader...")
        train_dataset = cur_dataset(cfg.NETWORK.NPOINTS, cfg.DATASET.TRAIN_F_LIST, cfg.DATASET.SMPL_PARAM_PREFIX,
                                    args.T, is_train=True, garment_template_prefix=cfg.DATASET.GARMENT_TEMPLATE_T_POSE_PREFIX,
                                    body_model_dict={'male': body_model_male, 'female': body_model_female, 'neutral': body_model_neutral})
        if args.dist_train:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, collate_fn = SeqPointSMPL_collate_fn,
                                                    shuffle = (train_sampler is None), num_workers = args.num_workers,
                                                    pin_memory = True, drop_last = True, sampler = train_sampler, timeout = 0,
                                                    worker_init_fn=worker_init_fn)
    else:
        train_dataloader = None
    if args.local_rank == 0:
        logger.info("Building Eval DataLoader...")
    eval_dataset = cur_dataset(cfg.NETWORK.NPOINTS, cfg.DATASET.EVAL_F_LIST, cfg.DATASET.SMPL_PARAM_PREFIX,
                            args.T, is_train=False, garment_template_prefix=cfg.DATASET.GARMENT_TEMPLATE_T_POSE_PREFIX,
                            body_model_dict={'male': body_model_male, 'female': body_model_female, 'neutral': body_model_neutral})
    if args.dist_train:
        rank, world_size = train_utils.get_dist_info()
        eval_sampler = train_utils.DistributedSampler(eval_dataset, world_size, rank, shuffle=False)
    else:
        eval_sampler = None
    eval_dataloader = torch.utils.data.DataLoader(dataset = eval_dataset, batch_size = args.batch_size, collate_fn = SeqPointSMPL_collate_fn,
                                                  shuffle = False, num_workers = args.num_workers,
                                                  pin_memory = True, drop_last = False, sampler = eval_sampler, timeout = 0)
    
    #-------------------------------- BUILDING MODEL --------------------------------#
    if args.local_rank == 0:
        logger.info("Building Model...")
    if args.MGN:
        model = PCALBSGarmentUseSegEncoderSegMGN(cfg = cfg, args = args).cuda()
    elif args.GarmentPCA:
        model = PCAGarmentEncoderSeg(cfg = cfg, args = args).cuda()
    elif args.GarmentPCALBS:
        model = PCALBSGarmentUseSegEncoderSeg(cfg = cfg, args = args).cuda()
    if args.syncbn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    
    #-------------------------------- BUILDING OPTIMIZER --------------------------------#
    if args.local_rank == 0:
        logger.info("Building Optimizer...")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    #-------------------------------- BUILD SCHEDULER --------------------------------#
    if args.local_rank == 0:
        logger.info("Building Scheduler...")
    scheduler = None
    if args.lr_sche:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 8)
    
    #-------------------------------- LOADING CKPT --------------------------------#
    epoch = -1
    other_state = {'best_v_l2': 10086}
    ckpt_fname = os.path.join(ckpt_dir, args.ckpt_name)
    if os.path.exists(ckpt_fname):
        if args.local_rank == 0:
            logger.info("Loading CKPT from {}".format(ckpt_fname))
        if args.GarmentPCA or args.GarmentPCALBS:
            PCA_params = list(map(lambda x: x[1], filter(lambda p: p[1].requires_grad and p[0].startswith('PCA_garment_encoder'), model.named_parameters())))
            LBS_params = list(map(lambda x: x[1], filter(lambda p: p[1].requires_grad and (not p[0].startswith('PCA_garment_encoder')), model.named_parameters())))
            if args.fix_PCA:
                for p in PCA_params:
                    p.requires_grad = False
                optimizer = torch.optim.Adam(LBS_params, lr=args.lr)
            else:
                optimizer = torch.optim.Adam(
                    [{'params': PCA_params},
                    {'params': LBS_params},],
                    lr = args.lr
                )
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        epoch, other_state = train_utils.load_params_with_optimizer_otherstate(model, ckpt_fname, to_cpu=args.dist_train,
                                                                               optimizer=optimizer, logger=logger)
    elif args.pretrained_model is not None and os.path.exists(args.pretrained_model):
        if args.local_rank == 0:
            logger.info("Loading pretrained CKPT from {}".format(args.pretrained_model))
        train_utils.load_pretrained_model(model, args.pretrained_model, to_cpu=args.dist_train, logger=logger)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.GarmentPCA_pretrain is not None and os.path.exists(args.GarmentPCA_pretrain):
        if args.local_rank == 0:
            logger.info("Loading pretrained CKPT from {}".format(args.GarmentPCA_pretrain))
        train_utils.load_pretrained_model(model, args.GarmentPCA_pretrain, to_cpu=args.dist_train, logger=logger)
        PCA_params = list(map(lambda x: x[1], filter(lambda p: p[1].requires_grad and p[0].startswith('PCA_garment_encoder'), model.named_parameters())))
        LBS_params = list(map(lambda x: x[1], filter(lambda p: p[1].requires_grad and (not p[0].startswith('PCA_garment_encoder')), model.named_parameters())))
        if args.fix_PCA:
            logger.info("Fixing PCA parameters.")
            for p in PCA_params:
                p.requires_grad = False
            optimizer = torch.optim.Adam(LBS_params, lr=args.lr)
        else:
            optimizer = torch.optim.Adam(
                [{'params': PCA_params},
                {'params': LBS_params},],
                lr = args.lr
            )

    #-------------------------------- ENABLE DATAPARALLEL --------------------------------#
    model.train()
    if args.dist_train:
        # model = SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            logger.info("Enabling Distributed Training...")
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % torch.cuda.device_count()], 
                                                    find_unused_parameters=True)
    
    #-------------------------------- ADD A WRITER --------------------------------#
    writer = None
    if args.local_rank == 0:
        logger.info("Building Writer...")
        writer = SummaryWriter(log_dir = os.path.join(args.output_dir, 'summary'))
    from utils.train_utils import merge_results, collect_decisions
    #-------------------------------- PACK AND RETURN --------------------------------#
    other_info = {
        'output_dir': args.output_dir,
        'ckpt_dir': ckpt_dir,
        'ckpt_fname': ckpt_fname,
        'body_model_male': body_model_male,
        'body_model_female': body_model_female,
    }
    return logger, train_dataloader, eval_dataloader, model, optimizer, body_model, \
           epoch, other_state, other_info, writer, scheduler

acc_list = [
    'total_loss_acc',
    'sem_seg_loss_acc',
    'garment_l2_loss_acc',
    'interpenetration_loss_acc',
    'garment_lap_loss_acc',
    'lbs_garment_l2_loss_acc',
    'lbs_garment_lap_loss_acc',
    'lbs_interpenetration_loss_acc',
    'garment_msre_acc',
    'lbs_garment_msre_acc',
    'garment_pca_coeff_l2_acc',
    'only_lbs_garment_msre_acc',
    'temporal_constraint_loss_acc',
    'acceleration_error_acc',
    'only_lbs_acceleration_error_acc'
]

def train_one_epoch_PCA(logger, dataloader, model, optimizer, body_model, writer, epoch, scheduler):
    np.random.seed()
    model.train()
    if args.fix_PCA:
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                # logger.info("Fixing BN: {}".format(classname))
                m.eval()
        model.module.PCA_garment_encoder.apply(set_bn_eval)
    if args.local_rank == 0:
        pbar = tqdm(total = len(dataloader), dynamic_ncols = True)
    acc_dict = {}
    for a in acc_list:
        acc_dict[a] = 0
    for i_iter, inputs in enumerate(dataloader):
        optimizer.zero_grad()
        output_dict = model(inputs['pcd_torch'].cuda(), body_model, inputs)
        if args.GarmentPCA:
            loss_dict = temporal_loss_PCA(output_dict, inputs, body_model, args)
        elif args.GarmentPCALBS:
            loss_dict = temporal_loss_PCA_LBS(output_dict, inputs, body_model, args)
        else:
            raise NotImplementedError
        total_loss = loss_dict['total_loss']

        if torch.isnan(total_loss):
            import pdb; pdb.set_trace()

        total_loss.backward()
        optimizer.step()
        if args.local_rank == 0:
            try:
                cur_lr = float(optimizer.lr)
            except:
                try:
                    cur_lr = optimizer.param_groups[1]['lr']
                except:
                    cur_lr = optimizer.param_groups[0]['lr']
            tqdm_log_dict = {'lr': cur_lr, 'l': total_loss.item()}
            if args.only_seg:
                tqdm_log_dict['seg'] = loss_dict['sem_seg_loss'].item()
            elif args.GarmentPCA:
                tqdm_log_dict['pca_msre'] = loss_dict['garment_msre'].item()
            if args.GarmentPCALBS:
                tqdm_log_dict['lbs_msre'] = loss_dict['lbs_garment_msre'].item()
                tqdm_log_dict['o_msre'] = loss_dict['only_lbs_garment_msre'].item()
            pbar.set_postfix(tqdm_log_dict)
            pbar.update(1)
            for i, (k, v) in enumerate(loss_dict.items()):
                try:
                    writer.add_scalar('Train/{}_{}'.format(str(i).zfill(2), k), v.item(), epoch * len(dataloader) + i_iter)
                except:
                    pass
            writer.add_scalar('LR', cur_lr, epoch * len(dataloader) + i_iter)
        for k, v in loss_dict.items():
            try:
                acc_dict[k + '_acc'] += v.item()
            except:
                pass
    if scheduler is not None:
        scheduler.step(acc_dict['total_loss_acc'] / len(dataloader))

    merged_dict = merge_results(acc_dict, os.path.join(args.output_dir, 'tmp'))

    if args.local_rank == 0:
        pbar.close()
        for k, v in merged_dict.items():
            if v == 0:
                continue
            lambda_k = k[:-4] + '_lambda'
            if lambda_k in args:
                logger.info("Average {}: {} * {}".format(k, v / len(dataloader), getattr(args, lambda_k)))
            else:
                logger.info("Average {}: {}".format(k, v / len(dataloader)))

def eval_one_epoch_PCA(logger, dataloader, model, body_model, writer, epoch):
    np.random.seed()
    model.eval()
    if args.local_rank == 0:
        pbar = tqdm(total = len(dataloader), dynamic_ncols = True)
    v_l2_loss_acc = 0

    acc_dict = {}
    for a in acc_list:
        acc_dict[a] = 0
    for i_iter, inputs in enumerate(dataloader):
        with torch.no_grad():
            output_dict = model(inputs['pcd_torch'].cuda(), body_model, inputs)
            # time_acc += output_dict['lbs_time']
            if args.GarmentPCA:
                loss_dict = temporal_loss_PCA(output_dict, inputs, body_model, args)
            elif args.GarmentPCALBS:
                loss_dict = temporal_loss_PCA_LBS(output_dict, inputs, body_model, args)
            else:
                raise NotImplementedError

            for k, v in loss_dict.items():
                try:
                    acc_dict[k + '_acc'] += v.item()
                except:
                    pass
            if args.GarmentPCA:
                if args.only_seg:
                    v_sqrt_l2_loss = loss_dict['sem_seg_loss']
                else:
                    v_sqrt_l2_loss = loss_dict['garment_msre']
            elif args.GarmentPCALBS:
                v_sqrt_l2_loss = loss_dict['lbs_garment_msre']
            else:
                raise NotImplementedError
        if args.local_rank == 0:
            if args.only_seg:
                pbar_postfix_dict = {
                    'sem_seg': loss_dict['sem_seg_loss'].item(),
                }
            elif args.GarmentPCA:
                pbar_postfix_dict = {
                    'pca_msre': loss_dict['garment_msre'].item(),
                }
            else:
                pbar_postfix_dict = {}
            if args.GarmentPCALBS:
                pbar_postfix_dict['lbs_msre'] = loss_dict['lbs_garment_msre'].item()
                pbar_postfix_dict['o_msre'] = loss_dict['only_lbs_garment_msre'].item()
            pbar.set_postfix(pbar_postfix_dict)
            pbar.update(1)
            writer.add_scalar('Eval/01_v_sqrt_l2_loss', v_sqrt_l2_loss.item(), epoch * len(dataloader) + i_iter)
        v_l2_loss_acc += v_sqrt_l2_loss.item()

    merged_dict = merge_results(acc_dict, os.path.join(args.output_dir, 'tmp'))
    if args.local_rank == 0:
        pbar.close()
        # logger.info("Average V L2 Loss: {}".format(v_l2_loss_acc / len(dataloader)))
        for k, v in merged_dict.items():
            if v == 0:
                continue
            lambda_k = k[:-4] + '_lambda'
            if lambda_k in args:
                logger.info("Average {}: {} * {}".format(k, v / len(dataloader), getattr(args, lambda_k)))
            else:
                logger.info("Average {}: {}".format(k, v / len(dataloader)))

        if args.GarmentPCA:
            if args.only_seg:
                return merged_dict['sem_seg_loss_acc'] / len(dataloader)
            else:
                return merged_dict['garment_msre_acc'] / len(dataloader)
        elif args.GarmentPCALBS:
            return merged_dict['lbs_garment_msre_acc'] / len(dataloader)
        else:
            raise NotImplementedError
    else:
        return None

def save_ckpt(logger, model, optimizer, epoch, other_state, ckpt_fname):
    if args.local_rank == 0:
        states = train_utils.checkpoint_state(model, optimizer, epoch, other_state)
        train_utils.save_checkpoint(states, ckpt_fname)
        logger.info("Saved ckpt to {}".format(ckpt_fname))

def main_PCA():
    logger, train_dataloader, eval_dataloader, model, optimizer, body_model, \
                  epoch, other_state, other_info, writer, scheduler = build()
    while(True):
        epoch += 1
        if epoch >= args.epoch_num:
            break
        if args.local_rank == 0:
            logger.info("TRAIN EPOCH {}".format(epoch))
        train_one_epoch_PCA(logger, train_dataloader, model, optimizer, body_model, writer, epoch, scheduler)
        if args.local_rank == 0:
            logger.info("FINISH TRAIN EPOCH {}".format(epoch))
            logger.info("This is {}".format(args.output_dir))
        if epoch % 1 == 0 or epoch == args.epoch_num - 1:
            if args.local_rank == 0:
                logger.info("EVAL EPOCH {}".format(epoch))
                curr_v_l2 = eval_one_epoch_PCA(logger, eval_dataloader, model, body_model, writer, epoch)
                logger.info("FINISH EVAL EPOCH {}".format(epoch))
                if curr_v_l2 < other_state['best_v_l2']:
                    other_state['best_v_l2'] = curr_v_l2
                    save_ckpt(logger, model, optimizer, epoch, other_state, other_info['ckpt_fname'])
            else:
                _ = eval_one_epoch_PCA(logger, eval_dataloader, model, body_model, writer, epoch)
    if args.local_rank == 0:
        logger.info("The best eval score: {}".format(other_state['best_v_l2']))

def main_PCA_eval():
    logger, train_dataloader, eval_dataloader, model, optimizer, body_model, \
                  epoch, other_state, other_info, writer, scheduler = build(dont_load_train=True)
    while(True):
        epoch += 1
        if args.local_rank == 0:
            logger.info("EVAL EPOCH {}".format(epoch))
            curr_v_l2 = eval_one_epoch_PCA(logger, eval_dataloader, model, body_model, writer, epoch)
            logger.info("FINISH EVAL EPOCH {}".format(epoch))
        else:
            _ = eval_one_epoch_PCA(logger, eval_dataloader, model, body_model, writer, epoch)
        break
    if args.local_rank == 0:
        logger.info("The best eval score: {}".format(other_state['best_v_l2']))

from utils.post_processing import process_single_frame
def eval_one_epoch_PCA_temporal_aggregation(logger, dataloader, model, body_model, writer, epoch):
    model.eval()
    if args.local_rank == 0:
        pbar = tqdm(total = len(dataloader), dynamic_ncols = True)
    v_l2_loss_acc = 0

    err_dict = {'MGN': args.MGN}

    for i_iter, inputs in enumerate(dataloader):
        with torch.no_grad():
            output_dict = model(inputs['pcd_torch'].cuda(), body_model, inputs)
            loss_dict = temporal_loss_PCA_LBS(output_dict, inputs, body_model, args)

            for ith in range(inputs['pose_np'].shape[0]):
                for frame in range(inputs['pose_np'].shape[1]):
                    process_single_frame(model, inputs, output_dict, ith, frame, body_model, save=True, post_process=False)

                    cur_seq = inputs['T_pcd_flist'][ith][frame].split('/')[-3]
                    cur_frame = inputs['T_pcd_flist'][ith][frame].split('/')[-2]
                    if cur_seq not in err_dict:
                        err_dict[cur_seq] = {}
                    err_dict[cur_seq][cur_frame] = loss_dict['lbs_garment_msre_list'][ith][frame].item()

        pbar.update(1)

if __name__ == '__main__':
    main_PCA()
