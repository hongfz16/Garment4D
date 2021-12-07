import torch
import os
import numpy as np
import pickle
import random
import shutil
import subprocess
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DistributedSampler as _DistributedSampler
from .config import args

class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def init_dist_slurm(batch_size, tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        batch_size:
        tcp_port:
        backend:
    Returns:
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    assert batch_size % total_gpus == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (batch_size, total_gpus)
    batch_size_each_gpu = batch_size // total_gpus
    rank = dist.get_rank()
    return batch_size_each_gpu, rank


def init_dist_pytorch(batch_size, tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    assert batch_size % num_gpus == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (batch_size, num_gpus)
    batch_size_each_gpu = batch_size // num_gpus
    rank = dist.get_rank()
    return batch_size_each_gpu, rank

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu

def checkpoint_state(model=None, optimizer=None, epoch=None, other_state=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state, 'other_state': other_state}

def save_checkpoint(state, filename):
    torch.save(state, filename)

def find_match_key(key, dic):
    # raise NotImplementedError
    for k in dic.keys():
        if '.'.join(k.split('.')[1:]) == key:
            return k
        if k == key:
            return k
    return None

def load_pretrained_model(model, filename, to_cpu=False, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError
    if args.local_rank == 0:
        logger.info('==> Loading parameters from pre-trained checkpoint {} to {}'.format(filename, 'CPU' if to_cpu else 'GPU'))
    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    if checkpoint.get('model_state', None) is not None:
        checkpoint = checkpoint.get('model_state')

    update_model_state = {}
    for key, val in checkpoint.items():
        match_key = find_match_key(key, model.state_dict())
        if match_key is None:
            print("Cannot find a matched key for {}".format(key))
            continue
        if model.state_dict()[match_key].shape == checkpoint[key].shape:
            update_model_state[match_key] = val
        else:
            print("Shape mis-match for {}".format(key))
            continue

    state_dict = model.state_dict()
    # state_dict.update(checkpoint)
    state_dict.update(update_model_state)
    model.load_state_dict(state_dict)

    if args.local_rank == 0:
        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

def load_params_with_optimizer(model, filename, to_cpu=False, optimizer=None, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    if args.local_rank == 0:
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    epoch = checkpoint.get('epoch', -1)

    model.load_state_dict(checkpoint['model_state'])

    if optimizer is not None:
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
            if args.local_rank == 0:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            except:
                if args.local_rank == 0:
                    logger.info('Optimizer could not be loaded.')

    if args.local_rank == 0:
        logger.info('==> Done')

    return epoch

def load_params_with_optimizer_otherstate(model, filename, to_cpu=False, optimizer=None, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    if args.local_rank == 0:
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
    loc_type = torch.device('cpu') if to_cpu else 'cuda:0'
    checkpoint = torch.load(filename, map_location=loc_type)
    epoch = checkpoint.get('epoch', -1)

    model.load_state_dict(checkpoint['model_state'])

    if optimizer is not None:
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
            if args.local_rank == 0:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            except:
                if args.local_rank == 0:
                    logger.info('Optimizer could not be loaded.')

    other_state = checkpoint.get('other_state', None)

    if args.local_rank == 0:
        logger.info('==> Done')

    return epoch, other_state

def merge_results(res_dict, tmp_dir):
    rank, world_size = get_dist_info()
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    dist.barrier()
    pickle.dump(res_dict, open(os.path.join(tmp_dir, 'res_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    def merge_from_dict(ori_dict, new_dict):
        for k in ori_dict.keys():
            ori_dict[k] += new_dict[k]
        return ori_dict

    for i in range(1, world_size):
        part_file = os.path.join(tmp_dir, 'res_part_{}.pkl'.format(i))
        res_dict = merge_from_dict(res_dict, pickle.load(open(part_file, 'rb')))
    
    for k in res_dict.keys():
        res_dict[k] /= world_size

    return res_dict

def collect_decisions(decision_file):
    dist.barrier()
    if not os.path.exists(decision_file):
        return None
    return pickle.load(open(decision_file, 'rb'))
