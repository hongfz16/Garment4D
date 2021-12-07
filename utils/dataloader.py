import os
import numpy as np
import torch
from torch.utils import data
import open3d as o3d
import pickle
from tqdm import tqdm

from smplx import build_layer
from smplx import parse_args, batch_rodrigues
from utils.mesh_utils import readOBJ
from utils import mesh_utils

from .config import args, cfg

label_dict = {
    'Body': 1,
    'Skirt': 2,
    'Dress': 3,
    'Jumpsuit': 4,
    'Top': 5,
    'Trousers': 6,
    'Tshirt': 7,
}
class_num = 7

def random_sample_pcd(pcd, n, retain_order=False):
    np.random.seed(0)
    if n == pcd.shape[0]:
        choice = np.arange(0, pcd.shape[0], dtype=np.int32)
    elif n < pcd.shape[0]:
        choice = np.random.choice(np.arange(0, pcd.shape[0], dtype=np.int32), n, replace=False)
        if retain_order:
            choice = np.sort(choice)
    else:
        try:
            choice = np.concatenate([
                np.arange(0, pcd.shape[0], dtype=np.int32),
                np.random.choice(np.arange(0, pcd.shape[0], dtype=np.int32), n - pcd.shape[0], replace=False)
            ], axis=0)
        except:
            choice = np.concatenate([
                np.arange(0, pcd.shape[0], dtype=np.int32),
                np.random.choice(np.arange(0, pcd.shape[0], dtype=np.int32), n - pcd.shape[0], replace=True)
            ], axis=0)
    if not retain_order:
        np.random.shuffle(choice)
    pcd = pcd[choice, :]
    return pcd, choice

class SeqPointSMPLDataset(data.Dataset):
    def __init__(self, npoints, data_f_list, smpl_param_prefix, T, is_train=True, garment_template_prefix=None, body_model_dict=None):
        self.npoints = npoints
        self.T = T
        self.garment_template_prefix = garment_template_prefix
        self.body_model_male = body_model_dict['male']
        self.body_model_female = body_model_dict['female']

        with open(data_f_list, 'r') as f:
            self.model_list = f.read().splitlines()

        self.exclude_seq_list = None
        if len(cfg.DATASET.EXCLUDE_SEQ_LIST) > 0:
            with open(cfg.DATASET.EXCLUDE_SEQ_LIST, 'r') as f:
                self.exclude_seq_list = f.read().splitlines()
                self.exclude_seq_list = [l.rstrip() for l in self.exclude_seq_list]

        seq_model_list = []
        last_pref = None
        for n in self.model_list:
            pref = n.split('/')[0]
            if self.exclude_seq_list is not None:
                if pref in self.exclude_seq_list:
                    continue
            if pref != last_pref:
                last_pref = pref
                seq_model_list.append([])
            seq_model_list[-1].append(n)
        self.T_list = []
        for seq in seq_model_list:
            if len(seq) < self.T:
                continue
            sample_time = int(len(seq) / (self.T)) + 1
            max_skip_num = min(int(len(seq) / self.T), 5)
            for i in range(sample_time):
                if is_train:
                    i_skip_list = [np.random.randint(1, max_skip_num + 1) for _ in range(self.T - 1)]
                    i_start = np.random.randint(0, len(seq) - sum(i_skip_list))
                    i_seq = [seq[i_start]]
                    acc = i_start
                    for j in range(self.T - 1):
                        acc += i_skip_list[j]
                        i_seq.append(seq[acc])
                    self.T_list.append(i_seq)
                else:
                    i_start = i * (self.T)
                    new_seq = seq[i_start : i_start + self.T]
                    if len(new_seq) < self.T:
                        new_seq += [seq[-1]] * (self.T - len(new_seq))
                    self.T_list.append(new_seq)

        self.T_smpl_param_flist = []
        self.T_garment_flist = []
        self.garment_template_flist = []
        for l in self.T_list:
            self.T_smpl_param_flist.append([os.path.join(smpl_param_prefix, s, 'smpl_param.pkl') for s in l])

            self.T_garment_flist.append(
                [
                    [
                        os.path.join(smpl_param_prefix, s, n.rstrip())
                        for n in open(os.path.join(smpl_param_prefix, s, 'garment_flist.txt'), 'r').readlines()
                    ]
                    for s in l
                ])
            self.garment_template_flist.append(os.path.join(self.garment_template_prefix, l[0].split('/')[0], '{}.obj'.format(cfg.GARMENT.NAME))) # hard coding

        self.remesh_garment_folder = cfg.DATASET.GARMENT_FOLDER
        
    def __len__(self):
        return len(self.T_list)

    def __getitem__(self, index):
        i_T_smpl_param_flist = self.T_smpl_param_flist[index]
        T_pose_list = []
        T_shape_list = []
        T_pcd_list = []
        T_garment_list = []
        T_ori_garment_list = []
        T_pcd_label_list = []
        T_smpl_vertices = []
        T_smpl_root_joints = []
        T_Tpose_smpl_vertices = []
        T_Tpose_smpl_root_joints = []
        T_zeropose_smpl_vertices = []
        T_J_regressor = []
        T_lbs_weights = []

        augmentation_info = {}

        #-------------- LOAD METADATA ---------------#
        with open(os.path.join('/'.join(i_T_smpl_param_flist[0].split('/')[:-1]), '../gender.pkl'), 'rb') as f:
            gender = pickle.load(f)
        with open('{}/{}/{}/PCACoeff_SS.pkl'.format(cfg.DATASET.ROOT_FOLDER, self.remesh_garment_folder, i_T_smpl_param_flist[0].split('/')[-3]), 'rb') as f:
            cur_PCACoeff = pickle.load(f)[:cfg.GARMENT.PCADIM]
        with open('{}/{}/{}/remesh_weights.pkl'.format(cfg.DATASET.ROOT_FOLDER, self.remesh_garment_folder, i_T_smpl_param_flist[0].split('/')[-3]), 'rb') as f:
            remesh_weights = pickle.load(f)
            remesh_ind = []
            remesh_w = []
            for r in remesh_weights:
                remesh_ind.append(r['ind'])
                remesh_w.append(r['w'])
            remesh_ind = np.stack(remesh_ind)
            remesh_w = np.stack(remesh_w)

        for i, spf in enumerate(i_T_smpl_param_flist):
            #-------------- LOAD SMPL ---------------#
            with open(spf, 'rb') as f:
                smpl_param_dict = pickle.load(f)
            pose_np = smpl_param_dict['pose'].astype(np.float32)
            shape_np = smpl_param_dict['shape'].astype(np.float32)
            T_pose_list.append(pose_np)
            T_shape_list.append(shape_np)

            inv_zrot = smpl_param_dict['zrot']
            zc, zs = np.cos(inv_zrot), np.sin(inv_zrot)
            inv_zrot_mat = np.array([[zc, -zs, 0],
                                     [zs,  zc, 0],
                                     [ 0,   0, 1]], np.float32)

            #-------------- LOAD GARMENT ---------------#
            cur_garment_flist = self.T_garment_flist[index][i]
            garment_vertices_list = []
            garment_label_list = []
            garment_vertices_dict = {}
            for garment_fname in cur_garment_flist:
                garment_vertices_list.append(np.fromfile(garment_fname, dtype=np.float32).reshape(-1, 3))
                garment_label_list.append(np.zeros(garment_vertices_list[-1].shape[0]) + label_dict[garment_fname.split('/')[-1][:-5]])
                garment_vertices_dict[garment_fname.split('/')[-1][:-5]] = garment_vertices_list[-1]
            
            ori_garment_vertices_no_sample = np.concatenate(garment_vertices_list, 0)
            ori_garment_label_no_sample = np.concatenate(garment_label_list, 0)
            ori_garment_vertices_no_sample = np.matmul(ori_garment_vertices_no_sample, inv_zrot_mat)
            ori_garment_vertices, garment_sample_ind = random_sample_pcd(ori_garment_vertices_no_sample, self.npoints)
            ori_garment_label = ori_garment_label_no_sample[garment_sample_ind]

            #-------------- Construct SMPL MODEL ---------------#
            batch_size = 1
            rot_poses = torch.from_numpy(pose_np).reshape(-1, 3)
            rot_poses = batch_rodrigues(rot_poses).view(batch_size, 24, 3, 3)
            beta = torch.from_numpy(shape_np).reshape(batch_size, 10)
            zero_pose = torch.zeros([batch_size, 72]).reshape(batch_size, 24, 3)
            zero_pose_rot = batch_rodrigues(zero_pose.reshape(-1, 3)).reshape(1, 24, 3, 3)
            T_pose = torch.zeros([batch_size, 72]).reshape(batch_size, 24, 3)
            T_pose[:, 0, 0] = np.pi / 2
            T_pose[:, 1, 2] = 0.15
            T_pose[:, 2, 2] = -0.15
            T_pose = T_pose.reshape(batch_size, 72)
            T_pose_rot = batch_rodrigues(T_pose.reshape(-1, 3)).reshape(1, 24, 3, 3)
            if gender == 0:
                assert self.body_model_female is not None
                so = self.body_model_female(betas = beta, body_pose = rot_poses[:, 1:], global_orient=rot_poses[:, 0, :, :].view(batch_size, 1, 3, 3))
                Tpose_so = self.body_model_female(betas = beta, body_pose = T_pose_rot[:, 1:], global_orient=T_pose_rot[:, 0, :, :].view(batch_size, 1, 3, 3))
                zeropose_so = self.body_model_female(betas = beta, body_pose = zero_pose_rot[:, 1:], global_orient=zero_pose_rot[:, 0, :, :].view(batch_size, 1, 3, 3))
                T_J_regressor.append(self.body_model_female.J_regressor)
                T_lbs_weights.append(self.body_model_female.lbs_weights)
            elif gender == 1:
                assert self.body_model_male is not None
                so = self.body_model_male(betas = beta, body_pose = rot_poses[:, 1:], global_orient=rot_poses[:, 0, :, :].view(batch_size, 1, 3, 3))
                Tpose_so = self.body_model_male(betas = beta, body_pose = T_pose_rot[:, 1:], global_orient=T_pose_rot[:, 0, :, :].view(batch_size, 1, 3, 3))
                zeropose_so = self.body_model_male(betas = beta, body_pose = zero_pose_rot[:, 1:], global_orient=zero_pose_rot[:, 0, :, :].view(batch_size, 1, 3, 3))
                T_J_regressor.append(self.body_model_male.J_regressor)
                T_lbs_weights.append(self.body_model_male.lbs_weights)
            else:
                raise NotImplementedError
            pcd_np = so['vertices'][0].numpy().reshape(-1, 3)
            choice = np.arange(0, pcd_np.shape[0], dtype=np.int32)
            np.random.shuffle(choice)
            pcd_np = pcd_np[choice]

            garment_vertices = ori_garment_vertices + so['joints'][0].numpy().reshape(-1, 3)[0]
            garment_vertices, garment_sample_ind = random_sample_pcd(garment_vertices, self.npoints // 2)
            garment_label = ori_garment_label[garment_sample_ind]

            pcd_label_np = np.zeros([pcd_np.shape[0] + garment_vertices.shape[0], 1], dtype=np.int32)
            pcd_label_np[:pcd_np.shape[0]] = 1 # body:  1
            pcd_label_np[pcd_np.shape[0]:, 0] = garment_label # cloth: other
            pcd_np = np.concatenate([pcd_np, garment_vertices], 0)
            
            pcd_np, choice = random_sample_pcd(pcd_np, self.npoints)
            pcd_label_np = pcd_label_np[choice, :]

            assert pcd_np.shape[0] == self.npoints

            T_pcd_list.append(pcd_np)
            T_pcd_label_list.append(pcd_label_np)
            
            cur_garment_vertices = garment_vertices_dict[cfg.GARMENT.NAME]
            cur_garment_vertices = np.matmul(cur_garment_vertices, inv_zrot_mat)
            cur_garment_vertices = mesh_utils.interpolateBarycentricCoorNumpy(cur_garment_vertices, remesh_ind, remesh_w)
            T_garment_list.append(cur_garment_vertices)

            T_smpl_vertices.append(so['vertices'].reshape(-1, 3))
            T_smpl_root_joints.append(so['joints'][0, 0, :].reshape(3))
            T_Tpose_smpl_vertices.append(Tpose_so['vertices'].reshape(-1, 3))
            T_Tpose_smpl_root_joints.append(Tpose_so['joints'][0, 0, :].reshape(3))
            T_zeropose_smpl_vertices.append(zeropose_so['vertices'].reshape(-1, 3))

        #-------------- LOAD GARMENT TEMPLATE ---------------#
        garment_template_fname = self.garment_template_flist[index]
        garment_template_vertices, garment_template_F, _, _ = readOBJ(garment_template_fname)
        garment_template_vertices = np.matmul(garment_template_vertices, inv_zrot_mat)
        garment_template_vertices = mesh_utils.interpolateBarycentricCoorNumpy(garment_template_vertices, remesh_ind, remesh_w)

        #-------------- COLLECT ALL THE DATA ---------------#
        data_tuple = (np.stack(T_pose_list), np.stack(T_shape_list), np.stack(T_pcd_list), augmentation_info,
                    np.stack(T_pcd_label_list), np.stack(T_garment_list), garment_template_vertices)
        data_tuple += (torch.stack(T_smpl_vertices), torch.stack(T_smpl_root_joints), T_Tpose_smpl_vertices[0],
                    T_Tpose_smpl_root_joints[0], torch.stack(T_zeropose_smpl_vertices), torch.stack(T_J_regressor),
                    torch.stack(T_lbs_weights), cur_PCACoeff)
        return data_tuple

def SeqPointSMPL_collate_fn(data):
    pose_np = np.stack([d[0] for d in data]).astype(np.float32)
    beta_np = np.stack([d[1] for d in data]).astype(np.float32)
    pcd_np = np.stack([d[2] for d in data]).astype(np.float32)
    augmentation_info = [d[3] for d in data]
    if len(augmentation_info[0].keys()) > 0:
        for i, _ in enumerate(augmentation_info):
            augmentation_info[i]['rotation_mat'] = torch.from_numpy(augmentation_info[i]['rotation_mat'])
    return_dict = {
        'pose_torch': torch.from_numpy(pose_np), ##
        'beta_torch': torch.from_numpy(beta_np),
        'pcd_torch': torch.from_numpy(pcd_np), ##
        'pose_np': pose_np,
        'beta_np': beta_np,
        'pcd_np': pcd_np,
        'augmentation_info': augmentation_info,
    }
    pcd_label_np = np.stack([d[4] for d in data]).astype(np.int32) - 1
    return_dict['pcd_label_torch'] = torch.from_numpy(pcd_label_np) ##
    return_dict['pcd_label_np'] = pcd_label_np
    garment_np = np.stack([d[5] for d in data]).astype(np.float32)
    return_dict['garment_torch'] = torch.from_numpy(garment_np) ##
    return_dict['garment_np'] = garment_np
    return_dict['garment_template_vertices'] = torch.from_numpy(np.stack([d[6] for d in data]).astype(np.float32)) ##
    return_dict['smpl_vertices_torch'] = torch.stack([d[7] for d in data]) ##
    return_dict['smpl_root_joints_torch'] = torch.stack([d[8] for d in data]) ##
    return_dict['Tpose_smpl_vertices_torch'] = torch.stack([d[9] for d in data]) ##
    return_dict['Tpose_smpl_root_joints_torch'] = torch.stack([d[10] for d in data]) ##
    return_dict['zeropose_smpl_vertices_torch'] = torch.stack([d[11] for d in data]) ##
    return_dict['T_J_regressor'] = torch.stack([d[12] for d in data]) ##
    return_dict['T_lbs_weights'] = torch.stack([d[13] for d in data]) ##
    return_dict['PCACoeff'] = torch.from_numpy(np.stack([d[14] for d in data])) ##
    return return_dict
    