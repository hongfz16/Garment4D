import numpy as np
import torch
from torch import nn
import pickle
import torch.nn.functional as F
from .pointnet2encoder import Pointnet2MSGSEG
import sys
sys.path.append('../')
from utils import mesh_utils
from smplx import batch_rodrigues
from .pygcn import layers
from .pygcn import utils as gcn_utils
import scipy.sparse as sp
from .pointnet2.pointnet2.pointnet2_utils import three_interpolate, three_nn, ball_query, grouping_operation, QueryAndGroup
from .pointnet2.pointnet2.pointnet2_modules import PointnetSAModuleMSG, PointnetSAModule

from smplx.smplx.lbs import batch_rigid_transform, vertices2joints, vertices2jointsB
from chamferdist import knn_points

import time

from utils.dataloader import label_dict, class_num

def quads2tris(F):
    out = []
    for f in F:
        if len(f) == 3: out += [f]
        elif len(f) == 4: out += [[f[0],f[1],f[2]],
                                [f[0],f[2],f[3]]]
        else: print("This should not happen...")
    return np.array(out, np.int32)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class PCAGarmentEncoderSeg(nn.Module):
    def __init__(self, cfg=None, args=None):
        super(PCAGarmentEncoderSeg, self).__init__()
        self.cfg = cfg
        self.args = args

        self.pointnet = Pointnet2MSGSEG(input_channels=0, bn=True, global_feat=False)

        if self.args.only_seg:
            return

        self.GarmentEncoder = nn.ModuleList()
        self.GarmentEncoder.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[self.pointnet.feat_channels_list[0], 32, 32], [self.pointnet.feat_channels_list[0], 64, 64]],
                use_xyz=True,
                bn=True
            )
        )
        self.GarmentEncoder.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[32, 64],
                mlps=[[32+64, 128, 128], [32+64, 256, 256]],
                use_xyz=True,
                bn=True
            )
        )
        self.GarmentSummarize = PointnetSAModule(
            mlp=[128+256, 512, 512], use_xyz=True,
            bn=True
        )
        self.PCAEncoder = nn.Sequential(
            nn.Conv1d(512, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1)
        )

        with open(self.cfg.GARMENT.PCACOMPONENTSFILE, 'rb') as fd:
            PCA_pkl = pickle.load(fd)
            self.PCA_comp = torch.from_numpy(PCA_pkl['components'][:self.cfg.GARMENT.PCADIM])
            self.PCA_mean = torch.from_numpy(PCA_pkl['mean'])
            self.PCA_expl = torch.from_numpy(PCA_pkl['explained'][:self.cfg.GARMENT.PCADIM])
            self.PCA_scale = torch.from_numpy(PCA_pkl['ss_scale'].astype(np.float32))

        self.remesh_cylinder_v, self.remesh_cylinder_f, _, _ = mesh_utils.readOBJ(self.cfg.GARMENT.TEMPLATE)
        self.remesh_cylinder_f = np.array(list(self.remesh_cylinder_f))
        self.garment_f_3 = quads2tris(self.remesh_cylinder_f).astype(np.int32)
        self.garment_v_num = self.remesh_cylinder_v.shape[0]

    def PCA_inverse_transform(self, coeff):
        assert coeff.shape[1] == self.cfg.GARMENT.PCADIM
        self.PCA_comp = self.PCA_comp.cuda()
        self.PCA_mean = self.PCA_mean.cuda()
        self.PCA_expl = self.PCA_expl.cuda()
        self.PCA_scale = self.PCA_scale.cuda()
        return ((torch.mm(coeff, self.PCA_comp) + self.PCA_mean) * self.PCA_scale).reshape(coeff.shape[0], -1, 3)

    def calc_segmentation_results(self, x, sem_logits, n, nbatch, T, feature):
        x = x.reshape(nbatch * T, -1, 3)
        feature = feature.transpose(1, 2).reshape(nbatch * T, 6890, feature.shape[-2])
        sem_logits = sem_logits.reshape(nbatch * T, -1, class_num)
        labels = torch.argmax(sem_logits, dim=2).detach()
        garment_v = []
        feat = []
        for i in range(nbatch * T):
            cur_x = x[i][labels[i]==(label_dict[self.cfg.GARMENT.NAME]-1), :]
            cur_f = feature[i][labels[i]==(label_dict[self.cfg.GARMENT.NAME]-1), :]
            if cur_x.shape[0] >= n:
                garment_v.append(cur_x[:n, :])
                feat.append(cur_f[:n, :])
            else:
                garment_v.append(torch.cat([cur_x, torch.zeros(n-cur_x.shape[0], 3, dtype=torch.float32).cuda()]))
                feat.append(torch.cat([cur_f, torch.zeros(n-cur_x.shape[0], feature.shape[-1], dtype=torch.float32).cuda()]))
        return torch.stack(garment_v), torch.stack(feat)

    def forward(self, x, body_model=None, batch=None):
        assert(x.size()[-1] >= 3)
        nbatch= x.size()[0]
        T = x.size()[1]
        N = x.size()[2]
        JN = 24
        x = x.view(nbatch * T, N, -1)
        output_dict = {}
        output_dict['middle_results'] = {}

        assert body_model is not None

        feat_global, sem_logits, feature_list, xyz_list = self.pointnet(x)

        output_dict['feat_global'] = feat_global
        output_dict['feature_list'] = feature_list
        output_dict['xyz_list'] = xyz_list
        output_dict['sem_logits'] = sem_logits

        if self.args.only_seg:
            return output_dict

        garment_v, garment_f = self.calc_segmentation_results(xyz_list[0], sem_logits, N // 4, nbatch, T, feature_list[0])
        garment_v = garment_v.reshape(nbatch * T, -1, 3)
        garment_f = garment_f.reshape(nbatch * T, -1, garment_f.shape[-1]).transpose(1, 2).contiguous()
        l_xyz, l_features = [garment_v], [garment_f]
        for i in range(len(self.GarmentEncoder)):
            li_xyz, li_features = self.GarmentEncoder[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        output_dict['garment_v_list'] = l_xyz
        output_dict['garment_f_list'] = l_features
        garment_summary = self.GarmentSummarize(l_xyz[-1], l_features[-1])[1].reshape(nbatch, T, 512)
        output_dict['garment_summary'] = garment_summary
        output_dict['garment_PCA_coeff'] = self.PCAEncoder(garment_summary.max(1)[0].reshape(nbatch, -1, 1)).reshape(nbatch, self.cfg.GARMENT.PCADIM)
        output_dict['tpose_garment'] = self.PCA_inverse_transform(output_dict['garment_PCA_coeff'])
        output_dict['garment_f_3'] = self.garment_f_3
        output_dict['PCABase'] = {
            'components': self.PCA_comp,
            'mean': self.PCA_mean,
            'explained': self.PCA_expl,
        }
        return output_dict


class PCALBSGarmentUseSegEncoderSeg(nn.Module):
    def __init__(self, cfg=None, args=None):
        super(PCALBSGarmentUseSegEncoderSeg, self).__init__()
        self.cfg = cfg
        self.args = args
        self.PCA_garment_encoder = PCAGarmentEncoderSeg(self.cfg, self.args)

        self.garment_radius_list = [0.1, 0.2, 0.4]
        self.garment_sample_num_list = [32, 16, 8]
        self.body_radius_list = [0.1, 0.2, 0.4]
        self.body_sample_num_list = [8, 16, 32]

        if self.cfg.GARMENT.NAME == 'Trousers':
            self.garment_radius_list = [0.1, 0.2, 0.4]
            self.garment_sample_num_list = [32, 8, 4]
            self.body_radius_list = [0.1, 0.2, 0.4]
            self.body_sample_num_list = [8, 16, 32]

        self.lbs_positional_encoding_dim = 3

        self.feat_num = 32 # positional encoding
        self.hidden_dim = 128 # GCN hidden dim
        self.graph_start_feature_dim = self.feat_num * 6 + self.lbs_positional_encoding_dim
        self.feat_num_output = self.feat_num

        self.body_query_group0 = QueryAndGroup(radius=self.body_radius_list[0], nsample=self.body_sample_num_list[0], use_xyz=True)
        self.body_query_group1 = QueryAndGroup(radius=self.body_radius_list[1], nsample=self.body_sample_num_list[1], use_xyz=True)
        self.body_query_group2 = QueryAndGroup(radius=self.body_radius_list[2], nsample=self.body_sample_num_list[2], use_xyz=True)
        self.body_query_group_list = [
            self.body_query_group0,
            self.body_query_group1,
            self.body_query_group2,
        ]
        self.body_positional_encoding0 = nn.Sequential(
            nn.Linear(6, self.feat_num),
            nn.ReLU(),
            nn.Linear(self.feat_num, self.feat_num_output),
        )
        self.body_positional_encoding1 = nn.Sequential(
            nn.Linear(6, self.feat_num),
            nn.ReLU(),
            nn.Linear(self.feat_num, self.feat_num_output),
        )
        self.body_positional_encoding2 = nn.Sequential(
            nn.Linear(6, self.feat_num),
            nn.ReLU(),
            nn.Linear(self.feat_num, self.feat_num_output),
        )
        self.body_positional_encoding_list = [
            self.body_positional_encoding0,
            self.body_positional_encoding1,
            self.body_positional_encoding2,
        ]

        self.garment_query_group0 = QueryAndGroup(radius=self.garment_radius_list[0], nsample=self.garment_sample_num_list[0], use_xyz=True)
        self.garment_query_group1 = QueryAndGroup(radius=self.garment_radius_list[1], nsample=self.garment_sample_num_list[1], use_xyz=True)
        self.garment_query_group2 = QueryAndGroup(radius=self.garment_radius_list[2], nsample=self.garment_sample_num_list[2], use_xyz=True)
        self.garment_query_group_list = [
            self.garment_query_group0,
            self.garment_query_group1,
            self.garment_query_group2,
        ]
        self.garment_positional_encoding_input_dim = [
            3 + 64,
            3 + 32 + 64,
            3 + 128 + 256,
        ]
        self.garment_positional_encoding0 = nn.Sequential(
            nn.Linear(self.garment_positional_encoding_input_dim[0], self.feat_num),
            nn.ReLU(),
            nn.Linear(self.feat_num, self.feat_num_output),
        )
        self.garment_positional_encoding1 = nn.Sequential(
            nn.Linear(self.garment_positional_encoding_input_dim[1], self.feat_num),
            nn.ReLU(),
            nn.Linear(self.feat_num, self.feat_num_output),
        )
        self.garment_positional_encoding2 = nn.Sequential(
            nn.Linear(self.garment_positional_encoding_input_dim[2], self.feat_num),
            nn.ReLU(),
            nn.Linear(self.feat_num, self.feat_num_output),
        )
        self.garment_positional_encoding_list = [
            self.garment_positional_encoding0,
            self.garment_positional_encoding1,
            self.garment_positional_encoding2,
        ]
    
        self.temporal_qkv_1 = nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=False)
        self.temporal_qkv_2 = nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=False)
        self.temporal_qkv_list = [
            self.temporal_qkv_1,
            self.temporal_qkv_2,
        ]

        self.lbs_graph_regress1 = nn.ModuleList([
            layers.GraphConvolution(self.graph_start_feature_dim, self.hidden_dim),
            layers.GraphConvolution(self.hidden_dim, self.hidden_dim),
            layers.GraphConvolution(self.hidden_dim, self.hidden_dim),
            layers.GraphConvolution(self.hidden_dim, 3),
        ])
        self.lbs_graph_regress2 = nn.ModuleList([
            layers.GraphConvolution(self.graph_start_feature_dim+self.hidden_dim, self.hidden_dim),
            layers.GraphConvolution(self.hidden_dim, self.hidden_dim),
            layers.GraphConvolution(self.hidden_dim, self.hidden_dim),
            layers.GraphConvolution(self.hidden_dim, 3),
        ])
        self.lbs_graph_regress3 = nn.ModuleList([
            layers.GraphConvolution(self.graph_start_feature_dim+self.hidden_dim, self.hidden_dim),
            layers.GraphConvolution(self.hidden_dim, self.hidden_dim),
            layers.GraphConvolution(self.hidden_dim, self.hidden_dim),
            layers.GraphConvolution(self.hidden_dim, 3),
        ])

        self.remesh_cylinder_f = self.PCA_garment_encoder.remesh_cylinder_f
        self.lbs_graph_regress = [self.lbs_graph_regress1, self.lbs_graph_regress2, self.lbs_graph_regress3]
        edges = np.zeros([2, self.remesh_cylinder_f.shape[0] * 4], dtype=np.int32)
        for i, f in enumerate(self.remesh_cylinder_f):
            if len(f) == 4:
                edges[:, i * 4 + 0] = np.array([f[0], f[1]], dtype=np.int32)
                edges[:, i * 4 + 1] = np.array([f[1], f[2]], dtype=np.int32)
                edges[:, i * 4 + 2] = np.array([f[2], f[3]], dtype=np.int32)
                edges[:, i * 4 + 3] = np.array([f[3], f[0]], dtype=np.int32)
            elif len(f) == 3:
                edges[:, i * 4 + 0] = np.array([f[0], f[1]], dtype=np.int32)
                edges[:, i * 4 + 1] = np.array([f[1], f[2]], dtype=np.int32)
                edges[:, i * 4 + 3] = np.array([f[2], f[0]], dtype=np.int32)
            else:
                raise NotImplementedError
        self.adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0, :], edges[1, :])),
                                shape=(self.PCA_garment_encoder.garment_v_num, self.PCA_garment_encoder.garment_v_num),
                                dtype=np.float32)
        self.adj = self.adj + self.adj.T.multiply(self.adj.T > self.adj) - self.adj.multiply(self.adj.T > self.adj)
        self.adj_old = self.adj.copy()
        self.adj = gcn_utils.normalize(self.adj + sp.eye(self.adj.shape[0]))
        self.adj = gcn_utils.sparse_mx_to_torch_sparse_tensor(self.adj).cuda()

        self.vf_fid = None
        self.vf_vid = None

    def lbs_garment_interpolation(self, pred_template_garment_v, Tpose_vertices, Tpose_root_joints, zeropose_vertices, body_model, gt_pose, T_J_regressor, T_lbs_weights, K=3):
        assert len(pred_template_garment_v.shape) == 3 and pred_template_garment_v.shape[2] == 3
        assert len(gt_pose.shape) == 3 and gt_pose.shape[2] == 72
        
        batch_size = pred_template_garment_v.shape[0]
        seq_length = gt_pose.shape[1]
        gt_pose_mat = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(batch_size * seq_length, 24, 3, 3)

        root_joint_pred_template_garment_v = pred_template_garment_v + Tpose_root_joints.reshape(batch_size, 3).unsqueeze(1)
        nnk = knn_points(root_joint_pred_template_garment_v, Tpose_vertices.reshape(batch_size, -1, 3), K=K)
        K64 = min(64, K)
        nn64 = knn_points(root_joint_pred_template_garment_v, Tpose_vertices.reshape(batch_size, -1, 3), K=K64)
        nn = knn_points(root_joint_pred_template_garment_v, Tpose_vertices.reshape(batch_size, -1, 3))

        inv_template_pose = torch.zeros([batch_size, 24, 3]).cuda()
        inv_template_pose[:, 0, 0] = -np.pi / 2
        inv_template_pose[:, 1, 1] = 0.15
        inv_template_pose[:, 2, 1] = -0.15
        inv_template_pose_mat = batch_rodrigues(inv_template_pose.reshape(-1, 3)).reshape(batch_size, 24, 3, 3)
        device, dtype = inv_template_pose.device, inv_template_pose.dtype
        
        inv_J = vertices2jointsB(T_J_regressor[:, 0, :, :].reshape(batch_size, T_J_regressor.shape[2], T_J_regressor.shape[3]),
                                 Tpose_vertices.reshape(batch_size, -1, 3))
        _, inv_A = batch_rigid_transform(inv_template_pose_mat, inv_J, body_model.parents, dtype=dtype)

        ##### INTERPOLATION
        num_joints = body_model.J_regressor.shape[0]
        inv_W = T_lbs_weights[:, 0, :, :].reshape(batch_size, -1, 1, num_joints).repeat(1, 1, K64, 1) # batch_size x num_body_v x K x num_joints
        inv_nn_W = torch.gather(inv_W, 1, nn64.idx.reshape(batch_size, -1, K64, 1).repeat(1, 1, 1, num_joints)) # batch_size x num_garment_v x K x num_joints
        interp_weights64 = 1 / nn64.dists.reshape(batch_size, -1, K64, 1)
        # interp_weights64 = torch.zeros_like(nnk.dists.reshape(batch_size, -1, K, 1)) + 1
        interp_weights64[torch.where(torch.isinf(interp_weights64))] = 0
        interp_weights64 = interp_weights64 / interp_weights64.sum(-2, keepdim=True)
        interp_weights64[torch.where(torch.isinf(interp_weights64))] = 0 # batch_size x num_garment_v x K x 1
        inv_nn_W = (inv_nn_W * interp_weights64).sum(-2) # batch_size x num_garment_v x num_joints
        inv_nn_T = torch.matmul(inv_nn_W, inv_A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
        del inv_W
        del interp_weights64
        del inv_nn_W
        ######

        ###### ORIGINAL
        # inv_W = T_lbs_weights[:, 0, :, :]
        # num_joints = body_model.J_regressor.shape[0]
        # inv_T = torch.matmul(inv_W, inv_A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
        # inv_nn_T = torch.gather(inv_T, 1, nn.idx.reshape(batch_size, -1, 1, 1).repeat(1, 1, inv_T.shape[2], inv_T.shape[3]))
        ######
        
        inv_homogen_coord = torch.ones([batch_size, root_joint_pred_template_garment_v.shape[1], 1], dtype=dtype, device=device)
        inv_v_posed_homo = torch.cat([root_joint_pred_template_garment_v, inv_homogen_coord], dim=2)
        inv_v_homo = torch.matmul(inv_nn_T, torch.unsqueeze(inv_v_posed_homo, dim=-1))
        inv_template_garment_v = inv_v_homo[:, :, :3, 0].reshape(batch_size, 1, -1, 3).repeat(1, seq_length, 1, 1).reshape(batch_size * seq_length, -1, 3)

        if torch.any(torch.isnan(inv_template_garment_v)):
            import pdb; pdb.set_trace()

        zero_pose_pred_shape_v = zeropose_vertices.reshape(batch_size * seq_length, -1, 3)
        J = vertices2jointsB(T_J_regressor.reshape(batch_size * seq_length, T_J_regressor.shape[2], T_J_regressor.shape[3]),
                             zero_pose_pred_shape_v)
        _, A = batch_rigid_transform(gt_pose_mat, J, body_model.parents, dtype=dtype)

        ###### INTERPOLATE
        interp_weights = 1 / nnk.dists.reshape(batch_size, -1, K, 1)
        # interp_weights = torch.zeros_like(nnk.dists.reshape(batch_size, -1, K, 1)) + 1
        interp_weights[torch.where(torch.isinf(interp_weights))] = 0
        interp_weights = interp_weights / interp_weights.sum(-2, keepdim=True)
        interp_weights[torch.where(torch.isinf(interp_weights))] = 0 # batch_size x num_garment_v x K x 1

        W = T_lbs_weights.reshape(batch_size * seq_length, -1, 1, num_joints).repeat(1, 1, K, 1) # batch_size*seq_length x num_body_v x K x num_joints
        nn_W = torch.gather(W, 1, nnk.idx.reshape(batch_size, 1, -1, K, 1).repeat(1, seq_length, 1, 1, num_joints).reshape(batch_size * seq_length, -1, K, num_joints)) # batch_size*seq_length x num_garment_v x K x num_joints
        nn_W = (nn_W * interp_weights.reshape(batch_size, 1, -1, K , 1).repeat(1, seq_length, 1, 1, 1).reshape(batch_size * seq_length, -1, K, 1)).sum(-2) # batch_size x num_garment_v x num_joints
        
        #### SMOOTH THE WEIGHTS
        if K > 1:
            adj = gcn_utils.normalize(self.adj_old) - sp.eye(self.adj_old.shape[0])
            adj = gcn_utils.sparse_mx_to_torch_sparse_tensor(adj).cuda()
            coeff = 0.1
            for it in range(100):
                nn_W = nn_W + coeff * torch.spmm(adj, nn_W.transpose(0, 1).reshape(-1, batch_size * seq_length * num_joints)).reshape(-1, batch_size * seq_length, num_joints).transpose(0, 1)
        ####

        nn_T = torch.matmul(nn_W, A.view(batch_size * seq_length, num_joints, 16)).view(batch_size * seq_length, -1, 4, 4)
        del nn_W
        del W
        del interp_weights
        ######

        ###### ORIGINAL
        # W = T_lbs_weights.reshape(batch_size * seq_length, T_lbs_weights.shape[2], T_lbs_weights.shape[3])
        # T = torch.matmul(W, A.view(batch_size * seq_length, num_joints, 16)).view(batch_size * seq_length, -1, 4, 4)
        # repeated_nn_idx = nn.idx.reshape(batch_size, 1, -1, 1, 1).repeat(1, seq_length, 1, T.shape[2], T.shape[3]).reshape(batch_size * seq_length, -1, T.shape[2], T.shape[3])
        # nn_T = torch.gather(T, 1, repeated_nn_idx)
        ######
        
        homogen_coord = torch.ones([batch_size * seq_length, inv_template_garment_v.shape[1], 1], dtype=dtype, device=device)
        v_posed_homo = torch.cat([inv_template_garment_v, homogen_coord], dim=2)
        v_homo = torch.matmul(nn_T, torch.unsqueeze(v_posed_homo, dim=-1))

        return v_homo[:, :, :3, 0].reshape(batch_size, seq_length, -1, 3), nn, inv_template_garment_v.reshape(batch_size, seq_length, -1, 3)

    def forward(self, x, body_model, batch):
        nbatch= x.size()[0]
        T = x.size()[1]
        N = x.size()[2]
        with torch.no_grad():
            output_dict = self.PCA_garment_encoder(x, body_model)
        lap_adj = sp.eye(self.adj_old.shape[0]) - gcn_utils.normalize(self.adj_old)
        output_dict['lap_adj'] = gcn_utils.sparse_mx_to_torch_sparse_tensor(lap_adj).cuda()

        garment_v_list = output_dict['garment_v_list']
        garment_f_list = output_dict['garment_f_list']

        body_v = batch['smpl_vertices_torch'].cuda().reshape(nbatch * T, -1, 3).contiguous()
        if self.vf_fid is None or self.vf_vid is None:
            self.vf_fid, self.vf_vid = mesh_utils.calc_body_mesh_info(body_model)
            self.vf_fid = self.vf_fid.cuda()
            self.vf_vid = self.vf_vid.cuda()
        body_vn = mesh_utils.compute_vnorms(body_v, torch.from_numpy(body_model.faces.astype(np.int64)).cuda(), self.vf_vid, self.vf_fid)
        body_vn = body_vn.float()

        regressed_garment_v = output_dict['tpose_garment'].reshape(nbatch, -1, 3)

        start_time = time.time()
        output_dict['lbs_pred_garment_v'], output_dict['lbs_nn'], output_dict['lbs_stage1_pred_garment_v'] = \
            self.lbs_garment_interpolation(regressed_garment_v, batch['Tpose_smpl_vertices_torch'].cuda(),
            batch['Tpose_smpl_root_joints_torch'].cuda(), batch['zeropose_smpl_vertices_torch'].cuda(),
            body_model, batch['pose_torch'].cuda(), batch['T_J_regressor'].cuda(),
            batch['T_lbs_weights'].cuda(), K=self.cfg.NETWORK.LBSK)
        end_time = time.time()
        output_dict['lbs_time'] = end_time - start_time

        # output_dict['middle_results']['offsets'] = []
        # output_dict['middle_results']['gcn_inputs'] = []
        iter_regressed_lbs_garment_v = []
        lbs_pred_garment_v = output_dict['lbs_pred_garment_v'].reshape(nbatch * T, -1, 3).contiguous()
        cur_garment_v = lbs_pred_garment_v
        garment_v_num = cur_garment_v.shape[1]
        lbs_iter_feat = []
        for regress_iter in range(self.cfg.NETWORK.ITERATION):
            body_pe_list = []
            for i in range(len(self.body_radius_list)):
                cur_body_qg = self.body_query_group_list[i](xyz=body_v, new_xyz=cur_garment_v, features=body_vn.transpose(1, 2).contiguous()) \
                                    .reshape(nbatch * T, 6, garment_v_num, self.body_sample_num_list[i])\
                                    .permute(0, 2, 3, 1)
                cur_body_pe = self.body_positional_encoding_list[i](cur_body_qg).max(-2)[0].reshape(nbatch * T, garment_v_num, self.feat_num)
                body_pe_list.append(cur_body_pe)
            garment_pe_list = []
            for i in range(len(self.garment_radius_list)):
                cur_garment_qg = self.garment_query_group_list[i](xyz=garment_v_list[i], new_xyz=cur_garment_v, features=garment_f_list[i])\
                                        .reshape(nbatch * T, self.garment_positional_encoding_input_dim[i], garment_v_num, self.garment_sample_num_list[i])\
                                        .permute(0, 2, 3, 1)
                cur_garment_pe = self.garment_positional_encoding_list[i](cur_garment_qg).max(-2)[0].reshape(nbatch * T, garment_v_num, self.feat_num)
                garment_pe_list.append(cur_garment_pe)
            cur_positional_encoding = cur_garment_v
            templates_feat = torch.cat([cur_positional_encoding] + body_pe_list + garment_pe_list, 2)
            if regress_iter > 0:
                last_feat = lbs_iter_feat[-2].reshape(nbatch, T, garment_v_num, self.hidden_dim)
                q, k, v = self.temporal_qkv_list[regress_iter - 1](last_feat).chunk(3, dim=-1)
                q = q.reshape(nbatch, T, garment_v_num * self.hidden_dim)
                k = k.reshape(nbatch, T, garment_v_num * self.hidden_dim)
                v = v.reshape(nbatch, T, garment_v_num * self.hidden_dim)
                qk = torch.matmul(q, k.transpose(1, 2)).reshape(nbatch, T, T) / np.sqrt(T)
                qk = torch.softmax(qk, dim=-1)
                v = torch.matmul(qk, v).reshape(nbatch * T, garment_v_num, self.hidden_dim)
                templates_feat = torch.cat([templates_feat, v], -1)
            for i, m in enumerate(self.lbs_graph_regress[regress_iter]):
                templates_feat = m(templates_feat, self.adj, False)
                if i != len(self.lbs_graph_regress[regress_iter]) - 1:
                    templates_feat = F.relu(templates_feat)
                lbs_iter_feat.append(templates_feat)
            regressed_garment_v = cur_garment_v + templates_feat
            cur_garment_v = regressed_garment_v
            iter_regressed_lbs_garment_v.append(cur_garment_v)
        output_dict['iter_regressed_lbs_garment_v'] = iter_regressed_lbs_garment_v

        return output_dict

class PCALBSGarmentUseSegEncoderSegMGN(nn.Module):
    def __init__(self, cfg=None, args=None):
        super(PCALBSGarmentUseSegEncoderSegMGN, self).__init__()
        self.cfg = cfg
        self.args = args
        self.PCA_garment_encoder = PCAGarmentEncoderSeg(self.cfg, self.args)

        self.remesh_cylinder_f = self.PCA_garment_encoder.remesh_cylinder_f
        edges = np.zeros([2, self.remesh_cylinder_f.shape[0] * 4], dtype=np.int32)
        for i, f in enumerate(self.remesh_cylinder_f):
            if len(f) == 4:
                edges[:, i * 4 + 0] = np.array([f[0], f[1]], dtype=np.int32)
                edges[:, i * 4 + 1] = np.array([f[1], f[2]], dtype=np.int32)
                edges[:, i * 4 + 2] = np.array([f[2], f[3]], dtype=np.int32)
                edges[:, i * 4 + 3] = np.array([f[3], f[0]], dtype=np.int32)
            elif len(f) == 3:
                edges[:, i * 4 + 0] = np.array([f[0], f[1]], dtype=np.int32)
                edges[:, i * 4 + 1] = np.array([f[1], f[2]], dtype=np.int32)
                edges[:, i * 4 + 3] = np.array([f[2], f[0]], dtype=np.int32)
            else:
                raise NotImplementedError
        self.adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0, :], edges[1, :])),
                                shape=(self.PCA_garment_encoder.garment_v_num, self.PCA_garment_encoder.garment_v_num),
                                dtype=np.float32)
        self.adj = self.adj + self.adj.T.multiply(self.adj.T > self.adj) - self.adj.multiply(self.adj.T > self.adj)
        self.adj_old = self.adj.copy()
        self.adj = gcn_utils.normalize(self.adj + sp.eye(self.adj.shape[0]))
        self.adj = gcn_utils.sparse_mx_to_torch_sparse_tensor(self.adj).cuda()

        self.vf_fid = None
        self.vf_vid = None

        self.displacement_encoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.PCA_garment_encoder.garment_v_num * 3)
        )

    def lbs_garment_MGN(self, pred_template_garment_v, Tpose_vertices, Tpose_root_joints, zeropose_vertices, body_model, gt_pose, T_J_regressor, T_lbs_weights, K=3):
        assert len(pred_template_garment_v.shape) == 4 and pred_template_garment_v.shape[-1] == 3
        assert len(gt_pose.shape) == 3 and gt_pose.shape[2] == 72
        assert K == 1
        
        batch_size = pred_template_garment_v.shape[0]
        seq_length = gt_pose.shape[1]
        gt_pose_mat = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(batch_size * seq_length, 24, 3, 3)

        root_joint_pred_template_garment_v = pred_template_garment_v + Tpose_root_joints.reshape(batch_size, 3).reshape(batch_size, 1, 1, 3).repeat(1, seq_length, 1, 1)
        root_joint_pred_template_garment_v = root_joint_pred_template_garment_v.reshape(batch_size * seq_length, -1, 3)
        new_Tpose_vertices = Tpose_vertices.reshape(batch_size, 1, -1, 3).repeat(1, seq_length, 1, 1).reshape(batch_size * seq_length, -1, 3)
        nn = knn_points(root_joint_pred_template_garment_v, new_Tpose_vertices.reshape(batch_size * seq_length, -1, 3), K=1)

        inv_template_pose = torch.zeros([batch_size * seq_length, 24, 3]).cuda()
        inv_template_pose[:, 0, 0] = -np.pi / 2
        inv_template_pose[:, 1, 1] = 0.15
        inv_template_pose[:, 2, 1] = -0.15
        inv_template_pose_mat = batch_rodrigues(inv_template_pose.reshape(-1, 3)).reshape(batch_size * seq_length, 24, 3, 3)
        device, dtype = inv_template_pose.device, inv_template_pose.dtype
        
        inv_J = vertices2jointsB(T_J_regressor.reshape(batch_size * seq_length, T_J_regressor.shape[2], T_J_regressor.shape[3]),
                                 new_Tpose_vertices.reshape(batch_size * seq_length, -1, 3))
        _, inv_A = batch_rigid_transform(inv_template_pose_mat, inv_J, body_model.parents, dtype=dtype)

        ##### ORIGINAL
        inv_W = T_lbs_weights.reshape(batch_size * seq_length, T_lbs_weights.shape[2], T_lbs_weights.shape[3])
        num_joints = body_model.J_regressor.shape[0]
        inv_T = torch.matmul(inv_W, inv_A.view(batch_size * seq_length, num_joints, 16)).view(batch_size * seq_length, -1, 4, 4)
        inv_nn_T = torch.gather(inv_T, 1, nn.idx.reshape(batch_size * seq_length, -1, 1, 1).repeat(1, 1, inv_T.shape[2], inv_T.shape[3]))
        #####
        
        inv_homogen_coord = torch.ones([batch_size * seq_length, root_joint_pred_template_garment_v.shape[1], 1], dtype=dtype, device=device)
        inv_v_posed_homo = torch.cat([root_joint_pred_template_garment_v, inv_homogen_coord], dim=2)
        inv_v_homo = torch.matmul(inv_nn_T, torch.unsqueeze(inv_v_posed_homo, dim=-1))
        inv_template_garment_v = inv_v_homo[:, :, :3, 0].reshape(batch_size * seq_length, -1, 3)

        if torch.any(torch.isnan(inv_template_garment_v)):
            import pdb; pdb.set_trace()

        zero_pose_pred_shape_v = zeropose_vertices.reshape(batch_size * seq_length, -1, 3)
        J = vertices2jointsB(T_J_regressor.reshape(batch_size * seq_length, T_J_regressor.shape[2], T_J_regressor.shape[3]),
                             zero_pose_pred_shape_v)
        _, A = batch_rigid_transform(gt_pose_mat, J, body_model.parents, dtype=dtype)

        ##### ORIGINAL
        W = T_lbs_weights.reshape(batch_size * seq_length, T_lbs_weights.shape[2], T_lbs_weights.shape[3])
        T = torch.matmul(W, A.view(batch_size * seq_length, num_joints, 16)).view(batch_size * seq_length, -1, 4, 4)
        repeated_nn_idx = nn.idx.reshape(batch_size * seq_length, -1, 1, 1).repeat(1, 1, T.shape[2], T.shape[3]).reshape(batch_size * seq_length, -1, T.shape[2], T.shape[3])
        nn_T = torch.gather(T, 1, repeated_nn_idx)
        #####
        
        homogen_coord = torch.ones([batch_size * seq_length, inv_template_garment_v.shape[1], 1], dtype=dtype, device=device)
        v_posed_homo = torch.cat([inv_template_garment_v, homogen_coord], dim=2)
        v_homo = torch.matmul(nn_T, torch.unsqueeze(v_posed_homo, dim=-1))

        return v_homo[:, :, :3, 0].reshape(batch_size, seq_length, -1, 3), nn, inv_template_garment_v.reshape(batch_size, seq_length, -1, 3)

    def forward(self, x, body_model, batch):
        nbatch= x.size()[0]
        T = x.size()[1]
        N = x.size()[2]
        with torch.no_grad():
            output_dict = self.PCA_garment_encoder(x, body_model)
        lap_adj = sp.eye(self.adj_old.shape[0]) - gcn_utils.normalize(self.adj_old)
        output_dict['lap_adj'] = gcn_utils.sparse_mx_to_torch_sparse_tensor(lap_adj).cuda()
        regressed_garment_v = output_dict['tpose_garment'].reshape(nbatch, -1, 3)

        garment_summary = output_dict['garment_summary']
        displacements = self.displacement_encoder(garment_summary).reshape(nbatch, T, self.PCA_garment_encoder.garment_v_num, 3)
        displacements = displacements * 0.05
        if torch.any(torch.isnan(displacements)):
            displacements[torch.where(torch.isnan(displacements))] = 0
        t_garment_displacement = regressed_garment_v.reshape(nbatch, 1, -1, 3).repeat(1, T, 1, 1) + displacements

        output_dict['lbs_pred_garment_v'], output_dict['lbs_nn'], output_dict['lbs_stage1_pred_garment_v'] = \
            self.lbs_garment_MGN(t_garment_displacement, batch['Tpose_smpl_vertices_torch'].cuda(),
            batch['Tpose_smpl_root_joints_torch'].cuda(), batch['zeropose_smpl_vertices_torch'].cuda(),
            body_model, batch['pose_torch'].cuda(), batch['T_J_regressor'].cuda(),
            batch['T_lbs_weights'].cuda(), K=1)

        lbs_pred_garment_v = output_dict['lbs_pred_garment_v'].reshape(nbatch * T, -1, 3).contiguous()
        iter_regressed_lbs_garment_v = [lbs_pred_garment_v]
        output_dict['iter_regressed_lbs_garment_v'] = iter_regressed_lbs_garment_v

        return output_dict
