import os
from random import choice
import numpy as np
import plotly.graph_objs as go
from utils.config import args, cfg
import torch

def quads2tris(F):
    out = []
    for f in F:
        if len(f) == 3: out += [f]
        elif len(f) == 4: out += [[f[0],f[1],f[2]],
                                [f[0],f[2],f[3]]]
        else: print("This should not happen...")
    return np.array(out, np.int32)

# Display mesh
def display(V, F, C=None, display_grid=False):
    if C is None:
        C = np.zeros_like(V) + 0.6
    F = quads2tris(F)
    layout = go.Layout(
             scene=dict(
                 aspectmode='data'
         ))
    tri_points = V[F]
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k%3][0] for k in range(4)]+[ None])
        Ye.extend([T[k%3][1] for k in range(4)]+[ None])
        Ze.extend([T[k%3][2] for k in range(4)]+[ None])
        
    if display_grid:
        fig = go.Figure(data=[
            go.Mesh3d(
                x=V[:,0],
                y=V[:,1],
                z=V[:,2],
                # i, j and k give the vertices of triangles
                i = F[:,0],
                j = F[:,1],
                k = F[:,2],
                vertexcolor = C,
                showscale=True
            ),
            go.Scatter3d(
                       x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       name='',
                       line=dict(color= 'rgb(70,70,70)', width=1))
        ], layout=layout)
    else:
        fig = go.Figure(data=[
            go.Mesh3d(
                x=V[:,0],
                y=V[:,1],
                z=V[:,2],
                # i, j and k give the vertices of triangles
                i = F[:,0],
                j = F[:,1],
                k = F[:,2],
                vertexcolor = C,
                showscale=True
            ),
        ], layout=layout)
    fig.show()
    
def writeOBJ(file, V, F, Vt=None, Ft=None):
    if not Vt is None:
        assert len(F) == len(Ft), 'Inconsistent data, mesh and UV map do not have the same number of faces'
        
    with open(file, 'w') as file:
        # Vertices
        for v in V:
            line = 'v ' + ' '.join([str(_) for _ in v]) + '\n'
            file.write(line)
        # UV verts
        if not Vt is None:
            for v in Vt:
                line = 'vt ' + ' '.join([str(_) for _ in v]) + '\n'
                file.write(line)
        # 3D Faces / UV faces
        if Ft:
            F = [[str(i+1)+'/'+str(j+1) for i,j in zip(f,ft)] for f,ft in zip(F,Ft)]
        else:
            F = [[str(i + 1) for i in f] for f in F]        
        for f in F:
            line = 'f ' + ' '.join(f) + '\n'
            file.write(line)
            
def readOBJ(file):
    V, Vt, F, Ft = [], [], [], []
    with open(file, 'r') as f:
        T = f.readlines()
    for t in T:
        # 3D vertex
        if t.startswith('v '):
            v = [float(n) for n in t.replace('v ','').split(' ')]
            V += [v]
        # UV vertex
        elif t.startswith('vt '):
            v = [float(n) for n in t.replace('vt ','').split(' ')]
            Vt += [v]
        # Face
        elif t.startswith('f '):
            idx = [n.split('/') for n in t.replace('f ','').split(' ')]
            idx = [i for i in idx if i[0]!='']
            f = [int(n[0]) - 1 for n in idx]
            F += [f]
            # UV face
            if '/' in t:
                f = [int(n[1]) - 1 for n in idx]
                Ft += [f]
    V = np.array(V, np.float32)
    Vt = np.array(Vt, np.float32)
    if Ft: assert len(F) == len(Ft), 'Inconsistent .obj file, mesh and UV map do not have the same number of faces' 
    else: Vt, Ft = None, None
    
    F = np.array(list(F))
    return V, F, Vt, Ft
    

from psbody.mesh import Mesh
from psbody.mesh.geometry.vert_normals import VertNormals
from psbody.mesh.geometry.tri_normals import TriNormals
from psbody.mesh.search import AabbTree
def laplacian(adj):
    """ Compute laplacian operator on part_mesh. This can be cached.
    """
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize
    connectivity = adj
    # connectivity is a sparse matrix, and np.clip can not applied directly on
    # a sparse matrix.
    connectivity.data = np.clip(connectivity.data, 0, 1)
    lap = normalize(connectivity, norm='l1', axis=1)
    lap = sp.eye(connectivity.shape[0]) - lap
    return lap

# inspired from frankengeist.body.ch.mesh_distance.MeshDistanceSquared
def get_nearest_points_and_normals(vert, base_verts, base_faces):

    fn = TriNormals(v=base_verts, f=base_faces).reshape((-1, 3))
    vn = VertNormals(v=base_verts, f=base_faces).reshape((-1, 3))

    tree = AabbTree(Mesh(v=base_verts, f=base_faces))
    nearest_tri, nearest_part, nearest_point = tree.nearest(vert, nearest_part=True)
    nearest_tri = nearest_tri.ravel().astype(np.long)
    nearest_part = nearest_part.ravel().astype(np.long)

    nearest_normals = np.zeros_like(vert)

    #nearest_part tells you whether the closest point in triangle abc is in the interior (0), on an edge (ab:1,bc:2,ca:3), or a vertex (a:4,b:5,c:6)
    cl_tri_idxs = np.nonzero(nearest_part == 0)[0].astype(np.int)
    cl_vrt_idxs = np.nonzero(nearest_part > 3)[0].astype(np.int)
    cl_edg_idxs = np.nonzero((nearest_part <= 3) & (nearest_part > 0))[0].astype(np.int)

    nt = nearest_tri[cl_tri_idxs]
    nearest_normals[cl_tri_idxs] = fn[nt]

    nt = nearest_tri[cl_vrt_idxs]
    npp = nearest_part[cl_vrt_idxs] - 4
    nearest_normals[cl_vrt_idxs] = vn[base_faces[nt, npp]]

    nt = nearest_tri[cl_edg_idxs]
    npp = nearest_part[cl_edg_idxs] - 1
    nearest_normals[cl_edg_idxs] += vn[base_faces[nt, npp]]
    npp = np.mod(nearest_part[cl_edg_idxs], 3)
    nearest_normals[cl_edg_idxs] += vn[base_faces[nt, npp]]

    nearest_normals = nearest_normals / (np.linalg.norm(nearest_normals, axis=-1, keepdims=True) + 1.e-10)

    return nearest_point, nearest_normals

def remove_interpenetration_fast(mesh, base, adj, L=None):
    """
    Laplacian can be cached.
    Computing laplacian takes 60% of the total runtime.
    """
    import scipy.sparse as sp
    from scipy.sparse import vstack, csr_matrix
    from scipy.sparse.linalg import spsolve

    eps = 0.008
#     eps = 0.01
    ww = 2.0
    nverts = mesh.v.shape[0]

    if L is None:
        L = laplacian(adj)

    nearest_points, nearest_normals = get_nearest_points_and_normals(mesh.v, base.v, base.f)
    direction = np.sign( np.sum((mesh.v - nearest_points) * nearest_normals, axis=-1) )

    mesh_vn = VertNormals(v=mesh.v, f=mesh.f).reshape((-1, 3))
    normal_dot_sign = np.sign(np.sum(mesh_vn * nearest_normals, axis=-1)).reshape(-1, 1)
    
    indices = np.where(direction <= -1e-6)[0]
    
#     print(np.where(normal_dot_sign[indices] < 0))
#     print(indices.shape)

    pentgt_points = (nearest_points[indices] - mesh.v[indices]) * normal_dot_sign[indices]
#     pentgt_points = mesh_vn[indices]
    pentgt_points = nearest_points[indices] \
                    + eps * pentgt_points / np.expand_dims(0.0001 + np.linalg.norm(pentgt_points, axis=1), 1)
#     pentgt_points = mesh.v[indices] + eps * pentgt_points / np.expand_dims(0.0001 + np.linalg.norm(pentgt_points, axis=1), 1)
    tgt_points = mesh.v.copy()
    tgt_points[indices] = ww * pentgt_points

    rc = np.arange(nverts)
    data = np.ones(nverts)
    data[indices] *= ww
    I = csr_matrix((data, (rc, rc)), shape=(nverts, nverts))

    A = vstack([L, I])
    b = np.vstack((
        L.dot(mesh.v),
        tgt_points
    ))

    res = spsolve(A.T.dot(A), A.T.dot(b))
    mres = Mesh(v=res, f=mesh.f)
    return mres, indices.shape[0]

from modules.pygcn.utils import normalize
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from plyfile import PlyData, PlyElement
def process_single_frame(model, vis_batch, output_dict, ith, frame, body_model, save=True, post_process=True):
    gt_garment_v = vis_batch['garment_np'][ith][frame]
    gt_body_v = vis_batch['pcd_np'][ith][frame]
    body_f = body_model.faces
    so = {
        'vertices': vis_batch['smpl_vertices_torch'][ith, frame].reshape(1, 6890, 3),
        'joints': vis_batch['smpl_root_joints_torch'][ith, frame].reshape(1, 1, 3)
    }
    root_joint = so['joints'][0].detach().cpu().numpy()[0].reshape(3)
    gt_body_v = so['vertices'][0].detach().cpu().numpy()
    t_so = {
        'vertices': vis_batch['Tpose_smpl_vertices_torch'][ith].reshape(1, 6890, 3),
        'joints': vis_batch['Tpose_smpl_root_joints_torch'][ith].reshape(1, 1, 3),
    }
    tpose_root_joint = t_so['joints'][0].detach().cpu().numpy()[0].reshape(3)
    tpose_gt_body_v = t_so['vertices'][0].detach().cpu().numpy()
    pred_tpose_garment_v = output_dict['tpose_garment'][ith].detach().cpu().numpy().reshape(-1, 3)
    gt_tpose_garment_v = vis_batch['garment_template_vertices'][ith].detach().cpu().numpy().reshape(-1, 3)
    garment_f = output_dict['garment_f_3']
    
    logits = output_dict['sem_logits'][ith * args.T + frame]
    labels = torch.argmax(logits, dim=1).detach().cpu().numpy()
    input_pcd = vis_batch['pcd_np'][ith][frame]
    gt_labels = vis_batch['pcd_label_np'][ith][frame].reshape(-1)
    save_pcd = np.concatenate([input_pcd.reshape(-1, 3), labels.reshape(-1, 1), gt_labels.reshape(-1, 1)], 1)
    def rgb(minimum, maximum, value):
        value = value.reshape(-1, 1)
        minimum, maximum = float(minimum), float(maximum)
        ratio = 2 * (value-minimum) / (maximum - minimum)
        b = (255*(1 - ratio)).astype(np.int32)
        r = (255*(ratio - 1)).astype(np.int32)
        b[b<0] = 0
        r[r<0] = 0
        g = 255 - b - r
        return np.concatenate([r, g, b], 1)
#     display_color = []
#     for v in input_pcd:
#         display_color.append(rgb(input_pcd.min(0)[2]-1e-6, input_pcd.max(0)[2]+1e-6, v[2]))
    display_color = rgb(input_pcd.min(0)[2]-1e-6, input_pcd.max(0)[2]+1e-6, input_pcd[:, 2])
    label_map = [
         [150, 150, 150],
         [245, 150, 100],
         [245, 230, 100],
         [250, 80, 100],
         [150, 60, 30],
         [255, 0, 0],
         [180, 30, 80],
    ]
    label_map = np.array(label_map)
    gt_color = label_map[gt_labels].reshape(-1, 3)
    pred_color = label_map[labels].reshape(-1, 3)
    
    joint_v = np.concatenate([gt_tpose_garment_v + tpose_root_joint, tpose_gt_body_v], 0)
    joint_f = np.concatenate([garment_f, body_f + gt_tpose_garment_v.shape[0]], 0)
    joint_c = np.concatenate([np.zeros_like(gt_tpose_garment_v) + 0.4, np.zeros_like(gt_body_v) + 0.6])
    
    pred_posed_garment_v = output_dict['iter_regressed_lbs_garment_v'][-1][ith * args.T + frame].detach().cpu().numpy()
    only_lbs_posed_garment_v = output_dict['lbs_pred_garment_v'][ith, frame].detach().cpu().numpy().reshape(-1, 3)
    gt_posed_garment_v = vis_batch['garment_torch'][ith, frame].numpy() + root_joint
    joint_v_gt = np.concatenate([gt_posed_garment_v, gt_body_v], 0)
    joint_v_only_lbs = np.concatenate([only_lbs_posed_garment_v, gt_body_v], 0)
    joint_v_pred = np.concatenate([pred_posed_garment_v, gt_body_v], 0)

    remove_interp_mesh_v = pred_posed_garment_v
    if post_process:
        lbs_pred_garment_v = output_dict['iter_regressed_lbs_garment_v'][-1][ith * args.T + frame].detach().cpu().numpy()
        smooth_lbs_pred_garment_v = lbs_pred_garment_v.copy()
        adj = normalize(model.module.adj_old) - sp.eye(model.module.adj_old.shape[0])
        coeff = 0.05
        neg_coeff = -0.052
        for it in range(100):
            if it % 2 == 0:
                smooth_lbs_pred_garment_v = smooth_lbs_pred_garment_v + coeff * adj.dot(smooth_lbs_pred_garment_v)
            else:
                smooth_lbs_pred_garment_v = smooth_lbs_pred_garment_v + neg_coeff * adj.dot(smooth_lbs_pred_garment_v)

        smooth_mesh_ = Mesh(v=smooth_lbs_pred_garment_v, f=garment_f)
        base_mesh_ = Mesh(v=gt_body_v, f=body_f)    
        remove_interp_mesh_ = smooth_mesh_
        for i in range(5):
            remove_interp_mesh_, interpenetration_point_num = remove_interpenetration_fast(remove_interp_mesh_, base_mesh_, model.module.adj_old)
            if interpenetration_point_num < 1:
                break
    
        remove_interp_mesh_v = remove_interp_mesh_.v
    
    if save:
        cur_seq = vis_batch['T_pcd_flist'][ith][frame].split('/')[-3]
        if args.MGN:
            cur_seq = 'MGN_' + cur_seq
        else:
            cur_seq = 'ours_' + cur_seq
        cur_frame = vis_batch['T_pcd_flist'][ith][frame].split('/')[-2]
        garment_name = cfg.GARMENT.NAME
        print(cur_seq, cur_frame, garment_name)
        # if not os.path.exists('../datasets/supp_drawing_assets/{}/{}'.format(cur_seq, cur_frame)):
        #     os.makedirs('../datasets/supp_drawing_assets/{}/{}'.format(cur_seq, cur_frame), exist_ok=True)
        
        # body_posed_gt_obj_fname = '../datasets/supp_drawing_assets/{}/{}/body.obj'.format(cur_seq, cur_frame)
        # if not os.path.exists(body_posed_gt_obj_fname):
        #     writeOBJ(body_posed_gt_obj_fname, gt_body_v, body_f)
        
        # garment_posed_obj_fname = '../datasets/supp_drawing_assets/{}/{}/{}.obj'.format(cur_seq, cur_frame, garment_name)
        # if os.path.exists(garment_posed_obj_fname):
        #     old_v, _, _, _ = readOBJ(garment_posed_obj_fname)
        #     new_v = remove_interp_mesh_v * 0.5 + old_v * 0.5
        #     writeOBJ(garment_posed_obj_fname, new_v, garment_f)
        # else:
        #     writeOBJ(garment_posed_obj_fname, remove_interp_mesh_v, garment_f)
        
        # garment_posed_gt_obj_fname = '../datasets/supp_drawing_assets/{}/{}/{}_gt.obj'.format(cur_seq, cur_frame, garment_name)
        # if not os.path.exists(garment_posed_gt_obj_fname):
        #     writeOBJ(garment_posed_gt_obj_fname, gt_posed_garment_v, garment_f)
        
        # garment_lbs_obj_fname = '../datasets/supp_drawing_assets/{}/{}/{}_only_lbs.obj'.format(cur_seq, cur_frame, garment_name)
        # if not os.path.exists(garment_lbs_obj_fname):
        #     writeOBJ(garment_lbs_obj_fname, only_lbs_posed_garment_v, garment_f)
        
        # garment_tpose_obj_fname = '../datasets/supp_drawing_assets/{}/{}_tpose.obj'.format(cur_seq, garment_name)
        # if not os.path.exists(garment_tpose_obj_fname):
        #     writeOBJ(garment_tpose_obj_fname, pred_tpose_garment_v + tpose_root_joint, garment_f)
        
        # body_tpose_obj_fname = '../datasets/supp_drawing_assets/{}/tpose_body.obj'.format(cur_seq)
        # if not os.path.exists(body_tpose_obj_fname):
        #     writeOBJ(body_tpose_obj_fname, tpose_gt_body_v, body_f)
            
        ply_dtype = [
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('red', 'u1'),
            ('green', 'u1'),
            ('blue', 'u1'),
        ]
        dis = np.concatenate([input_pcd, display_color], 1)
        dis = [tuple(list(l)) for l in dis]
        dis_ply = np.array(dis, dtype=ply_dtype)
        gt = np.concatenate([input_pcd, gt_color], 1)
        gt = [tuple(list(l)) for l in gt]
        gt_ply = np.array(gt, dtype=ply_dtype)
        pred = np.concatenate([input_pcd, pred_color], 1)
        pred = [tuple(list(l)) for l in pred]
        pred_ply = np.array(pred, dtype=ply_dtype)

        dis_el = PlyElement.describe(dis_ply, 'vertex')
        gt_el = PlyElement.describe(gt_ply, 'vertex')
        pred_el = PlyElement.describe(pred_ply, 'vertex')
        PlyData([dis_el]).write('../datasets/supp_drawing_assets/{}/{}/dis.ply'.format(cur_seq, cur_frame))
        PlyData([gt_el]).write('../datasets/supp_drawing_assets/{}/{}/gt.ply'.format(cur_seq, cur_frame))
        PlyData([pred_el]).write('../datasets/supp_drawing_assets/{}/{}/pred.ply'.format(cur_seq, cur_frame))
