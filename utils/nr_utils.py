import os
import torch
import numpy as np
import neural_renderer as nr
import matplotlib.pyplot as plt
from PIL import Image
from smplx import batch_rodrigues
from utils.config import args

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

def render_one_batch(output_dict, inputs, body_model, add_cloth=False):
    camera_distance = 1.5
    elevation = 0
    texture_size = 2

    # batch_size = output[0][0].shape[0]
    # seq_length = output[0][0].shape[1]
    # pred_poses = output[0][0].view(-1, 72)
    # pred_shapes = output[0][1].view(-1, 10)

    batch_size = inputs['pose_torch'].shape[0]
    seq_length = inputs['pose_torch'].shape[1]
    pred_poses = inputs['pose_torch'].view(-1, 72).cuda()
    pred_shapes = inputs['beta_torch'].view(-1, 10).cuda()

    pred_rot = batch_rodrigues(pred_poses.reshape(-1, 3)).view(-1, 24, 3, 3)
    pred_so = body_model(betas = pred_shapes, body_pose = pred_rot[:, 1:], global_orient = pred_rot[:, 0].reshape(-1, 1, 3, 3), pose2rot=False)
    vertices = pred_so['vertices']
    body_vertices = vertices.detach().cpu().numpy().copy()
    faces = torch.from_numpy(body_model.faces.astype(np.int32)).cuda().unsqueeze(0).repeat(vertices.shape[0], 1, 1)
    body_faces = faces.detach().cpu().numpy().copy()
    if add_cloth:
        # cloth_vertices = output[2].reshape(batch_size * seq_length, -1, 3)
        # cloth_faces = torch.from_numpy(output[4]).cuda().unsqueeze(0).repeat(cloth_vertices.shape[0], 1, 1)
        cloth_vertices = output_dict['iter_regressed_lbs_garment_v'][-1].reshape(batch_size * seq_length, -1, 3)
        cloth_faces = torch.from_numpy(output_dict['garment_f_3']).cuda().unsqueeze(0).repeat(cloth_vertices.shape[0], 1, 1)
        faces = torch.cat([faces, cloth_faces + vertices.shape[1]], 1)
        # vertices = torch.cat([vertices, cloth_vertices + pred_so['joints'][:, 0, :].unsqueeze(1)], 1)
        vertices = torch.cat([vertices, cloth_vertices], 1)
    textures = torch.ones(vertices.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    rot_mat = torch.from_numpy(np.array(
        [[ 1.,  0.,  0.],
        [ 0.,  0., -1.],
        [ 0.,  1.,  0.]], dtype=np.float32)).cuda()
    vertices = torch.matmul(vertices, rot_mat)

    renderer = nr.Renderer(camera_mode='look_at').cuda()
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, 45)
    images, _, _ = renderer(vertices, faces, textures)
    # import pdb; pdb.set_trace()
    images = images.detach().cpu().numpy().transpose(0, 2, 3, 1) * 256
    images[images == 256] = 255

    return images.reshape(batch_size, seq_length, 256, 256, 3), \
           body_vertices.reshape(batch_size, seq_length, -1, 3), \
           body_faces.reshape(batch_size, seq_length, -1, 3), \
           cloth_vertices.reshape(batch_size, seq_length, -1, 3).detach().cpu().numpy(), \
           cloth_faces.reshape(batch_size, seq_length, -1, 3).detach().cpu().numpy()

def save_obj(body_vertices, body_faces, cloth_vertices, cloth_faces, inputs):
    batch_size = body_vertices.shape[0]
    seq_length = body_vertices.shape[1]
    for b in range(batch_size):
        for s in range(seq_length):
            pcd_fname = inputs['T_pcd_flist'][b][s]
            seq_name = pcd_fname.split('/')[-3]
            frame_name = pcd_fname.split('/')[-2]
            save_folder = os.path.join(args.output_dir, 'pred_obj', seq_name, frame_name)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)
            writeOBJ(os.path.join(save_folder, 'body.obj'), body_vertices[b, s], body_faces[b, s])
            writeOBJ(os.path.join(save_folder, 'garment.obj'), cloth_vertices[b, s], cloth_faces[b, s])

def save_images(images, inputs, add_cloth=False):
    batch_size = images.shape[0]
    seq_length = images.shape[1]
    for b in range(batch_size):
        for s in range(seq_length):
            pcd_fname = inputs['T_pcd_flist'][b][s]
            seq_name = pcd_fname.split('/')[-3]
            frame_name = pcd_fname.split('/')[-2]
            if add_cloth:
                save_folder = os.path.join(args.output_dir, 'vis', seq_name+'_cloth')
            else:
                save_folder = os.path.join(args.output_dir, 'vis', seq_name)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)
            Image.fromarray(images[b, s].astype(np.uint8)).save(os.path.join(save_folder, frame_name+'.png'))
