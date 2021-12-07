import os
import torch
import torch_scatter
import numpy as np
import scipy.io as sio
import openmesh as om

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
    return V, F, Vt, Ft

def loadInfo(filename):
    '''
    this function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    del data['__globals__']
    del data['__header__']
    del data['__version__']
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray) and np.any([isinstance(item, sio.matlab.mio5_params.mat_struct) for item in elem]):
            dict[strg] = [None] * len(elem)
            for i,item in enumerate(elem):
                if isinstance(item, sio.matlab.mio5_params.mat_struct):
                    dict[strg][i] = _todict(item)
                else:
                    dict[strg][i] = item
        else:
            dict[strg] = elem
    return dict

def zRotMatrix(zrot):
    c, s = np.cos(zrot), np.sin(zrot)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], np.float32)

def calc_body_mesh_info(body_model):
    ommesh = om.TriMesh()
    om_v_list = []
    for v in body_model.v_template.detach().cpu().numpy().reshape(6890, 3):
        om_v_list.append(ommesh.add_vertex(v))
    for f in body_model.faces.astype(np.int32):
        ommesh.add_face([om_v_list[i] for i in f])
    vf_fid=torch.zeros(0,dtype=torch.long)
    vf_vid=torch.zeros(0,dtype=torch.long)
    for vid,fids in enumerate(ommesh.vertex_face_indices()):
        fids=torch.from_numpy(fids[fids>=0]).to(torch.long)
        vf_fid=torch.cat((vf_fid,fids),dim=0)
        vf_vid=torch.cat((vf_vid,fids.new_ones(fids.shape)*vid),dim=0)
    return vf_fid, vf_vid

def calc_garment_mesh_info(garment_v, garment_f):
    ommesh = om.TriMesh()
    om_v_list = []
    for v in garment_v:
        om_v_list.append(ommesh.add_vertex(v))
    for f in garment_f.astype(np.int32):
        ommesh.add_face([om_v_list[i] for i in f])
    vf_fid=torch.zeros(0,dtype=torch.long)
    vf_vid=torch.zeros(0,dtype=torch.long)
    for vid,fids in enumerate(ommesh.vertex_face_indices()):
        fids=torch.from_numpy(fids[fids>=0]).to(torch.long)
        vf_fid=torch.cat((vf_fid,fids),dim=0)
        vf_vid=torch.cat((vf_vid,fids.new_ones(fids.shape)*vid),dim=0)
    return vf_fid, vf_vid

def compute_fnorms(verts,tri_fs):
    v0=verts.index_select(-2,tri_fs[:,0])
    v1=verts.index_select(-2,tri_fs[:,1])
    v2=verts.index_select(-2,tri_fs[:,2])
    e01=v1-v0
    e02=v2-v0
    fnorms=torch.cross(e01,e02,-1)
    diss=fnorms.norm(2,-1).unsqueeze(-1)
    diss=torch.clamp(diss,min=1.e-6,max=float('inf'))
    fnorms=fnorms/diss
    return fnorms

def compute_vnorms(verts,tri_fs,vertex_index,face_index):
    fnorms=compute_fnorms(verts,tri_fs)
    vnorms=torch_scatter.scatter(fnorms.index_select(-2,face_index),vertex_index,-2,None,verts.shape[-2])
    diss=vnorms.norm(2,-1).unsqueeze(-1)
    diss=torch.clamp(diss,min=1.e-6,max=float('inf'))
    vnorms=vnorms/diss
    return vnorms

def interpolateBarycentricCoorNumpy(v, ind, w):
    indv = v[ind].reshape(ind.shape[0], 3, 3)
    rv = np.matmul(w.reshape(w.shape[0], 1, 3), indv).reshape(w.shape[0], 3)
    return rv