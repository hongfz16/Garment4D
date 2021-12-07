"""
Computes Lx and it's derivative, where L is the graph laplacian on the mesh with cotangent weights.
1. Given V, F, computes the cotangent matrix (for each face, computes the angles) in pytorch.
2. Then it's taken to NP and sparse L is constructed.
Mesh laplacian computation follows Alec Jacobson's gptoolbox.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np
from scipy import sparse

#############
### Utils ###
#############
def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src

########################################################################
################# Wrapper class for a  PythonOp ########################
##### All functions must only use torch Tensors as inputs/outputs ######
########################################################################
class Laplacian(torch.autograd.Function):
    # def __init__(self, faces):
    #     """
    #     Faces is B x F x 3, cuda torch Variabe.
    #     Reuse faces.
    #     """
    #     self.F_np = faces.data.cpu().numpy()
    #     self.F = faces.data
    #     self.L = None
    F_np = None
    F = None
    L = None

    @staticmethod
    def forward(ctx, V):
        """
        If forward is explicitly called, V is still a Parameter or Variable
        But if called through __call__ it's a tensor.
        This assumes __call__ was used.
        
        Input:
           V: B x N x 3
           F: B x F x 3
        Outputs: Lx B x N x 3
        
         Numpy also doesnt support sparse tensor, so stack along the batch
        """

        V_np = V.cpu().numpy()
        batchV = V_np.reshape(-1, 3)

        if Laplacian.L is None:
            # print('Computing the Laplacian!')
            # Compute cotangents
            C = cotangent(V, Laplacian.F)
            C_np = C.cpu().numpy()
            batchC = C_np.reshape(-1, 3)
            # Adjust face indices to stack:
            offset = np.arange(0, V.size(0)).reshape(-1, 1, 1) * V.size(1)
            F_np = Laplacian.F_np + offset
            batchF = F_np.reshape(-1, 3)

            rows = batchF[:, [1, 2, 0]].reshape(-1) #1,2,0 i.e to vertex 2-3 associate cot(23)
            cols = batchF[:, [2, 0, 1]].reshape(-1) #2,0,1 This works because triangles are oriented ! (otherwise 23 could be associated to more than 1 cot))

            # Final size is BN x BN
            BN = batchV.shape[0]
            L = sparse.csr_matrix((batchC.reshape(-1), (rows, cols)), shape=(BN,BN))
            L = L + L.T
            # np.sum on sparse is type 'matrix', so convert to np.array
            M = sparse.diags(np.array(np.sum(L, 1)).reshape(-1), format='csr')
            L = L - M
            # remember this
            Laplacian.L = L
            # TODO The normalization by the size of the voronoi cell is missing.
            # import matplotlib.pylab as plt
            # plt.ion()
            # plt.clf()
            # plt.spy(L)
            # plt.show()
            # import ipdb; ipdb.set_trace()

        # import pdb; pdb.set_trace()

        Lx = Laplacian.L.dot(batchV).reshape(V_np.shape)

        return convert_as(torch.Tensor(Lx), V)

    @staticmethod
    def backward(ctx, grad_out):
        """
        Just L'g = Lg
        Args:
           grad_out: B x N x 3
        Returns:
           grad_vertices: B x N x 3
        """
        g_o = grad_out.cpu().numpy()
        # Stack
        g_o = g_o.reshape(-1, 3)
        Lg = Laplacian.L.dot(g_o).reshape(grad_out.shape)

        return convert_as(torch.Tensor(Lg), grad_out)

class GarmentLaplacian(torch.autograd.Function):
    # def __init__(self, faces):
    #     """
    #     Faces is B x F x 3, cuda torch Variabe.
    #     Reuse faces.
    #     """
    #     self.F_np = faces.data.cpu().numpy()
    #     self.F = faces.data
    #     self.L = None
    F_np = None
    F = None
    L = None

    @staticmethod
    def forward(ctx, V):
        """
        If forward is explicitly called, V is still a Parameter or Variable
        But if called through __call__ it's a tensor.
        This assumes __call__ was used.
        
        Input:
           V: B x N x 3
           F: B x F x 3
        Outputs: Lx B x N x 3
        
         Numpy also doesnt support sparse tensor, so stack along the batch
        """

        V_np = V.cpu().numpy()
        batchV = V_np.reshape(-1, 3)

        if GarmentLaplacian.L is None:
            # print('Computing the Laplacian!')
            # Compute cotangents
            C = cotangent(V, GarmentLaplacian.F)
            C_np = C.cpu().numpy()
            batchC = C_np.reshape(-1, 3)
            # Adjust face indices to stack:
            offset = np.arange(0, V.size(0)).reshape(-1, 1, 1) * V.size(1)
            F_np = GarmentLaplacian.F_np + offset
            batchF = F_np.reshape(-1, 3)

            rows = batchF[:, [1, 2, 0]].reshape(-1) #1,2,0 i.e to vertex 2-3 associate cot(23)
            cols = batchF[:, [2, 0, 1]].reshape(-1) #2,0,1 This works because triangles are oriented ! (otherwise 23 could be associated to more than 1 cot))

            # Final size is BN x BN
            BN = batchV.shape[0]
            L = sparse.csr_matrix((batchC.reshape(-1), (rows, cols)), shape=(BN,BN))
            L = L + L.T
            # np.sum on sparse is type 'matrix', so convert to np.array
            M = sparse.diags(np.array(np.sum(L, 1)).reshape(-1), format='csr')
            L = L - M
            # remember this
            GarmentLaplacian.L = L
            # TODO The normalization by the size of the voronoi cell is missing.
            # import matplotlib.pylab as plt
            # plt.ion()
            # plt.clf()
            # plt.spy(L)
            # plt.show()
            # import ipdb; ipdb.set_trace()

        # import pdb; pdb.set_trace()

        Lx = GarmentLaplacian.L.dot(batchV).reshape(V_np.shape)

        return convert_as(torch.Tensor(Lx), V)

    @staticmethod
    def backward(ctx, grad_out):
        """
        Just L'g = Lg
        Args:
           grad_out: B x N x 3
        Returns:
           grad_vertices: B x N x 3
        """
        g_o = grad_out.cpu().numpy()
        # Stack
        g_o = g_o.reshape(-1, 3)
        Lg = GarmentLaplacian.L.dot(g_o).reshape(grad_out.shape)

        return convert_as(torch.Tensor(Lg), grad_out)

class OnetimeLaplacian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, V, F):
        """
        If forward is explicitly called, V is still a Parameter or Variable
        But if called through __call__ it's a tensor.
        This assumes __call__ was used.
        
        Input:
           V: B x N x 3
           F: B x F x 3
        Outputs: Lx B x N x 3
        
         Numpy also doesnt support sparse tensor, so stack along the batch
        """

        F_np = F.cpu().numpy()
        V_np = V.cpu().numpy()
        batchV = V_np.reshape(-1, 3)

        # Compute cotangents
        C = cotangent(V, F)
        C_np = C.cpu().numpy()
        batchC = C_np.reshape(-1, 3)
        # Adjust face indices to stack:
        offset = np.arange(0, V.size(0)).reshape(-1, 1, 1) * V.size(1)
        F_np = F_np + offset
        batchF = F_np.reshape(-1, 3)

        rows = batchF[:, [1, 2, 0]].reshape(-1) #1,2,0 i.e to vertex 2-3 associate cot(23)
        cols = batchF[:, [2, 0, 1]].reshape(-1) #2,0,1 This works because triangles are oriented ! (otherwise 23 could be associated to more than 1 cot))

        # Final size is BN x BN
        BN = batchV.shape[0]
        L = sparse.csr_matrix((batchC.reshape(-1), (rows, cols)), shape=(BN,BN))
        L = L + L.T
        # np.sum on sparse is type 'matrix', so convert to np.array
        M = sparse.diags(np.array(np.sum(L, 1)).reshape(-1), format='csr')
        L = L - M
        # TODO The normalization by the size of the voronoi cell is missing.

        # import pdb; pdb.set_trace()

        Lx = L.dot(batchV).reshape(V_np.shape)

        ctx.LapL = L

        if np.any(np.isnan(Lx)):
            import pdb; pdb.set_trace()

        return convert_as(torch.Tensor(Lx), V)

    @staticmethod
    def backward(ctx, grad_out):
        """
        Just L'g = Lg
        Args:
           grad_out: B x N x 3
        Returns:
           grad_vertices: B x N x 3
        """
        g_o = grad_out.cpu().numpy()
        # Stack
        g_o = g_o.reshape(-1, 3)
        Lg = ctx.LapL.dot(g_o).reshape(grad_out.shape)

        return convert_as(torch.Tensor(Lg), grad_out), None

def cotangent(V, F):
    """
    Input:
      V: B x N x 3
      F: B x F  x3
    Outputs:
      C: B x F x 3 list of cotangents corresponding
        angles for triangles, columns correspond to edges 23,31,12
    B x F x 3 x 3
    """
    indices_repeat = torch.stack([F, F, F], dim=2)

    #v1 is the list of first triangles B*F*3, v2 second and v3 third
    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())

    l1 = torch.sqrt(((v2 - v3)**2).sum(2)) #distance of edge 2-3 for every face B*F
    l2 = torch.sqrt(((v3 - v1)**2).sum(2))
    l3 = torch.sqrt(((v1 - v2)**2).sum(2))

    # semiperimieters
    sp = (l1 + l2 + l3) * 0.5

    # Heron's formula for area #FIXME why the *2 ? Heron formula is without *2 It's the 0.5 than appears in the (0.5(cotalphaij + cotbetaij))
    A = 2*torch.sqrt( sp * (sp-l1)*(sp-l2)*(sp-l3))

    # Theoreme d Al Kashi : c2 = a2 + b2 - 2ab cos(angle(ab))
    cot23 = (l2**2 + l3**2 - l1**2)
    cot31 = (l1**2 + l3**2 - l2**2)
    cot12 = (l1**2 + l2**2 - l3**2)

    # 2 in batch #proof page 98 http://www.cs.toronto.edu/~jacobson/images/alec-jacobson-thesis-2013-compressed.pdf
    C = torch.stack([cot23, cot31, cot12], 2) / torch.unsqueeze(A, 2) / 4

    C[torch.where(torch.isnan(C))] = 0
    C[torch.where(torch.isinf(C))] = 0

    return C

def teapot_smooth_test():
    import tqdm
    from time import time
    from ..external.neural_renderer import neural_renderer
    from ..utils.bird_vis import VisRenderer
    import scipy.misc

    obj_file = 'birds3d/external/neural_renderer/examples/data/teapot.obj'
    obj_file = 'birds3d/data/cow.obj'
    img_save_dir = 'birds3d/cachedir/laplacian/'

    verts, faces = neural_renderer.load_obj(obj_file)
    # Use NMR for rendering,,
    renderer = VisRenderer(256, faces)

    # verts = np.tile(verts[None, :, :], (3,1,1))
    # faces = np.tile(faces[None, :, :], (3,1,1))
    verts = verts[None, :, :]
    faces = faces[None, :, :]

    faces_var = Variable(torch.LongTensor(faces).cuda(), requires_grad=False)

    class TeapotModel(torch.nn.Module):
        def __init__(self):
            super(TeapotModel, self).__init__()
            vertices_var = torch.from_numpy(verts).cuda()
            # vertices_var = torch.from_numpy(verts)
            self.vertices_var = torch.nn.Parameter(vertices_var)
            self.laplacian = Laplacian(faces_var)

        def forward(self):
            return self.laplacian(self.vertices_var)

    opt_model = TeapotModel()
    optimizer = torch.optim.Adam(opt_model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    loop = tqdm.tqdm(range(300))
    print('Smoothing Vertices: ')
    for ix in loop:
        t0 = time()
        optimizer.zero_grad()
        Lx = opt_model.forward()
        # loss = 0.5 * torch.sum(Lx**2)
        loss = torch.norm(Lx.view(-1, Lx.size(2)), p=2, dim=1).mean()
        print('loss %.2g' % loss)
        loss.backward()
        if ix % 20 == 0:
            vert_np = opt_model.vertices_var[0].data.cpu().numpy()
            cams = Variable(torch.Tensor([1, 0, 0, 1, 0, 0, 0]).cuda(), requires_grad=False)
            rend_img = renderer(opt_model.vertices_var[0], cams)
            # import matplotlib.pyplot as plt
            # plt.ion()
            # plt.figure(1)
            # plt.clf()
            # plt.imshow(rend_img)
            scipy.misc.imsave(img_save_dir + 'iter_{:03d}.png'.format(ix), rend_img)

            # opt_model.laplacian.L = None
            # import ipdb; ipdb.set_trace()

        optimizer.step()

def test_laplacian():
    from ..utils import mesh
    verts, faces = mesh.create_sphere()

    # Pytorch-fy
    # verts = np.tile(verts[None, :, :], (3,1,1))
    # faces = np.tile(faces[None, :, :], (3,1,1))
    verts = verts[None, :, :]
    faces = faces[None, :, :]
    # verts = torch.nn.Parameter(torch.FloatTensor(verts).cuda())
    # faces = Variable(torch.IntTensor(faces).cuda(), requires_grad=False)

    verts = torch.nn.Parameter(torch.FloatTensor(verts))
    faces = Variable(torch.LongTensor(faces), requires_grad=False)

    laplacian = Laplacian(faces)

    # Dont do this:
    # y = laplacian.forward(verts, faces)
    Lx = laplacian(verts)

    L = laplacian.L.todense()

    from scipy.io import loadmat
    L_want = loadmat('birds3d/test/matlab/L_isosphere3.mat')['L']

    print(L- L_want)
    import ipdb; ipdb.set_trace()

class LaplacianLoss(object):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, faces, vert, toref=True):
        # Input:
        #  faces: B x F x 3
        self.toref = toref
        # V x V
        # self.laplacian = Laplacian(faces)
        Laplacian.F = faces.data
        Laplacian.F_np = faces.data.cpu().numpy()
        self.Lx = None
        self.B = vert.shape[0]
        tmp = Laplacian.apply(vert)
        self.curve_gt = torch.norm(tmp.view(-1, tmp.size(2)), p=2, dim=1).float()
        if not self.toref:
            self.curve_gt = self.curve_gt*0
    
    def __call__(self, verts):
        if verts.shape[0] < self.B:
            verts = torch.cat([verts, verts[0].reshape(-1, 6890, 3).repeat(self.B - verts.shape[0], 1, 1)], 0)
        self.Lx = Laplacian.apply(verts)
        # Reshape to BV x 3
        Lx = self.Lx.view(-1, self.Lx.size(2))
        loss = (torch.norm(Lx, p=2, dim=1).float()-self.curve_gt).mean()
        return loss

class GarmentLaplacianLoss(object):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, faces, vert, toref=True):
        # Input:
        #  faces: B x F x 3
        self.toref = toref
        # V x V
        # self.laplacian = Laplacian(faces)
        GarmentLaplacian.F = faces.data
        GarmentLaplacian.F_np = faces.data.cpu().numpy()
        self.Lx = None
        self.B = vert.shape[0]
        tmp = GarmentLaplacian.apply(vert)
        self.curve_gt = torch.norm(tmp.view(-1, tmp.size(2)), p=2, dim=1).float()
        if not self.toref:
            self.curve_gt = self.curve_gt*0

    def __call__(self, verts):
        if verts.shape[0] < self.B:
            verts = torch.cat([verts, verts[0].reshape(-1, 6890, 3).repeat(self.B - verts.shape[0], 1, 1)], 0)
        self.Lx = GarmentLaplacian.apply(verts)
        # Reshape to BV x 3
        Lx = self.Lx.view(-1, self.Lx.size(2))
        loss = torch.abs(torch.norm(Lx, p=2, dim=1).float()-self.curve_gt).mean()
        return loss

class OnetimeLaplacianLoss(object):
    def __init__(self, faces, verts):
        gt_lap = OnetimeLaplacian.apply(verts, faces.data)
        self.curve_gt = torch.norm(gt_lap.view(-1, gt_lap.size(2)), p=2, dim=1).float()
        self.faces = faces
        self.B = verts.shape[0]

    def __call__(self, verts):
        if verts.shape[0] < self.B:
            verts = torch.cat([verts, verts[0].reshape(-1, 6890, 3).repeat(self.B - verts.shape[0], 1, 1)], 0)
        Lx = OnetimeLaplacian.apply(verts, self.faces.data)
        Lx = Lx.view(-1, Lx.size(2))
        loss = torch.abs(torch.norm(Lx, p=2, dim=1).float()-self.curve_gt).mean()
        return loss

if __name__ == '__main__':
    teapot_smooth_test()
    # test_laplacian()
