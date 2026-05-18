import numpy as np
import torch
from torch import tensor
class MeshInfo:
    # Initialize the mesh information
    def __init__(self, cfg):
        """Initialize the mesh information."""
        geoX = cfg.geoX
        geoY = cfg.geoY
        nodesx = cfg.nodesx
        nodesy = cfg.nodesy
        device = cfg.device
        self.cfg = cfg
        self.geoX = geoX
        self.geoY = geoY
        self.nelx = nodesx - 1
        self.nely = nodesy - 1
        self.nEl = self.nelx * self.nely
        self.nGauss = 4
        self.ex = geoX / self.nelx
        self.ey = geoY / self.nely
        self.nodesx = nodesx
        self.nodesy = nodesy
        self.nNods = nodesx * nodesy
        # Get the node coordinates
        self.X, self.Y = np.meshgrid(np.linspace(0, geoX, nodesx),
                                     np.linspace(0, geoY, nodesy))
        self.coord = np.vstack((self.X.ravel(), self.Y.ravel())).T
        # Get the nodes ID
        meshNods = np.int32(np.reshape(np.arange(1, (nodesx * nodesy) + 1, 1), (nodesy, nodesx)))
        eleNodsID = np.reshape(
            meshNods[:-1, :-1], (self.nEl, 1)
        ) + np.array(
            [0, 1, self.nodesx + 1, self.nodesx]
        )
        self.eleNodsID = eleNodsID
        # Get the element node ID
        self.xEle = torch.from_numpy(
            np.take(self.coord[:, 0], eleNodsID - 1)
        ).float().to(device)
        self.yEle = torch.from_numpy(
            np.take(self.coord[:, 1], eleNodsID - 1)
        ).float().to(device)
        # Get the degree of freedom
        self.nDOF = 2 * nodesx * nodesy
        # Get DOF of each element
        ## Define the first DOF of each element at leftdown
        cVec = np.reshape(2 * meshNods[:-1, :-1] + 1, (self.nEl, 1))
        ## Define other DOF of each element
        cMat = cVec + np.array([-2, -1, 0, 1, 2 * self.nelx + 2, 2 * self.nelx + 3, 2 * self.nelx, 2 * self.nelx + 1])
        self.cMat = cMat
        # Set device
        self.device = device
        # Get K0
        self.K0 = self.get_K0()

    def get_K0(self):
        # Get shape function value at gauss point [nGauss,nShape]
        device = self.device
        nEl = self.nEl
        xEle = self.xEle
        yEle = self.yEle
        N, dNs, dNt, w = ShapeFunAtGauss(device)
        nGauss, nShape = N.shape
        # Get initial D0
        mu = 0.3
        D0 = tensor([[1, mu, 0], [mu, 1, 0],
                     [0, 0, 0.5 * (1 - mu)]],
                    device=device) \
             / (1 - mu ** 2)
        # Get the Jacobian matrix at each Gauss point
        # [nGauss, nEl, 1]
        dxds = dNs @ xEle.T
        dxdt = dNt @ xEle.T
        dyds = dNs @ yEle.T
        dydt = dNt @ yEle.T
        detJ = dxds * dydt - dyds * dxdt
        dxds = (dxds / detJ).unsqueeze(-1)
        dxdt = (dxdt / detJ).unsqueeze(-1)
        dyds = (dyds / detJ).unsqueeze(-1)
        dydt = (dydt / detJ).unsqueeze(-1)
        # Get grad(u) & grad(v) at each Gauss point in s-t coord
        duds = torch.zeros(nGauss, 1, nShape * 2).to(device)
        dudt = torch.zeros(nGauss, 1, nShape * 2).to(device)
        dvds = torch.zeros(nGauss, 1, nShape * 2).to(device)
        dvdt = torch.zeros(nGauss, 1, nShape * 2).to(device)
        duds[:, 0, 0::2] = dNs
        dudt[:, 0, 0::2] = dNt
        dvds[:, 0, 1::2] = dNs
        dvdt[:, 0, 1::2] = dNt
        # Get grad(u) & grad(v) at each Gauss point in x-y coord
        # [nGauss, nEl, 8]
        dudx = dydt @ duds - dyds @ dudt
        dudy = -dxdt @ duds + dxds @ dudt
        dvdx = dydt @ dvds - dyds @ dvdt
        dvdy = -dxdt @ dvds + dxds @ dvdt
        # Get strain
        # [nGauss, nEl, nShape * 2]
        ex = dudx
        ey = dvdy
        exy = dudy + dvdx
        # Get B at each Gauss point [4, nEl, 3, nShape * 2]
        B = torch.concatenate((ex, ey, exy), dim=-1).reshape((nGauss, nEl, 3, nShape * 2))
        # Calculate BDB without E
        D0detJ = torch.kron(detJ.reshape((1, -1)), D0).T.reshape((nGauss, nEl, 3, 3))
        K0 = w[0] * B.transpose(2, 3) @ D0detJ @ B
        return K0

    def get_ele_dof(self, dof):
        nEl = self.nEl
        cMat = self.cMat
        cMat = cMat.reshape((-1)) - 1
        total_dof = dof[:, cMat]
        dof_ele = total_dof.reshape((-1, nEl, 8))
        return dof_ele.unsqueeze(3)

    def get_IDlist(self, batch_size):
        device = self.device
        nNods = self.nNods
        eleNodsID = self.eleNodsID
        batch_list = nNods * np.arange(batch_size).reshape((-1, 1))
        NodsID = (eleNodsID - 1).reshape((1, -1))
        IDlist = NodsID + batch_list
        return torch.from_numpy(IDlist.reshape(-1)).long().to(device)

    def get_gauss_modulus(self, output):
        # Get batch size
        batch_size = output.shape[0]
        # Get device
        device = self.device
        # Change the Size of E to
        ## E_vec [Batch_size,nodesy*nodesx]
        E_vec = output.reshape((batch_size, -1))
        # Get modulus at each node in each element
        eleNodsID = self.eleNodsID
        nEl = self.nEl
        eleNodsID = eleNodsID.reshape((-1)) - 1
        Eele = E_vec[:, eleNodsID]
        ## Eele [batch_size,nEl,4]
        Eele = Eele.reshape((-1, nEl, 4))
        # Get modulus at each gauss point
        ## [batch_size,nEl,nGauss]
        N, _, _, _ = ShapeFunAtGauss(device)
        GaussE = Eele @ N.T
        return GaussE

    def get_tot_grad(self, gaussE, dof_ele, res_ele):
        # Get batch size
        batch_size = gaussE.shape[0]
        # Get device
        device = self.device
        # Calculate gradients for each element
        ## gradE [batch_size,nEl]
        gradE1, gradE2, gradE3, gradE4 = self.get_ele_grad(res_ele, dof_ele)
        # Get IDlist for modulus points
        IDlist = self.get_IDlist(batch_size)
        # Assemble all the gradients
        ## grad [batch_size,nodesy,nodesx]
        grad_all = torch.stack((gradE1, gradE2, gradE3, gradE4), dim=2)
        grad = torch.zeros(
            (batch_size * self.nNods), device=device
        ).scatter_reduce(0, IDlist, grad_all.reshape(-1), reduce='sum')
        return grad.reshape((batch_size, self.nodesy, self.nodesx))

    def get_ele_grad(self, res_ele, dof_ele):
        # Get device
        device = self.device
        # Get K0
        K0 = self.K0
        # Calculate derivatives of the stiffness matrix to node modulus
        ## dKdE [nEl,8,8]
        N, _, _, _ = ShapeFunAtGauss(device)
        dKdE1 = torch.einsum('i,i...->...', N[:, 0], K0)
        dKdE2 = torch.einsum('i,i...->...', N[:, 1], K0)
        dKdE3 = torch.einsum('i,i...->...', N[:, 2], K0)
        dKdE4 = torch.einsum('i,i...->...', N[:, 3], K0)
        # Calculate gradients for each element
        gradE1 = (torch.einsum('ijkl,jlm->ijkm',
                               res_ele.transpose(2, 3),
                               dKdE1
                               ) @ dof_ele).squeeze(-1).squeeze(-1)
        gradE1 *= 2
        gradE2 = (torch.einsum('ijkl,jlm->ijkm',
                               res_ele.transpose(2, 3),
                               dKdE2
                               ) @ dof_ele).squeeze(-1).squeeze(-1)
        gradE2 *= 2
        gradE3 = (torch.einsum('ijkl,jlm->ijkm',
                               res_ele.transpose(2, 3),
                               dKdE3
                               ) @ dof_ele).squeeze(-1).squeeze(-1)
        gradE3 *= 2
        gradE4 = (torch.einsum('ijkl,jlm->ijkm',
                               res_ele.transpose(2, 3),
                               dKdE4
                               ) @ dof_ele).squeeze(-1).squeeze(-1)
        gradE4 *= 2
        return gradE1, gradE2, gradE3, gradE4

class LocRes(MeshInfo):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_ele_res(self, gaussE, dof_ele, force_ele):
        # Get batch size
        batch_size = gaussE.shape[0]
        # Calculate stiffness matrix with modulus values
        ## totalK [nGauss,nEl,64]
        K0 = self.K0
        totalK = K0.reshape((4, -1, 64))
        ## Kele [batch_size,nEl,8,8]
        Kele = torch.einsum('ijk,kjl->ijl', gaussE, totalK).reshape((batch_size, -1, 8, 8))
        # Calculate residual forces for each element
        Rele = Kele @ dof_ele - force_ele
        return Rele

class GloRes(MeshInfo):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.K0Vec = self.get_K0_vec()
        self.Ivar = self.get_Ivar()

    def get_K0_vec(self):
        K0 = self.K0
        # Get upper triangle indices
        si = []
        for i in range(8):
            for j in range(i, 8):
                si.append(i * 8 + j)
        si = tensor(si, device=self.device)
        # Get triu of K0
        K0Vec = K0.reshape((4, self.nEl, -1))
        K0Vec = K0Vec[:, :, si]
        return K0Vec

    def get_Ivar(self):
        device = self.device
        cMat = self.cMat
        # Generate degree list of global stiffness matrix
        si = torch.tensor([], dtype=torch.int32, device=device)
        sii = torch.tensor([], dtype=torch.int32, device=device)
        for i in range(1, 9):
            si = torch.cat((si, torch.arange(i, 9, device=device)))
            sii = torch.cat((sii, torch.full((8 - i + 1,), i, device=device)))
        cMat_tensor = torch.from_numpy(cMat).to(device)
        ik = cMat_tensor[:, si - 1]
        jk = cMat_tensor[:, sii - 1]
        ik_vec = ik.reshape(1, -1).T
        jk_vec = jk.reshape(1, -1).T
        Ivar, _ = torch.sort(torch.cat((ik_vec, jk_vec), dim=1), descending=True, dim=1)
        return Ivar - 1

    def get_tot_K(self, gaussE):
        # Get batch size
        batch_size = gaussE.shape[0]
        # Get device
        device = gaussE.device
        ## GaussE [nEl,batch_size,nGauss]
        GaussE = gaussE.transpose(0, 1)
        ## K0Vec [nEl,nGauss,36]
        K0Vec = self.K0Vec
        K0Vec = K0Vec.transpose(0, 1)
        # Get Local Stiffness Matrix
        K_loc = GaussE @ K0Vec
        ## [batch_size,nEl,36]
        K_loc = K_loc.transpose(0, 1)
        # Get sparse matrix values
        nDOF = self.nDOF
        # Get Ivar for 3D total stiffness matrix
        # [3, batch_size*len(ik_vec)]
        Ivar = self.Ivar
        rep_num = Ivar.shape[0]
        Ivar = Ivar.repeat(batch_size, 1).T
        batch_list = torch.arange(batch_size, device=device).repeat(rep_num, 1).T.reshape((1, -1))
        Ivar = torch.vstack((batch_list, Ivar))
        K_triu = torch.sparse_coo_tensor(
            indices=Ivar,
            values=K_loc.reshape(-1),
            size=(batch_size, nDOF, nDOF)
        )
        K_triu = K_triu.coalesce()
        K_tril = K_triu.transpose(1, 2)
        # Generate total Stiffness Matrix
        K_triu = K_triu.to_dense()
        K_diag = torch.diagonal(K_triu, dim1=1, dim2=2)
        K_diagnal = torch.diag_embed(K_diag, dim1=1, dim2=2)
        K = K_triu + K_tril - K_diagnal
        K = K.to_sparse_coo()
        return K

def ShapeFunAtGauss(device):
    gpt = 1 / torch.sqrt(tensor(3.0))
    # Gauss Point clockwise from (-1,1)
    s = gpt * tensor([[-1], [1], [1], [-1]], device=device)
    t = gpt * tensor([[-1], [-1], [1], [1]], device=device)
    w = tensor([[1], [1], [1], [1]], device=device)
    # Node Counterclockwise From LeftDown
    N1 = (1.0 - s) * (1.0 - t) / 4.0
    N2 = (1.0 + s) * (1.0 - t) / 4.0
    N3 = (1.0 + s) * (1.0 + t) / 4.0
    N4 = (1.0 - s) * (1.0 + t) / 4.0
    N1s = -(1.0 - t) / 4.0
    N1t = -(1.0 - s) / 4.0
    N2s = (1.0 - t) / 4.0
    N2t = -(1.0 + s) / 4.0
    N3s = (1.0 + t) / 4.0
    N3t = (1.0 + s) / 4.0
    N4s = -(1.0 + t) / 4.0
    N4t = (1.0 - s) / 4.0
    N = torch.hstack([N1, N2, N3, N4])
    dNs = torch.hstack([N1s, N2s, N3s, N4s])
    dNt = torch.hstack([N1t, N2t, N3t, N4t])
    return N, dNs, dNt, w
