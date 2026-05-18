import torch

def LocMixloss(output, target, dof, force_ele, sample, gamma):
    # Get loss value and gradient of Local losses
    loss_phy, grad_phy = LocResloss(output.detach(), dof, force_ele, sample)
    # Get MSE loss and gradient
    mse = torch.nn.MSELoss()
    loss_mse = mse(output, target)
    batch_size = output.shape[0]
    nodesx = sample.nodesx
    nodesy = sample.nodesy
    grad_mse = 2 * (output - target) / (batch_size * nodesy * nodesx)
    # Calculate total loss and gradient
    loss = loss_mse + gamma * loss_phy
    grad = grad_mse + gamma * grad_phy
    return loss, loss_mse, loss_phy, grad

def GloMixloss(output, target, dof, force, sample, gamma):
    # Get loss value and gradient of Local losses
    loss_phy, grad_phy = GloResloss(output.detach(), dof, force, sample)
    # Get MSE loss and gradient
    mse = torch.nn.MSELoss()
    loss_mse = mse(output, target)
    batch_size = output.shape[0]
    nodesx = sample.nodesx
    nodesy = sample.nodesy
    grad_mse = 2 * (output - target) / (batch_size * nodesy * nodesx)
    # Calculate total loss and gradient
    loss = loss_mse + gamma * loss_phy
    grad = grad_mse + gamma * grad_phy
    return loss, loss_mse, loss_phy, grad

def LocResloss(output, dof, force_ele, sample):
    batch_size = output.shape[0]
    # Get modulus values at Gauss points
    gaussE = sample.get_gauss_modulus(output)
    # Get residual forces of elements
    dof_ele = sample.get_ele_dof(dof)
    res_ele = sample.get_ele_res(gaussE, dof_ele, force_ele)
    # Get loss value
    locResloss = torch.sum(
        torch.pow(torch.norm(res_ele, dim=(2, 3)), 2),
           dim=1)
    # Get Gradient
    grad = sample.get_tot_grad(gaussE, dof_ele, res_ele) / batch_size
    return locResloss.mean(), grad

def GloResloss(output, dof, force, sample):
    batch_size = output.shape[0]
    # Get modulus values at Gauss points
    gaussE = sample.get_gauss_modulus(output)
    # Compute per-element stiffness: K_e = sum_i E_i^(e) * K0^(e,i)
    # K0 shape: (4, n_el*64) -> reshape to (4, n_el, 64)
    totK = sample.K0.reshape((4, -1, 64))
    # Kele: (B, n_el, 8, 8)
    Kele = torch.einsum('ijk,kjl->ijl', gaussE, totK).reshape((batch_size, -1, 8, 8))
    # Get element DOF vectors: (B, n_el, 8, 1)
    dof_ele = sample.get_ele_dof(dof)
    # Per-element K_e @ u_e: (B, n_el, 8, 1)
    Ku_ele = Kele @ dof_ele
    # Assemble global K @ u via scatter_add
    # cMat: (nEl, 8) numpy int32, 1-indexed -> convert to 0-indexed long tensor
    cMat_idx = torch.from_numpy(sample.cMat.flatten() - 1).long().to(output.device)
    Ku_global = torch.zeros(batch_size, sample.nDOF, device=output.device)
    Ku_global.scatter_add_(
        1,
        cMat_idx.unsqueeze(0).expand(batch_size, -1),
        Ku_ele.squeeze(3).reshape(batch_size, -1)
    )
    # Residual: K @ u - F, shape (B, nDOF)
    res = Ku_global - force
    # Loss: ||K @ u - F||_2^2
    gloResloss = torch.pow(torch.norm(res, dim=1), 2)
    # Get element residuals and analytical gradient
    res_ele = sample.get_ele_dof(res)
    grad = sample.get_tot_grad(gaussE, dof_ele, res_ele) / batch_size
    return gloResloss.mean(), grad
