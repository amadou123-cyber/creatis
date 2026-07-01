##########################
# TV regularization
###########################

import torch
from math import sqrt


def pos(x_pos: torch.Tensor) -> torch.Tensor:
    """
    Compute pos(x) = max(0, x)

    Parameters
    ----------
    x_pos : torch.Tensor
        input tensor

    Return
    ------
    torch.Tensor
        ReLU(x_pos)
    """
    return torch.clamp(x_pos, min=0)


def torch_gradient(a: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of a using finite differences

    Parameters
    ----------
    a : torch.Tensor
        input tensor

    Returns
    -------
    torch.Tensor
        Spatial gradient(a) with shape (spatial_dims, *a.shape)
    """
    if a.dim() < 2:
        raise ValueError("torch_gradient expects at least 2D spatial data")

    # By convention in this repo, tensors are shaped (B, C, *spatial_dims).
    # We compute TV over spatial dimensions only.
    spatial_dims = a.dim() - 2 if a.dim() >= 3 else a.dim()
    spatial_axes = tuple(range(a.dim() - spatial_dims, a.dim()))

    # torch.gradient requires each differentiated axis size >= edge_order+1.
    # If a spatial axis is degenerate (size 1), its gradient is identically 0.
    grads = []
    for axis in spatial_axes:
        if a.shape[axis] < 2:
            grads.append(torch.zeros_like(a))
        else:
            grads.append(torch.gradient(a, dim=axis, edge_order=1)[0])
    return torch.stack(grads, dim=0)


def torch_gradient_div(a: torch.Tensor) -> torch.Tensor:
    """
    Compute backward gradient of a using finite differences

    Parameters
    ----------
    a : torch.Tensor
        input tensor

    Returns
    -------
    torch.Tensor
        Spatial backward gradient(a) with shape (spatial_dims, *a.shape)
    """
    if a.dim() < 2:
        raise ValueError("torch_gradient_div expects at least 2D spatial data")

    spatial_dims = a.dim() - 2 if a.dim() >= 3 else a.dim()
    spatial_axes = tuple(range(a.dim() - spatial_dims, a.dim()))

    grads = []
    for axis in spatial_axes:
        # edge_order=2 needs size >= 3; otherwise fall back to edge_order=1 (or 0 if degenerate)
        if a.shape[axis] < 2:
            grads.append(torch.zeros_like(a))
        elif a.shape[axis] < 3:
            grads.append(torch.gradient(a, dim=axis, edge_order=1)[0])
        else:
            grads.append(torch.gradient(a, dim=axis, edge_order=2)[0])
    return torch.stack(grads, dim=0)


def torch_divergence(u: torch.Tensor) -> torch.Tensor:
    """
    Compute divergence of u

    Parameters
    ----------
    u : torch.Tensor
        vector field with shape (dim, *spatial_dims)

    Returns
    -------
    torch.Tensor
        divergence(u)
    """
    if u.dim() < 3:
        raise ValueError(
            "torch_divergence expects a vector field shaped (spatial_dims, B, C, *spatial)"
        )

    spatial_dims = u.shape[0]
    a = u[0]
    if a.dim() < 2:
        raise ValueError("torch_divergence expects at least 2D spatial data")

    spatial_axes = tuple(range(a.dim() - spatial_dims, a.dim()))
    if len(spatial_axes) != spatial_dims:
        raise ValueError(
            f"Inconsistent shapes: u has {spatial_dims} components but inferred {len(spatial_axes)} spatial axes"
        )

    div = torch.zeros_like(a)
    for d, axis in enumerate(spatial_axes):
        if a.shape[axis] < 2:
            continue
        div = div + torch.gradient(u[d], dim=axis, edge_order=1)[0]
    return div


def torch_module(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2 norm along the first dimension

    Parameters
    ----------
    q : torch.Tensor
        tensor with shape (dim, *spatial_dims)

    Return
    ------
    torch.Tensor
        L2 norm along dim 0
    """
    return torch.linalg.norm(q, dim=0)


def torch_TV(x: torch.Tensor) -> float:
    """
    Compute TV norm of x

    Parameters
    ----------
    x : torch.array

    Returns
    -------
    res : float()
          TV norm of x
    """
    # x = crop(x,10)
    grad = torch_gradient(x)
    res = torch_module(grad)
    return float(torch.sum(res))


def div_zer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, "Both matrix should have the same size"
    new = torch.zeros_like(a)
    mask = b > 0
    new[mask] = a[mask] / b[mask]
    return new


def TV_dual_denoising(
    x: torch.Tensor,
    sensitivity: torch.Tensor,
    alpha: float,
    n_iter: int = 50,
    epsilon: float = 1e-8,
    fista: bool = False,
):
    """
    Compute dual denoising, from Maxim2018. Used for Poisson noise

    Parameters
    ----------
    x : np.array
        image to denoise
    sensitivity : np.array
        sensitivity
    alpha : float
            TV weight parameter
    n_iter : int
             Number of iteration
    epsilon : float
              Minimum value of the returned image
    fista : bool
            Whether to use heuristic FISTA acceleration or not
    Returns
    -------
    f : np.array
        denoised image
    """

    # Determine spatial dimensions (exclude batch and channel dimensions)
    spatial_dims = x.dim() - 2

    # crop sensitivity to get rid of zero value
    sensitivity[sensitivity < 0.1] = 0.1
    if spatial_dims == 2:
        den = sensitivity - 4 * alpha
        den[den < 0] = torch.min(sensitivity)
        # Lh = 8 * alpha**2 * sensitivity * x / den**2
        Lh = 8 * alpha**2 * x / den**2

    elif spatial_dims == 3:
        den = sensitivity - 6 * alpha
        den[den < 0] = torch.min(sensitivity)
        Lh = 12 * alpha**2 * sensitivity * x / den**2

    tau = 0.5 * alpha * div_zer(torch.ones_like(Lh), Lh)

    # Dual variable phi is a vector field with one component per spatial dimension.
    phi = torch.zeros((spatial_dims,) + x.shape, device=x.device, dtype=torch.float32)
    phi_ = torch.zeros_like(phi)
    tTV = 1

    for k in range(n_iter):
        z = pos(div_zer(sensitivity * x, sensitivity + alpha * torch_divergence(phi)))
        phi__ = phi - tau * torch_gradient(z)
        norm_phi = torch_module(phi__)
        denom = torch.maximum(norm_phi, torch.ones_like(norm_phi))
        phi__ = phi__ / denom.unsqueeze(0)

        # FISTA
        if fista:
            tTV_ = (1 + sqrt(1 + 4 * tTV**2)) * 0.5
            phi = phi__ + (tTV - 1) / tTV_ * (phi__ - phi_)
            tTV = tTV_
            phi_ = phi__
        else:
            phi = phi__

    x = div_zer(x, 1 + alpha * torch_divergence(phi) / sensitivity)
    # Ensure we don't return negative values
    x[x < epsilon] = epsilon

    return x
