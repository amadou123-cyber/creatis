# Implementation of PnP-MM and MLEM

import torch
import deepinv as dinv
from tqdm import tqdm
from tv_dual import TV_dual_denoising


def mlem(
    y: torch.Tensor,
    x0: torch.Tensor,
    stepsize: float,
    physics: dinv.physics.LinearPhysics,
    steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """
    Performs Richardson-Lucy deconvolution on an observed image.

    Args:
        y (torch.Tensor): The observed image
        x_0 (torch.Tensor): The initial estimate
        physics (deepinv.physics.LinearPhysics): The physics operator
        steps (int): Number of iterations
        verbose (bool): Whether to show progress bar
        keep_inter (bool): Whether to keep intermediate results

    Returns:
        torch.Tensor or tuple: The deconvolved image, and if keep_inter=True, list of intermediate results
    """
    xs = [x0.cpu().clone()] if keep_inter else None

    with torch.no_grad():
        recon = x0.clone()
        recon = recon.clamp(min=filter_epsilon)
        s = physics.A_adjoint(torch.ones_like(y))

        for step in tqdm(range(steps), desc="MLEM", disable=not verbose):
            mlem = (recon / s) * physics.A_adjoint(
                y / physics.A(recon).clamp(min=filter_epsilon)
            )
            recon = (1 - stepsize) * recon + stepsize * mlem

            if keep_inter:
                xs.append(recon.cpu().clone())

        return (recon, xs) if keep_inter else recon


def mlem_tv(
    y: torch.Tensor,
    x0: torch.Tensor,
    stepsize: float,
    physics: dinv.physics.LinearPhysics,
    steps: int,
    alpha: int,
    n_iter: int,
    fista,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """
    Performs Richardson-Lucy deconvolution on an observed image.

    Args:
        y (torch.Tensor): The observed image
        x_0 (torch.Tensor): The initial estimate
        physics (deepinv.physics.LinearPhysics): The physics operator
        steps (int): Number of iterations
        verbose (bool): Whether to show progress bar
        keep_inter (bool): Whether to keep intermediate results

    Returns:
        torch.Tensor or tuple: The deconvolved image, and if keep_inter=True, list of intermediate results
    """
    xs = [x0.cpu().clone()] if keep_inter else None

    with torch.no_grad():
        recon = x0.clone()
        recon = recon.clamp(min=filter_epsilon)
        s = physics.A_adjoint(torch.ones_like(y))

        for step in tqdm(
            range(steps),
            desc="MLEM-TV (FISTA)" if fista else "MLEM-TV",
            disable=not verbose,
        ):
            mlem = (recon / s) * physics.A_adjoint(
                y / physics.A(recon).clamp(min=filter_epsilon)
            )
            recon = (1 - stepsize) * recon + stepsize * mlem
            recon = TV_dual_denoising(
                recon,
                s,
                alpha=alpha,
                n_iter=n_iter,
                fista=fista,
            )
            if keep_inter:
                xs.append(recon.cpu().clone())

        return (recon, xs) if keep_inter else recon


def pnp_mm(
    y: torch.Tensor,
    x0: torch.Tensor,
    denoiser: torch.nn.Module,
    sigma: float,
    physics: torch.nn.Module,
    stepsize: float,
    steps: int,
    lambda_reg: float,
    verbose=True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """Plug-and-Play Majorize-Minimize algorithm for Poisson inverse problems."""

    with torch.no_grad():
        s = physics.A_adjoint(torch.ones_like(y))

        xk = x0.clone()
        xs = [x0.cpu().clone()] if keep_inter else None
        costs = []

        for step in tqdm(range(steps), desc="PnP-MM", disable=not verbose):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                recon = denoiser(xk, sigma)

            recon = recon.clamp(min=filter_epsilon)
            recon = lambda_reg * stepsize * recon + (1 - lambda_reg * stepsize) * xk

            # RL / EM updates
            Ax = physics.A(recon).clamp(min=filter_epsilon)
            z = recon * physics.A_adjoint(y / Ax)

            # Compute surrogate step
            sqrt_term = torch.sqrt((recon - s * stepsize) ** 2 + 4 * stepsize * z)
            xk = 0.5 * (recon - s * stepsize + sqrt_term)

            if keep_inter:
                xs.append(xk.cpu().clone())

            cost = torch.sum(Ax - y * torch.log(Ax)).item()
            costs.append(cost)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            xk = denoiser(xk, sigma)
        xk = xk.clamp(min=filter_epsilon)

        return (xk, xs, costs) if keep_inter else xk
