import torch
from tqdm import tqdm
import deepinv as dinv
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
images = [
    "celeba_example.jpg",
    "JAX_018_011_RGB.tif",
    "CBSD_0010.png",
    "FMD_TwoPhoton_MICE_R_gt_12_avg50.png",
    "mbappe.jpg",
    "leaves.png",
    "barbara.jpeg",
    "butterfly.png",
]

for image_name in images:
    print(f"------------------- Processing GD -> {image_name} --------------------")

    x = dinv.utils.demo.load_example(image_name, device=device, img_size=128)

    # You can also use other noise models but my algorithm is designed for Poisson noise
    gain = 1 / 60
    noise_model = dinv.physics.PoissonNoise(gain=gain)

    # Here you can replace with any other linear physics operator
    # I've had particularly good results on deblurring and tomography
    psf = dinv.physics.blur.gaussian_blur(sigma=1.6)
    physics = dinv.physics.Blur(
        filter=psf,
        noise_model=noise_model,
        padding="circular",
        device=device,
    )

    y = physics(x)

    # Just a quick look at the measurement
    dinv.utils.plot(
        [x, y], titles=["Original", "Blurred + Poisson Noise"], figsize=(8, 4)
    )

    # A baseline algorithm to compare to
    # Here since it's deconvolution I use Richardson-Lucy
    # which is more often called MLEM in tomography
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
        costs = []
        with torch.no_grad():
            recon = x0.clone()
            recon = recon.clamp(min=filter_epsilon)
            s = physics.A_adjoint(torch.ones_like(y))

            for step in tqdm(
                range(steps), desc="MLEM-" + image_name, disable=not verbose
            ):
                mlem = (recon / s) * physics.A_adjoint(
                    y / physics.A(recon).clamp(min=filter_epsilon)
                )
                recon = (1 - stepsize) * recon + stepsize * mlem

                if keep_inter:
                    xs.append(recon.cpu().clone())
                cost = torch.sum(
                    physics.A(recon) - y * torch.log(physics.A(recon))
                ).item()
                costs.append(cost)
            return (recon, xs, costs) if keep_inter else recon

    # Here since it's deconvolution I use Richardson-Lucy
    # which is more often called MLEM in tomography
    def mlem_regularization(
        y: torch.Tensor,
        x0: torch.Tensor,
        stepsize: float,
        physics: dinv.physics.LinearPhysics,
        steps: int,
        prior,
        lambda_reg: float,
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
        costs = []
        with torch.no_grad():
            recon = x0.clone()
            recon = recon.clamp(min=filter_epsilon)
            s = physics.A_adjoint(torch.ones_like(y))

            for step in tqdm(
                range(steps), desc="MLEM Reg-" + image_name, disable=not verbose
            ):
                mlem = (
                    recon
                    / (s + lambda_reg * prior.grad(recon).clamp(min=filter_epsilon))
                ) * physics.A_adjoint(y / physics.A(recon).clamp(min=filter_epsilon))
                recon = (1 - stepsize) * recon + stepsize * mlem

                if keep_inter:
                    xs.append(recon.cpu().clone())
                cost = torch.sum(
                    physics.A(recon) - y * torch.log(physics.A(recon))
                ).item()
                costs.append(cost)
            return (recon, xs, costs) if keep_inter else recon

    def gd(
        y: torch.Tensor,
        x0: torch.Tensor,
        stepsize: float,
        physics: dinv.physics.LinearPhysics,
        steps: int,
        verbose: bool = True,
        keep_inter: bool = False,
    ) -> torch.Tensor:

        xs = [x0.cpu().clone()] if keep_inter else None

        data_fidelity = dinv.optim.L2()
        prior_l2 = dinv.optim.Tikhonov()
        # A. Manual GD Loop
        x_manual = y.clone()
        lambd = 0.01
        costs = []
        for step in tqdm(range(steps), desc="GD-" + image_name, disable=not verbose):
            x_manual.requires_grad_(True)
            grad = data_fidelity.grad(x_manual, y, physics) + lambd * prior_l2.grad(
                x_manual
            )
            with torch.no_grad():
                x_manual = x_manual - stepsize * grad
                if keep_inter:
                    xs.append(x_manual.cpu().clone())
                cost = data_fidelity(x_manual, y, physics) + lambd * prior_l2(x_manual)
                costs.append(cost.item())
        return (x_manual, xs, costs) if keep_inter else x_manual

    def pgd(
        y: torch.Tensor,
        x0: torch.Tensor,
        stepsize: float,
        physics: dinv.physics.LinearPhysics,
        steps: int,
        verbose: bool = True,
        keep_inter: bool = False,
    ) -> torch.Tensor:
        xs = [x0.cpu().clone()] if keep_inter else None

        data_fidelity = dinv.optim.L2()
        prior = dinv.optim.TVPrior()
        lambd = 0.05
        norm_A2 = physics.compute_sqnorm(y, tol=1e-4, verbose=False).item()
        stepsize = 1.9 / norm_A2
        x_k = torch.zeros_like(x, device=device)
        costs = []
        with torch.no_grad():
            for step in tqdm(
                range(steps), desc="PGD-" + image_name, disable=not verbose
            ):
                u = x_k - stepsize * data_fidelity.grad(x_k, y, physics)
                x_k = prior.prox(u, gamma=stepsize * lambd)

                if keep_inter:
                    xs.append(x_k.cpu().clone())

                cost = data_fidelity(x_k, y, physics) + lambd * prior(x_k)
                costs.append(cost.item())
            return (x_k, xs, costs) if keep_inter else x_k

    # The implementation of PnP-MM used throughout the paper
    def pnp_mm(
        y: torch.Tensor,
        x0: torch.Tensor,
        denoiser: torch.nn.Module,
        sigma: float,
        physics: torch.Tensor,
        stepsize: float,
        steps: int,
        lambda_reg: float,
        verbose=True,
        keep_inter: bool = False,
        filter_epsilon: float = 1e-20,
    ) -> torch.Tensor:
        """
        Plug-and-Play Majorize-Minimize algorithm for Poisson inverse problems.

        Args:
            y: Observed data
            x0: Initial estimate
            denoiser: Neural network denoiser
            sigma: Noise level for denoiser
            physics: Physics operator with A() and A_adjoint() methods
            stepsize: Step size for updates
            steps: Number of iterations
            lambda_reg: Regularization parameter
            verbose: Show progress bar
            keep_inter: Store intermediate results
            filter_epsilon: Small value to prevent division by zero

        Returns:
            Final reconstruction or (reconstruction, intermediates) if keep_inter=True
        """
        with torch.no_grad():
            s = physics.A_adjoint(torch.ones_like(y))

            xk = x0.clone()
            xs = [x0.cpu().clone()] if keep_inter else None

            # From my experience, this speeds things up quite a lot
            # without significant loss of performances
            costs = []
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                for step in tqdm(
                    range(steps), desc="PNP_MM-" + image_name, disable=not verbose
                ):

                    recon = denoiser(xk, sigma).clamp(
                        min=filter_epsilon
                    )  # D_theta(x^{(n)})
                    recon = (
                        lambda_reg * stepsize * recon + (1 - lambda_reg * stepsize) * xk
                    )

                    # RL updates
                    z = recon * physics.A_adjoint(
                        y / physics.A(recon).clamp(min=filter_epsilon)
                    )

                    # Compute the square root
                    sqrt_term = torch.sqrt(
                        (recon - s * stepsize) ** 2 + 4 * stepsize * z
                    )

                    xk = 0.5 * (recon - s * stepsize + sqrt_term)

                    if keep_inter:
                        xs.append(xk.cpu().clone())
                    cost = torch.sum(
                        physics.A(recon) - y * torch.log(physics.A(recon))
                    ).item()
                    costs.append(cost)
                # final denoising step
                xk = denoiser(xk, sigma).clamp(min=filter_epsilon)

            return (xk, xs, costs) if keep_inter else xk

    # Load a pretrained gradient-step denoiser
    # This one is pretrained on natural images
    model_gs = dinv.models.GSDRUNet(pretrained="download").to(device)

    # Define hyperparameters for the algorithm
    # The product stepsize * lambda_reg should be less than 1
    # The noise level sigma is critical, bigger values will have stronger effects on low frequencies
    # and conversely for smaller values

    stepsize = 0.5
    lambda_reg = 0.9
    sigma = 15.0 / 255.0
    steps = 400

    x0 = physics.A_adjoint(y)

    x_mlem, xs_mlem, costs_mlem = mlem(
        y=y,
        x0=x0,
        physics=physics,
        stepsize=stepsize,
        steps=steps,
        verbose=True,
        keep_inter=True,
    )
    x_mlem_reg, xs_mlem_reg, costs_mlem_reg = mlem_regularization(
        y=y,
        x0=x0,
        physics=physics,
        stepsize=stepsize,
        steps=steps,
        prior=dinv.optim.Tikhonov(),
        lambda_reg=lambda_reg,
        verbose=True,
        keep_inter=True,
    )
    x_pnpmm, xs_pnpmm, costs_pnpmm = pnp_mm(
        y=y,
        x0=x0,
        denoiser=model_gs,
        sigma=sigma,
        physics=physics,
        stepsize=stepsize,
        steps=steps,
        lambda_reg=lambda_reg,
        verbose=True,
        keep_inter=True,
    )
    x_pgd, xs_pgd, costs_pgd = pgd(
        y=y,
        x0=x0,
        stepsize=stepsize,
        steps=steps,
        verbose=True,
        keep_inter=True,
        physics=physics,
    )
    x_gd, xs_gd, costs_gd = gd(
        y=y,
        x0=x0,
        stepsize=stepsize,
        steps=steps,
        verbose=True,
        keep_inter=True,
        physics=physics,
    )
    # Plot the results
    psnr = dinv.metric.PSNR()

    dinv.utils.plot(
        [x, y, x0, x_mlem, x_mlem_reg, x_pnpmm, x_pgd, x_gd],
        titles=[
            "Ground Truth",
            "Measurement",
            "Initialization",
            "MLEM",
            "MLEM + Reg",
            "PnP-MM",
            "PGD",
            "GD",
        ],
        subtitles=[
            "PSNR (dB):",
            f"{psnr(y, x).item():.2f}",
            f"{psnr(x0, x).item():.2f}",
            f"{psnr(x_mlem, x).item():.2f}",
            f"{psnr(x_mlem_reg, x).item():.2f}",
            f"{psnr(x_pnpmm, x).item():.2f}",
            f"{psnr(x_pgd, x).item():.2f}",
            f"{psnr(x_gd, x).item():.2f}",
        ],
        figsize=(12, 6),
    )

    # Evolution of the PSNR values
    x_cpu = x.cpu()
    psnr_values_mlem = [psnr(xi, x_cpu).item() for xi in xs_mlem]
    psnr_values_pnpmm = [psnr(xi, x_cpu).item() for xi in xs_pnpmm]
    psnr_values_pgd = [psnr(xi, x_cpu).item() for xi in xs_pgd]
    psnr_values_gd = [psnr(xi, x_cpu).item() for xi in xs_gd]
    psnr_values_mlem_reg = [psnr(xi, x_cpu).item() for xi in xs_mlem_reg]
    colors = {
        "MLEM": "#E74C3C",  # Rouge (Méthode classique, souvent instable)
        "PnP-MM": "#2ECC71",  # Vert (Méthode moderne, performante)
        "PGD": "#3498DB",  # Bleu (Standard de l'optimisation)
        "GD": "#F1C40F",  # Jaune/Or (Référence de base)
        "MLEM+Reg": "#9B59B6",  # Violet (MLEM avec régularisation)
    }

    plt.figure(figsize=(10, 6))

    # Tracé avec les nouvelles couleurs
    plt.plot(psnr_values_mlem, label="MLEM (Base)", color=colors["MLEM"], linewidth=2)
    plt.plot(
        psnr_values_pnpmm,
        label="PnP-MM (Deep Prior)",
        color=colors["PnP-MM"],
        linewidth=2.5,
    )
    plt.plot(
        psnr_values_pgd,
        label="PGD (Proximal)",
        color=colors["PGD"],
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        psnr_values_gd,
        label="GD (Gradient)",
        color=colors["GD"],
        linewidth=2,
        linestyle=":",
    )
    plt.plot(
        psnr_values_mlem_reg,
        label="MLEM + Tikhonov Reg",
        color=colors["MLEM+Reg"],
        linewidth=2,
    )
    # Cosmétique améliorée
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.title("Comparaison de la Convergence PSNR", fontsize=14, fontweight="bold")
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    # Graphs for costs
    # --- Affichage du Cost History ---
    plt.figure(figsize=(10, 6))

    # On utilise l'échelle Log car les échelles de coût diffèrent énormément entre Poisson et L2
    plt.plot(costs_mlem, label="MLEM (Poisson Loss)", color=colors["MLEM"], lw=2)
    plt.plot(costs_pnpmm, label="PnP-MM (Poisson Loss)", color=colors["PnP-MM"], lw=2.5)
    plt.plot(
        costs_pgd, label="PGD (L2 + TV)", color=colors["PGD"], lw=2, linestyle="--"
    )
    plt.plot(costs_gd, label="GD (L2 + Tikhonov)", color=colors["GD"], lw=2, linestyle=":")
    plt.plot(
        costs_mlem_reg,
        label="MLEM + Tikhonov Reg (Poisson Loss)",
        color=colors["MLEM+Reg"],
        lw=2,
    )
    plt.yscale("log")  # Très important pour voir la convergence
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cost Value (Log Scale)", fontsize=12)
    plt.title(f"Convergence du Coût - {image_name}", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
