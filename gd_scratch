import deepinv as dinv
import torch
import matplotlib.pyplot as plt

device = dinv.utils.get_device()
# Reduced maxiter for efficiency; 100-500 is usually enough for visual convergence with GD
maxiter = 100
images = [
    "celeba_example.jpg",
    "JAX_018_011_RGB.tif",
    "CT100_256x256_0.pt",
    "CBSD_0010.png",
    "FMD_TwoPhoton_MICE_R_gt_12_avg50.png",
    "mbappe.jpg",
    "leaves.png",
    "barbara.jpeg",
    "butterfly.png",
]

for image_name in images:
    print(f"------------------- Processing GD -> {image_name} --------------------")

    # 1. Setup Data
    x = dinv.utils.load_example(image_name, grayscale=True, img_size=256, device=device)
    physics = dinv.physics.Blur(
        filter=dinv.physics.blur.gaussian_blur(sigma=2.5), padding="circular"
    ).to(device)
    y = physics(x)

    metric = dinv.metric.PSNR()
    data_fidelity = dinv.optim.L2()
    prior_l2 = dinv.optim.Tikhonov()

    lambd = 0.01  # Added a small lambda to actually use the prior
    stepsize = 1.0

    # ---------------------------------------------------------
    # 2. Reconstructions
    # ---------------------------------------------------------

    # A. Manual GD Loop
    x_manual = y.clone()
    cost_history = []
    psnr_history = []

    for k in range(maxiter):
        x_manual.requires_grad_(True)
        grad = data_fidelity.grad(x_manual, y, physics) + lambd * prior_l2.grad(
            x_manual
        )

        with torch.no_grad():
            x_manual = x_manual - stepsize * grad

            # Record stats
            cost = data_fidelity(x_manual, y, physics) + lambd * prior_l2(x_manual)
            cost_history.append(cost.item())
            psnr_history.append(metric(x_manual, x).item())

    # B. Library GD (Zero Prior)
    model_gd = dinv.optim.GD(
        prior=None,
        data_fidelity=data_fidelity,
        stepsize=stepsize,
        max_iter=maxiter,
        verbose=True,
    )
    x_library = model_gd(y, physics)

    # ---------------------------------------------------------
    # 3. Combined Visualization
    # ---------------------------------------------------------

    # Create a figure with 2 rows: Top for Images, Bottom for Metrics
    fig = plt.figure(figsize=(15, 10))

    # Row 1: Images using DeepInv's plotting utility on a subplot grid
    # We'll plot them manually for better control within the subplots
    imgs = [x, y, x_manual, x_library]
    titles = [
        "Original",
        f"Degraded\n{metric(y,x).item():.2f}dB",
        f"Manual GD\n{metric(x_manual,x).item():.2f}dB",
        f"Library GD\n{metric(x_library,x).item():.2f}dB",
    ]

    for i, (img, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(2, 4, i + 1)
        ax.imshow(img.squeeze().cpu().detach().numpy(), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    # Row 2, Col 1-2: Convergence Curves
    ax_cost = fig.add_subplot(2, 4, 6)
    color_c = "tab:blue"
    ax_cost.plot(cost_history, color=color_c, linewidth=2)
    ax_cost.set_xlabel("Iteration")
    ax_cost.set_ylabel("Total Cost", color=color_c)
    ax_cost.tick_params(axis="y", labelcolor=color_c)
    ax_cost.grid(True, alpha=0.3)
    ax_cost.set_title("Convergence Cost")

    ax_psnr = fig.add_subplot(2, 4, 5)
    color_p = "tab:red"
    ax_psnr.plot(psnr_history, color=color_p, linestyle="--")
    ax_psnr.set_ylabel("PSNR (dB)", color=color_p)
    ax_psnr.set_xlabel("Iteration")
    ax_psnr.tick_params(axis="y", labelcolor=color_p)
    ax_psnr.set_title("Convergence PSNR")

    # Row 2, Col 4: Pixel-wise Error
    ax_err = fig.add_subplot(2, 4, 7)
    error_map = (x_manual - x).abs().squeeze().cpu().detach().numpy()
    im_err = ax_err.imshow(error_map, cmap="hot")
    ax_err.set_title("Manual GD Error Map")
    ax_err.axis("off")
    plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

    ax_err = fig.add_subplot(2, 4, 8)
    error_map = (x_library - x).abs().squeeze().cpu().detach().numpy()
    im_err = ax_err.imshow(error_map, cmap="hot")
    ax_err.set_title("Library GD Error Map")
    ax_err.axis("off")
    plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# %%
import deepinv as dinv
import torch
import matplotlib.pyplot as plt
from deepinv.optim import PGD

device = dinv.utils.get_device()
# Reduced maxiter for efficiency; 100-500 is usually enough for visual convergence with GD
maxiter = 100
images = [
    "celeba_example.jpg",
    "JAX_018_011_RGB.tif",
    "CT100_256x256_0.pt",
    "CBSD_0010.png",
    "FMD_TwoPhoton_MICE_R_gt_12_avg50.png",
    "mbappe.jpg",
    "leaves.png",
    "barbara.jpeg",
    "butterfly.png",
]

for image_name in images:
    print(f"------------------- Processing PGD -> {image_name} --------------------")

    # 1. Setup Data
    x = dinv.utils.load_example(image_name, grayscale=True, img_size=256, device=device)
    physics = dinv.physics.Blur(
        filter=dinv.physics.blur.gaussian_blur(sigma=2.5), padding="circular"
    ).to(device)
    y = physics(x)

    metric = dinv.metric.PSNR()
    data_fidelity = dinv.optim.L2()
    prior = dinv.optim.TVPrior()
    lambd = 0.05
    norm_A2 = physics.compute_sqnorm(y, tol=1e-5, verbose=False).item()
    print(f"Estimated norm of A^2: {norm_A2:.4f}")
    step_size = 1.9 / norm_A2
    maxiter = 100
    x_k = torch.zeros_like(x, device=device)
    cost_history = []
    psnr_history = []
    with torch.no_grad():
        for k in range(maxiter):
            u = x_k - step_size * data_fidelity.grad(x_k, y, physics)
            x_k = prior.prox(u, gamma=step_size * lambd)
            cost = data_fidelity(x_k, y, physics) + lambd * prior(x_k)
            cost_history.append(cost.item())
            psnr_history.append(metric(x_k, x).item())

    model = PGD(
        prior=prior,
        data_fidelity=data_fidelity,
        stepsize=step_size,
        sigma_denoiser=0.05,
        max_iter=maxiter,
        verbose=True,
    )
    x_library = model(y, physics)
    # ---------------------------------------------------------
    # 3. Combined Visualization
    # ---------------------------------------------------------

    # Create a figure with 2 rows: Top for Images, Bottom for Metrics
    fig = plt.figure(figsize=(15, 10))
    x_manual = x_k
    # Row 1: Images using DeepInv's plotting utility on a subplot grid
    # We'll plot them manually for better control within the subplots
    imgs = [x, y, x_manual, x_library]
    titles = [
        "Original",
        f"Degraded\n{metric(y,x).item():.2f}dB",
        f"Manual PGD\n{metric(x_manual,x).item():.2f}dB",
        f"Library PGD\n{metric(x_library,x).item():.2f}dB",
    ]

    for i, (img, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(2, 4, i + 1)
        ax.imshow(img.squeeze().cpu().detach().numpy(), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    # Row 2, Col 1-2: Convergence Curves
    ax_cost = fig.add_subplot(2, 4, 6)
    color_c = "tab:blue"
    ax_cost.plot(cost_history, color=color_c, linewidth=2)
    ax_cost.set_xlabel("Iteration")
    ax_cost.set_ylabel("Total Cost", color=color_c)
    ax_cost.tick_params(axis="y", labelcolor=color_c)
    ax_cost.grid(True, alpha=0.3)
    ax_cost.set_title("Convergence Cost")

    ax_psnr = fig.add_subplot(2, 4, 5)
    color_p = "tab:red"
    ax_psnr.plot(psnr_history, color=color_p, linestyle="--")
    ax_psnr.set_ylabel("PSNR (dB)", color=color_p)
    ax_psnr.set_xlabel("Iteration")
    ax_psnr.tick_params(axis="y", labelcolor=color_p)
    ax_psnr.set_title("Convergence PSNR")

    # Row 2, Col 4: Pixel-wise Error
    ax_err = fig.add_subplot(2, 4, 7)
    error_map = (x_manual - x).abs().squeeze().cpu().detach().numpy()
    im_err = ax_err.imshow(error_map, cmap="hot")
    ax_err.set_title("Manual PGD Error Map")
    ax_err.axis("off")
    plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

    ax_err = fig.add_subplot(2, 4, 8)
    error_map = (x_library - x).abs().squeeze().cpu().detach().numpy()
    im_err = ax_err.imshow(error_map, cmap="hot")
    ax_err.set_title("Library PGD Error Map")
    ax_err.axis("off")
    plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
