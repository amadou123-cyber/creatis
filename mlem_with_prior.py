import torch
import deepinv as dinv
from pathlib import Path
from torchvision import transforms
from deepinv.utils.demo import load_dataset
from deepinv.utils.plotting import plot, plot_curves
import matplotlib.pyplot as plt

# --- 1. Environment Setup ---
BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results"
torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def calculate_objective(x_current, prior, y_obs, physics, lamb, gain):
    ax = physics.A(x_current)
    fidelity = torch.sum(ax - y_obs * torch.log(ax + 1e-9))
    tv_val = prior(x_current)
    return fidelity.item() + (lamb * tv_val.item())
  
def get_best_iter(data_list):
    tensor_data = torch.tensor(data_list) if isinstance(data_list, list) else data_list
    return torch.argmax(tensor_data)
  
# --- 2. Loading images and applying transformers ---
img_size = 128 if torch.cuda.is_available() else 64
val_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)
dataset = load_dataset("set3c", transform=val_transform)
x = dataset[0].unsqueeze(0).to(device)  # ground-truth image

# --- 3. Define Physics (Blur + Poisson Noise) ---
filter_torch = dinv.physics.blur.gaussian_blur(sigma=(1.6, 1.6))
gain = 1 / 30 # the gain of poisson
physics = dinv.physics.BlurFFT(
    img_size=x.shape[1:],
    filter=filter_torch,
    device=device,
    noise_model=dinv.physics.PoissonNoise(
        gain=gain, normalize=True, clip_positive=True
    ),
)
y_blur = physics(x)
n_iter_outer = 100 # nbre d'iterations
data_fidelity = dinv.optim.PoissonLikelihood(gain=gain)
for lamb in [0.01]:
    print(f"\n--- Running Standard MLEM and Voichita-MLEM with lambda={lamb} ---")
    mlem_no_prior = dinv.optim.MLEM(
        data_fidelity=data_fidelity,
        prior=None,
        lambda_reg=lamb,
        max_iter=100,
        verbose=True,
    )
    x_mlem, psnr_mlem = mlem_no_prior(y_blur, physics, x_gt=x, compute_metrics=True)
  
    # --- 5. Iterative Chambolle-TV-MLEM Implementation ---
    # This implements:
    # 1. x_half = MLEM_step(x_k)
    # 2. x_k+1  = prox_TV(x_half)

    prior = dinv.optim.TVPrior(n_it_max=20)  # Inner Chambolle iterations
    model_tv = dinv.optim.MLEM(
        data_fidelity=data_fidelity,
        prior=prior,
        lambda_reg=0.02,
        max_iter=100,
        verbose=True,
    )
    x_mlem_prior, psnr_mlem_prior = model_tv(
        y_blur, physics, x_gt=x, compute_metrics=True
    )

    # Initialize x_k (MLEM is sensitive to zeros; use noisy image)
    x_k = y_blur.clone().detach().clamp(min=1e-4)

    # Precompute A^T * 1 for the MLEM denominator
    ones_y = torch.ones_like(y_blur)
    at_ones = physics.A_adjoint(ones_y).clamp(min=1e-9)

    pnr_values = []
    cost_history = []
    print("\nStarting Iterative (Voichita)...")
    for i in range(n_iter_outer + 1):
        # Step A: Multiplicative MLEM Update (Data Fidelity)
        ax_k = physics.A(x_k)
        ratio = y_blur / (ax_k + 1e-9)
        back_projection = physics.A_adjoint(ratio)

        x_half = (x_k / at_ones) * back_projection

        # Step B: Proximal Step
        x_k = prior.prox(x_half, gamma=lamb)

        # Physical Constraint: Non-negativity
        x_k = x_k.clamp(min=1e-9)

        # Track PSNR
        with torch.no_grad():
            current_cost = calculate_objective(x_k, prior, y_blur, physics, lamb, gain)
            cost_history.append(current_cost)
            pnr = dinv.metric.PSNR()(x_k, x)
            pnr_values.append(pnr)

        if i % 10 == 0 or i == n_iter_outer - 1:
            print(f"Iteration {i:03d} | PSNR: {pnr.item():.2f} dB", end=", ")

    x_final = x_k

    # --- 6. Results Visualization ---
    plot(
        [x, y_blur, x_mlem, x_final, x_mlem_prior],
        titles=[
            "Original",
            "Measurement",
            "St MLEM",
            "Voichita-MLEM",
            "MLEM with TV Prior",
        ],
        subtitles=[
            "PSNR:",
            f"{dinv.metric.PSNR()(y_blur, x).item():.2f} dB",
            f"{dinv.metric.PSNR()(x_mlem, x).item():.2f} dB",
            f"{dinv.metric.PSNR()(x_final, x).item():.2f} dB",
            f"{dinv.metric.PSNR()(x_mlem_prior, x).item():.2f} dB",
        ],
        tight=False,
    )
    plot_curves(psnr_mlem)
    plot_curves(psnr_mlem_prior)
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(psnr_mlem["cost"][0], "g-", linewidth=2)
    plt.title("Cost History Standard MLEM")
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function Value")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(cost_history, "g-", color="blue", linewidth=2)
    plt.title("Cost History Voichita-MLEM")
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function Value")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

    pnr_tensor = torch.tensor(pnr_values)

    # --- Graphique ---
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(
        pnr_tensor.detach().cpu().numpy(),
        "o-",
        label="Voichita",
        color="blue",
        linewidth=2,
    )
    # Plot Standard MLEM
    plt.plot(
        psnr_mlem["psnr"][0],
        "s--",
        label="St MLEM",
        color="red",
        alpha=0.6,
    )
    # Plot Standard MLEM with TV Prior
    plt.plot(
        psnr_mlem_prior["psnr"][0],
        "s--",
        label="MLEM with TV Prior",
        color="green",
        alpha=0.6,
    )
    # Indice max pour Voichita
    best_voichita = get_best_iter(pnr_values)
    plt.axvline(
        best_voichita.item(),  # .item() convertit l'indice en simple nombre
        color="blue",
        linestyle="--",
        alpha=0.5,
        label=f"Best Voichita (iter={best_voichita.item()}, PSNR={pnr_tensor[best_voichita]:.2f} dB)",
    )

    # Indice max pour St MLEM
    best_mlem = get_best_iter(psnr_mlem["psnr"][0])
    plt.axvline(
        best_mlem.item(),
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Best MLEM (iter={best_mlem.item()}, PSNR={psnr_mlem['psnr'][0][best_mlem]:.2f} dB)",
    )

    # Indice max pour MLEM with TV Prior
    best_mlem_prior = get_best_iter(psnr_mlem["psnr"][0])
    plt.axvline(
        best_mlem_prior.item(),
        color="green",
        linestyle="-",
        alpha=0.5,
        label=f"Best MLEM with TV Prior (iter={best_mlem_prior.item()}, PSNR={psnr_mlem_prior['psnr'][0][best_mlem_prior]:.2f} dB)",
    )
    plt.title("PSNR Evolution of standard MLEM vs Voichita-MLEM")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
