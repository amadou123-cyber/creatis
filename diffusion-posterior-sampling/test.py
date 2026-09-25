# %%
import warnings
import numpy as np
import torch
import torchvision.transforms as T
import deepinv as dinv
import matplotlib.pyplot as plt
from pnp_mm import pnp_mm, mlem, mlem_tv
from sample_condition import main
import os
import shutil
import torchvision.utils

warnings.filterwarnings("ignore")


def set_seed(seed: int):
    """Fixe les graines aléatoires pour garantir la reproductibilité stochastique."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """Convertit un tenseur Torch (1,C,H,W) ou (C,H,W) en tableau NumPy (H,W,C) float32 [0, 1]."""
    t = t.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    t = t.permute(1, 2, 0).clamp(0, 1).numpy().astype(np.float32)
    return t.clip(0, 1)


def build_deconvolution_physics(
    n_channels: int, img_size: int, gain: float, device: str
):
    """Initialise le modèle physique de déconvolution avec bruit de Poisson."""
    filter_torch = dinv.physics.functional.gaussian_blur(sigma=(2.0, 2.0))
    physics = dinv.physics.Blur(
        img_size=(n_channels, img_size, img_size),
        filter=filter_torch,
        padding="circular",
        device=device,
        noise_model=dinv.physics.PoissonNoise(
            gain=gain, normalize=True, clip_positive=True
        ),
    )
    return physics


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Appareil utilisé : {device}")

image_name = "leaves.png"
x = dinv.utils.load_example(
    image_name,
    img_size=256,
    resize_mode="resize",
).to(device)

if x.shape[1] == 4:
    x = x[:, :3, :, :]

import shutil

# enregistrer l'image d'entrée
print(f"Loaded image {image_name} with shape: {x.shape}")
output_dir = "/home/sow/projet_creatis/diffusion-posterior-sampling/data/samples"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, image_name)
torchvision.utils.save_image(x, output_path)
gain = 1 / 5
gain_val_str = "5"
task = "deconvolution"

params_config = {
    "RL": {"lambda_reg": 0, "max_iter": 1},
    "RL-TV": {
        "max_iter": 11,
        "n_it_max": 1820,
        "stepsize": 0.7598285174171517,
        "lambda_reg": 0.12750707320765098,
    },
    "DPS": {"scale": 1.288888931274414},
    "MM": {
        "steps": 1000,
        "stepsize": 0.33,
        "sigma": 56 / 255,
        "lambda_reg": 0.95,
    },
    "prox_em": {"tau": 0.07000000029802322},
    "ps++": {"scale": 0.09999999403953552},
}

config_task = "configs/deconvolution_5_config.yaml"
model_config = "configs/imagenet_model_config.yaml"
diffusion_config = "configs/diffusion_config.yaml"

psnr_fn = dinv.metric.PSNR()
ssim_fn = dinv.metric.SSIM()

model_gs = dinv.models.DRUNet(in_channels=3, out_channels=3, pretrained="download").to(
    device
)

# Configuration de l'expérience Monte Carlo
N_RUNS = 30
methods_all = ["Input", "PnP-MM", "DPS", "DPSP-grad", "DPSP-prox"]
stoch_methods = ["DPS", "DPSP-grad", "DPSP-prox"]  # méthodes stochastiques uniquement

# Stockage des images moyennes, des sommes de carrés (pour la variance) et métriques
images_stack = {m: np.zeros_like(tensor_to_np(x)) for m in methods_all}
images_sqsum_stack = {m: np.zeros_like(tensor_to_np(x)) for m in methods_all}
metrics = {m: {"psnr": [], "ssim": []} for m in methods_all}

# Initialisation de la physique
physics = build_deconvolution_physics(
    n_channels=3, img_size=256, gain=gain, device=device
)

# --------------------------------------------------
# 2. Calculs Déterministes (Hors Boucle)
# --------------------------------------------------
print("\n--- Génération de l'observation et calcul déterministe ---")

# Génération de la mesure
set_seed(12)  # Graine fixe pour la création de la mesure dégradée
y = physics(x)
x0 = physics.A_adjoint(y)

# Calcul unique de PnP-MM (Algorithme Déterministe)
with torch.no_grad():
    x_pnpmm = pnp_mm(
        y=y,
        x0=x0,
        denoiser=model_gs,
        physics=physics,
        stepsize=params_config["MM"]["stepsize"],
        steps=params_config["MM"]["steps"],
        lambda_reg=params_config["MM"]["lambda_reg"],
        sigma=params_config["MM"]["sigma"],
        verbose=True,
        keep_inter=False,
    )

# Enregistrement des résultats déterministes répétés N_RUNS fois
p_in, s_in = psnr_fn(x, y).item(), ssim_fn(x, y).item()
p_mm, s_mm = psnr_fn(x, x_pnpmm).item(), ssim_fn(x, x_pnpmm).item()

images_stack["Input"] = tensor_to_np(y)
images_stack["PnP-MM"] = tensor_to_np(x_pnpmm)
# Méthodes déterministes → variance nulle par construction
images_sqsum_stack["Input"] = images_stack["Input"] ** 2
images_sqsum_stack["PnP-MM"] = images_stack["PnP-MM"] ** 2

metrics["Input"]["psnr"] = [p_in] * N_RUNS
metrics["Input"]["ssim"] = [s_in] * N_RUNS
metrics["PnP-MM"]["psnr"] = [p_mm] * N_RUNS
metrics["PnP-MM"]["ssim"] = [s_mm] * N_RUNS

# --------------------------------------------------
# 3. Boucle Monte Carlo (Méthodes Stochastiques)
# --------------------------------------------------
print(f"\n--- Lancement des {N_RUNS} runs stochastiques ---")
for run_idx in range(N_RUNS):
    seed = run_idx + 1
    print(f"Run {run_idx + 1}/{N_RUNS} (Seed {seed})...")
    set_seed(seed)

    # 1. DPS
    sample_dps, psnr_dps, ssim_dps = main(
        model_config,
        diffusion_config,
        config_task,
        method="ps",
        params_method=params_config["DPS"],
    )
    img_dps = tensor_to_np(sample_dps)
    images_stack["DPS"] += img_dps / N_RUNS
    images_sqsum_stack["DPS"] += img_dps**2 / N_RUNS
    metrics["DPS"]["psnr"].append(psnr_dps)
    metrics["DPS"]["ssim"].append(ssim_dps)

    # 2. DPSP-prox (Prox_EM)
    sample_prox, psnr_prox, ssim_prox = main(
        model_config,
        diffusion_config,
        config_task,
        method="dpsp_prox",
        params_method=params_config["prox_em"],
    )
    img_prox = tensor_to_np(sample_prox)
    images_stack["DPSP-prox"] += img_prox / N_RUNS
    images_sqsum_stack["DPSP-prox"] += img_prox**2 / N_RUNS
    metrics["DPSP-prox"]["psnr"].append(psnr_prox)
    metrics["DPSP-prox"]["ssim"].append(ssim_prox)

    # 3. DPSP-grad (PS++)
    sample_pspp, psnr_pspp, ssim_pspp = main(
        model_config,
        diffusion_config,
        config_task,
        method="dpsp_grad",
        params_method=params_config["ps++"],
    )
    img_pspp = tensor_to_np(sample_pspp)
    images_stack["DPSP-grad"] += img_pspp / N_RUNS
    images_sqsum_stack["DPSP-grad"] += img_pspp**2 / N_RUNS
    metrics["DPSP-grad"]["psnr"].append(psnr_pspp)
    metrics["DPSP-grad"]["ssim"].append(ssim_pspp)

# --------------------------------------------------
# 3bis. Calcul des cartes de variance pixel-wise : Var = E[X²] - E[X]²
# --------------------------------------------------
images_var_stack = {}
for m in methods_all:
    var_map = images_sqsum_stack[m] - images_stack[m] ** 2
    images_var_stack[m] = np.clip(
        var_map, 0, None
    )  # évite les résidus négatifs numériques

# --------------------------------------------------
# 4. Affichage des Résultats
# --------------------------------------------------
print("\n" + "=" * 55)
print(f" RÉSULTATS MOYENS (GAIN = 1/5, N={N_RUNS} RUNS)")
print("=" * 55)
for m in methods_all:
    p_mean = np.mean(metrics[m]["psnr"])
    s_mean = np.mean(metrics[m]["ssim"])
    p_std = np.std(metrics[m]["psnr"])
    s_std = np.std(metrics[m]["ssim"])
    print(
        f"{m:<10} | PSNR Moy: {p_mean:.2f} ± {p_std:.2f} dB | "
        f"SSIM Moy: {s_mean:.3f} ± {s_std:.3f}"
    )

# --------------------------------------------------
# 5. Visualisation : Images Moyennes Reconstruites
# --------------------------------------------------
fig, axes = plt.subplots(1, len(methods_all), figsize=(4 * len(methods_all), 4))

for col, method in enumerate(methods_all):
    ax = axes[col]
    ax.imshow(images_stack[method])
    ax.set_xticks([])
    ax.set_yticks([])

    title_str = f"{method} (ours)" if method in ["DPSP-grad", "DPSP-prox"] else method
    ax.set_title(title_str, fontsize=16, fontweight="bold", pad=8)

    p_mean = np.mean(metrics[method]["psnr"])
    s_mean = np.mean(metrics[method]["ssim"])
    p_std = np.std(metrics[method]["psnr"])
    s_std = np.std(metrics[method]["ssim"])

    ax.set_xlabel(
        f"PSNR: {p_mean:.2f} ± {p_std:.2f} dB\nSSIM: {s_mean:.3f} ± {s_std:.3f}",
        fontsize=14,
        color="#333333",
        fontweight="bold",
        labelpad=6,
    )

plt.suptitle(
    f"Images Moyennes Reconstruites (Gain = 1/5, N={N_RUNS})",
    fontsize=18,
    fontweight="bold",
    y=1.05,
)
plt.tight_layout()

# Ligne ajoutée : affichage de l'incertitude (std) sur les métriques avant la sauvegarde
print(
    "Incertitude (std) sur les métriques :",
    {m: (np.std(metrics[m]["psnr"]), np.std(metrics[m]["ssim"])) for m in methods_all},
)

plt.savefig(f"moyennes_gain_1_5_{N_RUNS}.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSauvegardé → moyennes_gain_1_5_{N_RUNS}.png")

# --------------------------------------------------
# 5bis. Visualisation : Cartes de Variance Pixel-wise (incertitude de reconstruction)
# --------------------------------------------------
fig, axes = plt.subplots(1, len(stoch_methods), figsize=(4 * len(stoch_methods), 4))

for col, method in enumerate(stoch_methods):
    ax = axes[col]
    var_map_gray = images_var_stack[method].mean(axis=-1)  # moyenne sur les canaux RGB
    im = ax.imshow(var_map_gray, cmap="inferno")
    ax.set_xticks([])
    ax.set_yticks([])

    title_str = f"{method} (ours)" if method in ["DPSP-grad", "DPSP-prox"] else method
    ax.set_title(title_str, fontsize=16, fontweight="bold", pad=8)

    mean_var = var_map_gray.mean()
    ax.set_xlabel(
        f"Var. moy: {mean_var:.2e}", fontsize=13, fontweight="bold", labelpad=6
    )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(
    f"Variance (Gain = 1/5, N={N_RUNS})",
    fontsize=18,
    fontweight="bold",
    y=1.05,
)
plt.tight_layout()
plt.savefig(f"variance_gain_1_5_{N_RUNS}.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Sauvegardé → variance_gain_1_5_{N_RUNS}.png")

# --------------------------------------------------
# 6. Visualisation : Boxplots PSNR et SSIM
# --------------------------------------------------
box_methods = ["PnP-MM", "DPS", "DPSP-grad", "DPSP-prox"]
psnr_data = [metrics[m]["psnr"] for m in box_methods]
ssim_data = [metrics[m]["ssim"] for m in box_methods]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot PSNR
axes[0].boxplot(psnr_data, labels=box_methods, patch_artist=True)
axes[0].set_title("Distribution du PSNR (dB)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("PSNR (dB)", fontsize=12)
axes[0].grid(True, linestyle="--", alpha=0.5)

# Boxplot SSIM
axes[1].boxplot(ssim_data, labels=box_methods, patch_artist=True)
axes[1].set_title("Distribution du SSIM", fontsize=14, fontweight="bold")
axes[1].set_ylabel("SSIM", fontsize=12)
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.suptitle(f"Boxplots - Gain = 1/5 (N={N_RUNS})", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"psnr_ssim_gain_1_5_{N_RUNS}.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Sauvegardé → psnr_ssim_gain_1_5_{N_RUNS}.png")
