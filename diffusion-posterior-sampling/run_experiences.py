# %%
import os
import warnings
from pnp_mm import pnp_mm, mlem, mlem_tv
import torch
import torchvision
import torchvision.transforms as T
import deepinv as dinv
import matplotlib.pyplot as plt
import numpy as np
from sample_condition import main
import os
import shutil
import torchvision.utils

warnings.filterwarnings("ignore")
torch.manual_seed(1)
np.random.seed(1)


def tensor_to_np(t):
    t = t.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    t = t.permute(1, 2, 0).clamp(0, 1).numpy().astype(np.float32)
    return t.clip(0, 1)


def plot_results_grid(
    results,
    tasks,
    methods,
    gain_labs,
    output_suffix,
    dpi=150,
):
    for task in tasks:

        rows_items, cols_items = gain_labs, methods

        n_rows, n_cols = len(rows_items), len(cols_items)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False
        )

        for row, row_val in enumerate(rows_items):
            for col, col_val in enumerate(cols_items):
                ax = axes[row][col]

                gain_label = row_val
                method = col_val

                data = results[(task, gain_label)]

                # Affichage de l'image
                ax.imshow(data[method])
                ax.set_xticks([])
                ax.set_yticks([])

                # Titres (Première ligne)
                if row == 0:

                    title = (
                        f"{method} (ours)"
                        if method in ["DPSP-prox", "DPSP-grad"]
                        else method
                    )
                    ax.set_title(title, fontsize=22, fontweight="bold", pad=6)

                # Labels Y (Première colonne)
                if col == 0:

                    ax.set_ylabel(
                        f"gain = {gain_label}",
                        fontsize=25,
                        fontweight="bold",
                        labelpad=8,
                    )

                # Affichage des métriques PSNR / SSIM
                if method in data["psnr"]:
                    psnr_info = data["psnr"][method]
                    ssim_info = data["ssim"].get(method, "")

                    psnr_str = (
                        f"{psnr_info}"
                        if isinstance(psnr_info, str)
                        else f"{psnr_info:.2f}"
                    )
                    ax.set_xlabel(
                        f"{psnr_str}\n{ssim_info}",
                        fontsize=25,
                        color="#444444",
                        fontweight="bold",
                        labelpad=4,
                    )

        filename = f"results_{task}_{output_suffix}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.show()
        print(f"Saved → {filename}")


def model_deconvolution(n_channels, img_size, gain, device):
    filter_torch = dinv.physics.functional.gaussian_blur(sigma=(1.6, 1.6))
    physics = dinv.physics.Blur(
        img_size=(
            n_channels,
            img_size,
            img_size,
        ),
        filter=filter_torch,
        padding="circular",
        device=device,
        noise_model=dinv.physics.PoissonNoise(
            gain=gain, normalize=True, clip_positive=True
        ),
    )
    return physics


device = "cuda:0"

image_name = "leaves.png"
x = dinv.utils.load_example(
    image_name,
    img_size=256,
    resize_mode="resize",
).to(device)
if x.shape[1] == 4:
    x = x[:, :3, :, :]

# enregistrer l'image d'entrée
print(f"Loaded image {image_name} with shape: {x.shape}")
output_dir = "/home/sow/projet_creatis/diffusion-posterior-sampling/data/samples"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, image_name)
torchvision.utils.save_image(x, output_path)

gains = [1 / 100]
operateurs = ["deconvolution"]

params_config = {
    "deconvolution": {
        "100": {
            "RL": {"lambda_reg": 0, "max_iter": 10},
            "RL-TV": {"lambda_reg": 0.02857, "max_iter": 29, "n_it_max": 2000},
            "DPS": {"scale": 0.8999999761581421},
            "MM": {
                "stepsize": 1,
                "lambda_reg": 0.3,
                "sigma": 15.0 / 255.0,
                "steps": 1000,
            },
            "DPSP-PROX": {"tau": 0.3999999761581421},
            "DPSP-GRAD": {"scale": 0.800000011920929},
        },
        "60": {
            "RL": {"lambda_reg": 0, "max_iter": 16},
            "RL-TV": {"lambda_reg": 0.03571, "max_iter": 29, "n_it_max": 2000},
            "DPS": {"scale": 1.3777778148651123},
            "MM": {
                "stepsize": 0.3,
                "lambda_reg": 0.86,
                "sigma": 16 / 255,
                "steps": 1000,
            },
            "DPSP-PROX": {"tau": 0.29999998211860657},
            "DPSP-GRAD": {"scale": 0.45333331823349},
        },
        "5": {
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
            "DPSP-PROX": {"tau": 0.07000000029802322},
            "DPSP-GRAD": {"scale": 0.09999999403953552},
        },
    },
}

configs = {
    "deconvolution": {
        "100": "configs/deconvolution_100_config.yaml",
        "60": "configs/deconvolution_60_config.yaml",
        "5": "configs/deconvolution_5_config.yaml",
    },
}

# metrics
psnr_value = dinv.metric.PSNR()
ssim_value = dinv.metric.SSIM()

results = {}

for i in range(len(operateurs)):
    for gain in gains:
        task = operateurs[i].strip()
        gain_val_str = str(int(1 / gain))
        gain_label = f"1/{gain_val_str}"
        print(f"\n{'_' * 50}")
        print(f"Testing with gain: {gain:.4f} - {task}")
        print(f"{'_' * 50}")
        physics = model_deconvolution(
            n_channels=3, img_size=256, gain=gain, device=device
        )
        y = physics(x)

        cfg_mlem = params_config[task][gain_val_str]["RL"]
        cfg_dps = params_config[task][gain_val_str]["DPS"]
        pnp_config = params_config[task][gain_val_str]["MM"]
        tv_config = params_config[task][gain_val_str]["RL-TV"]
        config_task = configs[task][gain_val_str]
        model_config = "configs/imagenet_model_config.yaml"
        diffusion_config = "configs/diffusion_config.yaml"

        sample_dps, psnr_dps, ssim_dps = main(
            model_config,
            diffusion_config,
            config_task,
            method="ps",
            params_method=params_config[task][gain_val_str]["DPS"],
        )
        sample_dpsp_prox, psnr_dpsp_prox, ssim_dpsp_prox = main(
            model_config,
            diffusion_config,
            config_task,
            method="dpsp_prox",
            params_method=params_config[task][gain_val_str]["DPSP-PROX"],
        )
        sample_dpsp_grad, psnr_dpsp_grad, ssim_dpsp_grad = main(
            model_config,
            diffusion_config,
            config_task,
            method="dpsp_grad",
            params_method=params_config[task][gain_val_str]["DPSP-GRAD"],
        )

        with torch.no_grad():
            model_gs = dinv.models.DRUNet(
                in_channels=3, out_channels=3, pretrained="download"
            ).to(device)
            x_pnpmm = pnp_mm(
                y=y,
                x0=physics.A_adjoint(y),
                denoiser=model_gs,
                physics=physics,
                stepsize=pnp_config["stepsize"],
                steps=pnp_config["steps"],
                lambda_reg=pnp_config["lambda_reg"],
                sigma=pnp_config["sigma"],
                verbose=True,
                keep_inter=False,
            )
            x0 = physics.A_adjoint(y)
            stepsize = 1
            x_mlem_scratch = mlem(
                y=y,
                x0=x0,
                physics=physics,
                stepsize=stepsize,
                steps=cfg_mlem["max_iter"],
                verbose=True,
                keep_inter=False,
            )
            x_mlem_tv_scratch_fista = mlem_tv(
                y=y,
                x0=x0,
                physics=physics,
                stepsize=tv_config.get("stepsize", stepsize),
                steps=tv_config["max_iter"],
                alpha=tv_config["lambda_reg"],
                n_iter=tv_config["n_it_max"],
                fista=False,
                verbose=True,
                keep_inter=False,
            )

        results[(task, gain_label)] = {
            "Input": tensor_to_np(y),
            "RL": tensor_to_np(x_mlem_scratch),
            "RL-TV": tensor_to_np(x_mlem_tv_scratch_fista),
            "PnP-MM": tensor_to_np(x_pnpmm),
            "DPS": tensor_to_np(sample_dps),
            "DPSP-grad": tensor_to_np(sample_dpsp_grad),
            "DPSP-prox": tensor_to_np(sample_dpsp_prox),
            "Label": tensor_to_np(x),
            "psnr": {
                "Input": (
                    f"PSNR: {psnr_value(x, y).item():.2f} dB"
                    if task == "deconvolution"
                    else "Projections"
                ),
                "RL": f"PSNR: {psnr_value(x, x_mlem_scratch).item():.2f} dB",
                "RL-TV": f"PSNR: {psnr_value(x, x_mlem_tv_scratch_fista).item():.2f} dB",
                "PnP-MM": f"PSNR: {psnr_value(x, x_pnpmm).item():.2f} dB",
                "DPS": f"PSNR: {psnr_dps:.2f} dB",
                "DPSP-grad": f"PSNR: {psnr_dpsp_grad:.2f} dB",
                "DPSP-prox": f"PSNR: {psnr_dpsp_prox:.2f} dB",
            },
            "ssim": {
                "Input": (
                    f"SSIM: {ssim_value(x, y).item():.3f}"
                    if task == "deconvolution"
                    else "Projections"
                ),
                "RL": f"SSIM: {ssim_value(x, x_mlem_scratch).item():.3f}",
                "RL-TV": f"SSIM: {ssim_value(x, x_mlem_tv_scratch_fista).item():.3f}",
                "PnP-MM": f"SSIM: {ssim_value(x, x_pnpmm).item():.3f}",
                "DPS": f"SSIM: {ssim_dps:.3f}",
                "DPSP-grad": f"SSIM: {ssim_dpsp_grad:.3f}",
                "DPSP-prox": f"SSIM: {ssim_dpsp_prox:.3f}",
            },
        }

GAIN_LABS = ["1/100"]


CONFIGS = [
    (["Input", "RL", "RL-TV", "PnP-MM"], operateurs, "mlem_pnp"),
    (["Input", "DPS", "Label"], operateurs, "dps"),
    (["Input", "RL", "RL-TV", "PnP-MM", "DPS"], operateurs, "existant_approach"),
    (["Input", "DPS", "DPSP-grad", "DPSP-prox"], operateurs, "dps_and_myapproach"),
    (["Input", "DPSP-grad", "DPSP-prox", "Label"], operateurs, "contribution"),
    (
        [
            "Input",
            "PnP-MM",
            "DPS",
            "DPSP-grad",
            "DPSP-prox",
            "Label",
        ],
        operateurs,
        "compraison_approches",
    ),
]

for methods, tasks, suffix in CONFIGS:
    plot_results_grid(results, tasks, methods, GAIN_LABS, output_suffix=suffix)
