# %%
import os
import sys
import yaml
import subprocess
import warnings
from PIL import Image
from pnp_mm import pnp_mm, mlem, mlem_tv

warnings.filterwarnings("ignore")
import torch
import torchvision
import torchvision.transforms as T
import deepinv as dinv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from run import main


def tensor_to_np(t):
    """Convert a (1,C,H,W) or (C,H,W) torch tensor to (H,W,C) float32 numpy for imshow."""
    t = t.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    t = t.permute(1, 2, 0).clamp(0, 1).numpy().astype(np.float32)
    return t.clip(0, 1)


def model_deconvolution(n_channels, img_size, gain, device):
    # Pass sigma as a tuple to explicitly enforce a 2D blur kernel structure
    filter_torch = dinv.physics.functional.gaussian_blur(sigma=(2.0, 2.0))

    physics = dinv.physics.Blur(
        img_size=(
            n_channels,
            img_size,
            img_size,
        ),  # Corrected to a 3D tuple: (3, 256, 256)
        filter=filter_torch,
        padding="circular",
        device=device,
        noise_model=dinv.physics.PoissonNoise(
            gain=gain, normalize=True, clip_positive=True
        ),
    )
    return physics


def compare_to_butterfly(img_path, middle_path, image, image_name="butterfly.png"):
    ext_img = Image.open(img_path).convert("RGB")
    middle_img = Image.open(middle_path).convert("RGB")
    int_img = Image.open(image).convert("RGB")
    transform = T.Compose([T.Resize(int_img.size), T.ToTensor()])
    ext_tensor = transform(ext_img).unsqueeze(0)
    middle_tensor = transform(middle_img).unsqueeze(0)
    init_tensor = transform(int_img).unsqueeze(0)
    # psnr_input = dinv.metric.PSNR()(init_tensor, middle_tensor).item()
    psnr_score = dinv.metric.PSNR()(init_tensor, ext_tensor).item()
    return psnr_score, ext_tensor, init_tensor, middle_tensor


def model_tomo(n_channels, img_size, gain, device):
    physics = dinv.physics.TomographyWithAstra(
        img_size=[256, 256],
        angles=180,
        n_detector_pixels=512,
        noise_model=dinv.physics.PoissonNoise(gain=gain),
    ).to(device)
    return physics


device = "cuda"
print(f"Le device est : {device}")

image_name = "butterfly.png"
x = dinv.utils.load_example(
    image_name,
    img_size=256,
    resize_mode="resize",
).to(device)
output_dir = "/home/sow/diffusion-posterior-sampling/data/samples"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, image_name)
torchvision.utils.save_image(x, output_path)

gains = [1 / 100]
info = ["deconvolution"]

params_config = {
    "deconvolution": {
        "100": {
            "MLEM": {"lambda_reg": 0, "max_iter": 10},
            "MLEM-TV": {"lambda_reg": 0.02857, "max_iter": 29, "n_it_max": 2000},
            "DPS": {"alpha": 1, "weight": 3.5278306239269366},
            "MM": {
                "stepsize": 1,
                "lambda_reg": 0.3,
                "sigma": 15.0 / 255.0,
                "steps": 1000,
            },
        },
        "60": {
            "MLEM": {"lambda_reg": 0, "max_iter": 16},
            "MLEM-TV": {"lambda_reg": 0.03571, "max_iter": 29, "n_it_max": 2000},
            "DPS": {"alpha": 1, "weight": 3.543215632979817},
            "MM": {
                "stepsize": 0.3,
                "lambda_reg": 0.86,
                "sigma": 16 / 255,
                "steps": 1000,
            },
        },
        "5": {
            "MLEM": {"lambda_reg": 0, "max_iter": 1},
            "MLEM-TV": {
                "max_iter": 11,
                "n_it_max": 1820,
                "stepsize": 0.7598285174171517,
                "lambda_reg": 0.12750707320765098,
            },
            "DPS": {"weight": 3.8757547810544954, "alpha": 1},
            "MM": {
                "steps": 1000,
                "stepsize": 0.33,
                "sigma": 56 / 255,
                "lambda_reg": 0.95,
            },
        },
    },
}

psnr_fn = dinv.metric.PSNR()
configs = {
    "deconvolution": {
        "100": "configs/deconvolution_100_config.yaml",
        "60": "configs/deconvolution_60_config.yaml",
        "5": "configs/deconvolution_5_config.yaml",
    },
}

results = {}
model_gs = dinv.models.DRUNet(in_channels=3, out_channels=3, pretrained="download").to(
    device
)
configs = {
    "tomography": {
        "100": "configs/tomo_100_config.yaml",
        "60": "configs/tomo_60_config.yaml",
        "5": "configs/tomo_5_config.yaml",
    },
    "deconvolution": {
        "100": "configs/deconvolution_100_config.yaml",
        "60": "configs/deconvolution_60_config.yaml",
        "5": "configs/deconvolution_5_config.yaml",
    },
}
for i in range(len(info)):
    for gain in gains:
        task = info[i].strip()
        gain_val_str = str(int(1 / gain))
        gain_label = f"1/{gain_val_str}"

        print(f"\n{'_' * 50}")
        print(f"Testing with gain: {gain:.4f} - {task}")
        print(f"{'_' * 50}")

        if task == "tomography":
            physics = model_tomo(n_channels=3, img_size=256, gain=gain, device=device)
        else:
            physics = model_deconvolution(
                n_channels=3, img_size=256, gain=gain, device=device
            )

        y = physics(x)
        data_fidelity_ct = dinv.optim.PoissonLikelihood(gain=gain)
        # Load specific parameter sets dynamically
        cfg_mlem = params_config[task][gain_val_str]["MLEM"]
        cfg_dps = params_config[task][gain_val_str]["DPS"]
        pnp_config = params_config[task][gain_val_str]["MM"]
        tv_config = params_config[task][gain_val_str]["MLEM-TV"]
        config_task = configs[task][gain_val_str]
        model_config = "configs/imagenet_model_config.yaml"
        diffusion_config = "configs/diffusion_config.yaml"
        sample, psnr, ssim = main(model_config, diffusion_config, config_task)
        name = "deconvolution_" + gain_val_str
        in_path = f"./results/configs/{name}/recon/00000.png"
        middle_path = f"./results/configs/{name}/input/00000.png"
        out_path = f"./results/configs/{name}/label/00000.png"
        psnr_score, ext_tensor, init_tensor, middle_tensor = compare_to_butterfly(
            in_path, middle_path, out_path
        )
        ssim_value = dinv.metric.SSIM()
        print(
            f"PSNR score: {psnr:.2f} dB",
            f"PSNR score (ext): {psnr_score:.2f} dB",
            f"SSIM score (ext): {ssim:.2f}",
        )
        print(ssim_value(x, y).item())
        dps = dinv.sampling.DPS(
            model_gs,
            schedule="vp",
            num_steps=1000,
            weight=float(cfg_dps["weight"]),
            alpha=1,
            verbose=True,
            device=device,
            dtype=torch.float32,
            rng=torch.Generator(device=device),
            minus_one_one=False,
        )
        with torch.no_grad():
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
            x_dps = dps(y, physics)

        results[(task, gain_label)] = {
            "Ground truth": tensor_to_np(x),
            "Measurement": tensor_to_np(y),
            "MLEM": tensor_to_np(x_mlem_scratch),
            "MLEM-TV": tensor_to_np(x_mlem_tv_scratch_fista),
            "PnP-MM": tensor_to_np(x_pnpmm),
            "DPS": tensor_to_np(x_dps),
            "Ours": tensor_to_np(ext_tensor),
            "psnr": {
                "Measurement": (
                    f"{psnr_fn(x, y).item():.2f} dB"
                    if task == "deconvolution"
                    else "Projections"
                ),
                "MLEM": f"{psnr_fn(x, x_mlem_scratch).item():.2f} dB",
                "MLEM-TV": f"{psnr_fn(x, x_mlem_tv_scratch_fista).item():.2f} dB",
                "PnP-MM": f"{psnr_fn(x, x_pnpmm).item():.2f} dB",
                "DPS": f"{psnr_fn(x, x_dps).item():.2f} dB",
                "Ours": f"{psnr:.2f} dB",
            },
            "ssim": {
                "Measurement": (
                    f"{ssim_value(x, y).item():.4f}"
                    if task == "deconvolution"
                    else "Projections"
                ),
                "MLEM": f"{ssim_value(x, x_mlem_scratch).item():.4f}",
                "MLEM-TV": f"{ssim_value(x, x_mlem_tv_scratch_fista).item():.4f}",
                "PnP-MM": f"{ssim_value(x, x_pnpmm).item():.4f}",
                "DPS": f"{ssim_value(x, x_dps).item():.4f}",
                "Ours": f"{ssim:.4f}",
            },
        }


METHODS = ["Ground truth", "Measurement", "MLEM", "MLEM-TV", "PnP-MM", "DPS", "Ours"]
GAIN_LABS = ["1/100"]
TASKS = info

for task in TASKS:
    n_rows = len(GAIN_LABS)
    n_cols = len(METHODS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False
    )

    for row, gain_label in enumerate(GAIN_LABS):
        data = results[(task, gain_label)]
        for col, method in enumerate(METHODS):
            ax = axes[row][col]
            ax.imshow(data[method])
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(method, fontsize=22, fontweight="bold", pad=6)

            if col == 0:
                ax.set_ylabel(
                    f"gain = {gain_label}", fontsize=25, fontweight="bold", labelpad=8
                )

            if method in data["psnr"]:
                information = data["psnr"][method]
                ssim_info = data["ssim"].get(method, None)
                if type(information) is str:
                    ax.set_xlabel(
                        f"{information}\n{ssim_info}",
                        fontsize=25,
                        color="#444444",
                        fontweight="bold",
                        labelpad=4,
                    )
                else:
                    ax.set_xlabel(
                        f"{information:.2f} \n{ssim_info}",
                        fontsize=25,
                        color="#444444",
                        fontweight="bold",
                        labelpad=4,
                    )

    plt.tight_layout()
    plt.savefig(f"results_{task}_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → results_{task}_comparison.png")
