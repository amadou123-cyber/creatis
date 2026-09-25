from functools import partial
import os
import yaml
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
import warnings

warnings.filterwarnings("ignore")


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(
    model_config,
    diffusion_config,
    task_config,
    save_dir="./results",
    method="ps_prox",
    params_method={"tau": 0.1},
):
    logger = get_logger(method)

    device_str = "cuda"
    logger.info(f"Device: {device_str}")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(model_config)
    diffusion_config = load_yaml(diffusion_config)
    task_configs = load_yaml(task_config)

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_configs["measurement"]
    operator = get_operator(device=device, **measure_config["operator"])
    noiser = get_noise(**measure_config["noise"])
    noise = measure_config["noise"]
    noise_info = ""
    for key, value in noise.items():
        if key == "name":
            key = "noise"
        noise_info += f" / {key}: {value}"
    logger.info(f"Operation: {measure_config['operator']['name']}{noise_info}")

    # Prepare conditioning method
    cond_config = task_configs["conditioning"]
    cond_config["method"] = method
    cond_config["params"] = params_method
    cond_method = get_conditioning_method(
        cond_config["method"], operator, noiser, **cond_config["params"]
    )
    measurement_cond_fn = cond_method.conditioning
    params_config = task_configs["conditioning"]["params"]
    params_info = ""
    method_used = task_configs["conditioning"]["method"]
    for i, j in params_config.items():
        params_info += f" / {i}: {j}"
    logger.info(f"Conditioning method: {method_used}{params_info}")
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(
        sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn
    )

    # Working directory
    name = "_".join(task_config.split("_")[:-1])
    name = method_used + f"/" + name
    out_path = os.path.join(save_dir, name)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ["input", "recon", "progress", "label"]:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_configs["data"]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask
    if measure_config["operator"]["name"] == "inpainting":
        mask_gen = mask_generator(**measure_config["mask_opt"])

    # Do Inference
    for i, ref_img in enumerate(loader):
        fname = str(i).zfill(5) + ".png"
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config["operator"]["name"] == "inpainting":
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else:
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)

        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        img_tensor, psnr_value, ssim_value = sample_fn(
            method=method_used,
            ground=ref_img,
            x_start=x_start,
            measurement=y_n,
            record=True,
            save_root=out_path,
        )

        plt.imsave(os.path.join(out_path, "input", fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, "label", fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, "recon", fname), clear_color(img_tensor))

        return img_tensor, psnr_value, ssim_value


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from deepinv.datasets import CBSD68
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
    from torch.utils.data import DataLoader

    warnings.filterwarnings("ignore")

    model_config = "configs/imagenet_model_config.yaml"
    diffusion_config = "configs/diffusion_config.yaml"
    task_configs = "configs/deconvolution_100_config.yaml"
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    dataset = CBSD68(root="CBSD68", download=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    taus = torch.linspace(0.01, 1.5, steps=30).tolist()

    best_tau = None
    best_mean_psnr = -1.0

    for tau in taus:
        print(f"\n================ Test Tau = {tau:.4f} ================")
        psnr_scores = []

        for i, x_true in enumerate(dataloader):
            image_name = f"image_{i+1:02d}.png"
            output_dir = (
                "/home/sow/projet_creatis/diffusion-posterior-sampling/data/samples"
            )
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"orig_{image_name}")
            torchvision.utils.save_image(x_true, output_path)

            img_tensor, psnr_value, ssim_value = main(
                model_config,
                diffusion_config,
                task_configs,
                save_dir="./results",
                method="ps",
                params_method={"scale": tau},
            )

            psnr_scores.append(psnr_value)
            print(
                f"        Image {i+1}/{len(dataset)} | Tau: {tau:.2f} | PSNR: {psnr_value:.2f} dB"
            )

        mean_psnr = sum(psnr_scores) / len(psnr_scores)
        print(f"--> PSNR Moyen pour tau={tau:.4f} : {mean_psnr:.2f} dB")

        if mean_psnr > best_mean_psnr:
            best_mean_psnr = mean_psnr
            best_tau = tau

    print("\n================ RÉSULTAT FINAL ================")
    print(
        f"Meilleur Tau global : {best_tau:.4f} avec PSNR moyen de {best_mean_psnr:.2f} dB"
    )
