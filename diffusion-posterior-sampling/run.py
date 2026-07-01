from functools import partial
import os
import argparse
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

model_config = "configs/imagenet_model_config.yaml"
diffusion_config = "configs/diffusion_config.yaml"
task_configs = "configs/deconvolution_5_config.yaml"


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(
    model_config,
    diffusion_config,
    task_configs,
    gpu=1,
    save_dir="./results",
    scale=0,
    tau=0,
):
    # logger
    logger = get_logger()

    # Device setting
    device_str = "cuda"
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(model_config)
    diffusion_config = load_yaml(diffusion_config)
    task_config = load_yaml(task_configs)
    # task_config["conditioning"]["params"]["scale"] = scale
    # task_config["conditioning"]["params"]["tau"] = tau
    # assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    # "learn_sigma must be the same for model and diffusion configuartion."

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config["measurement"]
    operator = get_operator(device=device, **measure_config["operator"])
    noiser = get_noise(**measure_config["noise"])
    l = measure_config["noise"]
    o = ""
    for i, j in l.items():
        if i == "name":
            i = "noise"
        o += f" / {i}: {j}"
    logger.info(f"Operation: {measure_config['operator']['name']}{o}")

    # Prepare conditioning method
    cond_config = task_config["conditioning"]
    cond_method = get_conditioning_method(
        cond_config["method"], operator, noiser, **cond_config["params"]
    )
    measurement_cond_fn = cond_method.conditioning
    l = task_config["conditioning"]["params"]
    o = ""
    for i, j in l.items():
        o += f" / {i}: {j}"
    logger.info(f"Conditioning method: {task_config['conditioning']['method']}{o}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(
        sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn
    )

    # Working directory
    name = "_".join(task_configs.split("_")[:-1])
    out_path = os.path.join(save_dir, name)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ["input", "recon", "progress", "label"]:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config["data"]
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
        logger.info(f"Inference for image {i}")
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
        x_start = torch.randn(ref_img.shape, device=device)
        img_tensor, psnr_value, ssim_value = sample_fn(
            method=task_config["conditioning"]["method"],
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
    main(
        model_config,
        diffusion_config,
        task_configs,
    )
