"""This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n."""

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel
import deepinv as dinv
from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m

# =================
# Operation classes
# =================

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name="noise")
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device

    def forward(self, data):
        return data

    def transpose(self, data):
        return data

    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name="deconvolution")
class DeconvolutionOperator(LinearOperator):
    def __init__(
        self, n_channels: int, img_size: int, device: str = "cuda", sigma: float = 2.0
    ):
        self.device = device
        self.sigma = sigma
        filter_torch = dinv.physics.functional.gaussian_blur(sigma=(sigma, sigma))
        self.physics = dinv.physics.Blur(
            img_size=(
                n_channels,
                img_size,
                img_size,
            ),
            filter=filter_torch,
            padding="circular",
            device=device,
        )

    def forward(self, data, **kwargs):
        """A x  — applies blur."""
        if data.dtype != torch.float32:
            data = data.float()
        x = data
        # print(
        #     f"[DEBUG] forward - x shape: {x.shape}, min: {x.min():.4f}, max: {x.max():.4f}"
        # )
        return self.physics(data)

    def transpose(self, data, **kwargs):
        """A^T x — adjoint."""
        if data.dtype != torch.float32:
            data = data.float()
        x = data
        # print(
        #     f"[DEBUG] forward (transpose) - x shape: {x.shape}, min: {x.min():.4f}, max: {x.max():.4f}"
        # )
        return self.physics.A_adjoint(data)


@register_operator(name="super_resolution")
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1 / scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)


@register_operator(name="motion_blur")
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(
            blur_type="motion", kernel_size=kernel_size, std=intensity, device=device
        ).to(
            device
        )  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)

    def forward(self, data, **kwargs):
        # A^T * A
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name="gaussian_blur")
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(
            blur_type="gaussian", kernel_size=kernel_size, std=intensity, device=device
        ).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        # For symmetric Gaussian blur, the transpose operation (adjoint)
        # is equivalent to running the forward convolution over the data.
        return self.conv(data)

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name="tomography")
class TomographyOperator(LinearOperator):
    def __init__(self, img_size, angles, device="cuda"):
        self.device = device
        import deepinv as dinv

        self.physics = dinv.physics.TomographyWithAstra(
            img_size=[img_size, img_size], angles=180, n_detector_pixels=512
        ).to(device)

    def forward(self, data, **kwargs):
        out = self.physics.A(data)
        return out

    def transpose(self, data, **kwargs):
        # Backprojection step
        out = self.physics.A_adjoint(data)
        return out


@register_operator(name="inpainting")
class InpaintingOperator(LinearOperator):
    """This operator get pre-defined mask and return masked image."""

    def __init__(self, device):
        self.device = device

    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get("mask", None).to(self.device)
        except:
            raise ValueError("Require mask")

    def transpose(self, data, **kwargs):
        return data

    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data)


@register_operator(name="phase_retrieval")
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device

    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude


@register_operator(name="nonlinear_blur")
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)

    def prepare_nonlinear_blur_model(self, opt_yml_path):
        """
        Nonlinear deblur requires external codes (bkse).
        """
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path))
        blur_model = blur_model.to(self.device)
        return blur_model

    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred


# =============
# Noise classes
# =============


__NOISE__ = {}


def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls

    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass


@register_noise(name="clean")
class Clean(Noise):
    def forward(self, data):
        return data


@register_noise(name="gaussian")
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


import torch
import deepinv as dinv


@register_noise(name="poisson")
class PoissonNoise(Noise):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

        # On appelle DIRECTEMENT la classe officielle de deepinv.
        # normalize=True permet de gérer automatiquement la mise à l'échelle.
        self.dinv_poisson = dinv.physics.PoissonNoise(gain=rate, normalize=True)

    def forward(self, data):
        """
        Applique le bruit de Poisson de deepinv.
        Prend en entrée une image dans [-1, 1] et retourne une image dans [-1, 1].
        """
        # 1. Passage de l'intervalle [-1, 1] à [0, 1] (requis par deepinv)
        data_normalized = (data + 1.0) / 2.0
        data_normalized = data_normalized.clamp(0.0, 1.0)

        # 2. Utilisation directe du générateur de deepinv
        # (S'exécute nativement sur le GPU/CPU actuel de votre tenseur data)
        noisy_data = self.dinv_poisson(data_normalized)

        # 3. Retour à l'intervalle initial [-1, 1]
        noisy_data = noisy_data * 2.0 - 1.0
        return noisy_data.clamp(-1.0, 1.0)
