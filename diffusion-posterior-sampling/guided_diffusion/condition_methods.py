from abc import ABC, abstractmethod
import torch

__CONDITIONING_METHOD__ = {}


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        """
        Original DPS gradient computation (Chung et al. 2022).
        - Gaussian: gradient of ||y - A(x_0_hat)||
        - Poisson:  gradient of KL(y || A(x_0_hat))  [unstable, kept for reference]
        Pour le bruit de poisson, on utilise le code pour le cas gaussien,
        modification par rapport à Chung et al. 2022 'or self.noiser.__name__ == "poisson"'
        """

        if self.noiser.__name__ == "gaussian" or self.noiser.__name__ == "poisson":
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        elif self.noiser.__name__ == "poisson":
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / (measurement.abs() + 1e-7)
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, norm

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Original methods (unchanged)
# ---------------------------------------------------------------------------


@register_conditioning_method(name="vanilla")
class Identity(ConditioningMethod):
    def conditioning(self, x_t, **kwargs):
        return x_t


@register_conditioning_method(name="projection")
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name="mcg")
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(
        self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs
    ):

        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t -= norm_grad * self.scale
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm


@register_conditioning_method(name="ps")
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t -= norm_grad * self.scale
        return x_t, norm


@register_conditioning_method(name="ps+")
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get("num_sampling", 5)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm


# ---------------------------------------------------------------------------
# New methods for Poisson noise (DPSP-grad and DPSP-prox)
# ---------------------------------------------------------------------------


@register_conditioning_method(name="dpsp_grad")
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 0.1)
        self.eps = kwargs.get("eps", 1e-6)

    def _to_01(self, x):
        if x.min() < -0.5:
            return (x + 1.0) / 2.0, 0.5
        else:
            return x, 1.0

    def poisson_log_likelihood(self, x, y):
        x_01, scale = self._to_01(x)
        y_01, _ = self._to_01(y)
        x_01 = torch.clamp(x_01, min=0.0, max=1.0)
        y_01 = torch.clamp(y_01, min=0.0, max=1.0)
        Ax = self.operator.forward(x_01)
        Ax = torch.clamp(Ax, min=self.eps, max=1.0)
        loss = torch.sum(Ax - y_01 * torch.log(Ax + self.eps))
        return loss

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        loss = self.poisson_log_likelihood(x_0_hat, measurement)
        grad_xt = torch.autograd.grad(loss, x_prev)[0]
        norm = torch.norm(grad_xt) + 1e-8
        x_corrected = x_t - self.scale * (self._to_01(x_0_hat)[0]) * grad_xt
        x_corrected = torch.clamp(x_corrected, -1.0, 1.0)
        return x_corrected, norm


@register_conditioning_method(name="dpsp_prox")
class PosteriorSamplingPoissonProxEM(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.tau = kwargs.get("tau", 0.05)
        self.eps = kwargs.get("eps", 1e-8)
        self.iter_count = 0

    def prox_em_majorant(self, x, x_tilde, y, tau, **kwargs):
        y = torch.clamp(y, -1.0, 1.0)
        x = torch.clamp(x, -1.0, 1.0)
        x_tilde = torch.clamp(x_tilde, -1.0, 1.0)
        x_pos = (x + 1.0) / 2.0
        x_tilde_pos = (x_tilde + 1.0) / 2.0
        y_pos = (y + 1.0) / 2.0
        x_pos = torch.clamp(x_pos, 0.0, 1.0)
        x_tilde_pos = torch.clamp(x_tilde_pos, 0.0, 1.0)
        y_pos = torch.clamp(y_pos, 0.0, 1.0)
        ones_y = torch.ones_like(y_pos)
        s = self.operator.transpose(ones_y, **kwargs)
        s = torch.clamp(s, min=self.eps)
        Ax_tilde = self.operator.forward(x_tilde_pos, **kwargs)
        Ax_tilde = torch.clamp(Ax_tilde, self.eps, 1.0)
        ratio = y_pos / (Ax_tilde + self.eps)
        ratio = torch.clamp(ratio, min=1e-4, max=1e4)
        adj = self.operator.transpose(ratio, **kwargs)
        x_em = (x_tilde_pos / s) * adj

        linear_diff = x_pos - tau * s
        sqrt_arg = linear_diff**2 + 4 * tau * s * x_em
        sqrt_arg = torch.clamp(sqrt_arg, min=self.eps)
        sqrt_term = torch.sqrt(sqrt_arg + self.eps**2)

        prox_val_pos = 0.5 * (linear_diff + sqrt_term)
        prox_val_pos = torch.clamp(prox_val_pos, 0.0, 1.0)
        prox_val = prox_val_pos * 2.0 - 1.0
        return prox_val

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        self.iter_count += 1
        prox = self.prox_em_majorant(
            x=x_0_hat,
            x_tilde=x_t,
            tau=self.tau,
            y=measurement,
            **kwargs,
        )
        return prox
