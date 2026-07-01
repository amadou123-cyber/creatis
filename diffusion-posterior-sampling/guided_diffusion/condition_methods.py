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
        """
        if self.noiser.__name__ == "gaussian":
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

    def prox_em_majorant(self, x_t, y, tau, **kwargs):

        # 1. Modèle direct
        Ax_t = self.operator.forward(x_t, **kwargs)
        Ax_t = torch.clamp(Ax_t, min=1e-6)

        ones_y = torch.ones_like(y).to(x_t.dtype)
        s = self.operator.transpose(ones_y, **kwargs)
        s = torch.clamp(s, min=1e-8)
        ratio = (y / Ax_t).to(x_t.dtype)
        adjoint_term = self.operator.transpose(ratio, **kwargs)
        x_em = (x_t / s) * adjoint_term

        linear_diff = x_t - tau * s
        sqrt_arg = (linear_diff**2) + 4 * tau * s * x_em
        sqrt_arg = torch.clamp(sqrt_arg, min=0.0)
        sqrt_term = torch.sqrt(sqrt_arg + 1e-16)

        prox_val = 0.5 * (linear_diff + sqrt_term)

        return prox_val

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
        # NO conversions! Work directly in [-1, 1] like ps/ps+
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
# My implemented methods (new)
# ---------------------------------------------------------------------------
@register_conditioning_method(name="ps_prox")
class PosteriorSamplingPoissonProxEM(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.tau = kwargs.get("tau", 0.05)
        self.eps = kwargs.get("eps", 1e-8)
        self.iter_count = 0

    def prox_em_majorant(self, x, x_tilde, y, tau, **kwargs):

        # 1. Clamp des entrées
        y = torch.clamp(y, -1.0, 1.0)
        x = torch.clamp(x, -1.0, 1.0)
        x_tilde = torch.clamp(x_tilde, -1.0, 1.0)

        # 2. Conversion [-1, 1] → [0, 1]
        x_pos = (x + 1.0) / 2.0
        x_tilde_pos = (x_tilde + 1.0) / 2.0
        y_pos = (y + 1.0) / 2.0

        # clamping in [0, 1]
        x_pos = torch.clamp(x_pos, 0.0, 1.0)
        x_tilde_pos = torch.clamp(x_tilde_pos, 0.0, 1.0)
        y_pos = torch.clamp(y_pos, 0.0, 1.0)

        # 3. s = A^T(1)
        ones_y = torch.ones_like(y_pos)
        s = self.operator.transpose(ones_y, **kwargs)
        s = torch.clamp(s, min=self.eps)

        # 4. A(x_tilde)
        Ax_tilde = self.operator.forward(x_tilde_pos, **kwargs)
        Ax_tilde = torch.clamp(Ax_tilde, self.eps, 1.0)

        # 5. Ratio (avec clamping sévère pour éviter l'explosion)
        ratio = y_pos / (Ax_tilde + self.eps)
        ratio = torch.clamp(ratio, min=1e-4, max=1e4)

        # 6. Adjoint
        adj = self.operator.transpose(ratio, **kwargs)

        # 7. Mise à jour EM
        x_em = (x_tilde_pos / s) * adj

        # 8. Opérateur proximal
        linear_diff = x_pos - tau * s
        sqrt_arg = linear_diff**2 + 4 * tau * s * x_em
        sqrt_arg = torch.clamp(sqrt_arg, min=self.eps)
        sqrt_term = torch.sqrt(sqrt_arg + self.eps**2)

        prox_val_pos = 0.5 * (linear_diff + sqrt_term)
        prox_val_pos = torch.clamp(prox_val_pos, 0.0, 1.0)

        # 9. Retour à [-1, 1]
        prox_val = prox_val_pos * 2.0 - 1.0

        return prox_val

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        self.iter_count += 1

        if self.iter_count == 1:
            x_t = x_prev
            x_0_hat = measurement
            x_0_hat = torch.clamp(x_0_hat, -1.0, 1.0)

        prox = self.prox_em_majorant(
            x=x_0_hat,
            x_tilde=x_t,
            tau=self.tau,
            y=measurement,
            **kwargs,
        )
        return prox


class BregmanPotential:
    """Bregman potential functions for different noise models."""

    @staticmethod
    def poisson_entropy(x):
        """Boltzmann-Shannon entropy: φ(x) = x log x - x"""
        x = torch.clamp(x, min=1e-8)
        return x * torch.log(x) - x

    @staticmethod
    def poisson_hessian(x):
        """Hessian of Poisson entropy: ∇²φ(x) = 1/x"""
        x = torch.clamp(x, min=1e-8)
        return 1.0 / x

    @staticmethod
    def poisson_hessian_inv(x):
        """Inverse Hessian: (∇²φ(x))⁻¹ = x"""
        return torch.clamp(x, min=1e-8)

    @staticmethod
    def bregman_divergence(x, y, potential="poisson"):
        """Dφ(x || y) = φ(x) - φ(y) - ⟨∇φ(y), x-y⟩"""
        if potential == "poisson":
            # ∇φ(y) = log(y)
            phi_x = x * torch.log(torch.clamp(x, min=1e-8)) - x
            phi_y = y * torch.log(torch.clamp(y, min=1e-8)) - y
            grad_phi_y = torch.log(torch.clamp(y, min=1e-8))
            return phi_x - phi_y - grad_phi_y * (x - y)
        else:
            raise NotImplementedError


@register_conditioning_method(name="bregman_dps")
class BregmanDPS(ConditioningMethod):
    """
    Implémentation de l'approche Bregman pour DPS (équation 33).
    Applique un préconditionnement multiplicatif (diag(x)) au gradient de vraisemblance.
    """

    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.beta = kwargs.get("beta", 0.1)  # Pas de temps de Langevin
        self.tau = kwargs.get("tau", 0.5)  # Pour le calcul du gradient KL (optionnel)
        self.eps = kwargs.get("eps", 1e-8)
        self.iter_count = 0

    def kl_gradient(self, x_0_hat, measurement, **kwargs):
        """
        Calcule le gradient de KL(y || A x_hat) par rapport à x_0_hat.
        Ici, on utilise l'approximation standard de DPS :
        ∇_x KL = A^T ( y / (A x_hat) - 1 )
        """
        # x_0_hat est en [-1, 1], on le passe en [0, 1] pour la physique
        x_pos = (x_0_hat + 1.0) / 2.0
        x_pos = torch.clamp(x_pos, 0.0, 1.0)
        y_pos = (measurement + 1.0) / 2.0
        y_pos = torch.clamp(y_pos, 0.0, 1.0)

        Ax = self.operator.forward(x_pos, **kwargs)
        Ax = torch.clamp(Ax, self.eps, 1.0)

        # Gradient de la KL de Poisson
        grad_pos = self.operator.transpose(y_pos / (Ax + self.eps) - 1.0, **kwargs)
        # On ramène le gradient dans l'espace [-1, 1] (attention à la chaîne de dérivation)
        # La conversion x_pos = (x + 1)/2 donne dx_pos/dx = 0.5, donc grad_x = 0.5 * grad_pos
        # Mais comme on va préconditionner, on garde la version [0,1] pour le moment.
        return grad_pos  # Gradient par rapport à x_pos

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        """
        Applique l'étape de Langevin avec préconditionnement Bregman.

        x_{t+1} = x_t - (beta/2) x_t - beta * diag(x_t) * (score + grad_kl) + sqrt(beta) * eps
        """
        self.iter_count += 1

        # 1. On travaille dans l'espace [0, 1] pour le préconditionneur
        x_pos = (x_t + 1.0) / 2.0
        x_pos = torch.clamp(x_pos, 0.0, 1.0)

        # 2. Score du modèle (en [-1, 1]) - on le convertit en [0,1] pour la cohérence ?
        # Le score est un gradient par rapport à x_t en [-1,1].
        # Pour simplifier, on utilise directement le score en [-1,1] et on le convertit.
        # La dérivée de la transformation est 0.5, on multiplie le score par 0.5 pour l'avoir en [0,1].
        # Mais attention, le réseau est entraîné sur [-1,1], donc on garde le score en [-1,1]
        # et on applique le préconditionneur en [-1,1] ? Non, le préconditionneur diag(x) demande x>0.

        # STRATÉGIE : On calcule le gradient de vraisemblance en [0,1], on le préconditionne,
        # on le convertit en [-1,1], et on l'ajoute au score euclidien standard.

        # 3. Gradient de vraisemblance (en [0,1])
        grad_kl_pos = self.kl_gradient(x_0_hat, measurement, **kwargs)
        grad_kl_pos = torch.clamp(grad_kl_pos, -1e3, 1e3)  # Stabilité

        # 4. Préconditionnement : diag(x) * grad_kl
        precond_grad_kl = x_pos * grad_kl_pos

        # 5. Conversion du gradient préconditionné vers [-1, 1] (dérivée de la transformation)
        # grad_kl = 0.5 * precond_grad_kl (car x = 2*x_pos - 1)
        grad_kl = 0.5 * precond_grad_kl

        # 6. Score du réseau (en [-1, 1])
        # On suppose que x_t est déjà en [-1,1]
        score = kwargs.model(x_t, self.iter_count)  # Adaptez selon votre API
        # Si vous n'avez pas noiser.score, utilisez model directement.
        # Pour l'exemple, on suppose que vous avez un modèle.

        # 7. Gradient total
        grad_total = score + grad_kl

        # 8. Terme de dérive (drift) en [-1, 1] ?
        # Le terme - (beta/2) * x_t est en [-1, 1].
        drift = -(self.beta / 2.0) * x_t

        # 9. Bruit de Langevin
        noise = torch.randn_like(x_t) * torch.sqrt(
            torch.tensor(self.beta, device=x_t.device)
        )

        # 10. Mise à jour finale
        x_next = x_t + drift - self.beta * grad_total + noise
        x_next = torch.clamp(x_next, -1.0, 1.0)

        # Distance (monitoring)
        Ax_hat = self.operator.forward(x_0_hat, **kwargs)
        Ax_hat = torch.clamp(Ax_hat, 1e-8)
        norm = torch.linalg.norm(measurement / (Ax_hat + 1e-8) - 1)

        return x_next, norm


@register_conditioning_method(name="ps_bregman")
class PosteriorSamplingBregman(ConditioningMethod):
    """
    Natural gradient DPS with Bregman preconditioning.
    Implements: x_{t+1} = x_t - β(t)/2 * x_t - β(t) * x_t * (sθ + ∇KL) + sqrt(β(t)) * ε
    for Poisson geometry where ∇²φ(x)⁻¹ = x
    """

    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.beta = kwargs.get("scale", 0.1)  # Step size
        self.tau = kwargs.get("tau", 0.05)  # For KL approximation
        self.use_clip = kwargs.get("use_clip", True)
        self.scale_kl = kwargs.get("scale_kl", 0.01)  # Scale KL gradient
        self.iter_count = 0

        # Bregman potential
        self.potential = BregmanPotential()

    def preconditioned_gradient(self, x, grad):
        """
        Apply Bregman preconditioner: (∇²φ(x))⁻¹ * grad
        For Poisson: ∇²φ(x) = 1/x, so preconditioner = x
        """
        # Ensure non-negativity
        x_pos = torch.clamp(x, min=1e-8)
        # Natural gradient: x * grad (element-wise)
        return x_pos * grad

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        self.iter_count += 1

        # 1. Compute standard gradient (from DPS)
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )

        # Handle case where gradient is None
        if norm_grad is None:
            norm_grad = torch.zeros_like(x_t)

        # 2. Apply natural gradient preconditioning
        # Convert to positive space [0,1]
        x_prev_pos = (x_prev + 1.0) / 2.0
        x_prev_pos = torch.clamp(x_prev_pos, min=1e-8)

        # Natural gradient: x * ∇KL
        natural_grad = self.preconditioned_gradient(x_prev_pos, norm_grad)

        # Scale the natural gradient
        natural_grad = natural_grad * self.scale_kl

        # 3. Get denoiser score (sθ)
        # Approximate from x_0_hat
        score = self.approximate_score(x_t, x_0_hat)

        # 4. Natural gradient update (Equation 33)
        beta_t = self.get_step_size(self.beta, kwargs)

        # Convert x_t to positive space
        x_t_pos = (x_t + 1.0) / 2.0
        x_t_pos = torch.clamp(x_t_pos, min=1e-8)

        # Combined gradient (score + likelihood)
        combined_grad = score + natural_grad

        # Natural gradient update
        # x_new = x - β/2 * x - β * x * (score + ∇KL)
        x_new_pos = x_t_pos - beta_t * x_t_pos - beta_t * x_t_pos * combined_grad

        # Add noise (from diffusion process)
        # FIX: Convert beta_t to tensor for sqrt
        beta_tensor = torch.tensor(beta_t, device=x_t.device, dtype=x_t.dtype)
        noise = torch.randn_like(x_t_pos)
        x_new_pos = x_new_pos + torch.sqrt(beta_tensor) * noise

        # Clamp and convert back to [-1, 1]
        if self.use_clip:
            x_new_pos = torch.clamp(x_new_pos, min=1e-8, max=1.0)

        x_t = x_new_pos * 2.0 - 1.0
        x_t = torch.clamp(x_t, -1.0, 1.0)

        # Monitor KL divergence
        Ax = self.operator.forward(x_0_hat, **kwargs)
        kl = torch.linalg.norm(measurement / (Ax + 1e-8) - 1)

        return x_t, kl

    def get_step_size(self, beta, kwargs):
        """Adaptive step size with cosine annealing."""
        progress = kwargs.get("progress", 0.0)
        if "total_iterations" in kwargs:
            progress = min(1.0, self.iter_count / kwargs["total_iterations"])

        # Cosine annealing: start at beta, end at beta/10
        cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        return beta * (0.1 + 0.9 * cosine.item())

    def approximate_score(self, x_t, x_0_hat):
        """
        Approximate the score function from x_0_hat.
        In practice, you'd get this from the diffusion model.
        """
        # Better approximation using the relationship between x_t and x_0_hat
        # For DDPM: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        # So score ≈ (x_0_hat - x_t) / sqrt(1-ᾱ_t)

        # Simple approximation
        diff = x_0_hat - x_t
        std = x_t.std() + 1e-8
        return diff / std
