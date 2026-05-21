import torch
import deepinv as dinv
import matplotlib.pyplot as plt
from deepinv.utils import load_example
import torch
import gc

gc.collect()

# Clear unused caches
torch.cuda.empty_cache()

# Configuration du device
device = dinv.utils.get_device()

img_size = 64
num_angles = 120


def to_1c(x):
    return x.mean(dim=1, keepdim=True)


def to_3c(x):
    return x.repeat(1, 3, 1, 1)


class PhysicsWrapper:
    def __init__(self, physics):
        self.physics = physics

    def A(self, x):
        return self.physics.A(to_1c(x))

    def A_adjoint(self, y):
        return to_3c(self.physics.A_adjoint(y))

    def __call__(self, x):
        return self.A(x)


maximums = []

# noises for gaussian and poisson
noises = [
    (0.05, 1 / 30),
    (0.1, 1 / 20),
    (0.2, 1 / 10),
    (0.3, 1 / 5),
    (0.5, 1 / 2),
    (0.7, 1 / 1.5),
    (1.0, 1 / 1),
]
for sigma, gain_ct in noises:
    print(
        f"\n\n{'='*60}\nTesting with Sigma = {sigma} and Gain CT = {gain_ct}\n{'='*60}"
    )
    # --- Simulation ---
    eta_values = torch.linspace(0, 1, 15)  # 15 points of etas
    results = []
    for eta in eta_values:
        print(
            f"\nTesting with Eta = {eta:.2f} ({eta_values.tolist().index(eta.item())+1}/{len(eta_values)})"
        )
        # --- chargement de l'image et simulation des mesures ---
        x_true = load_example(
            "SheppLogan.png", img_size=img_size, grayscale=True, device=device
        )
        x_true = x_true / x_true.max()

        physics_gauss = dinv.physics.Tomography(
            img_width=img_size,
            angles=num_angles,
            device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=sigma),
        )
        physics_poisson = dinv.physics.Tomography(
            img_width=img_size,
            angles=num_angles,
            device=device,
            noise_model=dinv.physics.PoissonNoise(
                gain=gain_ct, clip_positive=True, normalize=True
            ),
        )

        y_gauss = physics_gauss(x_true)
        y_poisson = physics_poisson(x_true)

        physics_gauss_wrapped = PhysicsWrapper(physics_gauss)
        physics_poisson_wrapped = PhysicsWrapper(physics_poisson)

        # Prior : Modèle de diffusion
        backbone = dinv.models.DiffUNet().to(device)

        # metric pour évaluer la qualité de la reconstruction
        metric = dinv.metric.PSNR()

        torch.manual_seed(30)
        with torch.no_grad():
            x_init_gauss = physics_gauss_wrapped.A_adjoint(y_gauss)
            x_init_poisson = physics_poisson_wrapped.A_adjoint(y_poisson)

        dps = dinv.sampling.DPS(
            backbone,
            data_fidelity=dinv.optim.data_fidelity.L2(),
            max_iter=1000,
            eta=eta.item(),
            verbose=True,
            device=device,
        )

        # DPS sur les modèles de bruit Gaussien et Poisson
        res_g = dps(y_gauss, physics_gauss_wrapped, x_init=x_init_gauss)
        res_p = dps(y_poisson, physics_poisson_wrapped, x_init=x_init_poisson)

        # Calcul PSNR
        psnr_g = metric(to_1c(res_g), x_true).item()
        psnr_p = metric(to_1c(res_p), x_true).item()

        # Stockage dans results
        results.append((eta.item(), psnr_g, psnr_p))

    # --- Extraction des Maximums ---
    eta_list, g_list, p_list = zip(*results)
    max_g_idx = g_list.index(max(g_list))
    max_p_idx = p_list.index(max(p_list))

    maximums.append(
        {
            "sigma": sigma,
            "gain_ct": gain_ct,
            "best_gauss": (eta_list[max_g_idx], g_list[max_g_idx]),
            "best_poisson": (eta_list[max_p_idx], p_list[max_p_idx]),
            "max_g_idx": max_g_idx,
            "max_p_idx": max_p_idx,
        }
    )

    # --- Graphiquepour voir l'évolution des PSNR des bruits ---
    plt.figure(figsize=(10, 5))
    plt.plot(eta_list, g_list, "o-", label="Gaussien", color="blue", linewidth=2)
    plt.plot(eta_list, p_list, "s-", label="Poisson", color="red", linewidth=2)
    plt.axvline(
        eta_list[max_g_idx],
        color="blue",
        linestyle="--",
        alpha=0.5,
        label=f"Best Gauss (iter={max_g_idx}, PSNR={g_list[max_g_idx]:.2f} dB,Eta={eta_list[max_g_idx]:.2f})",
    )
    plt.axvline(
        eta_list[max_p_idx],
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Best Poisson (iter={max_p_idx}, PSNR={p_list[max_p_idx]:.2f} dB,Eta={eta_list[max_p_idx]:.2f})",
    )

    plt.title(
        f"Performance de DPS en fonction de la stochasticité Eta (Sigma = {sigma} and Gain = {gain_ct})"
    )
    plt.xlabel("Paramètre Eta")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


print("\n\nRésumé des Meilleurs Résultats :")
for max_result in maximums:
    print(f"Sigma = {max_result['sigma']}, Gain CT = {max_result['gain_ct']}:")
    print(
        f"  - Best Gauss: Eta = {max_result['best_gauss'][0]}, PSNR = {max_result['best_gauss'][1]:.2f} dB"
    )
    print(
        f"  - Best Poisson: Eta = {max_result['best_poisson'][0]}, PSNR = {max_result['best_poisson'][1]:.2f} dB"
    )

plt.plot(
    [m["sigma"] for m in maximums],
    [m["best_gauss"][1] for m in maximums],
    "o-",
    label="Best Gauss PSNR",
    color="blue",
    linewidth=2,
)
plt.plot(
    [m["gain_ct"] for m in maximums],
    [m["best_poisson"][1] for m in maximums],
    "s-",
    label="Best Poisson PSNR",
    color="red",
    linewidth=2,
)
plt.title("Meilleure Performance de DPS en fonction du Bruit")
plt.xlabel("Paramètre Sigma")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
