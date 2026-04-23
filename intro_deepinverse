
import deepinv as dinv
import torch
from deepinv.optim import PGD

device = dinv.utils.get_device()
print(device)
images = ["mbappe.jpg","leaves.png", "barbara.jpeg", "butterfly.png"]
for image in images:
    print(f"---------- Processing {image} ----------")
    x = dinv.utils.load_example(image, device=device)
    print(x.shape)

    physics = dinv.physics.Inpainting(img_size=x.shape[1:], mask=0.5, device=device)
    y = physics(x)
    print(y.shape)
    dinv.utils.plot([x, y], titles=["Original", "Inpainted"])

    data_fidelity = dinv.optim.L2()
    prior = dinv.optim.TVPrior()
    print(data_fidelity)
    print(prior)
    lambd = 0.05
    norm_A2 = physics.compute_sqnorm(y, tol=1e-4, verbose=False).item()
    print(f"Estimated norm of A^2: {norm_A2:.4f}")
    step_size = 1.9 / norm_A2
    maxiter = 100
    x_k = torch.zeros_like(x, device=device)
    cost_history = torch.zeros(maxiter, device=device)
    with torch.no_grad():
        for k in range(maxiter):
            if k % 4 == 1:
                print(f"Iteration {k+1}/{maxiter}", end=" ")
            u = x_k - step_size * data_fidelity.grad(x_k, y, physics)
            x_k = prior.prox(u, gamma=step_size * lambd)
            cost = data_fidelity(x_k, y, physics) + lambd * prior(x_k)
            cost_history[k] = cost.item()
    print()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(cost_history.detach().cpu().numpy(), marker="o")
    plt.title("Cost history")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid()
    plt.show()

    metric = dinv.metric.PSNR()

    dinv.utils.plot(
        {
            f"Originale": x,
            f"Degradée \n {metric(y, x).item():.2f} dB": y,
            f"Recon scratch \n {metric(x_k, x).item():.2f} dB": x_k,
        }
    )

    x_k = x_k.clone()

    denoiser = dinv.models.DRUNet(device=device)  # Load a pretrained denoiser

    with torch.no_grad():  # disable autodifferentiation
        for it in range(maxiter):
            u = x_k - step_size * data_fidelity.grad(x_k, y, physics)  # Gradient step
            x_k = denoiser(u, sigma=0.05)  # replace prox by denoising step

    dinv.utils.plot(
        {
            f"Originale": x,
            f"Degradé\n {metric(y, x).item():.2f} dB": y,
            f"Recon Denoiser\n {metric(x_k, x).item():.2f} dB": x_k,
        }
    )


    class Algo(dinv.models.Reconstructor):
        def __init__(self, data_fidelity, prior, stepsize, lambd, max_iter):
            super().__init__()
            self.data_fidelity = data_fidelity
            self.prior = prior
            self.stepsize = stepsize
            self.lambd = lambd
            self.max_iter = max_iter

        def forward(self, y, physics, **kwargs):
            """ forward process of the reconstructor, takes measurements and physics as input and returns the reconstruction """
            x_k = torch.zeros_like(y, device=y.device)  # initial guess

            # Disable autodifferentiation, remove this if you want to unfold
            with torch.no_grad():
                for _ in range(self.max_iter):
                    u = x_k - self.stepsize * self.data_fidelity.grad(
                        x_k, y, physics
                    )  # Gradient step
                    x_k = self.prior.prox(
                        u, gamma=self.lambd * self.stepsize
                    )  # Proximal step

            return x_k


    tv_algo = Algo(data_fidelity, prior, step_size, lambd, maxiter)

    # Standard reconstructor forward pass
    x_hat = tv_algo(y, physics)

    dinv.utils.plot(
        {
            f"Ground truth": x,
            f"Measurements\n {metric(y, x).item():.2f} dB": y,
            f"Recon Class PGD\n {metric(x_hat, x).item():.2f} dB": x_hat,
        }
    )

    print("Algo from the deep inverse library")
    prior = dinv.optim.PnP(denoiser=denoiser)  # prior with prox via denoising step


    model = PGD(
        prior=prior,
        data_fidelity=data_fidelity,
        stepsize=step_size,
        sigma_denoiser=0.05,
        max_iter=maxiter,verbose=True
    )

    x_hat = model(y, physics,init=tv_algo(y, physics))

    dinv.utils.plot(
        {
            f"Ground truth": x,
            f"Measurements\n {metric(y, x).item():.2f} dB": y,
            f"Recon Deep\n {metric(x_hat, x).item():.2f} dB": x_hat,
        }
    )
