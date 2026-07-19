import random
import torch
import torch.nn as nn


class MixStyle(nn.Module):

    def __init__(
        self,
        p=0.0,
        alpha=0.1,
        eps=1e-6,
    ):
        super().__init__()

        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):

        if (
            not self.training
            or self.p <= 0
            or random.random() > self.p
        ):
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)

        var = x.var(
            dim=[2, 3],
            keepdim=True,
            unbiased=False,
        )

        sigma = (var + self.eps).sqrt()

        x_norm = (x - mu) / sigma

        perm = torch.randperm(
            B,
            device=x.device,
        )

        mu2 = mu[perm]

        sigma2 = sigma[perm]

        lam = torch.distributions.Beta(
            self.alpha,
            self.alpha,
        ).sample((B, 1, 1, 1)).to(x.device)

        mu_mix = lam * mu + (1 - lam) * mu2

        sigma_mix = lam * sigma + (1 - lam) * sigma2

        return x_norm * sigma_mix + mu_mix