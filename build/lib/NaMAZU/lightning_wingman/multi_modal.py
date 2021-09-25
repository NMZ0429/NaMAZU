import torch
from torch import Tensor


class StochasticFusion(torch.nn.Module):
    def __init__(self, num_modalities: int, mu: Tensor = None, rho: Tensor = None):
        super().__init__()

        if mu is None:
            mu = torch.randn(num_modalities)
        if rho is None:
            rho = torch.randn(num_modalities)

        self.mu = torch.nn.Parameter(mu)  # type: ignore
        self.rho = torch.nn.Parameter(rho)  # type: ignore
        self.register_buffer("eps_w", torch.Tensor(self.mu.shape))
        self.sigma = None
        self.w = None

    def sample(self):
        """
        Samples weights by sampling form a Normal distribution, multiplying by a sigma, which is 
        a function from a trainable parameter, and adding a mean
        sets those weights as the current ones
        returns:
            torch.tensor with same shape as self.mu and self.rho
        """

        self.eps_w.data.normal_()  # type: ignore
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.mu + self.sigma * self.eps_w
        return self.w
