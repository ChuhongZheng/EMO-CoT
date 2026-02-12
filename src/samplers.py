import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "gaussian_ood": GaussianOODSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


class GaussianSampler(DataSampler):
    def __init__(self, n_dims):
        super().__init__(n_dims)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None):
        xs_b = torch.randn(b_size, n_points, self.n_dims)
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class GaussianOODSampler(DataSampler):
    """
    OOD (Out-of-Distribution) Gaussian sampler.
    Samples x ~ N(0, c * Λ) where:
      - c is a global scale factor (default 4.0)
      - Λ is a fixed diagonal matrix per sampler instance, with diagonal
        entries λ_i ~ Exponential(1), introducing unequal variances
        across dimensions.
    """

    def __init__(self, n_dims, c: float = 4.0):
        super().__init__(n_dims)
        self.c = float(c)
        # Sample a fixed Λ for this sampler: shape (n_dims,)
        # Tensor.exponential_(lambd) draws from Exp(lambd).
        self.lambda_diag = torch.empty(self.n_dims).exponential_(1.0)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None):
        # Base standard normal samples: (batch, points, dims)
        xs_b = torch.randn(b_size, n_points, self.n_dims)

        # Per-dimension standard deviation: sqrt(c * λ_i), shape (1, 1, dims)
        std_per_dim = torch.sqrt(self.c * self.lambda_diag).view(1, 1, -1)
        xs_b = xs_b * std_per_dim

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
