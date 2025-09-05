from flax import linen as nn
from jax import Array
from jax import numpy as jnp

from svae_control.config import NNConfig, NNDistConfig
from svae_control.registry import NNType
from svae_control.utils.dataclass_utils import DiagGaussianParams, GaussianParams
from svae_control.utils.utils import lie_params_to_constrained

INITIALIZER = nn.initializers.variance_scaling(
    scale=0.1, mode="fan_in", distribution="truncated_normal"
)


class NoOp(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x


class MLP(nn.Module):
    config: NNConfig

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for h in self.config.hidden_dims:
            x = nn.Dense(features=h, kernel_init=INITIALIZER)(x)
            x = nn.relu(x)
        return x


class CNN(nn.Module):
    config: NNConfig

    @nn.compact
    def __call__(self, x: Array) -> Array:
        kernel_size = self.config.nn_args.get("kernel_size", (3, 3))
        kernel_strides = self.config.nn_args.get("kernel_strides", (1, 1))
        pool_window_shape = self.config.nn_args.get("pool_window_shape", (2, 2))
        pool_strides = self.config.nn_args.get("pool_strides", (2, 2))
        for h in self.config.hidden_dims:
            x = nn.Conv(features=h, kernel_size=kernel_size, strides=kernel_strides)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=pool_window_shape, strides=pool_strides)
        return x.flatten()


def get_backbone(config: NNConfig) -> nn.Module:
    if NNType(config.nn_type) == NNType.CNN:
        return CNN(config)
    elif NNType(config.nn_type) == NNType.MLP:
        return MLP(config)
    elif NNType(config.nn_type) == NNType.NoOp:
        return NoOp()
    raise ValueError(f"Unknown NNType: {config.nn_type}")


class GaussianNN(nn.Module):
    config: NNDistConfig
    output_dim: int

    @nn.compact
    def __call__(self, x: Array) -> GaussianParams:
        backbone = get_backbone(self.config.backbone)
        x = backbone(x)
        mean = MLP(self.config.head)(x)
        mean = nn.Dense(self.output_dim)(mean)
        cov_output_dim = self.output_dim * (self.output_dim + 1) // 2
        cov_flat = MLP(self.config.head)(x)
        cov_flat = nn.Dense(cov_output_dim)(cov_flat)
        cov = lie_params_to_constrained(cov_flat, self.output_dim)
        return GaussianParams(mean, cov)


class DiagGaussianNN(nn.Module):
    config: NNDistConfig
    output_dim: int
    eps: float = 1e-4

    @nn.compact
    def __call__(self, x: Array) -> GaussianParams:
        backbone = get_backbone(self.config.backbone)
        x = backbone(x)
        mean = MLP(self.config.head)(x)
        mean = nn.Dense(self.output_dim)(mean)
        cov = MLP(self.config.head)(x)
        cov = nn.softplus(nn.Dense(self.output_dim)(cov))
        cov += self.eps
        return DiagGaussianParams(mean, cov)


class DiagGaussianFixedVarNN(DiagGaussianNN):

    @nn.compact
    def __call__(self, x: Array) -> GaussianParams:
        backbone = get_backbone(self.config.backbone)
        x = backbone(x)
        mean = MLP(self.config.head)(x)
        mean = nn.Dense(self.output_dim)(mean)
        cov = jnp.ones(self.output_dim) * self.config.kwargs["var_scale"]
        return DiagGaussianParams(mean, cov)
