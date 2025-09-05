from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import Array

if TYPE_CHECKING:
    from svae_control.utils.dataclass_utils import GaussianParams

NetworkParams = Union[FrozenDict, dict[str, Any]]


def get_normal_dist(params: GaussianParams) -> tfd.Distribution:
    if params.is_diag:
        return tfd.MultivariateNormalDiag(params.mean, params.cov)
    return tfd.MultivariateNormalFullCovariance(params.mean, params.cov)


def lie_params_to_constrained(out_flat: Array, dim: int, eps: float = 1e-4) -> Array:
    D, A = out_flat[:dim], out_flat[dim:]
    D = nn.softplus(D) + eps
    # Build a skew-symmetric matrix
    S = jnp.zeros((dim, dim))
    i1, i2 = jnp.tril_indices(dim - 1)
    S = S.at[i1 + 1, i2].set(A)
    S = S - S.T
    O = jax.scipy.linalg.expm(S)
    J: Array = O.T @ jnp.diag(D) @ O
    return J


def R2_inferred_vs_actual_z(posterior_means: Array, true_z: Array) -> Array:
    # true_z_shape = true_z.shape
    # posterior_means = posterior_means.reshape(-1, true_z_shape[-1])
    # true_z = true_z.reshape(-1, true_z_shape[-1])
    _, res, _, _ = jax.numpy.linalg.lstsq(posterior_means, true_z)
    r2 = 1 - res / jnp.sum((true_z - true_z.mean()) ** 2)
    return r2


def inv_softplus(x: Array | float, eps: float = 1e-4) -> Array:
    return jnp.log(jnp.exp(x - eps) - 1)


def get_extreme_eigenvalues(A: Array) -> tuple[Array, Array]:
    eigs = jnp.linalg.eigvalsh(A)
    return jnp.min(eigs), jnp.max(eigs)


def clip_sv(A, eps):
    """
    Clip the SVs of a matrix to be less than 1-EPS.
    """
    u, s, vt = jnp.linalg.svd(A)
    return u @ jnp.diag(jnp.clip(s, 0., 1.-eps)) @ vt