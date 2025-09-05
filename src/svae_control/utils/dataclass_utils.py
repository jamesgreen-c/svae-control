import jax
from flax.struct import dataclass
from jax import Array
from jax import numpy as jnp


@dataclass
class GaussianParams:
    mean: Array
    cov: Array

    @property
    def is_diag(self) -> bool:
        return False

    @property
    def full_cov(self) -> Array:
        return self.cov

    def __getitem__(self, idx: slice | int) -> "GaussianParams":
        assert self.mean.ndim == 2, "Trying to index an unbatched Gaussian"
        return GaussianParams(self.mean[idx], self.cov[idx])

    def __len__(self) -> int:
        if self.mean.ndim == 1:
            return 1
        return self.mean.shape[0]


@dataclass
class DiagGaussianParams(GaussianParams):
    @property
    def is_diag(self) -> bool:
        return True

    @property
    def full_cov(self) -> Array:
        ndims = self.cov.ndim
        fn = jnp.diag
        for _ in range(ndims - 1):
            fn = jax.vmap(fn)
        return fn(self.cov)


@dataclass
class LGSSMSmoothedPosterior:
    smoothed_means: Array
    smoothed_covs: Array
    smoothed_ExxnT: Array

    @property
    def pairwise_covs(self) -> Array:
        return self.smoothed_ExxnT + jnp.einsum(
            "ti,tj->tij", self.smoothed_means[:-1], self.smoothed_means[1:]
        )


@dataclass
class LDSParams:
    A: Array
    b: Array

    @property
    def Q(self) -> Array:
        return jnp.eye(self.A.shape[0]) - self.A @ self.A.T

    @property
    def Q0(self) -> Array:
        return jnp.eye(self.A.shape[0]) # - self.A @ self.A.T

    @property
    def m0(self) -> Array:
        return jnp.zeros(self.A.shape[0])

    @staticmethod
    def init(latent_dim: int) -> "LDSParams":
        return LDSParams(A=jnp.eye(latent_dim) * 0.5, b=jnp.zeros(latent_dim))


@dataclass
class GaussianChainMarginalParams:
    marginals: list[GaussianParams]
    pairwise_marginals: list[GaussianParams]


@dataclass
class InputData:
    obs: Array
    targets: Array

    def __getitem__(self, idx: int | slice | Array) -> "InputData":
        return InputData(self.obs[idx], self.targets[idx])

    def __len__(self) -> int:
        return self.obs.shape[0]

    @property
    def T(self) -> int:
        return self.obs.shape[1]

    def get_time_slice(self, starts: Array, ends: Array) -> "InputData":
        assert starts.ndim == 1 and ends.ndim == 1
        assert len(starts) == len(self) == len(ends)
        obs = jnp.array(
            [self.obs[i, s:e] for i, (s, e) in enumerate(zip(starts, ends))]
        )
        targets = jnp.array(
            [self.targets[i, s:e] for i, (s, e) in enumerate(zip(starts, ends))]
        )
        return InputData(obs, targets)

    def get_batch(self, batch_idxs: Array, start_idxs: Array, T: int) -> "InputData":
        end_idxs = start_idxs + T
        seq_len = self.obs.shape[1]
        assert jnp.all(end_idxs < seq_len), "Batch goes out of bounds"
        obs = jnp.array(
            [self.obs[i, s:e] for i, s, e in zip(batch_idxs, start_idxs, end_idxs)]
        )

        targets = jnp.array(
            [self.targets[i, s:e] for i, s, e in zip(batch_idxs, start_idxs, end_idxs)]
        )
        return InputData(obs=obs, targets=targets)
