import jax
import numpy as np
from dynamax.linear_gaussian_ssm.inference import (
    lgssm_smoother,
    make_lgssm_params,
)
from flax import linen as nn
from jax import Array
from jax import numpy as jnp
from jax import random as jr

from svae_control.config import Config
from svae_control.utils.dataclass_utils import (
    GaussianParams,
    InputData,
    LDSParams,
    LGSSMSmoothedPosterior,
)
from svae_control.utils.utils import (
    NetworkParams,
    R2_inferred_vs_actual_z,
    get_extreme_eigenvalues,
    get_normal_dist,
)


class SVAE:

    def __init__(
        self,
        generation_network: nn.Module,
        recognition_network: nn.Module,
        config: Config,
        key: Array,
    ) -> None:
        self.generation_net = generation_network
        self.recognition_net = recognition_network
        self.config = config
        self.T = config.data.T
        self.latent_dim = config.model.latent_dim
        self.key = key

    def rollout(
        self,
        init_mean: Array,
        init_cov: Array,
        prior_params: LDSParams,
        key: Array,
        T: int,
    ) -> Array:
        As = jnp.tile(prior_params.A[None], (T, 1, 1))
        bs = jnp.concatenate(
            [init_mean[None], jnp.tile(prior_params.b[None], (T - 1, 1))]
        )
        Qs = jnp.concatenate(
            [
                init_cov[None],
                jnp.tile(prior_params.Q[None], (T - 1, 1, 1)),
            ]
        )

        keys = jr.split(key, T)
        biases = jax.vmap(jr.multivariate_normal)(keys, bs, Qs)
        init_elems = (As, biases)

        @jax.vmap
        def recursion(
            elem1: tuple[Array, Array], elem2: tuple[Array, Array]
        ) -> tuple[Array, Array]:
            A1, b1 = elem1
            A2, b2 = elem2
            return A2 @ A1, A2 @ b1 + b2

        sample: Array = jax.lax.associative_scan(recursion, init_elems)[1]
        return sample

    def sample_posterior_latents(
        self, post_params: LGSSMSmoothedPosterior, num_samples: int, key: Array
    ) -> Array:
        """
        sample z_t ~ p(z_t|x_{1:T}, a_{1:T})
        """
        keys = jr.split(key, self.T)
        means = post_params.smoothed_means
        covs = post_params.smoothed_covs
        return jax.vmap(jr.multivariate_normal, (0, 0, 0, None))(
            keys, means, covs, (num_samples,)
        )

    def get_post(
        self, prior_params: LDSParams, conjugate_params: GaussianParams
    ) -> LGSSMSmoothedPosterior:
        """
        runs kalman smoother to get posterior over latents
        """
        params = make_lgssm_params(
            initial_mean=prior_params.m0,
            initial_cov=prior_params.Q0,
            dynamics_weights=prior_params.A,
            dynamics_cov=prior_params.Q,
            dynamics_bias=prior_params.b,
            emissions_weights=jnp.eye(self.latent_dim),
            emissions_cov=conjugate_params.full_cov,
            emissions_bias=jnp.zeros(self.latent_dim),
        )

        smoothed = lgssm_smoother(params, conjugate_params.mean)

        return LGSSMSmoothedPosterior(
            smoothed_means=smoothed.smoothed_means,
            smoothed_covs=smoothed.smoothed_covariances,
            smoothed_ExxnT=smoothed.smoothed_cross_covariances,
        )

    def get_post_marginals(
        self, post_params: LGSSMSmoothedPosterior
    ) -> tuple[GaussianParams, GaussianParams]:
        """
        returns Gaussian params for p(z_t|x_{1:T}, a_{1:T}) and p(z_t, z_{t+1}|x_{1:T}, a_{1:T})
        """
        # p(z_t|x_{1:T}, a_{1:T})
        marginals = GaussianParams(
            post_params.smoothed_means, post_params.smoothed_covs
        )

        # E[z_t]E[z_{t+1}]^T
        ExExnT = jnp.einsum(
            "ti, tj -> tij",
            post_params.smoothed_means[:-1],
            post_params.smoothed_means[1:],
        )

        # E[(z_t, z_{t+1})]
        pairwise_means = jnp.concatenate(
            [post_params.smoothed_means[:-1], post_params.smoothed_means[1:]], axis=-1
        )

        # E[z_tz_{t+1}^T] - E[z_t]E[z_{t+1}]^T
        pairwise_cross_covs = post_params.smoothed_ExxnT - ExExnT

        # Cov[(z_t, z_{t+1})] = [[S_t, S_{t,t+1}], [S_{t,t+1}^T, S_{t+1}]]
        pairwise_covs = jnp.block(
            [
                [post_params.smoothed_covs[:-1], pairwise_cross_covs],
                [pairwise_cross_covs.transpose(0, 2, 1), post_params.smoothed_covs[1:]],
            ]
        )

        # p(z_t, z_{t+1}|x_{1:T}, a_{1:T})
        pairwise_marginals = GaussianParams(pairwise_means, pairwise_covs)
        return marginals, pairwise_marginals

    def get_prior_marginals(
        self, prior_params: LDSParams
    ) -> tuple[GaussianParams, GaussianParams]:
        """
        returns Gaussian params for p(z_t|a_{1:T}) and p(z_t, z_{t+1}|a_{1:T})
        """
        bs = jnp.concatenate(
            [prior_params.m0[None], jnp.tile(prior_params.b[None], (self.T - 1, 1))]
        )
        covs = jnp.concatenate(
            [
                prior_params.Q0[None],
                jnp.tile(prior_params.Q[None], (self.T - 1, 1, 1)),
            ]
        )
        As = jnp.tile(prior_params.A[None], (self.T, 1, 1))

        @jax.vmap
        def recursion(
            elem1: tuple[Array, Array, Array], elem2: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            b1, Q1, A1 = elem1
            b2, Q2, A2 = elem2
            return A2 @ b1 + b2, A2 @ Q1 @ A2.T + Q2, A2 @ A1

        init_elems = (bs, covs, As)
        means, covs, _ = jax.lax.associative_scan(recursion, init_elems)

        # p(z_t|a_{1:T})
        marginals = GaussianParams(means, covs)

        # p(z_t, z_{t+1}|a_{1:T})
        S1, S2 = covs[:-1], covs[1:]
        S12: Array = S1 @ prior_params.A.T
        pairwise_marginals = GaussianParams(
            jnp.concatenate([means[:-1], means[1:]], axis=-1),
            jnp.block([[S1, S12], [S12.transpose(0, 2, 1), S2]]),
        )
        return marginals, pairwise_marginals

    def kl_posterior_prior(
        self, prior_params: LDSParams, post_params: LGSSMSmoothedPosterior
    ) -> Array:
        """
        KL(q||p) = sum_t KL(q(z_t, z_{t+1}) || p(z_t, z_{t+1})) - sum_{t=2}^{T-1} KL(q(z_t) || p(z_t))
        """
        prior_marginals, prior_pairwise = self.get_prior_marginals(prior_params)
        post_marginals, post_pairwise = self.get_post_marginals(post_params)

        def kl_single(post: GaussianParams, prior: GaussianParams) -> Array:
            prior_dist = get_normal_dist(prior)
            post_dist = get_normal_dist(post)
            kl: Array = post_dist.kl_divergence(prior_dist)
            return kl

        kl_marginals = jax.vmap(kl_single)(post_marginals[1:-1], prior_marginals[1:-1])
        kl_pairwise = jax.vmap(kl_single)(post_pairwise, prior_pairwise)
        return kl_pairwise.sum() - kl_marginals.sum()

    def free_energy(
        self,
        params: tuple[LDSParams, NetworkParams, NetworkParams],
        data: InputData,
        key: Array,
        training: bool = True,
    ) -> dict[str, Array]:
        """
        -F = E_q[log p(x|z)] - KL(q(z|x, a)||p(z|a))
        """
        prior_params, generation_net_params, recognition_net_params = params

        # phi(z|x) (vmap across time dimension)
        conjugate_params: GaussianParams = jax.vmap(self.recognition_net.apply, (None, 0))(  # type: ignore
            recognition_net_params, data.obs
        )

        # mask out a window of observations
        num_masks = self.config.training.num_masks
        if num_masks and training:
            key, dropout_key = jr.split(key)
            T = len(data)
            D = self.config.model.latent_dim
            mask_size = self.config.training.mask_size // num_masks
            # potential dropout mask
            mask = jnp.ones(T, jnp.bool)
            for _ in range(num_masks):
                start_id = jr.choice(dropout_key, T - mask_size + 1)
                mask *= jnp.array(jnp.arange(T) >= start_id) * jnp.array(
                    jnp.arange(T) < start_id + mask_size
                )
            # uninformative potential
            infinity = self.config.training.uninformative_potential
            uninf_mean = np.zeros((T, D))
            uninf_cov = np.tile(np.eye(D) * infinity, (T, 1, 1))

            # replace masked parts with uninformative potentials
            conjugate_params = GaussianParams(
                mean=mask[..., None] * conjugate_params.mean
                + (1 - mask[..., None]) * uninf_mean,
                cov=mask[..., None, None] * conjugate_params.full_cov
                + (1 - mask[..., None, None]) * uninf_cov,
            )

        # get q(z|x) by kalman smoothing with emissions as phi(z|x)
        post_params = self.get_post(prior_params, conjugate_params)

        # sample z_t ~ q(z_t|x_{1:T})
        latents = self.sample_posterior_latents(
            post_params, self.config.training.num_samples, key
        )

        # generate x_t ~ p(x_t|z_t)
        generative_params: GaussianParams = jax.vmap(self.generation_net.apply, (None, 0))(  # type: ignore
            generation_net_params, latents
        )

        def log_prob_single(params: GaussianParams, x: Array) -> Array:
            lp: Array = get_normal_dist(params).log_prob(x.flatten())
            return lp

        # mc estimate of E_q[log p(x|z)] (mean across samples but sum across time)
        obs_log_prob = jax.vmap(log_prob_single)(generative_params, data.obs)
        obs_log_prob = obs_log_prob.mean(axis=-1).sum()

        # compute KL(q(z|x)||p(z)) in closed form
        kl = self.kl_posterior_prior(prior_params, post_params)

        # normalise by TD_x
        kl /= data.obs.size
        obs_log_prob /= data.obs.size

        r2 = R2_inferred_vs_actual_z(post_params.smoothed_means, data.targets)

        if self.config.run_diagnostics:
            results_dict = self.run_diagnostics(
                prior_params, conjugate_params, post_params
            )
        else:
            results_dict = {}

        results_dict["obs_ll"] = obs_log_prob
        results_dict["kl"] = kl
        results_dict["r2_post"] = r2

        return results_dict

    def run_diagnostics(
        self,
        prior_params: LDSParams,
        conjugate_params: GaussianParams,
        post_params: LGSSMSmoothedPosterior,
    ) -> dict[str, Array]:
        results_dict = {}
        Q, Q0 = prior_params.Q, prior_params.Q0
        results_dict["Q_min_eig"], results_dict["Q_max_eig"] = get_extreme_eigenvalues(
            Q
        )
        results_dict["Q0_min_eig"], results_dict["Q0_max_eig"] = (
            get_extreme_eigenvalues(Q0)
        )
        results_dict["conj_min_eig"], results_dict["conj_max_eig"] = jax.vmap(
            get_extreme_eigenvalues
        )(conjugate_params.full_cov)
        results_dict["smoothed_min_eig"], results_dict["smoothed_max_eig"] = jax.vmap(
            get_extreme_eigenvalues
        )(post_params.smoothed_covs)
        return results_dict

    def compute_loss(
        self,
        params: tuple[LDSParams, NetworkParams, NetworkParams],
        data_batch: InputData,
        beta: float,
        key: Array,
        training: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        keys = jr.split(key, len(data_batch))
        free_energy = jax.vmap(self.free_energy, (None, 0, 0, None))
        results_dict = free_energy(params, data_batch, keys, training)
        loss = (beta * results_dict["kl"] - results_dict["obs_ll"]).mean()
        for k, v in results_dict.items():
            if "min" in k:
                results_dict[k] = jnp.min(v)
            elif "max" in k:
                results_dict[k] = jnp.max(v)
            else:
                results_dict[k] = jnp.mean(v)
        results_dict["loss"] = loss
        return loss, results_dict

    def get_prior_means(
        self,
        init_state: Array,
        prior_params: LDSParams,
        T: int,
    ) -> Array:
        As = jnp.tile(prior_params.A[None], (T, 1, 1))
        bs = jnp.concatenate(
            [init_state[None], jnp.tile(prior_params.b[None], (T - 1, 1))]
        )

        @jax.vmap
        def recursion(
            elem1: tuple[Array, Array], elem2: tuple[Array, Array]
        ) -> tuple[Array, Array]:
            A1, b1 = elem1
            A2, b2 = elem2
            return A2 @ A1, A2 @ b1 + b2

        init_elems = (As, bs)
        means: Array = jax.lax.associative_scan(recursion, init_elems)[1]
        return means

    def generate_trajectory(
        self,
        params: tuple[LDSParams, NetworkParams, NetworkParams],
        prev_obs: Array,
        T_gen: int,
        sample: bool = False,
        key: Array | None = None,
    ) -> tuple[Array, Array]:
        prior_params, generation_net_params, recognition_net_params = params

        # phi(z|x) (vmap across time dimension)
        conjugate_params: GaussianParams = jax.vmap(self.recognition_net.apply, (None, 0))(  # type: ignore
            recognition_net_params, prev_obs
        )

        post = self.get_post(prior_params, conjugate_params)

        if sample:
            assert key is not None
            key, subkey = jr.split(key)
            latents = self.rollout(
                post.smoothed_means[-1],
                post.smoothed_covs[-1],
                prior_params,
                subkey,
                T_gen,
            )
        else:
            latents = self.get_prior_means(post.smoothed_means[-1], prior_params, T_gen)
        latents = latents[1:]  # remove init latent

        generative_params: GaussianParams = jax.vmap(self.generation_net.apply, (None, 0))(  # type: ignore
            generation_net_params, latents
        )

        gen_obs = generative_params.mean
        gen_obs = gen_obs.reshape(-1, *self.config.data.obs_shape)

        return gen_obs, latents

    def apply(
            self,
            params: tuple[LDSParams, NetworkParams, NetworkParams],
            obs: Array
    ) -> GaussianParams:
        prior_params, _, rec_params = params

        # phi(z|x) (vmap across time dimension)
        conjugate_params: GaussianParams = jax.vmap(self.recognition_net.apply, (None, 0))(  # type: ignore
            rec_params, obs
        )
        post = self.get_post(prior_params, conjugate_params)
        return post