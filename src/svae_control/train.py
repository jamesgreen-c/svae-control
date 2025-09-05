from typing import Any, Callable, Union

import jax
import numpy as np
import optax
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

import wandb
from svae_control.config import Config
from svae_control.svae import SVAE
from svae_control.utils.dataclass_utils import InputData, LDSParams
from svae_control.utils.setup_utils import (
    dummy_logger,
    get_encoder_dummy_input,
)
from svae_control.utils.utils import NetworkParams, clip_sv


class Trainer:
    opt_states: list[optax.OptState]
    opts: list[optax.GradientTransformation]
    itr: int

    def __init__(
        self,
        model: SVAE,
        prior_params: LDSParams,
        config: Config,
        key: Array,
        beta_schedule: Callable[[int], float],
        logger: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.key, get_net_key, rec_net_key = jr.split(key, 3)
        self.model = model
        generation_net_params = model.generation_net.init(
            get_net_key, jnp.ones((config.model.latent_dim,))
        )
        recognition_net_params = model.recognition_net.init(
            rec_net_key, get_encoder_dummy_input(config)
        )
        self.params = [prior_params, generation_net_params, recognition_net_params]
        self.opts = [
            getattr(optax, config.model.prior.optim.optim_type)(
                config.model.prior.optim.lr
            ),
            getattr(optax, config.model.decoder.optim.optim_type)(
                config.model.decoder.optim.lr
            ),
            getattr(optax, config.model.encoder.optim.optim_type)(
                config.model.encoder.optim.lr
            ),
        ]
        self.opt_states = [opt.init(p) for opt, p in zip(self.opts, self.params)]
        self.config = config
        self.logger = logger if logger is not None else dummy_logger
        self.itr = 0
        self.beta_schedule = beta_schedule

    def train_step(
        self,
        data: InputData,
        params: list[Union[LDSParams, NetworkParams]],
        opt_states: list[optax.OptState],
        key: Array,
    ) -> tuple[
        Array,
        dict[str, Array],
        list[Union[LDSParams, NetworkParams]],
        list[optax.OptState],
    ]:
        beta = self.beta_schedule(self.itr)
        loss: Array
        (loss, aux), grads = jax.value_and_grad(self.model.compute_loss, has_aux=True)(
            params, data, beta, key
        )

        new_params: list[Union[LDSParams, NetworkParams]] = []
        new_opt_states: list[optax.OptState] = []
        for param, grad, opt_state, opt in zip(params, grads, opt_states, self.opts):
            updates, new_opt_state = opt.update(grad, opt_state, param)
            new_param = optax.apply_updates(param, updates)
            new_params.append(new_param)
            new_opt_states.append(new_opt_state)

        return loss, aux, new_params, new_opt_states

    def val_step(
        self,
        data: InputData,
        params: tuple[LDSParams, NetworkParams, NetworkParams],
        key: Array,
    ) -> dict[str, Array]:
        _, results_dict = self.model.compute_loss(params, data, 1.0, key, False)
        results_dict = {"val_" + k: v for k, v in results_dict.items()}
        return results_dict

    def generate_val_trajectory(
        self,
        data: InputData,
        params: tuple[LDSParams, NetworkParams, NetworkParams],
    ) -> wandb.Image:
        traj = data[0]
        N = self.config.eval.plot_traj_len
        T = self.config.eval.plot_traj_context
        assert T + N <= len(traj)
        shape = self.config.data.obs_shape
        prev_data = traj[:T]
        sample = self.config.eval.plot_traj_sample
        if sample:
            self.key, subkey = jr.split(self.key)
        else:
            subkey = None
        gen_obs, gen_latents = self.model.generate_trajectory(
            params, prev_data.obs, N, sample, subkey
        )
        C = N // self.config.eval.plot_traj_num
        gen_obs = jnp.concat([o for i, o in enumerate(gen_obs) if i % C == 0], axis=1)
        true_obs = traj[T : T + N].obs.reshape(N, *shape)
        true_obs = jnp.concat([o for i, o in enumerate(true_obs) if i % C == 0], axis=1)
        img_array = jnp.concat((gen_obs, true_obs), axis=0)
        images = wandb.Image(
            np.array(img_array),
            caption=f"Top: Generated, Bottom: Ground Truth, Generated Length = {N}, Context Length = {T}, Plot Intervals = {C}",
        )
        return images

    def get_batch(self, data: InputData, key: Array) -> InputData:
        batch_indices = jr.randint(
            key, (self.config.training.batch_size,), 0, len(data)
        )
        batch = data[batch_indices]
        if self.config.training.random_starts:
            batch_starts = jr.randint(
                key, (self.config.training.batch_size,), 0, batch.T - self.config.data.T
            )
        else:
            batch_starts = jnp.zeros(self.config.training.batch_size, dtype=jnp.int32)
        batch = batch.get_time_slice(batch_starts, batch_starts + self.config.data.T)
        return batch

    def train(
        self, train_data: InputData, val_data: InputData
    ) -> tuple[list[Array], list[Array], list[Union[LDSParams, NetworkParams]]]:
        train_step = jax.jit(self.train_step) if self.config.jit else self.train_step
        val_step = jax.jit(self.val_step) if self.config.jit else self.val_step

        loss_tot: list[Array] = []
        r2_tot: list[Array] = []

        pbar = tqdm(range(self.config.training.num_iters))
        for self.itr in pbar:
            self.key, subkey1, subkey2 = jr.split(self.key, 3)
            data_batch = self.get_batch(train_data, subkey1)

            loss, results_dict, self.params, self.opt_states = train_step(
                data_batch, self.params, self.opt_states, subkey2
            )
            if self.itr % self.config.eval.eval_every == 0:
                starts = jnp.zeros(len(val_data), dtype=jnp.int32)
                val_batch = val_data.get_time_slice(starts, starts + self.config.data.T)
                val_results_dict = val_step(val_batch, tuple(self.params), self.key)
                if self.config.eval.plot_traj_len:
                    val_results_dict["generated_val_trajectory"] = (
                        self.generate_val_trajectory(val_batch, tuple(self.params))  # type: ignore
                    )

                results_dict.update(val_results_dict)

            self._stabilize_params()

            loss_tot.append(loss)
            r2_tot.append(results_dict["r2_post"])
            to_print = {
                f"{k}": f"{v:.3f}"
                for k, v in results_dict.items()
                if k
                in [
                    "loss",
                    "r2_post",
                    "kl",
                    "obs_ll",
                    "val_loss",
                    "val_r2_post",
                    "val_kl",
                    "val_obs_ll",
                ]
            }
            self.logger(results_dict)

            pbar.set_postfix(**to_print)

        return loss_tot, r2_tot, self.params

    def _stabilize_params(self) -> None:
        prior = self.params[0]  # LDSParams
        prior = prior.replace(
            A=clip_sv(prior.A, 1e-3),
            b=jnp.zeros_like(prior.b),  # dont learn control 
        )
        self.params[0] = prior 