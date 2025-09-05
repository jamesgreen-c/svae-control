from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Union

import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from jax import Array
from jax import random as jr

from svae_control.config import BetaConfig, Config
from svae_control.registry import BetaSchedule, NNType
from svae_control.utils.dataclass_utils import InputData

NetworkParams = Union[FrozenDict, dict[str, Any]]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="src/svae_control/yaml_configs/pendulum.yaml",
    )
    return parser.parse_args()


def dummy_logger(x: dict[str, Any]) -> None:
    return


def get_beta_schedule(cfg: BetaConfig) -> Callable[[int], float]:
    if BetaSchedule(cfg.beta_schedule) == BetaSchedule.CONSTANT:
        return lambda _: cfg.beta_start
    elif BetaSchedule(cfg.beta_schedule) == BetaSchedule.LINEAR:
        sched: Callable[[int], float] = optax.linear_schedule(
            cfg.beta_start, cfg.beta_end, cfg.beta_steps, cfg.beta_warmup
        )
        return sched
    else:
        raise NotImplementedError(f"beta schedule {cfg.beta_schedule} not implemented")


def load_dataset(config: Config, key: Array) -> tuple[InputData, InputData]:
    datadir = Path("data") / config.data.dataset
    obs = jnp.load(datadir / "obs.npy")
    targets = jnp.load(datadir / "targets.npy")
    if NNType(config.model.encoder.backbone.nn_type) != NNType.CNN:
        obs = obs.reshape(obs.shape[0], obs.shape[1], -1)
        assert config.data.obs_dim_flat == obs.shape[-1]
    else:
        assert (
            obs.ndim == 4 or obs.ndim == 5
        ), f"Cannot use CNN encoder on non-image data of shape {obs.shape}"
        if obs.ndim == 4:
            obs = obs[..., None]
        assert (
            tuple(config.data.obs_shape) == obs.shape[-3:]
        ), f"Obs shape mismatch: {obs.shape}, {config.data.obs_shape}"

    if config.data.normalise_pixels:
        obs = obs / 255.0
    if config.data.normalise_obs:
        obs = (obs - obs.mean()) / obs.std()
    data = InputData(obs=obs, targets=targets)
    N = obs.shape[0]
    train_num = int(config.data.train_frac * N)
    true_idxs = jnp.arange(train_num)
    train_idxs = jnp.zeros(N, dtype=bool).at[true_idxs].set(True)
    # train_idxs = jr.permutation(key, train_idxs)
    return data[train_idxs], data[~train_idxs]


def get_encoder_dummy_input(config: Config) -> jnp.ndarray:
    nn_type = NNType(config.model.encoder.backbone.nn_type)
    if nn_type == NNType.CNN:
        return jnp.ones(config.data.obs_shape)
    return jnp.ones((config.data.obs_dim_flat,))
