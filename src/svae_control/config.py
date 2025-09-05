from dataclasses import field
from pathlib import Path
from typing import Sequence, Type, get_args

import yaml
from dacite import from_dict
from flax.struct import dataclass
from jax import numpy as jnp


from svae_control.registry import BaseEnum, BetaSchedule, Dataset, NNDistType, NNType


def check_enum(val: str, enum: Type[BaseEnum]) -> None:
    assert val in enum.members(), f"Unknown type {val} not in {get_args(enum)}"


@dataclass
class OptimiserConfig:
    lr: float
    optim_type: str
    lr_schedule: str = "constant"
    weight_decay: float = 0.0


@dataclass
class NNConfig:
    nn_type: str
    # for CNN this sequence denotes the number of convolutional filters per layer
    hidden_dims: Sequence[int] = field(default_factory=lambda: [256, 256])
    nn_args: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_enum(self.nn_type, NNType)


@dataclass
class NNDistConfig:
    dist_type: str
    backbone: NNConfig
    head: NNConfig
    optim: OptimiserConfig
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_enum(self.dist_type, NNDistType)


@dataclass
class LDSPriorConfig:
    optim: OptimiserConfig
    init_dynamics_noise_scale: float = 0.1


@dataclass
class DataConfig:
    dataset: str
    obs_shape: Sequence[int]
    T: int
    normalise_obs: bool = True
    normalise_pixels: bool = True
    train_frac: float = 0.8

    # def __post_init__(self) -> None:
    #     check_enum(self.dataset, Dataset)

    @property
    def obs_dim_flat(self) -> int:
        return int(jnp.prod(jnp.array(self.obs_shape)))


@dataclass
class BetaConfig:
    beta_schedule: str = "constant"
    beta_start: float = 1.0
    beta_end: float = 1.0
    beta_steps: int = 1000
    beta_warmup: int = 0

    def __post_init__(self) -> None:
        check_enum(self.beta_schedule, BetaSchedule)


@dataclass
class ModelConfig:
    latent_dim: int
    encoder: NNDistConfig
    decoder: NNDistConfig
    prior: LDSPriorConfig


@dataclass
class EvalConfig:
    eval_every: int = 100
    plot_traj_len: int = 20
    plot_traj_context: int = 20
    plot_traj_num: int = 10
    plot_traj_sample: bool = False


@dataclass
class TrainingConfig:
    num_samples: int = 1

    batch_size: int = 128
    seed: int = 0

    mask_size: int = 0
    num_masks: int = 0
    uninformative_potential: float = 1e5

    num_iters: int = 50000
    random_starts: bool = False
    beta: BetaConfig = field(default_factory=BetaConfig)


@dataclass
class WandBConfig:
    use_wandb: bool = False
    mode: str = "online"
    project: str = "svae_control"
    entity: str = "gatsby-sahani"


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    eval: EvalConfig
    wandb: WandBConfig
    jit: bool = True
    run_diagnostics: bool = False

    @staticmethod
    def from_yaml(path: Path) -> "Config":
        with path.open("r") as f:
            config = yaml.safe_load(f)
        return from_dict(data_class=Config, data=config)
