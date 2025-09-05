from dataclasses import asdict
from pathlib import Path

import jax
from jax import random as jr
from matplotlib import pyplot as plt

import wandb
from svae_control import network
from svae_control.config import Config
from svae_control.svae import SVAE
from svae_control.train import Trainer
from svae_control.utils.dataclass_utils import LDSParams
from svae_control.utils.setup_utils import get_args, get_beta_schedule, load_dataset

jax.config.update("jax_disable_jit", False)
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_check_tracer_leaks", False)

if __name__ == "__main__":
    args = get_args()
    config = Config.from_yaml(Path(args.config))

    wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        config=asdict(config),
    )

    key = jr.PRNGKey(config.training.seed)
    data_key, model_key, trainer_key = jr.split(key, 3)

    train_data, val_data = load_dataset(config, data_key)
    prior = LDSParams.init(config.model.latent_dim)
    enc = getattr(network, config.model.encoder.dist_type)(
        config.model.encoder, config.model.latent_dim
    )
    dec = getattr(network, config.model.decoder.dist_type)(
        config.model.decoder, config.data.obs_dim_flat
    )

    model = SVAE(
        generation_network=dec, recognition_network=enc, config=config, key=model_key
    )
    trainer = Trainer(
        model=model,
        config=config,
        prior_params=prior,
        beta_schedule=get_beta_schedule(config.training.beta),
        key=trainer_key,
        logger=wandb.log,
    )

    loss, r2, trained_params = trainer.train(train_data, val_data)

    outdir = Path("out")
    outdir.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(loss)
    ax[0].set_title("Loss")
    ax[1].plot(r2)
    ax[1].set_title("R2")
    plt.savefig(outdir / "loss_r2_jit.png")
