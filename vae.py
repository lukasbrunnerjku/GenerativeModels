import torch
from scipy.stats import truncnorm
import torch.nn as nn
import torchvision
from pytorch_lightning.utilities import cli
import pytorch_lightning as pl
from torch import nn
import torch

from utils.dataset import PokeDataModule
from utils.model import Encoder, Decoder

# https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed


class VAE(pl.LightningModule):
    def __init__(
        self,
        size: int = 32,
        lr: float = 0.0001,
        hidden_channels: int = 16,
        truncation_trick: bool = False,
        truncation_factor: float = 1.0,
        latent_dim: int = 100,
        enc_out_dim: int = 512,
        log_every_n_steps: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.encoder = Encoder(
            out_channels=enc_out_dim, hidden_channels=hidden_channels, size=size
        )
        self.decoder = Decoder(
            in_channels=enc_out_dim,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            size=size,
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.validation_z = self.sample_noise(batch_size=2)

    def forward(self, z) -> torch.Tensor:
        return torch.tanh(self.decoder(z))

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, _ = batch
        optimizer = self.optimizers()

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()  # B x hidden_dim

        # decoded
        x_hat = self(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # KL is non-negative and distributions are equal if KL is zero
        kl = self.kl_divergence(z, mu, std)

        # in its original formulation the ELBO is a lower bound
        # on the log-likelihood of the data
        # mathematically speaking: ELBO <= log(p(x))
        # therefore by increasing the ELBO we increase the log-likelihood,
        # in pytorch we take -ELBO as objective function we want to minimize
        elbo = kl - recon_loss
        elbo = elbo.mean()

        optimizer.zero_grad(set_to_none=True)
        elbo.backward()
        optimizer.step()

        if self.global_step % self.hparams.log_every_n_steps == 0:
            self.log_generated_images(
                "generated from random noise", x_hat, self.global_step
            )
            self.log_gradients(self.encoder)
            self.log_gradients(self.decoder)

        if self.current_epoch % int(self.hparams.log_every_n_steps / 2) == 0:
            self.generate_latent_walk()

        self.log_dict(
            {
                "elbo": elbo,
                "kl": kl.mean(),
                "recon_loss": recon_loss.mean(),
            }
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def log_gradients(self, net):
        cname = net.__class__.__name__ + "/Gradients"
        for name, module in net.named_modules():
            # name is ie. 'layer1.0.conv1'
            if isinstance(module, nn.Conv2d):
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight", module.weight.grad.data, self.global_step
                )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias", module.bias.grad.data, self.global_step
                    )

    def sample_noise(self, batch_size) -> torch.Tensor:
        if (
            self.hparams.truncation_trick
        ):  # avoid low density latent regions (especially a problem in small datasets)
            z = (
                torch.as_tensor(
                    truncnorm.rvs(-1, 1, size=(batch_size, self.hparams.latent_dim))
                )
                * self.hparams.truncation_factor
            )
        else:
            z = torch.randn(batch_size, self.hparams.latent_dim)
        z = z.to(next(self.decoder.parameters()))
        return z

    def postprocess(self, image):
        return (image * 0.5 + 0.5).clamp(0, 1)  # from [-1, 1] to [0, 1]

    def log_generated_images(
        self, tag: str, imgs: torch.Tensor, global_step: int
    ) -> None:
        grid = torchvision.utils.make_grid(self.postprocess(imgs))
        self.logger.experiment.add_image(tag, grid, global_step)

    def generate_latent_walk(self, steps: int = 10):
        z_start = self.sample_noise(batch_size=1)
        z_end = self.sample_noise(batch_size=1)
        alpha = torch.linspace(0.0, 1.0, steps=steps)[
            :, None, None, None
        ]  # steps x1x1x1
        z_intp = alpha * z_end + (1.0 - alpha) * z_start
        generated_images = self(z_intp)
        for i, img in enumerate(self.postprocess(generated_images)):
            self.logger.experiment.add_image("latent walk", img, i)

    def log_weights(self, net):
        cname = net.__class__.__name__
        for name, module in net.named_modules():
            # name is ie. 'layer1.0.conv1'
            if isinstance(module, nn.Conv2d):
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight", module.weight.data, self.current_epoch
                )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias", module.bias.data, self.current_epoch
                    )

    def on_epoch_end(self) -> None:
        z = self.validation_z.to(next(self.decoder.parameters()))
        imgs = self(z)
        self.log_generated_images(
            "generated from validation noise", imgs, self.current_epoch
        )
        self.log_weights(self.encoder)
        self.log_weights(self.decoder)


if __name__ == "__main__":
    pl.seed_everything(seed=42)
    trainer_defaults = dict(
        logger=pl.loggers.TensorBoardLogger(
            "lightning_logs", name="vae", default_hp_metric=False
        ),
    )
    cli.LightningCLI(
        model_class=VAE,
        datamodule_class=PokeDataModule,
        trainer_defaults=trainer_defaults,
    )
