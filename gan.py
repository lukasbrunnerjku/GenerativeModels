import numpy as np
import torch
from scipy.stats import truncnorm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.utilities import cli

from utils.dataset import PokeDataModule
from utils.augmentation import DiffAugment
from utils.model import Generator, Discriminator, SpectralNorm

# TODO:
# hinge loss!
# more depth!
# https://machinelearningmastery.com/a-gentle-introduction-to-the-biggan/
# https://arxiv.org/pdf/2002.02117v1.pdf  # smooth convs avoid checkerboard?


def weights_init(m):
    # usage: model.apply(weights_init)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m.weight.data.normal_(0.0, 0.02)


def make_trainable(net, val):
    for p in net.parameters():
        p.requires_grad_(val)


def noise_factory(
    noise_std: float = 0.2, gamma: float = 0.99, every_nth_call: int = 26
):
    decay = 1.0
    calls = 0

    def add_noise(imgs):
        nonlocal decay, calls
        calls += 1
        if calls % every_nth_call == 0:
            decay *= gamma
        return decay * noise_std * torch.randn_like(imgs) + imgs

    return add_noise


class GAN(pl.LightningModule):
    def __init__(
        self,
        size: int = 32,
        lr: float = 0.0002,
        lr_generator: float = 0.0002,
        lr_discriminator: float = 0.0002,
        weight_clip_val: float = 0.01,
        truncation_trick: bool = False,
        truncation_factor: float = 1.0,
        n_critic: int = 5,
        b1: float = 0.5,
        b2: float = 0.999,
        latent_dim: int = 100,
        spectral_norm: bool = False,
        use_interpolate: bool = False,
        wasserstein: bool = True,
        hidden_channels: int = 16,
        diffaug_probability: float = 1.0,
        log_every_n_steps: int = 50,
        gradient_clip_algorithm: str = "value",
        gradient_clip_val: float = 10.0,
        add_noise: bool = False,
        label_smoothing: bool = False,
        custom_weight_init: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator(
            in_channels=self.hparams.latent_dim,
            hidden_channels=hidden_channels,
            size=size,
            spectral_norm=spectral_norm,
            use_interpolate=use_interpolate,
        )
        self.discriminator = Discriminator(
            hidden_channels=hidden_channels,
            wasserstein=wasserstein,
            size=size,
            spectral_norm=spectral_norm,
            use_interpolate=use_interpolate,
        )

        if custom_weight_init and not spectral_norm:
            self.generator.apply(weights_init)
            self.discriminator.apply(weights_init)

        if not spectral_norm:
            self.clamp_parameters_to_cube()

        self.validation_z = self.sample_noise(batch_size=2)
        self.add_noise_real = noise_factory()
        self.add_noise_fake = noise_factory()

    def forward(self, z) -> torch.Tensor:
        return self.generator(z)

    def training_step(self, batch, batch_idx):

        if self.hparams.wasserstein:

            imgs, _ = batch
            opt_g, opt_d = self.optimizers()

            # train discriminator
            with torch.no_grad():
                z = self.sample_noise(imgs.size(0))
                generated_imgs = self(z)

            p = np.random.random()
            if p < self.hparams.diffaug_probability:
                imgs = DiffAugment(imgs, policy="color,translation,cutout")
                generated_imgs = DiffAugment(
                    generated_imgs, policy="color,translation,cutout"
                )

            if self.hparams.add_noise:
                imgs = self.add_noise_real(imgs)
                generated_imgs = self.add_noise_fake(generated_imgs)

            make_trainable(self.discriminator, True)
            opt_d.zero_grad(set_to_none=True)
            d_loss = -(
                self.discriminator(imgs).mean()
                - self.discriminator(generated_imgs).mean()
            )
            d_loss.backward()
            opt_d.step()
            if not self.hparams.spectral_norm:
                self.clamp_parameters_to_cube()
            self.log("d_loss", d_loss, True)

            if self.global_step % self.hparams.log_every_n_steps == 0:
                # for n, p in self.discriminator.named_parameters():
                #     print("name:", n, "has gradient:", p.grad is not None)
                self.log_gradients(self.discriminator)

            # train generator
            if self.global_step % self.hparams.n_critic == self.hparams.n_critic - 1:
                z = self.sample_noise(imgs.size(0))
                generated_imgs = self(z)

                p = np.random.random()
                if p < self.hparams.diffaug_probability:
                    generated_imgs = DiffAugment(
                        generated_imgs, policy="color,translation,cutout"
                    )

                make_trainable(self.discriminator, False)
                opt_g.zero_grad(set_to_none=True)
                g_loss = -self.discriminator(generated_imgs).mean()
                g_loss.backward()
                opt_g.step()
                self.log("g_loss", g_loss, True)

            if self.global_step % self.hparams.log_every_n_steps == 0:
                self.log_generated_images(
                    "generated from random noise", generated_imgs, self.global_step
                )
                if self.global_step >= self.hparams.n_critic - 1:
                    self.log_gradients(self.generator)

        else:
            imgs, _ = batch
            opt_g, opt_d = self.optimizers()

            # train discriminitator
            make_trainable(self.discriminator, True)

            z = self.sample_noise(imgs.size(0))
            generated_imgs = self(z)

            p = np.random.random()
            if p < self.hparams.diffaug_probability:
                imgs = DiffAugment(imgs, policy="color,translation,cutout")
                generated_imgs = DiffAugment(
                    generated_imgs, policy="color,translation,cutout"
                )

            if self.hparams.add_noise:
                imgs = self.add_noise_real(imgs)
                generated_imgs = self.add_noise_fake(generated_imgs)

            if self.hparams.label_smoothing:
                real = torch.empty(imgs.size(0))
                real.uniform_(0.7, 1.2)
            else:
                real = torch.ones(imgs.size(0))
            real = real.to(imgs)

            opt_d.zero_grad(set_to_none=True)
            real_loss = F.binary_cross_entropy(self.discriminator(imgs).view(-1), real)

            if self.hparams.label_smoothing:
                fake = torch.empty(imgs.size(0))
                fake.uniform_(0.0, 0.3)
            else:
                fake = torch.zeros(imgs.size(0))
            fake = fake.to(imgs)

            fake_loss = F.binary_cross_entropy(
                self.discriminator(generated_imgs.detach()).view(-1), fake
            )

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()
            self.log("d_loss", d_loss, True)

            # train generator
            if self.global_step % self.hparams.n_critic == self.hparams.n_critic - 1:
                make_trainable(self.discriminator, False)
                self.log_generated_images(
                    "generated from random noise", generated_imgs, self.global_step
                )

                opt_g.zero_grad(set_to_none=True)
                g_loss = F.binary_cross_entropy(
                    self.discriminator(generated_imgs).view(-1), real
                )
                g_loss.backward()
                opt_g.step()
                self.log("g_loss", g_loss, True)

        if self.current_epoch % int(self.hparams.log_every_n_steps / 2) == 0:
            self.generate_latent_walk()

    def configure_optimizers(self):
        if self.hparams.wasserstein:
            opt_g = torch.optim.RMSprop(
                self.generator.parameters(), lr=self.hparams.lr_generator
            )
            opt_d = torch.optim.RMSprop(
                self.discriminator.parameters(), lr=self.hparams.lr_discriminator
            )
        else:
            opt_g = torch.optim.Adam(
                self.generator.parameters(),
                lr=self.hparams.lr,
                betas=(self.hparams.b1, self.hparams.b2),
            )
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.hparams.lr,
                betas=(self.hparams.b1, self.hparams.b2),
            )
        return opt_g, opt_d

    def log_gradients(self, net):
        cname = net.__class__.__name__
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                name = f"{name}_Conv2d"
                if module.weight.grad is not None:  # can happen due to SpectralNorm
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_weight_grad",
                        module.weight.grad.data,
                        self.global_step,
                    )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias_grad",
                        module.bias.grad.data,
                        self.global_step,
                    )
            elif isinstance(module, nn.ConvTranspose2d):
                name = f"{name}_ConvTranspose2d"
                if module.weight.grad is not None:  # can happen due to SpectralNorm
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_weight_grad",
                        module.weight.grad.data,
                        self.global_step,
                    )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias_grad",
                        module.bias.grad.data,
                        self.global_step,
                    )
            elif isinstance(module, nn.LayerNorm):
                name = f"{name}_Layernorm"
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight_grad",
                    module.weight.grad.data,
                    self.global_step,
                )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias_grad",
                        module.bias.grad.data,
                        self.global_step,
                    )
            elif isinstance(module, SpectralNorm):
                name = f"{name}_SpectralNorm"
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight_grad",
                    module.module.weight_bar.grad.data,
                    self.global_step,
                )

    def sample_noise(self, batch_size) -> torch.Tensor:
        if (
            self.hparams.truncation_trick
        ):  # avoid low density latent regions (especially a problem in small datasets)
            z = (
                torch.as_tensor(
                    truncnorm.rvs(
                        -1, 1, size=(batch_size, self.hparams.latent_dim, 1, 1)
                    )
                )
                * self.hparams.truncation_factor
            )
        else:
            z = torch.randn(batch_size, self.hparams.latent_dim, 1, 1)
        z = z.to(next(self.generator.parameters()))
        return z

    def clamp_parameters_to_cube(self):
        for name, module in self.discriminator.named_modules():
            if isinstance(module, SpectralNorm):
                """
                (0): SpectralNorm(
                    (module): Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
                )
                """
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                for p in module.parameters():
                    p.data.clamp_(
                        -self.hparams.weight_clip_val, self.hparams.weight_clip_val
                    )

    def postprocess(self, image):
        return (image * 0.5 + 0.5).clamp(0, 1)  # from [-1, 1] to [0, 1]

    def log_generated_images(
        self, tag: str, imgs: torch.Tensor, global_step: int
    ) -> None:
        grid = torchvision.utils.make_grid(self.postprocess(imgs))
        self.logger.experiment.add_image(tag, grid, global_step)

    def manual_clip_gradients(self, net):
        if self.hparams.gradient_clip_algorithm == "value":
            torch.nn.utils.clip_grad_value_(
                net.parameters(), self.hparams.gradient_clip_val
            )
        else:  # 'norm'
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), self.hparams.gradient_clip_val
            )

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
            if isinstance(module, nn.Conv2d):
                name = f"{name}_Conv2d"
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight", module.weight.data, self.current_epoch
                )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias", module.bias.data, self.current_epoch
                    )
            elif isinstance(module, nn.ConvTranspose2d):
                name = f"{name}_ConvTranspose2d"
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight", module.weight.data, self.current_epoch
                )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias", module.bias.data, self.current_epoch
                    )
            elif isinstance(module, nn.LayerNorm):
                name = f"{name}_Layernorm"
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight", module.weight.data, self.current_epoch
                )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias", module.bias.data, self.current_epoch
                    )
            elif isinstance(module, SpectralNorm):
                name = f"{name}_SpectralNorm"
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight",
                    module.module.weight_bar.data,
                    self.current_epoch,
                )

    def log_graph(self):
        class GAN(nn.Sequential):
            pass

        z = self.validation_z.to(next(self.generator.parameters()))
        if self.current_epoch == 0:
            self.logger.experiment.add_graph(GAN(self.generator, self.discriminator), z)

    def on_epoch_end(self) -> None:
        z = self.validation_z.to(next(self.generator.parameters()))
        imgs = self(z)
        self.log_generated_images(
            "generated from validation noise", imgs, self.current_epoch
        )
        self.log_weights(self.generator)
        self.log_weights(self.discriminator)

    def on_fit_start(self) -> None:
        self.log_graph()


if __name__ == "__main__":
    pl.seed_everything(seed=42)
    trainer_defaults = dict(
        logger=pl.loggers.TensorBoardLogger(
            "lightning_logs", name="gan", default_hp_metric=False
        ),
    )
    cli.LightningCLI(
        model_class=GAN,
        datamodule_class=PokeDataModule,
        trainer_defaults=trainer_defaults,
    )
