from path import Path
from glob import glob
from typing import Optional, List, Callable
import numpy as np
import cv2 as cv
import torch
import math
from scipy.stats import truncnorm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.utilities import cli
from torch.utils.data import DataLoader, Dataset

# TODO: # https://machinelearningmastery.com/a-gentle-introduction-to-the-biggan/


def DiffAugment(x, policy="", channels_first=True):
    # https://arxiv.org/pdf/2006.10738.pdf
    # https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/DiffAugmentation.py
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(","):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (
        torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    ) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (
        torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    ) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(
        -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device
    )
    translation_y = torch.randint(
        -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = (
        x_pad.permute(0, 2, 3, 1)
        .contiguous()[grid_batch, grid_x, grid_y]
        .permute(0, 3, 1, 2)
    )
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(
        0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    offset_y = torch.randint(
        0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(
        grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1
    )
    grid_y = torch.clamp(
        grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1
    )
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


class PokeDataset(Dataset):
    # https://www.kaggle.com/kvpratama/pokemon-images-dataset

    def __init__(
        self,
        data_dir: str = ".",
        image_size: int = 32,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        data_dir = Path(data_dir) / "pokemon_jpg" / "*.jpg"
        self.files = glob(data_dir)
        self.images: Optional[List[torch.Tensor]] = None
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=image_size),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.transform = transform

    @property
    def preloaded(self) -> bool:
        return self.images is not None

    def imread(self, file) -> torch.Tensor:
        img: np.ndarray = cv.cvtColor(cv.imread(file), cv.COLOR_BGR2RGB)
        return self.preprocess(img)

    def preload(self) -> None:
        self.images = [self.imread(file) for file in self.files]

    def __getitem__(self, idx) -> torch.Tensor:
        if self.preloaded:
            img = self.images[idx]
        else:
            img = self.imread(self.files[idx])

        # add pseudo class "label"
        if self.transform is not None:
            img = self.transform(img)

        # we need to create additional data that does not serverly alter
        # the original distribution thus light rotations are added
        angle = np.random.uniform(-10, +10)
        img = TF.rotate(img, angle, fill=1.0)

        return img, 0

    def __len__(self):
        return len(self.files)


class PokeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = ".",
        batch_size: int = 64,
        num_workers: int = 0,
        size: int = 32,
        preload_data: bool = False,
        flip_probability: float = 0.0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.preload_data = preload_data
        self.flip_probability = flip_probability
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        transform = None
        if self.flip_probability > 0.0:
            transform = transforms.RandomVerticalFlip(p=self.flip_probability)
        self.dataset = PokeDataset(
            data_dir=self.data_dir, image_size=self.size, transform=transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )


class Generator(nn.Sequential):
    def __init__(
        self,
        in_channels=100,
        out_channels=3,
        hidden_channels=16,
        size=32,
        spectral_norm: bool = False,
    ):
        # 1x1 -> size x size
        assert size % 2 == 0

        sub_modules = []
        dim = fin = fout = None
        intermediate = int(math.log2(size) - 3)
        for i in reversed(range(intermediate + 1)):
            if i == intermediate:
                dim = 4
                fout = 2**i
                conv = nn.ConvTranspose2d(
                    in_channels, fout * hidden_channels, 4, 1, 0, bias=False
                )
                if spectral_norm:
                    conv = SpectralNorm(conv)
                sub_modules.extend(
                    [
                        conv,
                        nn.LayerNorm((fout * hidden_channels, dim, dim)),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
            else:
                dim *= 2
                fin = fout
                fout = 2**i
                conv = nn.ConvTranspose2d(
                    fin * hidden_channels, fout * hidden_channels, 4, 2, 1, bias=False
                )
                if spectral_norm:
                    conv = SpectralNorm(conv)
                sub_modules.extend(
                    [
                        conv,
                        nn.LayerNorm((fout * hidden_channels, dim, dim)),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )

        conv = nn.ConvTranspose2d(hidden_channels, out_channels, 4, 2, 1, bias=True)
        if spectral_norm:
            conv = SpectralNorm(conv)
        sub_modules.extend([conv, nn.Tanh()])
        super().__init__(*sub_modules)


class Discriminator(nn.Sequential):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        hidden_channels=16,
        size=32,
        wasserstein=False,
        spectral_norm: bool = False,
    ):
        # size x size -> 1x1
        assert size % 2 == 0

        sub_modules = []
        fin = fout = None
        dim = size
        intermediate = int(math.log2(size) - 3)
        for i in range(intermediate + 1):
            if i == 0:
                dim = int(dim / 2)
                fout = 2**i
                conv = nn.Conv2d(
                    in_channels, fout * hidden_channels, 4, 2, 1, bias=False
                )
                if spectral_norm:
                    conv = SpectralNorm(conv)
                sub_modules.extend(
                    [
                        conv,
                        nn.LayerNorm((fout * hidden_channels, dim, dim)),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
            else:
                dim = int(dim / 2)
                fin = fout
                fout = 2**i
                conv = nn.Conv2d(
                    fin * hidden_channels, fout * hidden_channels, 4, 2, 1, bias=False
                )
                if spectral_norm:
                    conv = SpectralNorm(conv)
                sub_modules.extend(
                    [
                        conv,
                        nn.LayerNorm((fout * hidden_channels, dim, dim)),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )

        assert dim == 4
        conv = nn.Conv2d(fout * hidden_channels, out_channels, 4, 1, 0, bias=True)
        if spectral_norm:
            conv = SpectralNorm(conv)
        sub_modules.extend([conv, nn.Identity() if wasserstein else nn.Sigmoid()])
        super().__init__(*sub_modules)


def weights_init(m):
    # usage: model.apply(weights_init)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m.weight.data.normal_(0.0, 0.008)


def make_trainable(net, val):
    for p in net.parameters():
        p.requires_grad_(val)


def noise_factory(
    noise_std: float = 0.4, gamma: float = 0.99, every_nth_call: int = 26
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


def l2normalize(v, eps=1e-4):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        _w = w.view(height, -1)
        for _ in range(self.power_iterations):
            v = l2normalize(torch.matmul(_w.t(), u))
            u = l2normalize(torch.matmul(_w, v))

        sigma = u.dot((_w).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


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
        )
        self.discriminator = Discriminator(
            hidden_channels=hidden_channels,
            wasserstein=wasserstein,
            size=size,
            spectral_norm=spectral_norm,
        )

        if custom_weight_init and not spectral_norm:
            self.generator.apply(weights_init)
            self.discriminator.apply(weights_init)

        if not spectral_norm:
            self.clamp_parameters_to_cube()

        self.validation_z = self.sample_noise(batch_size=8)
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
                # self.log_gradients(self.discriminator)
                # if self.global_step >= self.hparams.n_critic - 1:
                #     self.log_gradients(self.generator)

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
        cname = net.__class__.__name__ + "/Gradients"
        for name, module in net.named_modules():
            if isinstance(module, SpectralNorm):
                """
                (0): SpectralNorm(
                    (module): Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
                )
                """
                continue
            else:  # name of Conv2d or ConvTranspose2d -> ie. 0.module
                name = name.split(".module")[0]
            # name is ie. '8' = the position of that module in nn.Sequential
            if isinstance(module, nn.Conv2d):
                name = f"{int(name):02d}_Conv2d"
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight", module.weight.grad.data, self.global_step
                )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias", module.bias.grad.data, self.global_step
                    )
            elif isinstance(module, nn.ConvTranspose2d):
                name = f"{int(name):02d}_ConvTranspose2d"
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight", module.weight.grad.data, self.global_step
                )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias", module.bias.grad.data, self.global_step
                    )
            elif isinstance(module, nn.LayerNorm):
                name = f"{int(name):02d}_Layernorm"
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
        self, tag: str, imgs: torch.Tensor, global_step: int, max_imgs: int = 2
    ) -> None:
        grid = torchvision.utils.make_grid(self.postprocess(imgs[:max_imgs]))
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
        generated_images = self.generator(z_intp)
        for i, img in enumerate(self.postprocess(generated_images)):
            self.logger.experiment.add_image("latent walk", img, i)

    def log_weights(self, net):
        cname = net.__class__.__name__
        for name, module in net.named_modules():
            if isinstance(module, SpectralNorm):
                """
                (0): SpectralNorm(
                    (module): Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
                )
                """
                continue
            else:  # name of Conv2d or ConvTranspose2d -> ie. 0.module
                name = name.split(".module")[0]

            # name is ie. '8' = the position of that module in nn.Sequential
            if isinstance(module, nn.Conv2d):
                name = f"{int(name):02d}_Conv2d"
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight", module.weight.data, self.current_epoch
                )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias", module.bias.data, self.current_epoch
                    )
            elif isinstance(module, nn.ConvTranspose2d):
                name = f"{int(name):02d}_ConvTranspose2d"
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight", module.weight.data, self.current_epoch
                )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias", module.bias.data, self.current_epoch
                    )
            elif isinstance(module, nn.LayerNorm):
                name = f"{int(name):02d}_Layernorm"
                self.logger.experiment.add_histogram(
                    f"{cname}/{name}_weight", module.weight.data, self.current_epoch
                )
                if module.bias is not None:
                    self.logger.experiment.add_histogram(
                        f"{cname}/{name}_bias", module.bias.data, self.current_epoch
                    )

    def log_graph(self):
        z = self.validation_z.to(next(self.generator.parameters()))
        if self.current_epoch == 0:
            self.logger.experiment.add_graph(
                nn.Sequential(self.generator, self.discriminator), z
            )

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
        logger=pl.loggers.TensorBoardLogger("lightning_logs", default_hp_metric=False),
    )
    cli.LightningCLI(
        model_class=GAN,
        datamodule_class=PokeDataModule,
        trainer_defaults=trainer_defaults,
    )
