from subprocess import call
import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2 as cv
import pytest
import pdb


class ActNorm(nn.Module):
    def __init__(
        self,
        channels: int,
        ndim: int,
        global_affine_init: float = 1.0,
        global_affine_type: str = "SOFTPLUS",
    ) -> None:
        super().__init__()

        if global_affine_type == "SIGMOID":
            global_scale = 2.0 - np.log(10.0 / global_affine_init - 1.0)
            self.global_scale_activation = lambda a: 10 * torch.sigmoid(a - 2.0)
        elif global_affine_type == "SOFTPLUS":
            global_scale = 2.0 * np.log(np.exp(0.5 * 10.0 * global_affine_init) - 1)
            self.softplus = nn.Softplus(beta=0.5)
            self.global_scale_activation = lambda a: 0.1 * self.softplus(a)
        elif global_affine_type == "EXP":
            global_scale = np.log(global_affine_init)
            self.global_scale_activation = lambda a: torch.exp(a)
        else:
            raise ValueError(
                'Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"'
            )

        self.global_scale = nn.Parameter(
            torch.ones(1, channels, *([1] * ndim)) * float(global_scale)
        )
        self.global_offset = nn.Parameter(torch.zeros(1, channels, *([1] * ndim)))
        self.ndim = ndim

    def forward(self, x, rev=False):
        s = self.global_scale_activation(self.global_scale)

        if rev:
            y = (x - self.global_offset) / s
            logdet = -torch.sum(
                torch.log(torch.abs(s)), dim=tuple(range(1, s.ndim))
            )  # B,
        else:
            y = s * x + self.global_offset
            logdet = torch.sum(
                torch.log(torch.abs(s)), dim=tuple(range(1, s.ndim))
            )  # B,

        return y, logdet


def test_norm(channels=8):
    actnorm = ActNorm(channels, 0)
    x = torch.randn(1, channels)
    y, logdet = actnorm(x)
    assert y.shape == x.shape
    assert logdet.shape == x.shape[:1]
    assert torch.allclose(actnorm(y, rev=True)[0], x)

    actnorm = ActNorm(channels, 1)
    x = torch.randn(1, channels, 5)
    y, logdet = actnorm(x)
    assert y.shape == x.shape
    assert logdet.shape == x.shape[:1]
    assert torch.allclose(actnorm(y, rev=True)[0], x)

    actnorm = ActNorm(channels, 2)
    x = torch.randn(1, channels, 5, 5)
    y, logdet = actnorm(x)
    assert y.shape == x.shape
    assert logdet.shape == x.shape[:1]
    assert torch.allclose(actnorm(y, rev=True)[0], x)

    actnorm = ActNorm(channels, 3)
    x = torch.randn(1, channels, 5, 5, 5)
    y, logdet = actnorm(x)
    assert y.shape == x.shape
    assert logdet.shape == x.shape[:1]
    assert torch.allclose(actnorm(y, rev=True)[0], x)


class Coupling(nn.Module):
    def __init__(
        self,
        channels: int,
        subnet_constructor: callable = None,
        clamp_activation: str = "ATAN",
        clamp: float = 2.0,
    ) -> None:
        super().__init__()

        assert channels % 2 == 0

        self.subnet1 = subnet_constructor(channels // 2, channels)
        self.subnet2 = subnet_constructor(channels // 2, channels)
        self.channels = channels
        self.clamp = clamp

        if clamp_activation == "ATAN":
            self.f_clamp = lambda u: 0.636 * torch.atan(u)
        elif clamp_activation == "TANH":
            self.f_clamp = torch.tanh
        elif clamp_activation == "SIGMOID":
            self.f_clamp = lambda u: 2.0 * (torch.sigmoid(u) - 0.5)
        elif clamp_activation == "NONE":
            self.f_clamp = lambda u: u
            self.clamp = 1.0
        else:
            raise ValueError(f'Unknown clamp activation "{clamp_activation}"')

    def forward(self, x, rev=False):
        if rev:
            y1, y2 = torch.split(x, self.channels // 2, dim=1)

            o2 = self.subnet2(y2)
            log_s2, b2 = o2[:, : self.channels // 2], o2[:, self.channels // 2 :]
            log_s2 = self.clamp * self.f_clamp(log_s2)
            x1 = (y1 - b2) * torch.exp(-log_s2)

            o1 = self.subnet1(x1)
            log_s1, b1 = o1[:, : self.channels // 2], o1[:, self.channels // 2 :]
            log_s1 = self.clamp * self.f_clamp(log_s1)
            x2 = (y2 - b1) * torch.exp(-log_s1)

            log_s = torch.cat((log_s2, log_s1), dim=1)
            logdet = -torch.sum(log_s, dim=tuple(range(1, log_s.ndim)))  # B,
            x = torch.cat((x1, x2), dim=1)

            return x, logdet

        else:
            x1, x2 = torch.split(x, self.channels // 2, dim=1)

            o1 = self.subnet1(x1)
            log_s1, b1 = o1[:, : self.channels // 2], o1[:, self.channels // 2 :]
            log_s1 = self.clamp * self.f_clamp(log_s1)
            y2 = torch.exp(log_s1) * x2 + b1

            o2 = self.subnet2(y2)
            log_s2, b2 = o2[:, : self.channels // 2], o2[:, self.channels // 2 :]
            log_s2 = self.clamp * self.f_clamp(log_s2)
            y1 = torch.exp(log_s2) * x1 + b2

            log_s = torch.cat((log_s2, log_s1), dim=1)
            logdet = torch.sum(log_s, dim=tuple(range(1, log_s.ndim)))  # B,
            y = torch.cat((y1, y2), dim=1)

            return y, logdet


def linear(in_channels, out_channels, hidden_channels=64):
    return nn.Sequential(
        nn.Linear(in_channels, hidden_channels, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_channels, out_channels, bias=True),
    )


def conv1d(in_channels, out_channels, hidden_channels=64):
    return nn.Sequential(
        nn.Conv1d(in_channels, hidden_channels, 3, 1, 1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv1d(hidden_channels, out_channels, 3, 1, 1, bias=True),
    )


def conv2d(in_channels, out_channels, hidden_channels=64):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, out_channels, 3, 1, 1, bias=True),
    )


def conv3d(in_channels, out_channels, hidden_channels=64):
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_channels, 3, 1, 1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv3d(hidden_channels, out_channels, 3, 1, 1, bias=True),
    )


def test_coupling(channels=8):

    couple = Coupling(channels, linear)
    x = torch.randn(1, channels)
    y, logdet = couple(x)
    assert y.shape == x.shape
    assert logdet.shape == x.shape[:1]
    assert torch.allclose(couple(y, rev=True)[0], x, atol=1e-7)

    couple = Coupling(channels, conv1d)
    x = torch.randn(1, channels, 5)
    y, logdet = couple(x)
    assert y.shape == x.shape
    assert logdet.shape == x.shape[:1]
    assert torch.allclose(couple(y, rev=True)[0], x, atol=1e-7)

    couple = Coupling(channels, conv2d)
    x = torch.randn(1, channels, 5, 5)
    y, logdet = couple(x)
    assert y.shape == x.shape
    assert logdet.shape == x.shape[:1]
    assert torch.allclose(couple(y, rev=True)[0], x, atol=1e-7)

    couple = Coupling(channels, conv3d)
    x = torch.randn(1, channels, 5, 5, 5)
    y, logdet = couple(x)
    assert y.shape == x.shape
    assert logdet.shape == x.shape[:1]
    assert torch.allclose(couple(y, rev=True)[0], x, atol=1e-7)


class Permute(nn.Module):
    def __init__(self, channels: int, ndim: int) -> None:
        super().__init__()

        w = np.zeros((channels, channels))
        for i, j in enumerate(np.random.permutation(channels)):
            w[i, j] = 1.0

        self.permute_function = {0: F.linear, 1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[
            ndim
        ]

        self.register_buffer(
            "w_perm", torch.FloatTensor(w.reshape(channels, channels, *[1] * ndim))
        )
        self.register_buffer(
            "w_perm_rev",
            torch.FloatTensor(w.T.reshape(channels, channels, *[1] * ndim)),
        )
        self.logdet = 0.0

    def forward(self, x, rev=False):
        if rev:
            return self.permute_function(x, self.w_perm_rev), self.logdet
        else:
            return self.permute_function(x, self.w_perm), self.logdet


def test_permute(channels=8):
    perm = Permute(channels, 0)
    x = torch.randn(1, channels)
    y, _ = perm(x)
    assert torch.allclose(perm(y, rev=True)[0], x)

    perm = Permute(channels, 1)
    x = torch.randn(1, channels, 5)
    y, _ = perm(x)
    assert torch.allclose(perm(y, rev=True)[0], x)

    perm = Permute(channels, 2)
    x = torch.randn(1, channels, 5, 5)
    y, _ = perm(x)
    assert torch.allclose(perm(y, rev=True)[0], x)

    perm = Permute(channels, 3)
    x = torch.randn(1, channels, 5, 5, 5)
    y, _ = perm(x)
    assert torch.allclose(perm(y, rev=True)[0], x)


def subnet_conv_1x1(in_channels, out_channels, hidden_channels=128):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding="same"),
        nn.ReLU(),
        nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding="same"),
    )


def subnet_conv_3x3(in_channels, out_channels, hidden_channels=128):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding="same"),
        nn.ReLU(),
        nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding="same"),
    )


class FlowSequential(nn.Sequential):
    def __init__(self, *args: nn.Module):
        super().__init__(*args)

    def forward(self, x, rev=False):
        sumlogdet = 0.0
        modules = reversed([m for m in iter(self)]) if rev else self
        for module in modules:
            x, logdet = module(x, rev)
            sumlogdet += logdet
        return x, sumlogdet


def test_flowsequential(channels=8):

    seq = FlowSequential(
        ActNorm(channels, 0),
        Permute(channels, 0),
        Coupling(channels, linear),
        ActNorm(channels, 0),
        Permute(channels, 0),
        Coupling(channels, linear),
    )
    x = torch.randn(1, channels)
    y, sumlogdet = seq(x)
    x_rev, sumlogdet_rev = seq(y, rev=True)
    assert torch.allclose(x_rev, x, atol=1e-7)
    assert torch.allclose(sumlogdet_rev, -sumlogdet, atol=1e-7)

    seq = FlowSequential(
        ActNorm(channels, 1),
        Permute(channels, 1),
        Coupling(channels, conv1d),
        ActNorm(channels, 1),
        Permute(channels, 1),
        Coupling(channels, conv1d),
    )
    x = torch.randn(1, channels, 5)
    y, sumlogdet = seq(x)
    x_rev, sumlogdet_rev = seq(y, rev=True)
    assert torch.allclose(x_rev, x, atol=1e-7)
    assert torch.allclose(sumlogdet_rev, -sumlogdet, atol=1e-7)

    seq = FlowSequential(
        ActNorm(channels, 2),
        Permute(channels, 2),
        Coupling(channels, conv2d),
        ActNorm(channels, 2),
        Permute(channels, 2),
        Coupling(channels, conv2d),
    )
    x = torch.randn(1, channels, 5, 5)
    y, sumlogdet = seq(x)
    x_rev, sumlogdet_rev = seq(y, rev=True)
    assert torch.allclose(x_rev, x, atol=1e-7)
    assert torch.allclose(sumlogdet_rev, -sumlogdet, atol=1e-7)

    seq = FlowSequential(
        ActNorm(channels, 3),
        Permute(channels, 3),
        Coupling(channels, conv3d),
        ActNorm(channels, 3),
        Permute(channels, 3),
        Coupling(channels, conv3d),
    )
    x = torch.randn(1, channels, 5, 5, 5)
    y, sumlogdet = seq(x)
    x_rev, sumlogdet_rev = seq(y, rev=True)
    assert torch.allclose(x_rev, x, atol=1e-7)
    assert torch.allclose(sumlogdet_rev, -sumlogdet, atol=1e-7)


class FastFlow(nn.Module):
    def __init__(
        self,
        channels: int,
        ndim: int,
        subnet_constructor1: callable,
        subnet_constructor2: callable,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()

        blocks = []
        for k in range(num_blocks):
            subnet_constructor = (
                subnet_constructor1 if (k % 2 == 0) else subnet_constructor2
            )
            blocks.extend(
                [
                    ActNorm(channels, ndim),
                    Permute(channels, ndim),
                    Coupling(channels, subnet_constructor),
                ]
            )
        self.blocks = FlowSequential(*blocks)

    def forward(self, x, rev=False):
        return self.blocks(x, rev)


def test_fastflow(channels=8, num_blocks=2):

    # TODO: numerical stability issues in Coupling ie. with 8 blocks won't pass test!

    fastflow = FastFlow(channels, 2, subnet_conv_3x3, subnet_conv_1x1, num_blocks)
    x = torch.randn(1, channels, 5, 5)
    y, sumlogdet = fastflow(x)
    x_rev, sumlogdet_rev = fastflow(y, rev=True)
    assert torch.allclose(x_rev, x, atol=1e-7)
    assert torch.allclose(sumlogdet_rev, -sumlogdet, atol=1e-7)


if __name__ == "__main__":
    test_permute(channels=8)
    test_coupling(channels=8)
    test_norm(channels=8)
    test_flowsequential(channels=8)
    test_fastflow(channels=8)
