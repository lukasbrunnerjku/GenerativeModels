import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2 as cv
import pytest
import pdb


class FastFlowCoupling(nn.Module):
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
        else:
            raise ValueError(f'Unknown clamp activation "{clamp_activation}"')

    def forward(self, x, rev=False):
        x1, x2 = torch.split(x, self.channels // 2, dim=1)

        if rev:
            pass
        else:
            o1 = self.subnet1(x1)
            s1, b1 = o1[:, : self.channels // 2], o1[:, self.channels // 2 :]
            s1 = torch.exp(self.clamp * self.f_clamp(s1))
            y2 = s1 * x2 + b1

            o2 = self.subnet2(y2)
            s2, b2 = o2[:, : self.channels // 2], o2[:, self.channels // 2 :]
            s2 = torch.exp(self.clamp * self.f_clamp(s2))
            y1 = s2 * x1 + b2

            s = torch.cat((s2, s1), dim=1)
            logdet = torch.sum(torch.log(s), dim=tuple(range(1, s.ndim)))  # B,
            y = torch.cat((y1, y2), dim=1)

        return y, logdet


def test_coupling(channels=8):
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

    couple = FastFlowCoupling(channels, linear)
    x = torch.randn(1, channels)
    y, logdet = couple(x)
    assert y.shape == x.shape
    assert logdet.shape == x.shape[:1]

    couple = FastFlowCoupling(channels, conv1d)
    x = torch.randn(1, channels, 5)
    y, logdet = couple(x)
    assert y.shape == x.shape
    assert logdet.shape == x.shape[:1]

    couple = FastFlowCoupling(channels, conv2d)
    x = torch.randn(1, channels, 5, 5)
    y, logdet = couple(x)
    assert y.shape == x.shape
    assert logdet.shape == x.shape[:1]

    couple = FastFlowCoupling(channels, conv3d)
    x = torch.randn(1, channels, 5, 5, 5)
    y, logdet = couple(x)
    assert y.shape == x.shape
    assert logdet.shape == x.shape[:1]


class Permute(nn.Module):
    def __init__(self, channels: int, ndim: int = 0) -> None:
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
    perm = Permute(channels, ndim=0)
    x = torch.randn(1, channels)
    y, _ = perm(x)
    assert torch.allclose(perm(y, rev=True)[0], x)

    perm = Permute(channels, ndim=1)
    x = torch.randn(1, channels, 5)
    y, _ = perm(x)
    assert torch.allclose(perm(y, rev=True)[0], x)

    perm = Permute(channels, ndim=2)
    x = torch.randn(1, channels, 5, 5)
    y, _ = perm(x)
    assert torch.allclose(perm(y, rev=True)[0], x)

    perm = Permute(channels, ndim=3)
    x = torch.randn(1, channels, 5, 5, 5)
    y, _ = perm(x)
    assert torch.allclose(perm(y, rev=True)[0], x)


if __name__ == "__main__":
    test_permute(channels=8)
    test_coupling(channels=8)

    # pdb.set_trace()
