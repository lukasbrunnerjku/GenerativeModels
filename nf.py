import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2 as cv
import pytest
import pdb


class Permute(nn.Module):
    def __init__(self, channels, ndim: int = 0) -> None:
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
    # pdb.set_trace()
