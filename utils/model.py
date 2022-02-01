import torch.nn as nn
from torch.nn import functional as F
import torch
import math


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


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, 1, 2, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            Interpolate(scale_factor=2),
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Sequential(
            Interpolate(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        x = self.upsample(x)

        out += x
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=512,
        hidden_channels=32,
        num_blocks=4,
        size=32,
    ):
        super().__init__()
        assert math.log2(size) >= num_blocks

        # overall spatial reduction by 2^num_blocks (default: 2^4 = 16)
        blocks = []
        blocks.append(EncoderBlock(in_channels, hidden_channels, 32))
        in_ch = 32
        for _ in range(num_blocks - 2):  # intermediate blocks
            out_ch = min(512, 2 * in_ch)  # restric network width
            blocks.append(EncoderBlock(in_ch, hidden_channels, out_ch))
            in_ch = out_ch
        blocks.append(EncoderBlock(in_ch, hidden_channels, out_channels))

        self.blocks = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=512,
        latent_dim=100,
        out_channels=3,
        hidden_channels=32,
        num_blocks=4,
        size=32,
    ):
        super().__init__()
        assert size >= 2 ** (num_blocks + 1)
        self.in_channels = in_channels
        self.linear = nn.Linear(latent_dim, 4 * in_channels)

        # overall spatial increase by 2^num_blocks (default: 2^4 = 16)
        blocks = []
        in_ch = in_channels
        for _ in range(num_blocks - 1):  # intermediate blocks
            out_ch = max(32, int(in_ch / 2))  # restrict network width
            blocks.append(DecoderBlock(in_ch, hidden_channels, out_ch))
            in_ch = out_ch
        blocks.append(DecoderBlock(in_ch, hidden_channels, out_channels))

        self.blocks = nn.Sequential(*blocks)
        self.interpolate = Interpolate(size=size)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), self.in_channels, 2, 2)
        x = self.blocks(x)
        x = self.interpolate(x)
        return x
