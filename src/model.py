import torch
import torch.nn as nn


def _safe_groups(channels):
    for g in [32, 16, 8, 4]:
        if channels % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.GroupNorm(_safe_groups(out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = _safe_groups(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.block(x)


class ConvVAE(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.z_dim = z_dim

        # Encoder
        self.enc_conv1 = ConvBlock(3, 64, downsample=True)   # 64 x 16 x 16
        self.enc_res1 = ResBlock(64)
        self.enc_conv2 = ConvBlock(64, 128, downsample=True)  # 128 x 8 x 8
        self.enc_res2 = ResBlock(128)
        self.enc_conv3 = ConvBlock(128, 256, downsample=True) # 256 x 4 x 4
        self.enc_res3 = ResBlock(256)

        self.enc_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(512, z_dim)
        self.logvar_head = nn.Linear(512, z_dim)

        # Decoder
        self.dec_fc = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256 * 4 * 4),
            nn.SiLU(),
        )

        self.dec_res1 = ResBlock(256)
        self.dec_up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv1 = ConvBlock(256, 128, downsample=False)  # 128 x 8 x 8

        self.dec_res2 = ResBlock(128)
        self.dec_up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv2 = ConvBlock(128, 64, downsample=False)   # 64 x 16 x 16

        self.dec_res3 = ResBlock(64)
        self.dec_up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv3 = ConvBlock(64, 64, downsample=False)    # 64 x 32 x 32

        self.dec_out = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv1(x)
        h = self.enc_res1(h)
        h = self.enc_conv2(h)
        h = self.enc_res2(h)
        h = self.enc_conv3(h)
        h = self.enc_res3(h)
        h = self.enc_fc(h)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, 256, 4, 4)
        h = self.dec_res1(h)
        h = self.dec_up1(h)
        h = self.dec_conv1(h)
        h = self.dec_res2(h)
        h = self.dec_up2(h)
        h = self.dec_conv2(h)
        h = self.dec_res3(h)
        h = self.dec_up3(h)
        h = self.dec_conv3(h)
        return self.dec_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
