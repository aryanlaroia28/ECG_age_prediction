import torch
import torch.nn as nn
from fds import FDS

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=False,
    )

def noop(x): 
    return x


# ---------------------------------------------------------------------
# Inception Block
# ---------------------------------------------------------------------

class InceptionBlock1d(nn.Module):
    def __init__(self, ni, nb_filters, kss, stride=1, bottleneck_size=32):
        super().__init__()

        self.bottleneck = (
            conv(ni, bottleneck_size, 1, stride)
            if bottleneck_size > 0 else noop
        )

        self.convs = nn.ModuleList([
            conv(
                bottleneck_size if bottleneck_size > 0 else ni,
                nb_filters,
                ks
            ) for ks in kss
        ])

        self.conv_bottle = nn.Sequential(
            nn.MaxPool1d(3, stride, padding=1),
            conv(ni, nb_filters, 1),
        )

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d((len(kss) + 1) * nb_filters),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x0 = self.bottleneck(x)
        out = torch.cat(
            [c(x0) for c in self.convs] + [self.conv_bottle(x)],
            dim=1
        )
        return self.bn_relu(out)


# ---------------------------------------------------------------------
# Residual Shortcut
# ---------------------------------------------------------------------

class Shortcut1d(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.conv = conv(ni, nf, 1)
        self.bn = nn.BatchNorm1d(nf)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, out):
        return self.relu(out + self.bn(self.conv(x)))


# ---------------------------------------------------------------------
# Inception Backbone
# ---------------------------------------------------------------------

class InceptionBackbone(nn.Module):
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual):
        super().__init__()
        assert depth % 3 == 0

        self.depth = depth
        self.use_residual = use_residual
        n_ks = len(kss) + 1

        self.blocks = nn.ModuleList([
            InceptionBlock1d(
                input_channels if i == 0 else n_ks * nb_filters,
                nb_filters,
                kss,
                bottleneck_size=bottleneck_size,
            )
            for i in range(depth)
        ])

        self.shortcuts = nn.ModuleList([
            Shortcut1d(
                input_channels if i == 0 else n_ks * nb_filters,
                n_ks * nb_filters,
            )
            for i in range(depth // 3)
        ])

    def forward(self, x):
        res = x
        for i in range(self.depth):
            x = self.blocks[i](x)
            if self.use_residual and i % 3 == 2:
                x = self.shortcuts[i // 3](res, x)
                res = x
        return x


# ---------------------------------------------------------------------
# Inception1D WITH FDS (ResNet1DWangFDS-compatible)
# ---------------------------------------------------------------------

class Inception1DFDS(nn.Module):
    def __init__(
        self,
        input_channels=1,
        num_classes=1,
        kernel_size=40,
        depth=6,
        bottleneck_size=32,
        nb_filters=32,
        use_residual=True,

        # -------- FDS params --------
        fds=False,
        bucket_num=None,
        bucket_start=None,
        start_update=None,
        start_smooth=None,
        kernel=None,
        ks=None,
        sigma=None,
        momentum=None,

        dropout=None,
    ):
        super().__init__()

        self.fds = fds
        self.start_smooth = start_smooth

        kss = [kernel_size, kernel_size // 2, kernel_size // 4]
        kss = [k - 1 if k % 2 == 0 else k for k in kss]

        # Backbone
        self.backbone = InceptionBackbone(
            input_channels=input_channels,
            kss=kss,
            depth=depth,
            bottleneck_size=bottleneck_size,
            nb_filters=nb_filters,
            use_residual=use_residual,
        )

        feat_dim = (len(kss) + 1) * nb_filters

        # Pool + Linear (same as ResNet1DWangFDS)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(feat_dim, num_classes)

        # FDS
        if self.fds:
            self.FDS = FDS(
                feature_dim=feat_dim,
                bucket_num=bucket_num,
                bucket_start=bucket_start,
                start_update=start_update,
                start_smooth=start_smooth,
                kernel=kernel,
                ks=ks,
                sigma=sigma,
                momentum=momentum,
            )

        # Dropout
        self.use_dropout = dropout is not None
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, targets=None, epoch=None):
        x = self.backbone(x)
        x = self.global_pool(x)

        encoding = x.squeeze(-1)      # (B, C)
        encoding_s = encoding

        if self.training and self.fds:
            if epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        if self.use_dropout:
            encoding_s = self.dropout(encoding_s)

        out = self.linear(encoding_s)

        if self.training and self.fds:
            return out, encoding
        else:
            return out


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def inception1d_fds(**kwargs):
    return Inception1DFDS(**kwargs)
