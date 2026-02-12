import torch
import torch.nn as nn
from fds import FDS

"""



"""
# ---------------------------------------------------------------------
# Utility layers
# ---------------------------------------------------------------------

def conv1d(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=False
    )


# ---------------------------------------------------------------------
# Basic Residual Block (Wang-style)
# ---------------------------------------------------------------------

class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=(5, 3), downsample=None):
        super().__init__()

        self.conv1 = conv1d(inplanes, planes, kernel_size[0], stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv1d(planes, planes, kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# ---------------------------------------------------------------------
# ResNet1D Wang with FDS
# ---------------------------------------------------------------------

class ResNet1DWangFDS(nn.Module):

    def __init__(
        self,
        input_channels=1,
        num_classes=1,
        inplanes=128,
        kernel_size=(5, 3),

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

        self.inplanes = inplanes
        self.fds = fds
        self.start_smooth = start_smooth

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                inplanes,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.layer1 = self._make_layer(kernel_size)
        self.layer2 = self._make_layer(kernel_size)
        self.layer3 = self._make_layer(kernel_size)

        # Pool + head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(inplanes, num_classes)

        # FDS
        if self.fds:
            self.FDS = FDS(
                feature_dim=inplanes,
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

    def _make_layer(self, kernel_size):
        return BasicBlock1d(
            self.inplanes,
            self.inplanes,
            stride=1,
            kernel_size=kernel_size,
        )

    def forward(self, x, targets=None, epoch=None):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)
        encoding = x.squeeze(-1)          # (B, C)
        encoding_s = encoding

        # -------- FDS --------
        if self.training and self.fds:
            if epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        # -------- Dropout --------
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

def resnet1d_wang_fds(**kwargs):
    return ResNet1DWangFDS(**kwargs)
