"""
The hourglass module is able to process features across multiple scales and consolidate the results
to best capture the spatial relationships associated with keypoints for the task of pose estimation.

Code inspired by hourglass modules from other research repositories:

    The CornerNet model for object detection utilises an hourglass backbone.
    * https://github.com/princeton-vl/CornerNet/tree/master/models
    BSD 3-Clause License: Copyright (c) 2018, University of Michigan

    The original hourglass model written for the task of human pose estimation.
    * https://github.com/princeton-vl/pytorch_stacked_hourglass
    BSD 3-Clause License: Copyright (c) 2019, princeton-vl
"""

from torch import nn

Pool = nn.MaxPool2d


class convolution(nn.Module):
    """
    Standard convolution module that is a component of the hourglass module.
    """

    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (kernel_size, kernel_size), padding=(pad, pad), stride=(stride, stride),
                              bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim, eps=1e-5, momentum=0.1) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class residual(nn.Module):
    """
    The residual module is used for up-sampling and down-sampling convolutions within the hourglass.
    """

    def __init__(self, inp_dim, out_dim, stride=1):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim, eps=1e-5, momentum=0.1)

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim, eps=1e-5, momentum=0.1)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


class HourglassBlock(nn.Module):
    def __init__(self, n_hourglass_layers, inp_dim, bn=None, modules=None, increases=None, use_conv=True):
        super(HourglassBlock, self).__init__()
        assert increases is not None
        assert modules is not None
        assert len(increases) == n_hourglass_layers
        assert len(modules) == n_hourglass_layers + 1
        cur_mod = modules[0]
        nf = inp_dim + increases[0]

        self.up1 = self.make_residuals(inp_dim, inp_dim, cur_mod)
        # Lower branch
        if use_conv:
            self.extra = nn.Conv2d(inp_dim, inp_dim, (2, 2), padding=(0, 0), stride=(2, 2), bias=False)  # pool replacement
        else:
            self.extra = Pool(2, 2)
        self.low1 = self.make_residuals(inp_dim, nf, cur_mod)
        self.n_hourglass_layers = n_hourglass_layers
        # Recursive hourglass
        if self.n_hourglass_layers > 1:
            self.low2 = HourglassBlock(n_hourglass_layers - 1, nf, bn=bn, increases=increases[1:], modules=modules[1:],
                                       use_conv=use_conv)
        else:
            self.low2 = self.make_residuals(nf, nf, modules[1])
        self.low3 = self.make_residuals_revr(nf, inp_dim, cur_mod)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    @staticmethod
    def make_residuals(inp_dim, out_dim, num_modules):
        layers = [residual(inp_dim, out_dim)]
        for _ in range(num_modules - 1):
            layers.append(residual(out_dim, out_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def make_residuals_revr(inp_dim, out_dim, num_modules):
        layers = []
        for _ in range(num_modules - 1):
            layers.append(residual(inp_dim, inp_dim))
        layers.append(residual(inp_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        up1 = self.up1(x)
        extra1 = self.extra(x)
        low1 = self.low1(extra1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2
