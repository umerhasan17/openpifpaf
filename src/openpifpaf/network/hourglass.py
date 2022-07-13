from torch import nn

Pool = nn.MaxPool2d


class convolution(nn.Module):
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


# class Conv(nn.Module):
#     def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
#         super(Conv, self).__init__()
#         self.inp_dim = inp_dim
#         self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
#         self.relu = None
#         self.bn = None
#         if relu:
#             self.relu = nn.ReLU()
#         if bn:
#             self.bn = nn.BatchNorm2d(out_dim)
#
#     def forward(self, x):
#         assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x


class residual(nn.Module):
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


# class Residual(nn.Module):
#     def __init__(self, inp_dim, out_dim):
#         super(Residual, self).__init__()
#         self.relu = nn.ReLU()
#         self.bn1 = nn.BatchNorm2d(inp_dim)
#         self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
#         self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
#         self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
#         self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
#         self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
#         self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
#         if inp_dim == out_dim:
#             self.need_skip = False
#         else:
#             self.need_skip = True
#
#     def forward(self, x):
#         if self.need_skip:
#             residual = self.skip_layer(x)
#         else:
#             residual = x
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn3(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#         out += residual
#         return out


class HourglassBlock(nn.Module):
    def __init__(self, n, f, bn=None, increases=None, use_conv=True):
        super(HourglassBlock, self).__init__()
        assert increases is not None
        assert len(increases) == n
        nf = f + increases[0]
        self.up1 = residual(f, f)
        # Lower branch
        if use_conv:
            self.extra = nn.Conv2d(f, f, (2, 2), padding=(0, 0), stride=(2, 2), bias=False)  # pool replacement
        else:
            self.extra = Pool(2, 2)
        self.low1 = residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = HourglassBlock(n - 1, nf, bn=bn, increases=increases[1:], use_conv=use_conv)
        else:
            self.low2 = residual(nf, nf)
        self.low3 = residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        extra1 = self.extra(x)
        low1 = self.low1(extra1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        # print(f'UP1 SHAPE: {up1.shape}, UP2 SHAPE: {up2.shape}')
        return up1 + up2
