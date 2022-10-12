import torch
import torch.nn as nn

EPS = 1e-6

OPS = {
    # LFS ops
    ## arity=2
    'add': lambda: Add(),
    'mul': lambda: Mul(),

    ## arity=1
    'neg': lambda: Neg(),
    'abs': lambda: Abs(),
    'inv': lambda: Inv(),
    'log': lambda: Log(),
    'exp': lambda: Exp(),
    'tanh': lambda: Tanh(),
    'square': lambda: Square(),
    'sqrt': lambda: Sqrt(),

    # NAS ops
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}


# LFS ops
class Add(nn.Module):

    def __init__(self):
        super(Add, self).__init__()
        self.arity = 2

    def forward(self, x1, x2):
        return x1 + x2


class Mul(nn.Module):

    def __init__(self):
        super(Mul, self).__init__()
        self.arity = 2

    def forward(self, x1, x2):
        return x1 * x2


class Neg(nn.Module):

    def __init__(self):
        super(Neg, self).__init__()
        self.arity = 1

    def forward(self, x1, x2):
        return -x1


class Abs(nn.Module):

    def __init__(self):
        super(Abs, self).__init__()
        self.arity = 1

    def forward(self, x1, x2):
        return torch.abs(x1)


class Inv(nn.Module):

    def __init__(self):
        super(Inv, self).__init__()
        self.arity = 1

    def forward(self, x1, x2):
        return 1 / (x1 + EPS)


class Log(nn.Module):

    def __init__(self):
        super(Log, self).__init__()
        self.arity = 1

    def forward(self, x1, x2):
        return torch.sign(x1) * torch.log(torch.abs(x1) + EPS)


class Exp(nn.Module):

    def __init__(self):
        super(Exp, self).__init__()
        self.arity = 1

    def forward(self, x1, x2):
        return torch.exp(x1)


class Tanh(nn.Module):

    def __init__(self):
        super(Tanh, self).__init__()
        self.arity = 1
        self.op = nn.Tanh()

    def forward(self, x1, x2):
        return self.op(x1)


class Square(nn.Module):

    def __init__(self):
        super(Square, self).__init__()
        self.arity = 1


    def forward(self, x1, x2):
        return x1**2

class Sqrt(nn.Module):

    def __init__(self):
        super(Sqrt, self).__init__()
        self.arity = 1


    def forward(self, x1, x2):
        return torch.sign(x1) * torch.sqrt(torch.abs(x1) + EPS)




# NAS ops
class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
