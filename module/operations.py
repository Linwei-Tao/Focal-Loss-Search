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
    'iden1': lambda: Iden1(),
    'iden2': lambda: Iden2(),
    'one_plus': lambda: One_plus(),
    'one_minus': lambda: One_minus(),
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
        return x1 ** 2


class Sqrt(nn.Module):

    def __init__(self):
        super(Sqrt, self).__init__()
        self.arity = 1

    def forward(self, x1, x2):
        return torch.sign(x1) * torch.sqrt(torch.abs(x1) + EPS)


class Iden1(nn.Module):

    def __init__(self):
        super(Iden1, self).__init__()
        self.arity = 1

    def forward(self, x1, x2):
        return x1

class Iden2(nn.Module):

    def __init__(self):
        super(Iden2, self).__init__()
        self.arity = 1

    def forward(self, x1, x2):
        return x2


class One_plus(nn.Module):

    def __init__(self):
        super(One_plus, self).__init__()
        self.arity = 1

    def forward(self, x1, x2):
        return 1 + x1

class One_minus(nn.Module):

    def __init__(self):
        super(One_minus, self).__init__()
        self.arity = 1

    def forward(self, x1, x2):
        return 1 - x1