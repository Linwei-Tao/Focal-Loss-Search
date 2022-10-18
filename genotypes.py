from collections import namedtuple

PRIMITIVES = [
    'add',
    'mul',
    'neg',
    'abs',
    'inv',
    'log',
    'exp',
    'tanh',
    'square',
    'sqrt'
]

# -add(tanh(abs(1)), tanh(mul(inv(p_k), abs(1))))*log(p_k)