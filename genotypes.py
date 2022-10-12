from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

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

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1), ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0), ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1), ('skip_connect', 0),
        ('avg_pool_3x3', 0), ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1), ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1), ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1), ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1), ('sep_conv_5x5', 0),
        ('skip_connect', 3), ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2), ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0), ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0), ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1), ('skip_connect', 1),
        ('skip_connect', 0), ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0), ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0), ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0), ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0), ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0), ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[
        ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
        ('skip_connect', 0), ('sep_conv_3x3', 1),
        ('skip_connect', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('skip_connect', 2),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0), ('max_pool_3x3', 1),
        ('skip_connect', 2), ('max_pool_3x3', 0),
        ('max_pool_3x3', 0), ('skip_connect', 2),
        ('skip_connect', 2), ('avg_pool_3x3', 0),
    ],
    reduce_concat=[2, 3, 4, 5]
)

DARTS_V2 = Genotype(
    normal=[
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1), ('skip_connect', 0),
        ('skip_connect', 0), ('dil_conv_3x3', 2),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0), ('max_pool_3x3', 1),
        ('skip_connect', 2), ('max_pool_3x3', 1),
        ('max_pool_3x3', 0), ('skip_connect', 2),
        ('skip_connect', 2), ('max_pool_3x3', 1),
    ],
    reduce_concat=[2, 3, 4, 5]
)

DARTS = DARTS_V2

PC_DARTS_cifar = Genotype(
    normal=[
        ('sep_conv_3x3', 1), ('skip_connect', 0),
        ('sep_conv_3x3', 0), ('dil_conv_3x3', 1),
        ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
        ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_5x5', 1), ('max_pool_3x3', 0),
        ('sep_conv_5x5', 1), ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)
    ],
    reduce_concat=range(2, 6)
)

PC_DARTS_image = Genotype(
    normal=[
        ('skip_connect', 1), ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 0), ('skip_connect', 1),
        ('sep_conv_3x3', 1), ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_3x3', 0), ('skip_connect', 1),
        ('dil_conv_5x5', 2), ('max_pool_3x3', 1),
        ('sep_conv_3x3', 2), ('sep_conv_3x3', 1),
        ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)
    ],
    reduce_concat=range(2, 6)
)

PCDARTS = PC_DARTS_cifar

# FLOPs: 554.486410 M, Params: 3.505726 M
# CIFAR-10 = 97.30
PVLL_NAS = Genotype(
    normal=[
        ('sep_conv_5x5', 0), ('dil_conv_3x3', 1),
        ('sep_conv_3x3', 1), ('sep_conv_5x5', 0),
        ('avg_pool_3x3', 3), ('dil_conv_5x5', 1),
        ('sep_conv_3x3', 1), ('max_pool_3x3', 0)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_5x5', 1), ('max_pool_3x3', 0),
        ('max_pool_3x3', 1), ('dil_conv_3x3', 2),
        ('dil_conv_5x5', 2), ('sep_conv_5x5', 0),
        ('dil_conv_3x3', 1), ('skip_connect', 2)
    ],
    reduce_concat=range(2, 6)
)

arch_rl = Genotype(
    normal=[
        ('max_pool_3x3', 1), ('dil_conv_3x3', 1),
        ('dil_conv_5x5', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 3), ('avg_pool_3x3', 3),
        ('sep_conv_5x5', 3), ('sep_conv_5x5', 3)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_5x5', 0), ('dil_conv_5x5', 0),
        ('avg_pool_3x3', 2), ('avg_pool_3x3', 2),
        ('max_pool_3x3', 2), ('max_pool_3x3', 3),
        ('sep_conv_3x3', 2), ('max_pool_3x3', 2)
    ],
    reduce_concat=range(2, 6)
)

# 85.9560, searched with LSTM estimator
arch_lstm_0 = Genotype(
    normal=[
        ('dil_conv_5x5', 0), ('dil_conv_5x5', 1),
        ('dil_conv_5x5', 2), ('skip_connect', 0),
        ('avg_pool_3x3', 1), ('avg_pool_3x3', 3),
        ('skip_connect', 3), ('sep_conv_3x3', 4)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0), ('dil_conv_5x5', 1),
        ('skip_connect', 0), ('max_pool_3x3', 1),
        ('max_pool_3x3', 1), ('sep_conv_3x3', 0),
        ('dil_conv_3x3', 3), ('dil_conv_3x3', 1)
    ],
    reduce_concat=range(2, 6)
)

# searched with GAE estimator
# FLOPs: 464.409226 M, Params: 2.927926 M
# search  = 86.0080
# retrain = 97.1800
arch_gae_0 = Genotype(
    normal=[
        ('dil_conv_5x5', 1), ('dil_conv_3x3', 0),
        ('skip_connect', 1), ('sep_conv_3x3', 2),
        ('avg_pool_3x3', 2), ('sep_conv_5x5', 3),
        ('dil_conv_5x5', 2), ('avg_pool_3x3', 0)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_3x3', 1), ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1), ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 0), ('max_pool_3x3', 3),
        ('dil_conv_5x5', 4), ('sep_conv_3x3', 1)
    ],
    reduce_concat=range(2, 6)
)

# search-EXP-20210714-132905-gae-wo_diw, 2021-07-14 15:12:10,998
# gae, w/o diw
# FLOPs: 508.203658 M, Params: 3.228814 M
# search = 85.7440
# retrain = 97.2600
arch_gae_1 = Genotype(
    normal=[
        ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
        ('skip_connect', 0), ('avg_pool_3x3', 2),
        ('skip_connect', 0), ('sep_conv_5x5', 1),
        ('dil_conv_3x3', 1), ('dil_conv_5x5', 2)],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 1), ('dil_conv_5x5', 0),
        ('dil_conv_3x3', 1), ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2), ('dil_conv_3x3', 3),
        ('sep_conv_5x5', 4), ('skip_connect', 1)],
    reduce_concat=range(2, 6)
)

# search-EXP-20210714-232739-gae-with_diw, 2021-07-15 01:31:13,122
# gae, with diw
# FLOPs: 428.190346 M, Params: 2.732878 M
# search = 86.5200
# retrain = 96.7900
arch_gae_2 = Genotype(
    normal=[
        ('dil_conv_5x5', 0), ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0), ('dil_conv_5x5', 1),
        ('skip_connect', 1), ('dil_conv_3x3', 2),
        ('dil_conv_3x3', 1), ('skip_connect', 3)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('dil_conv_5x5', 0), ('sep_conv_5x5', 1),
        ('skip_connect', 1), ('dil_conv_3x3', 0),
        ('avg_pool_3x3', 0), ('dil_conv_3x3', 3),
        ('sep_conv_3x3', 0), ('skip_connect', 4)
    ],
    reduce_concat=range(2, 6)
)

# 427: search-EXP-20210714-232739-gae-with_diw.txt:2021-07-14 23:48:23,519
# Params: 3.763846, FLOPs: 597.34081
# retrain = 97.0000
arch_fs0 = Genotype(
    normal=[
        ('sep_conv_5x5', 0), ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_5x5', 1),
        ('skip_connect', 1), ('dil_conv_5x5', 0),
        ('sep_conv_3x3', 4), ('sep_conv_5x5', 2)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('dil_conv_3x3', 1), ('sep_conv_5x5', 0),
        ('skip_connect', 0), ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0), ('sep_conv_5x5', 1),
        ('max_pool_3x3', 3), ('sep_conv_5x5', 2)
    ],
    reduce_concat=range(2, 6)
)

# 429: search-EXP-20210714-232739-gae-with_diw.txt:2021-07-14 23:50:50,520
# Params: 3.763846, FLOPs: 597.34081
# retrain = 97.1000
arch_fs1 = Genotype(
    normal=[
        ('sep_conv_5x5', 0), ('max_pool_3x3', 1),
        ('sep_conv_5x5', 1), ('sep_conv_3x3', 0),
        ('dil_conv_5x5', 0), ('skip_connect', 1),
        ('sep_conv_3x3', 4), ('sep_conv_5x5', 2)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('dil_conv_3x3', 1), ('sep_conv_5x5', 0),
        ('skip_connect', 0), ('dil_conv_5x5', 2),
        ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
        ('max_pool_3x3', 3), ('sep_conv_5x5', 2)
    ],
    reduce_concat=range(2, 6)
)

# 428: search-EXP-20210714-232739-gae-with_diw.txt:2021-07-14 23:49:42,603
# Params: 3.732094, FLOPs: 593.995402
# retrain = 97.0900
arch_fs2 = Genotype(
    normal=[
        ('sep_conv_5x5', 0), ('max_pool_3x3', 1),
        ('sep_conv_5x5', 1), ('sep_conv_3x3', 0),
        ('dil_conv_5x5', 0), ('skip_connect', 1),
        ('sep_conv_3x3', 4), ('sep_conv_5x5', 2)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('dil_conv_3x3', 1), ('sep_conv_5x5', 0),
        ('skip_connect', 0), ('max_pool_3x3', 2),
        ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
        ('max_pool_3x3', 3), ('sep_conv_5x5', 2)
    ],
    reduce_concat=range(2, 6)
)

# 5: search-EXP-20210616-205435-lstm.txt:2021-06-16 23:16:58,672
# Params: 3.715678, FLOPs: 590.871178
arch_fs3 = Genotype(
    normal=[
        ('sep_conv_5x5', 1), ('skip_connect', 0),
        ('dil_conv_3x3', 2), ('dil_conv_5x5', 1),
        ('avg_pool_3x3', 3), ('sep_conv_5x5', 0),
        ('sep_conv_5x5', 3), ('sep_conv_5x5', 2)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 1), ('sep_conv_3x3', 0),
        ('dil_conv_5x5', 1), ('sep_conv_3x3', 0),
        ('skip_connect', 3), ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 3), ('sep_conv_5x5', 2)
    ],
    reduce_concat=range(2, 6)
)

# 180: search-EXP-20210617-032117-gae.txt:2021-06-17 05:34:02,901
# Params: 3.688462, FLOPs: 574.586506
arch_fs4 = Genotype(
    normal=[
        ('dil_conv_5x5', 1), ('dil_conv_5x5', 0),
        ('sep_conv_3x3', 2), ('dil_conv_3x3', 0),
        ('dil_conv_3x3', 2), ('sep_conv_3x3', 3),
        ('dil_conv_5x5', 2), ('sep_conv_3x3', 3)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('dil_conv_3x3', 1), ('skip_connect', 0),
        ('sep_conv_5x5', 2), ('sep_conv_5x5', 1),
        ('avg_pool_3x3', 0), ('skip_connect', 2),
        ('dil_conv_3x3', 0), ('sep_conv_3x3', 2)
    ],
    reduce_concat=range(2, 6)
)

# CIFAR-10: err=2.62
# ImageNet:
# flops: 507.022888M
# params: 4.618048M
arch_d0_0 = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_3x3', 1),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 3),
            ('sep_conv_3x3', 3), ('max_pool_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1),
            ('skip_connect', 1), ('max_pool_3x3', 2),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 2),
            ('dil_conv_5x5', 3), ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6))

# qsub -v arch_name="arch_d0_1" retrain-cifar.pbs
# CIFAR-10: acc=97.2900, err=2.71
arch_d0_1 = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_3x3', 1),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('max_pool_3x3', 1),
            ('sep_conv_5x5', 1), ('max_pool_3x3', 3)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1),
            ('skip_connect', 1), ('max_pool_3x3', 2),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 2),
            ('dil_conv_5x5', 4), ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6))

# CIFAR-10: err=2.50
# ImageNet:
# flops: 583.575784M
# params: 5.268448M
arch_d1_0 = Genotype(
    normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 2),
            ('skip_connect', 0), ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 3), ('dil_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('dil_conv_5x5', 1), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 0), ('skip_connect', 1),
            ('dil_conv_3x3', 1), ('sep_conv_5x5', 0),
            ('dil_conv_5x5', 1), ('dil_conv_5x5', 3)],
    reduce_concat=range(2, 6))

# qsub -v arch_name="arch_d1_1" retrain-cifar.pbs
# CIFAR-10: acc=97.4100, err=2.59
arch_d1_1 = Genotype(
    normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 2),
            ('sep_conv_5x5', 4), ('dil_conv_3x3', 2)],
    normal_concat=range(2, 6),
    reduce=[('dil_conv_5x5', 1), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 0), ('skip_connect', 1),
            ('dil_conv_3x3', 1), ('sep_conv_5x5', 0),
            ('dil_conv_5x5', 1), ('dil_conv_5x5', 3)],
    reduce_concat=range(2, 6))

arch_d1_1m = Genotype(
    normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 2),
            ('sep_conv_5x5', 4), ('dil_conv_3x3', 2)],
    normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_3x3', 0), ('skip_connect', 1),  # 1
        ('dil_conv_5x5', 1), ('dil_conv_5x5', 2),  # 3
        ('dil_conv_3x3', 1), ('sep_conv_5x5', 0),  # 2
        ('dil_conv_5x5', 1), ('max_pool_3x3', 0),  # 0
    ],
    reduce_concat=range(2, 6))

# search: source=73.4210 domain=0.5008 target=41.7460
# CIFAR-10: acc=97.40, err=2.60
arch_d2_0 = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 2), ('sep_conv_5x5', 0),
            ('max_pool_3x3', 0), ('skip_connect', 2),
            ('sep_conv_5x5', 1), ('dil_conv_3x3', 4)],
    normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('sep_conv_5x5', 1),
            ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('avg_pool_3x3', 2), ('avg_pool_3x3', 3),
            ('dil_conv_5x5', 4), ('dil_conv_5x5', 0)],
    reduce_concat=range(2, 6))
