from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')

PRIMITIVES = [
    'none',
    'skip_connect',
    'sep_conv_3x3',
    #'sep_conv_5x5',
    'dil_conv_3x3',
    #'dil_conv_5x5',
    'sep_conv_3x3_spatial',
    #'sep_conv_5x5_spatial',
    'dil_conv_3x3_spatial',
    #'dil_conv_5x5_spatial',
    'SE',
    'SE_A_M',
    'CBAM'

]

Attention = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3_spatial', 0), ('sep_conv_3x3', 1), ('CBAM', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 3), ('dil_conv_3x3', 2)], normal_concat=range(1, 5))
DARTS = Attention


