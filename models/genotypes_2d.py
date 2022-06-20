from collections import namedtuple

Genotype = namedtuple('Genotype_2D', 'cell cell_concat')

PRIMITIVES = [
    'skip_connect',
    'conv_3x3',
    'conv_5x5',
    'conv_7x7'
    ]



PRIMITIVES_Li = [
    'skip_connect',
    'depthwise-separable_conv_3x3',
    'depthwise-separable_conv_5x5',
    'atrous_conv_3x3',
    'atrous_conv_5x5',
    'average_pooling_3x3',
    'max_pooling_3x3'
    ]