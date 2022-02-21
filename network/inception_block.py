import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    bn_axis = 3

    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def InceptionBlock_s(x, inplanes, mix_id=0, channel_axis=3):

    branch1x1 = conv2d_bn(x, inplanes, 1, 1)

    branch5x5 = conv2d_bn(x, int(inplanes/4*3), 1, 1)
    branch5x5 = conv2d_bn(branch5x5, inplanes, 5, 5)

    branch3x3dbl = conv2d_bn(x, inplanes, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, int(inplanes*1.5), 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, int(inplanes*1.5), 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, inplanes//2, 1, 1)

    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed{}'.format(mix_id))
    
    return x


def InceptionBlock_m(x, inplanes, mix_id=0, channel_axis=3):

    branch1x1 = conv2d_bn(x, inplanes, 1, 1)

    branch7x7 = conv2d_bn(x, int(inplanes/4*3), 1, 1)
    branch7x7 = conv2d_bn(branch7x7, inplanes, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, inplanes, 7, 1)
    branch7x7 = conv2d_bn(branch7x7, inplanes, 3, 3)

    branch3x3dbl = conv2d_bn(x, inplanes, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, inplanes, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, inplanes, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, inplanes, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, inplanes, 1, 1)

    x = layers.concatenate(
        [branch1x1, branch7x7, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed{}'.format(mix_id))
    
    return x

