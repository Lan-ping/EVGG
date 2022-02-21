
from network.rep_vgg_block import RepVGGBlock
from network.selayer import SELayer
from network.inception_block import InceptionBlock_s, InceptionBlock_m

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


def EVGG(input_shape=None, pooling='avg'):


    img_input = layers.Input(shape=input_shape)

    # Block 1
    x = RepVGGBlock(in_channels=3, out_channels=64,
                    kernel_size=3, stride=2, padding=1, deploy=False)(img_input)
    x = SELayer(64)(x)

    # Block 2
    x = InceptionBlock_s(x, 64, mix_id=1)
    x = SELayer(256)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)


    # Block 3
    x = InceptionBlock_s(x, 64, mix_id=2)
    x = SELayer(256)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = InceptionBlock_m(x, 128, mix_id=3)
    x = SELayer(512)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = InceptionBlock_m(x, 256, mix_id=4)
    x = SELayer(1024)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)


    inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='evgg')

    return model
