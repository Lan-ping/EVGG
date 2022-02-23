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

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups = 1):
    result = models.Sequential()
    result.add(layers.Conv2D(out_channels,kernel_size = kernel_size,strides=stride,padding="same",name="conv",use_bias=False))
    result.add(layers.BatchNormalization(name="bn"))
    return result

def RepVGGBlock(x, in_channels, out_channels, kernel_size, stride = 1, padding=0, groups = 1):
    rbr_identity = layers.BatchNormalization() if out_channels==in_channels and stride ==1 else None
    rbr_dense = conv_bn(in_channels = in_channels,out_channels=out_channels,kernel_size = kernel_size,stride = stride,padding = padding,groups = groups)
    padding_l1 = padding - kernel_size //2
    rbr_1x1=conv_bn(in_channels=in_channels,out_channels = out_channels,kernel_size = 1,stride = stride,padding=padding_l1,groups = groups)
    nonlinearity = layers.ReLU()
    id_out = rbr_identity(x) if rbr_identity is not None else 0
    dense_out = rbr_dense(x)
    x = nonlinearity(id_out + dense_out + rbr_1x1(x))
    return x

class SELayer(tf.keras.Model):

    def __init__(self,
                 inplanes,
                 kernel_initializer='glorot_normal'):
        super(SELayer, self).__init__()

        self.pool1 = layers.GlobalAveragePooling2D(data_format='channels_last')

        self.conv1 = layers.Conv2D(inplanes//4,
                                            kernel_size=[1, 1],
                                            strides=1,
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer)
        self.bn1 = layers.BatchNormalization(fused=True,
                                              momentum=0.997,
                                              epsilon=1e-5)
        self.conv2 = layers.Conv2D(inplanes,
                                            kernel_size=[1, 1],
                                            strides=1,
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer)

        self.attention_act = layers.Activation('relu')

    def call(self, inputs, training=False):

        se_pool = tf.expand_dims(self.pool1(inputs), axis=1)
        se_pool = tf.expand_dims(se_pool, axis=2)

        se_conv1 = self.conv1(se_pool)
        se_bn1 = self.bn1(se_conv1, training=training)
        se_conv2 = self.conv2(se_bn1)

        attention = self.attention_act(se_conv2)

        outputs = inputs*attention
        return outputs

def EVGG(input_shape=None, pooling='max', use_se = False):


    img_input = layers.Input(shape=input_shape)

    # Block 1
    x = RepVGGBlock(img_input, 3, 64, 3, stride = 1, padding=1)
    x = RepVGGBlock(x, 64, 64, 3, stride = 1, padding=1)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)


    # Block 2
    if use_se:
        x = SELayer(64)(x)
    branch1x1 = conv2d_bn(x, 32, 1, 1)

    branch7x7 = conv2d_bn(x, 16, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 32, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 32, 7, 1)

    branch7x7dbl = conv2d_bn(x, 16, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 32, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 32, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 32, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 32, 1, 7)

    branch_pool = layers.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                            axis=3,
                            name='mixed1')
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    if use_se:
        x = SELayer(128)(x)
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch7x7 = conv2d_bn(x, 48, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 64, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 64, 7, 1)

    branch7x7dbl = conv2d_bn(x, 48, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 64, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 64, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 64, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 64, 1, 7)

    branch_pool = layers.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                            axis=3,
                            name='mixed2')
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    if use_se:
        x = SELayer(256)(x)
    branch1x1 = conv2d_bn(x, 128, 1, 1)

    branch5x5 = conv2d_bn(x, 96, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 128, 5, 5)

    branch3x3dbl = conv2d_bn(x, 96, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 128, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 128, 3, 3)

    branch_pool = layers.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 128, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                            axis=3,
                            name='mixed4')
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    if use_se:
        x = SELayer(512)(x)
    branch3x3 = conv2d_bn(x, 256, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 224, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 256, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 256, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
                            axis=3,
                            name='mixed3')

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)
    inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='evgg')

    return model

if __name__ == '__main__':
    model = EVGG()
    model.summary()
