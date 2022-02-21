import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


def batch_norm():
    return layers.BatchNormalization(fused=True,
                                              momentum=0.997,
                                              epsilon=1e-5)


class HS(tf.keras.Model):

    def __init__(self):
        super(HS, self).__init__()

    def call(self, inputs):
        x = inputs*tf.nn.relu6(inputs+3.)/6.
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
        self.bn1 = batch_norm()
        self.conv2 = layers.Conv2D(inplanes,
                                            kernel_size=[1, 1],
                                            strides=1,
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer)

        self.attention_act = HS()

    def call(self, inputs, training=False):

        se_pool = tf.expand_dims(self.pool1(inputs), axis=1)
        se_pool = tf.expand_dims(se_pool, axis=2)

        se_conv1 = self.conv1(se_pool)
        se_bn1 = self.bn1(se_conv1, training=training)
        se_conv2 = self.conv2(se_bn1)

        attention = self.attention_act(se_conv2)

        outputs = inputs*attention
        return outputs
