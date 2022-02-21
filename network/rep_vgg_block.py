import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import numpy as np


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = models.Sequential()
    result.add(layers.Conv2D(out_channels, kernel_size=kernel_size,
               strides=stride, padding="same", name="conv", use_bias=False))
    result.add(layers.BatchNormalization(name="bn"))
    return result


class RepVGGBlock(layers.Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):

        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1
        padding_l1 = padding - kernel_size // 2
        self.nonlinearity = layers.ReLU()

        if(deploy):
            self.rbr_reparam = layers.Conv2D(out_channels, kernel_size=kernel_size, strides=stride,
                                             padding=padding, dilation_rate=dilation, groups=groups, use_bias=True)
        else:

            self.rbr_identity = layers.BatchNormalization(
            ) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, stride=stride, padding=padding_l1, groups=groups)
            print("RepVGG Block, identity= ", self.rbr_identity)

    def call(self, inputs):
        if(hasattr(self, "rbr_reparam")):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if(self.rbr_identity is None):
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.rbr_dense(inputs)+self.rbr_1x1(inputs)+id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1)+kernelid, bias3x3+bias1x1+biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if(kernel1x1 is None):
            return 0
        else:
            return tf.pad(kernel1x1, [[1, 1], [1, 1]], mode='CONSTANT', constant_values=0)

    def _fuse_bn_tensor(self, branch):
        if(branch is None):
            return 0, 0
        if(isinstance(branch, models.Sequential)):  # 模型conv_bn的
            kernel = branch.get_layer("conv").get_weights()[
                0]  # 0 weights 1 bias
            bn_params = branch.get_layer("bn").get_weights()
            running_mean = bn_params[0]
            running_var = bn_params[1]
            gamma = bn_params[2]  # weight
            beta = bn_params[3]  # bias
        else:  # 直接过来的identity
            assert isinstance(branch, layers.BatchNormalization)
            if(not hasattr(self, 'id_tensor')):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (3, 3, self.in_channels, input_dim), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[1, 1, i, i % input_dim] = 1
                # torch.from_numpy(kernel_value).to(branch.weight.device)#TODO
                self.id_tensor = tf.convert_to_tensor(kernel_value)
            kernel = self.id_tensor
            bn_params = branch.get_weights()
            running_mean = bn_params[0]
            running_var = bn_params[1]
            gamma = bn_params[2]  # weight
            beta = bn_params[3]  # bias
        std = tf.math.sqrt(running_var + 10e-8)
        t = (gamma / std).reshape(1, 1, 1, -1)

        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.numpy(), bias.numpy()
