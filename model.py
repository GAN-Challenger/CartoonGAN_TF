# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import InputLayer, Conv2d, BatchNormLayer, ElementwiseLayer, DeConv2d
from tensorlayer.layers import MaxPool2d, FlattenLayer, DenseLayer
import time


def generator(input, is_train=False, reuse=False):
    """
    Cartoon GAN generator neural network
    :param input: TF Tensor
        input tensor
    :param is_train: boolean
        train or test flag
    :param reuse: boolean
        whether to reuse the neural network
    :return:
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    gamma_init = tf.random_normal_initializer(1.0, stddev=0.02)

    with tf.variable_scope('CartoonGAN_G', reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        n = InputLayer(input, name='g_input')
        n = Conv2d(n, 64, (7, 7), (1, 1), act=None, padding='SAME', W_init=w_init, name='k7n64s1/c')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='k7n64s1/b_r')

        with tf.variable_scope('down_conv'):
            n = Conv2d(n, 128, (3, 3), (2, 2), padding='SAME', W_init=w_init, name='k3n128s2/c1')
            n = Conv2d(n, 128, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='k3n128s1/c2')
            n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='k3n128/b_r')

            n = Conv2d(n, 256, (3, 3), (2, 2), padding='SAME', W_init=w_init, name='k3n256s2/c1')
            n = Conv2d(n, 256, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='k3n256s1/cc')
            n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='k3n256/b_r')

        with tf.variable_scope('residual_blocks'):
            for i in range(8):
                nn = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                            name='k3n256s1/c1/%s' % i)
                nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                                    name='k3n256s1/b1/%s' % i)
                nn = Conv2d(nn, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                            name='k3n256s1/c2/%s' % i)
                nn = BatchNormLayer(nn, is_train=is_train, gamma_init=gamma_init, name='k3n256s1/b2/%s' % i)
                nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
                n = nn

        with tf.variable_scope('up_conv'):
            n = DeConv2d(n, n_filter=128, filter_size=(3, 3), out_size=(128, 128), strides=(2, 2), padding='SAME', W_init=w_init, name='k3n128s05/c1')
            n = Conv2d(n, 128, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='k3n128s1/c2')
            n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='k3n128/b_r')

            n = DeConv2d(n, n_filter=64, filter_size=(3, 3), out_size=(256, 256), strides=(2, 2), padding='SAME', W_init=w_init, name='k3n64s05/c1')
            n = Conv2d(n, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='k3n64s1/c2')
            n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='k3n64/b_r')

        n = Conv2d(n, 3, (7, 7), (1, 1), act=None, padding='SAME', W_init=w_init, name='g_output')

    return n


def discriminator(input, is_train=False, reuse=False):
    """
    Cartoon GAN discriminator neural network
    :param input: TF Tensor
        input tensor
    :param is_train: boolean
        train or test flag
    :param reuse: boolean
        whether to reuse the discriminator neural network
    :return:
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.0, stddev=0.02)
    leaky_relu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope('CartoonGAN_D', reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        n = InputLayer(input, name='d_input')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=leaky_relu, padding='SAME', W_init=w_init, name='block1/c')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=leaky_relu, padding='SAME', W_init=w_init, name='block2/c1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='block2/c2')
        n = BatchNormLayer(n, act=leaky_relu, is_train=is_train, gamma_init=gamma_init, name='block2/b')

        n = Conv2d(n, 128, (3, 3), (2, 2), act=leaky_relu, padding='SAME', W_init=w_init, name='block3/c1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='block3/c2')
        n = BatchNormLayer(n, act=leaky_relu, is_train=is_train, gamma_init=gamma_init, name='block3/b')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='block4/c')
        n = BatchNormLayer(n, act=leaky_relu, is_train=is_train, gamma_init=gamma_init, name='block4/b')

        n = Conv2d(n, 1, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='d_output')
        n = FlattenLayer(n)
        n = DenseLayer(n, n_units=1, name='d_output')
        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

    return n, logits, n.outputs


def vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    :param rgb:
        rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    :param reuse:
        whether to reuse the vgg network
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                            padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                            padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                            padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        conv4_4 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                            padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                            padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        # discriminator, swap the sigmoid activation function with the linear activation
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv4_4

