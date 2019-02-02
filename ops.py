
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))


def lrelu(inputs):
    return tf.nn.leaky_relu(inputs, alpha=0.2)


def relu(inputs):
    return tf.nn.relu(inputs)


def apply_bias(x, data_format, lr_mult=1.0):
    shape = int_shape(x)
    assert(len(shape) == 2 or len(shape) == 4)
    if len(shape) == 2:
        channels = shape[1]
    else:
        channels = shape[3] if data_format == 'NHWC' else shape[1]
    b = tf.get_variable(
        'bias', shape=[channels], initializer=tf.initializers.zeros())
    b = _lr_mult(lr_mult)(b)
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        if data_format == 'NHWC':
            return x + tf.reshape(b, [1, 1, 1, -1])
        else:
            return x + tf.reshape(b, [1, -1, 1, 1])


def _lr_mult(alpha):
    @tf.custom_gradient
    def _lr_mult(x):
        def grad(dy):
            return dy * alpha * tf.ones_like(x)
        return x, grad
    return _lr_mult


def dense(name, x, fmaps, data_format, has_bias=True, lr_mult=1.0):
    with tf.variable_scope(name):
        if len(x.shape) > 2:
            x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
        w = get_weight([x.shape[1].value, fmaps], gain=1.0, lr_mult=lr_mult)
        w = tf.cast(w, x.dtype)
        x = tf.matmul(x, w)
        if has_bias:
            x = apply_bias(x, data_format, lr_mult=lr_mult)
        return x


def get_weight(shape, gain=np.sqrt(2), fan_in=None, lr_mult=1.0):
    if fan_in is None:
        fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init
    wscale = tf.constant(np.float32(std), name='wscale')
    w = tf.get_variable('weight', shape=shape,
                        initializer=tf.random_normal_initializer())
    w = w * wscale
    w = _lr_mult(lr_mult)(w)
    return w


def conv2d(name, x, fmaps, kernel, data_format, has_bias=True, gain=np.sqrt(2), lr_mult=1.0):
    with tf.variable_scope(name):
        assert kernel >= 1 and kernel % 2 == 1
        w = get_weight([kernel, kernel, x.shape[3 if data_format ==
                                                'NHWC' else 1].value, fmaps], gain=gain, lr_mult=lr_mult)
        w = tf.cast(w, x.dtype)
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1],
                         padding='SAME', data_format=data_format)
        if has_bias:
            x = apply_bias(x, data_format, lr_mult=lr_mult)
        return x


def conv2d_down(name, x, fmaps, kernel, data_format, gain=np.sqrt(2), has_bias=True, lr_mult=1.0):
    with tf.variable_scope(name):
        assert kernel >= 1 and kernel % 2 == 1
        w = get_weight([kernel, kernel, x.shape[3 if data_format ==
                                                'NHWC' else 1].value, fmaps], gain=gain, lr_mult=lr_mult)
        w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
        w = tf.cast(w, x.dtype)
        x = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1] if data_format == 'NHWC' else [
                         1, 1, 2, 2], padding='SAME', data_format=data_format)
        if has_bias:
            x = apply_bias(x, data_format, lr_mult=lr_mult)
        return x


def conv2d_up(name, x, fmaps, kernel, data_format, gain=np.sqrt(2), has_bias=True, lr_mult=1.0):
    with tf.variable_scope(name):
        assert kernel >= 1 and kernel % 2 == 1
        c = x.shape[3 if data_format == 'NHWC' else 1].value
        w = get_weight([kernel, kernel, fmaps, c], gain=gain,
                       fan_in=(kernel**2)*c, lr_mult=lr_mult)
        w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
        w = tf.cast(w, x.dtype)
        if data_format == 'NHWC':
            os = [tf.shape(x)[0], x.shape[1] * 2, x.shape[2] * 2, fmaps]
        else:
            os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
        x = tf.nn.conv2d_transpose(x, w, os, strides=[1, 2, 2, 1] if data_format == 'NHWC' else [
                                   1, 1, 2, 2], padding='SAME', data_format=data_format)
        if has_bias:
            x = apply_bias(x, data_format, lr_mult=lr_mult)
        return x


def adaptive_instance_norm(name, x, z, data_format, epsilon=1e-5, lr_mult=1.0):
    with tf.variable_scope(name):
        batch_size = x.get_shape()[0]
        batch_size = int(batch_size)
        ch = x.get_shape()[3] if data_format == 'NHWC' else x.get_shape()[1]
        ch = int(ch)
        z = _lr_mult(lr_mult)(z)
        z = dense('fc', z, ch * 3, 'NC', lr_mult=lr_mult)
        if data_format == 'NHWC':
            z = tf.reshape(z, [batch_size, 1, 1, ch * 3])
            x_scale, x_offset, n_scale = tf.split(z, 3, axis=3)
        else:
            z = tf.reshape(z, [batch_size, ch * 3, 1, 1])
            x_scale, x_offset, n_scale = tf.split(z, 3, axis=1)
        x = x + n_scale * tf.random_normal(shape=tf.shape(x))
        mean, variance = tf.nn.moments(
            x, axes=[1, 2] if data_format == 'NHWC' else [2, 3], keep_dims=True)
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x-mean)*inv
        return x_scale*normalized + x_offset


def minibatch_stddev_layer(x, data_format, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])
        s = x.shape
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)
        y = tf.cast(y, x.dtype)
        if data_format == 'NHWC':
            y = tf.tile(y, [group_size, s[1], s[2], 1])
            return tf.concat([x, y], axis=3)
        else:
            y = tf.tile(y, [group_size, 1, s[2], s[3]])
            return tf.concat([x, y], axis=1)


def instance_norm(name, x, data_format, epsilon=1e-5, lr_mult=1.0):
    with tf.variable_scope(name):
        if data_format == 'NHWC':
            sh = [1, 1, 1, x.get_shape()[3]]
        else:
            sh = [1, x.get_shape()[1], 1, 1]
        scale = tf.get_variable("scale", sh, initializer=tf.random_normal_initializer(
            1.0, 0.01, dtype=tf.float32))
        scale = _lr_mult(lr_mult)(scale)
        noise_scale = tf.get_variable(
            "noise_scale", sh, initializer=tf.random_normal_initializer(1.0, 0.01, dtype=tf.float32))
        noise_scale = _lr_mult(lr_mult)(noise_scale)
        offset = tf.get_variable(
            "offset", sh, initializer=tf.constant_initializer(0.0))
        offset = _lr_mult(lr_mult)(offset)
        mean, variance = tf.nn.moments(
            x, axes=[1, 2] if data_format == 'NHWC' else [2, 3], keep_dims=True)
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x-mean)*inv
        return scale*normalized + noise_scale * tf.random_normal(shape=tf.shape(x)) + offset


def upscale2d(x, data_format, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    with tf.variable_scope('Upscale2D'):
        if data_format == 'NHWC':
            s = x.shape
            x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
            x = tf.tile(x, [1, 1, factor, 1, factor, 1])
            x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        else:
            s = x.shape
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x


def downscale2d(x, data_format, factor=2):
    with tf.variable_scope('Downscale2D'):
        assert isinstance(factor, int) and factor >= 1
        if factor == 1:
            return x
        if data_format == 'NHWC':
            ksize = [1, factor, factor, 1]
        else:
            ksize = [1, 1, factor, factor]
        x = tf.nn.avg_pool(x, ksize=ksize, strides=ksize,
                           padding='VALID', data_format=data_format)
        return x


def maxpool2d(x, data_format, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    if data_format == 'NHWC':
        ksize = [1, factor, factor, 1]
    else:
        ksize = [1, 1, factor, factor]
        x = tf.nn.max_pool(x, ksize=ksize, strides=ksize,
                           padding='SAME', data_format=data_format)
    return x
