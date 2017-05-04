from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

def conv2d(x, input_filters, output_filters, strides, kernel, name=None):
	with tf.variable_scope('conv'):

		w_shape = [kernel, kernel, input_filters, output_filters]
		weight = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.05), name = 'weight')
		strides = [1, strides, strides, 1]

		return tf.nn.relu(tf.nn.conv2d(x, weight, strides=strides, padding='SAME', name=name))

def deconv2d(x, input_filters, output_filters, strides, kernel, output_shape, name=None):
	with tf.variable_scope('deconv'):

		w_shape = [kernel, kernel, output_filters, input_filters]
		weight = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.05), name='weight')
		strides = [1, strides, strides, 1]

		return tf.nn.relu(tf.nn.conv2d_transpose(x, weight, output_shape, strides=strides, padding='SAME', name=name))

def residual(x, filters, strides, kernel, name=None):
	with tf.variable_scope('residual'):
		conv1 = conv2d(x, filters, filters, strides, kernel, name+'_1')
		conv2 = conv2d(conv1, filters, filters, strides, kernel, name+'_2')

		add_conv1 = tf.add(conv1, conv2)

		return tf.nn.relu(add_conv1)

def net(x):
	conv1 = conv2d(x, 1, 32, 4, 11, 'conv1')
	conv2 = conv2d(conv1, 32, 64, 2, 3, 'conv2')
	conv3 = conv2d(conv2, 64, 256, 2, 3, 'conv3')
	conv4 = residual(conv3, 256, 1, 3, 'res_conv')
	add1 = tf.add(conv3, conv4, name='add1')
	deconv1 = deconv2d(add1, 256, 64, 2, 3, conv2.get_shape(), 'deconv1')
	add2 = tf.add(deconv1, conv2, name='add2')
	deconv2 = deconv2d(add2, 64, 32, 2, 3, conv1.get_shape(), 'deconv2')
	add3 = tf.add(deconv2, conv1, name='add3')
	deconv3 = deconv2d(add3, 32, 1, 4, 11, x.get_shape(), 'deconv3')

	return deconv3