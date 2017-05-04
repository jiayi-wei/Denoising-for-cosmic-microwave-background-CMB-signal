import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, add, UpSampling2D, MaxPooling2D
from keras.initializers import TruncatedNormal
from keras.layers.advanced_activations import LeakyReLU

def conv2d(x, output_filters, kernels, strides, activation='relu', name=None):
	kernel_init = TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
	bias_init = TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
	return Conv2D(filters=output_filters, kernel_size=kernels, strides=strides, 
		name=name, padding='same', activation=activation, 
		kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

def deconv2d(x, output_filters, kernels, strides, activation='relu', name=None):
	kernel_init = TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
	bias_init = TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
	return Conv2DTranspose(filters=output_filters, kernel_size=kernels, strides=strides,
		name=name, padding='same', activation=activation, 
		kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

def residual(x, output_filters, kernels, strides, activation='relu', name=None):
	conv1 = conv2d(x, output_filters, kernels, strides, activation, name + '_1')
	conv2 = conv2d(conv1, output_filters, kernels, strides, activation, name + '_2')
	add1 = add([conv1, conv2])
	return LeakyReLU(alpha=0.3)(add1)

def uppool(x, kernel, name=None):
	return UpSampling2D(x, size=2, name=name)

def pool(x, kernel, stride, name=None):
	return MaxPooling2D(x, pool_size=2, strides=2, padding='same', name=name)

def net(x, activation='relu', name=None):
	conv1 = conv2d(x, 32, 7, 2, activation, name + 'conv1')
	conv2 = conv2d(conv1, 64, 3, 2, activation, name + 'conv2')
	conv3 = conv2d(conv2, 256, 3, 2, activation, name + 'conv3')
	res1 = residual(conv3, 256, 3, 1, activation, name + 'res1')
	add1 = add([conv3, res1])
	deconv1 = deconv2d(add1, 64, 3, 2, activation, name + 'deconv1')
	add2 = add([conv2, deconv1])
	deconv2 = deconv2d(add2, 32, 3, 2, activation, name + 'deconv2')
	add3 = add([conv1, deconv2])
	deconv3 = deconv2d(add3, 1, 7, 2, activation, name + 'deconv3')
	return deconv3

def net_pool(x, activation='relu', name=None):
	conv1 = conv2d(x, 32, 5, 2, activation, name + 'conv1')
	pool1 = pool(conv1, 2, 2, name + 'pool1')
	conv2 = conv2d(pool1, 64, 3, 1, activation, name + 'conv2')
	pool2 = pool(conv2, 2, 2, name + 'pool2')
	conv3 = conv2d(pool2, 256, 3, 1, activation, name + 'conv3')
	res1 = residual(conv3, 256, 3, 1, activation, name + 'res1')
	add1 = add([conv3, res1])
	deconv1 = deconv2d(add1, 64, 3, 1, activation, name + 'deconv1')
	uppool1 = uppool(deconv1, 2, name + 'uppool1')
	add2 = add([pool2, uppool1])
	deconv2 = deconv2d(add2, 32, 3, 1, activation, name + 'deconv2')
	uppool2 = uppool(deconv2, 2, name + 'uppool2')
	add3 = add([pool1, uppool2])
	deconv3 = deconv2d(add3, 1, 5, 2, activation, name + 'deconv3')
	return deconv3
