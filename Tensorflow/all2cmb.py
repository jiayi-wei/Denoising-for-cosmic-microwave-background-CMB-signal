import numpy as np
import sys
sys.path.append('./tmp/')
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, merge, add, Lambda, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, MaxPooling2D,Conv2DTranspose
from keras.models import Model
from keras.layers.merge import Add
from keras.initializers import RandomNormal
from keras import optimizers
from keras import backend as K
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16
import functools
import net

input_dim = 1024
nb_epoch = 50000
batch_size = 16
lr = 1e-6
block = "block_4"
modelname = "all2cmb_3"
input_shape = (input_dim,input_dim,1)

all_path = "/home/sedlight/workspace/shl/data/train/img_data/band1_all_%s/"%(block)
noise_path = "/home/sedlight/workspace/shl/data/train/img_data/band1_noise_%s/"%(block)
cmb_path = "/home/sedlight/workspace/shl/data/train/img_data/band1_cmb_%s/"%(block)

val_all_path = "/home/sedlight/workspace/shl/data/test/img_data/band1_all_%s/"%(block)
val_noise_path = "/home/sedlight/workspace/shl/data/test/img_data/band1_noise_%s/"%(block)
val_cmb_path = "/home/sedlight/workspace/shl/data/test/img_data/band1_cmb_%s/"%(block)

modelpath="model/band1_%s_%s.h5"%(block, modelname)
modelpng = "model/%s.png" %(modelname)
train_file = "data/train/train.txt"
test_file = "data/test/test.txt"

def mse(y_true, y_pred):
	return K.mean(K.square(y_true - y_pred), axis=-1)
def my_mlse(y_true, y_pred):
	return K.mean(K.log(K.square(y_true - y_pred) + 1), axis=-1)
def generate_arrays_from_file(allpath,noisepath,cmbpath):
	allfiles = os.listdir(allpath)
	noisefiles = os.listdir(noisepath)
	cmbfiles = os.listdir(cmbpath)
	allfiles.sort()
	noisefiles.sort()
	cmbfiles.sort()
	num = len(allfiles)
	while 1:
		for i in xrange(num):
			#if allfiles[i].replace("all","noise") == noisefile[i]:
			all_data = np.load(allpath+allfiles[i])
			noise_data = np.load(noisepath+noisefiles[i])
			cmb_data = np.load(cmbpath+cmbfiles[i])
			all_data = np.stack([all_data],axis=0)
			noise_data = np.stack([noise_data],axis=0)
			cmb_data = np.stack([cmb_data],axis=0)
			yield (all_data,[noise_data,cmb_data])


def conv2d(x,filters,kernel_size,strides,activation='relu',name=None):
	kernel_init = RandomNormal(mean=0.0, stddev=0.05, seed=None)
	bias_init = RandomNormal(mean=0.0, stddev=0.05, seed=None)
	return Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,name=name,padding='same',activation=activation,kernel_initializer=kernel_init,bias_initializer=bias_init)(x)

def deconv2d(x,filters,kernel_size,strides,activation='relu',name=None):
	kernel_init = RandomNormal(mean=0.0, stddev=0.05, seed=None)
	bias_init = RandomNormal(mean=0.0, stddev=0.05, seed=None)
	return Conv2DTranspose(filters=filters,kernel_size=kernel_size,strides=strides,name=name,padding='same',activation=activation,kernel_initializer=kernel_init,bias_initializer=bias_init)(x)

def residual(x,filters,kernel_size,strides,activation='relu',name=None):
	conv1 = conv2d(x,filters,kernel_size,strides,activation=activation,name=None)
	conv2 = conv2d(conv1,filters,kernel_size,strides,activation=activation,name=None)
	add1 = add([conv1,conv2])
	return keras.layers.advanced_activations.LeakyReLU(alpha=0.3)(add1)

def predict(inputs,activation):

	conv1 = conv2d(inputs,32,11,4,activation)
	conv2 = conv2d(conv1,64,3,2,activation)
	conv3 = conv2d(conv2,256,3,2,activation)
	conv4 = residual(conv3,256,3,1,activation)
	add_conv1 = add([conv3,conv4])
	deconv1 = deconv2d(add_conv1,64,1,2,activation)
	add_conv2 = add([conv2,deconv1])
	deconv2 = deconv2d(add_conv2,32,3,2,activation)
	add_conv3 = add([conv1,deconv2])
	output = deconv2d(add_conv3,1,11,4,activation)
	return output

all_input = Input(shape=input_shape)
noise_pre = predict(all_input,"relu")
neg_noise_pre = Lambda(lambda x: -x)(noise_pre)
cmb_input = add([all_input,neg_noise_pre])
cmb_pre = predict(cmb_input,"relu")

model = Model(all_input,[noise_pre,cmb_pre])


print model.summary()
plot_model(model, show_shapes=True, to_file=modelpng)

adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.9999, epsilon=1e-10)
model.compile(optimizer=adam,loss=my_mlse,metrics=['mse'])

callback = keras.callbacks.ModelCheckpoint(modelpath, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.fit_generator(generate_arrays_from_file(all_path,noise_path,cmb_path),steps_per_epoch=batch_size, epochs=nb_epoch, verbose=2, callbacks=[callback],validation_data=generate_arrays_from_file(val_all_path,val_noise_path,val_cmb_path),validation_steps=batch_size)
