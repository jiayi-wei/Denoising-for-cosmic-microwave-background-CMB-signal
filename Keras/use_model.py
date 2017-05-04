import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
'''
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)
'''
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
#from keras.initializations import uniform
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

input_dim = 1024
vmin = -400
vmax = 500

num = 1999188
block = "block_4"
mode = "cmb"
modelname = "all2cmb_fine_1"

data_path = '/home/sedlight/workspace/shl/data/test/img_data/'
all_file = "/home/sedlight/workspace/shl/data/test/img_data/band1_all_%s/band1_%d_all_%s.npy" %(block,num,block)
noise_file = "/home/sedlight/workspace/shl/data/test/img_data/band1_noise_%s/band1_%d_noise_%s.npy" %(block,num,block)
cmb_file = "/home/sedlight/workspace/shl/data/test/img_data/band1_cmb_%s/band1_%d_cmb_%s.npy" %(block,num,block)
png_file = "model/%s_%d.png" %(block,num)
model_file = 'model/%s/model.hs' %(modelname)


def model_use(model_file,all_file,noise_file,cmb_file,png_file,mode):
        	
	model = load_model(model_file)
	
	all_input = np.load(all_file)
	noise_input = np.load(noise_file)
	cmb_input = np.load(cmb_file)
	all_input = np.stack([all_input],axis=0)
	noise_input = np.stack([noise_input],axis=0)
	cmb_input = np.stack([cmb_input],axis=0)

	all_img = np.squeeze(all_input)
	noise_img = np.squeeze(noise_input)
	cmb_img = np.squeeze(cmb_input)

	pre = model.predict(all_input)
	pre_img = np.squeeze(pre)
	
	
	pre_error = pre_img-cmb_img if mode=="cmb" else pre_img-noise_img
	pre_mse = ((pre_error ** 2).mean(axis=None))
	print "mse between predict and ground-truth is ",str(pre_mse)
	print "mae between predict and ground-truth is ",str(np.absolute(pre_error).mean())

	print 'predict min_val is',str(np.absolute(pre_error).min())
	print 'predict max_val is',str(np.absolute(pre_error).max())


	plt.subplot(3, 3, 1)
	plt.title('all')
	plt.imshow(all_img, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)
	plt.colorbar()

	plt.subplot(3, 3, 2)
	plt.title('noise')
	plt.imshow(noise_img, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)
	plt.colorbar()

	plt.subplot(3, 3, 3)
	plt.title('cmb')
	plt.imshow(cmb_img, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)
	plt.colorbar()

	plt.subplot(3, 3, 4)
	plt.title('predict error')
	plt.imshow(pre_error, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)
	plt.colorbar()

	plt.subplot(3, 3, 5)
	plt.title('predict ' + mode)
	plt.imshow(pre_img, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)
	plt.colorbar()

	plt.subplot(3, 3, 6)
	plt.title('predict cmb')
	plt.imshow(all_img - pre_img, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)
	plt.colorbar()
	
	plt.subplot(3, 3, 7)
	plt.title('hist of pre_error')
	plt.hist(pre_error, bins=10,normed=True,range=(-30,20))
	#plt.imshow(all_img-noise_img, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)
	#plt.colorbar()

	plt.savefig(png_file, dpi='figure', bbox_inches='tight')

if __name__=="__main__":

	model_use(model_file,all_file,noise_file,cmb_file,png_file,mode)
	#model = load_model(model_file,custom_objects={'my_msle':my_msle})
	#print model.get_config()
