import reader
import net
import functions

from keras.models import Model
from keras import optimizers
from keras.utils import plot_model
from keras import callbacks
from keras.layers import Lambda, Input, add
from keras import backend as K

import os
os.environ['CUDA_VISIBLE_DEVICES']="1"

model_name = 'all2noise_pool_1'
#model_name = 'all2noise_try_2'
model_path = 'model/%s/model.hs'%model_name
model_pic = 'model/%s/model.png'%model_name
model_log = 'model/%s/logs'%model_name

lr = 1e-4
batch_size = 16
epoch = 200000


def create_model_folder(model_name):
	if not os.path.exists('model/%s'%model_name):
	    os.makedirs('model/%s'%model_name)

def my_msle(y_true, y_pred):
	return K.mean(K.log(K.square(y_true - y_pred) + 1), axis=-1)

def main():
	create_model_folder(model_name)

	all_real = Input(shape=(1024, 1024, 1))
	noise_pre = net.net_pool(all_real, activation='relu', name='noise_')
	#neg_noise_pre = Lambda(lambda x: -x)(noise_pre)
	#cmb_fake = add([all_real, neg_noise_pre])
	#cmb_pre = net.net(cmb_fake, activation='relu', name='cmb_')

	#model = Model(all_real, [noise_pre, cmb_pre])
	model = Model(all_real, noise_pre)

	print model.summary()
	plot_model(model, show_shapes=True, to_file=model_pic)

	adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.9999, epsilon=1e-10)

	model.compile(optimizer=adam, loss='my_mlpse', metrics=['mse', 'mae'])

	callback = callbacks.ModelCheckpoint(model_path, monitor='val_loss', 
		verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)

	tensorBD = callbacks.TensorBoard(log_dir=model_log, histogram_freq=5)

	model.fit_generator(reader.get_data_noise(), steps_per_epoch=batch_size, epochs=epoch, 
		verbose=2, callbacks=[callback, tensorBD], validation_data=reader.test_data_noise(), 
		validation_steps=batch_size)

if __name__ == '__main__':
	main()
