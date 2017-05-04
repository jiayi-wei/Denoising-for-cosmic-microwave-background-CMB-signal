import reader

from keras.models import load_model
from keras import optimizers
from keras.utils import plot_model
from keras import callbacks
from keras import backend as K

import os
os.environ['CUDA_VISIBLE_DEVICES']="1"

model_name = 'all2cmb_fine_1'
load_path = 'model/all2cmb_fix_noise/model.hs'

model_path = 'model/%s/model.hs'%model_name
model_pic = 'model/%s/model.png'%model_name
model_log = 'model/%s/logs'%model_name

lr = 1e-5
batch_size = 16
epoch = 200000

def create_model_folder(model_name):
	if not os.path.exists('model/%s'%model_name):
	    os.makedirs('model/%s'%model_name)

def main():
	create_model_folder(model_name)
	model = load_model(load_path)

	for f in model.layers:
		f.trainable = True

	print model.summary()
	plot_model(model, show_shapes=True, to_file=model_pic)

	adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.9999, epsilon=1e-10)

	model.compile(optimizer=adam, loss='mlpse', metrics=['mae', 'mse'])

	callback = callbacks.ModelCheckpoint(model_path, monitor='val_loss', 
		verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)

	tensorBD = callbacks.TensorBoard(log_dir=model_log, histogram_freq=5)

	model.fit_generator(reader.get_data(), steps_per_epoch=batch_size, epochs=epoch, 
		verbose=2, callbacks=[callback, tensorBD], validation_data=reader.test_data(), 
		validation_steps=batch_size)

if __name__ == '__main__':
	main()
