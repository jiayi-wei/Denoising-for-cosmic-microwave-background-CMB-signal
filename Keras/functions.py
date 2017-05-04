import numpy as np
from keras import backend as Kbackend
from keras.layers import Lambda, multiply
	
def loss_pixel(y_true, y_pred):
	reci_y_true = Lambda(lambda x: (1/x))(y_true)
	return Kbackend.mean(multiply([Kbackend.abs(y_true - y_pred), reci_y_true]), axis=-1)

def loss_log(y_true, y_pred):
	return Kbackend.mean(Kbackend.log(Kbackend.square(y_true - y_pred) + 1), axis=-1)

