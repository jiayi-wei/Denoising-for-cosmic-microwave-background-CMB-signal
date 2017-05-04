from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import model_1
import data_process
import reader
import time
import os

os.environ['CUDA_VISIBLE_DEVICES']="1"

Batch_size = 4
Epoch = 1000
band = '1'
patch = '5'

def main():
	with tf.device('/gpu:0'):
		#real all signal
		all_signal = tf.placeholder(tf.float32, shape=[Batch_size, 1024, 1024 ,1])
		
		with tf.variable_scope('get_noise'):
			noise_signal_immitate = model_1.net(all_signal)
		
		#the real noise signal
		noise_signal_real = tf.placeholder(tf.float32, shape=[Batch_size, 1024, 1024, 1])

		#loss of noise part. Compute the square error of immitation_noise and real_noise
		loss_all = tf.reduce_mean(tf.abs(noise_signal_real - noise_signal_immitate))

		train_op = tf.train.AdamOptimizer(learning_rate = 0.000001).minimize(loss_all)

	saver = tf.train.Saver()
	#initialization
	init = tf.global_variables_initializer()

	#all_data contains cmd, all and noise file pathes 
	all_data = data_process.do_it(reader.load_data(band, patch))

	acount = all_data._numbers

	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.5

	with tf.Session(config=config) as sess:
		sess.run(init)
		total_batch = int(acount / Batch_size)
		iteration = 0
		print ("Train Begins!")
		start_time = time.time()
		for i in range(Epoch):
			for j in range(total_batch):
				#get a batch of data to feed
				batch_all, batch_noise, batch_cmd = all_data.next_batch(Batch_size)

				feed_dict = {all_signal: batch_all, noise_signal_real: batch_noise}
					
				#One iteration
				iteration += 1
				_, loss_iter = sess.run([train_op, loss_all], feed_dict=feed_dict)

				if iteration % 20 == 0:
					timer = time.strftime("%Y-%m-%d %H:%M:%S")
					print('Time is %s.'%(timer), 'At iteration :%d' % (iteration), 'loss at this time: {:.9f}'.format(loss_iter))
				if iteration % 200 == 0:
					print('\nEvery iter takes: {:.9f}\n'.format((time.time() - start_time) / float(200)))
					start_time = time.time()
				if iteration % 5000 == 0:
					saver.save(sess, 'model_tmp/field8_3_layer.ckpt', global_step=iteration)
		saver.save(sess, 'model_tmp/field8_3_layer.ckpt')
		print ("Train ends! And model has been save!")

if __name__ == '__main__':
	main()
