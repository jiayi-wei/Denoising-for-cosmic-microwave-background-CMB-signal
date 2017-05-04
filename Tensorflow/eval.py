import tensorflow as tf
import numpy as np
import model_new
import data_process
import reader
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES']="1"

Batch_size = 4
Epoch = 1000
band = '1'
patch = '5'
lr = 1e-6

vmin = -500
vmax = 500

def main():
	with tf.device('/gpu:0'):
		#real all signal
		all_signal = tf.placeholder(tf.float32, shape=[Batch_size, 1024, 1024 ,1])
		with tf.variable_scope('get_noise'):
			noise_signal_immitate = model_new.net(all_signal)
		
		#the real noise signal
		noise_signal_real = tf.placeholder(tf.float32, shape=[Batch_size, 1024, 1024, 1])
		
		#loss of noise part. Compute the square error of immitation_noise and real_noise
		loss_all = tf.reduce_mean(tf.pow(noise_signal_real - noise_signal_immitate, 2))

		train_op = tf.train.AdamOptimizer(learning_rate = lr, beta1=0.9, beta2=0.9999, epsilon=1e-10).minimize(loss_all)

	saver = tf.train.Saver()
	#initialization
	init = tf.global_variables_initializer()

	#all_data contains cmd, all and noise file pathes 
	all_data = data_process.do_it(reader.load_data(band, patch))

	with tf.Session() as sess:
		all_, noise_, cmb_ = all_data.next_batch(Batch_size)
			
		feed_dict = {all_signal: all_, noise_signal_real: noise_}
		saver.restore(sess, 'model_tmp/new_model.ckpt')
		noise_mm = sess.run(noise_signal_immitate, feed_dict=feed_dict)
		cmap = matplotlib.cm.jet

		for i in range(4):
			all_img = np.squeeze(all_[i])
			ax = plt.subplot(231)
			ax.set_title('all image')
			ax.imshow(all_img,cmap=cmap)
			#ax.colorbar()
			
			noise_img = np.squeeze(noise_[i])
			bx = plt.subplot(232)
			bx.set_title('noise image')
			bx.imshow(noise_img,cmap=cmap)
			#bx.colorbar()

			cmb_img = np.squeeze(cmb_[i])
			cx = plt.subplot(233)
			cx.set_title('cmb image')
			cx.imshow(cmb_img,cmap=cmap)
			#cx.colorbar()
			noise_pred_img = np.squeeze(noise_mm[i])
			dx = plt.subplot(234)
			dx.set_title('diff')
			dx.imshow(all_img - noise_pred_img - cmb_img,cmap=cmap)
			dd = all_img - noise_pred_img - cmb_img
			
			print np.mean(np.power(dd, 2))
			print np.amax(dd)
			print np.amin(dd)
			print np.mean(np.absolute(dd))
			print ""

			
			ex = plt.subplot(235)
			ex.set_title('predict noise')
			ex.imshow(noise_pred_img,cmap=cmap)
			#dx.colorbar()

			fx = plt.subplot(236)
			fx.set_title('all_image - predict noise')
			fx.imshow(all_img - noise_pred_img,cmap=cmap)
			#fx.colorbar()

			

			plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
			nu = str(i)
			plt.savefig('52_'+nu+'.png')
			plt.show()

		for i in range(4):
			all_img = np.squeeze(all_[i])
			plt.subplot(2, 3, 1)
			plt.title('all')
			plt.imshow(all_img, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)
			

			noise_img = np.squeeze(noise_[i])
			plt.subplot(2, 3, 2)
			plt.title('noise')
			plt.imshow(noise_img, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)
			

			cmb_img = np.squeeze(cmb_[i])
			plt.subplot(2, 3, 3)
			plt.title('cmb')
			plt.imshow(cmb_img, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)
			

			noise_pred_img = np.squeeze(noise_mm[i])
			plt.subplot(2, 3, 5)
			plt.title('pre noise')
			plt.imshow(noise_pred_img, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)
			

			plt.subplot(2, 3, 6)
			plt.title('all-pre_noise')
			plt.imshow(all_img-noise_pred_img, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)
			
			plt.subplot(2, 3, 4)
			plt.title('all-pre_noise-cmb_img')
			plt.imshow(all_img-noise_pred_img-cmb_img, aspect='auto',cmap='jet', vmin=vmin, vmax=vmax)

			plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
			plt.show()
			nu = str(i)
			plt.savefig('5_same_'+nu+'.png')


if __name__ == '__main__':
	main()
