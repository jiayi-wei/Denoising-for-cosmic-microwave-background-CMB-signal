import numpy as np
import os

all_path = '/home/sedlight/workspace/shl/data/train/img_data/band1_all_block_4/'
noise_path = '/home/sedlight/workspace/shl/data/train/img_data/band1_noise_block_4/'
cmb_path = '/home/sedlight/workspace/shl/data/train/img_data/band1_cmb_block_4/'

def get_data():
	all_file = os.listdir(all_path)
	cmb_file = os.listdir(cmb_path)
	noise_file = os.listdir(noise_path)

	all_file.sort()
	cmb_file.sort()
	noise_file.sort()

	while 1:
		for i,name in enumerate(all_file):
			all_data = np.stack([np.load(all_path + all_file[i])], axis=0)
			noise_data = np.stack([np.load(noise_path + noise_file[i])], axis=0)
			cmb_data = np.stack([np.load(cmb_path + cmb_file[i])], axis=0)

			yield(all_data, [noise_data, cmb_data])


def get_data_noise():
	all_file = os.listdir(all_path)
	#cmb_file = os.listdir(cmb_path)
	noise_file = os.listdir(noise_path)

	all_file.sort()
	#cmb_file.sort()
	noise_file.sort()

	while 1:
		for i,name in enumerate(all_file):
			all_data = np.stack([np.load(all_path + all_file[i])], axis=0)
			noise_data = np.stack([np.load(noise_path + noise_file[i])], axis=0)
			#cmb_data = np.stack([np.load(cmb_path + cmb_file[i])], axis=0)

			yield(all_data, noise_data)


all_path_test = '/home/sedlight/workspace/shl/data/test/img_data/band1_all_block_4/'
noise_path_test = '/home/sedlight/workspace/shl/data/test/img_data/band1_noise_block_4/'
cmb_path_test = '/home/sedlight/workspace/shl/data/test/img_data/band1_cmb_block_4/'

def test_data():
	all_file = os.listdir(all_path_test)
	cmb_file = os.listdir(cmb_path_test)
	noise_file = os.listdir(noise_path_test)

	all_file.sort()
	cmb_file.sort()
	noise_file.sort()

	while 1:
		for i,name in enumerate(all_file):
			all_data = np.stack([np.load(all_path_test + all_file[i])], axis=0)
			noise_data = np.stack([np.load(noise_path_test + noise_file[i])], axis=0)
			cmb_data = np.stack([np.load(cmb_path_test + cmb_file[i])], axis=0)

			yield(all_data, [noise_data, cmb_data])


def test_data_noise():
	all_file = os.listdir(all_path_test)
	#cmb_file = os.listdir(cmb_path_test)
	noise_file = os.listdir(noise_path_test)

	all_file.sort()
	#cmb_file.sort()
	noise_file.sort()

	while 1:
		for i,name in enumerate(all_file):
			all_data = np.stack([np.load(all_path_test + all_file[i])], axis=0)
			noise_data = np.stack([np.load(noise_path_test + noise_file[i])], axis=0)
			#cmb_data = np.stack([np.load(cmb_path_test + cmb_file[i])], axis=0)

			yield(all_data, noise_data)
