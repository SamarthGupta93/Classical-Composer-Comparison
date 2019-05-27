import os
import numpy as np
from midi_to_statematrix import midiToNoteStateMatrix
import constants
from keras.utils import to_categorical
import random

composer_token_map = {
	constants.BEETHOVEN:0,
	constants.MOZART:1
}


def get_batch(data_map, batch_size=64, timesteps=50):
	x_batch = []
	y_batch = []
	i=0
	for batch_count in range(batch_size):
		# Randomly choose a composer dataset
		composer, composer_data = select_composer(data_map)
		# Randomly choose start index of timesteps
		start_idx = np.random.randint(len(composer_data)-timesteps)
		# Pick timesteps from start idx. This will be one state input
		end_idx = start_idx + timesteps
		state_input = np.expand_dims(composer_data[start_idx:end_idx], axis=0)

		# Add to x_batch, y_batch
		if i==0:
			x_batch = state_input
			y_batch = np.array(composer_token_map[composer])
		else:
			x_batch = np.vstack([x_batch, state_input])
			y_batch = np.vstack([y_batch, np.array(composer_token_map[composer])])
		i+=1

	#x_batch = np.array(x_batch)
	return x_batch, y_batch


def randomized_generator(data_map, batch_size=64, timesteps=50, n_classes=5):
	# x_batch - shape: (batch_size, timesteps, input_dim)
	while True:
		x_batch, y_batch = get_batch(data_map, batch_size, timesteps)
		y_batch = to_categorical(y_batch, num_classes=n_classes)
		yield(x_batch, y_batch)


def select_composer(data_map):
	return random.choice(list(data_map.items()))


def create_dataset():
	chopin_data = get_composer_data(constants.chopin_path)
	print("Chopin: {}".format(chopin_data.shape))
	np.save(file=constants.dataset_path+'chopin', arr=chopin_data)
	beethoven_data = get_composer_data(constants.beethoven_path)
	print("Beethoven: {}".format(beethoven_data.shape))
	np.save(file=constants.dataset_path+'beethoven', arr=beethoven_data)
	mozart_data = get_composer_data(constants.mozart_path)
	print("Mozart: {}".format(mozart_data.shape))
	np.save(file=constants.dataset_path+'mozart', arr=mozart_data)
	schubert_data = get_composer_data(constants.schubert_path)
	print("Schubert: {}".format(schubert_data.shape))
	np.save(file=constants.dataset_path+'schubert', arr=schubert_data)
	schumann_data = get_composer_data(constants.schumann_path)
	print("Schumann: {}".format(schumann_data.shape))
	np.save(file=constants.dataset_path+'schumann', arr=schumann_data)



def get_datamap(composers, split=True, split_idx=[0.8,0.9]):
	chopin_data, beethoven_data, mozart_data, schubert_data, schumann_data = None, None, None, None, None
	if composers[constants.CHOPIN]: chopin_data = np.load(file=constants.dataset_path+'chopin.npy')
	if composers[constants.BEETHOVEN]: beethoven_data = np.load(file=constants.dataset_path+'beethoven.npy')
	if composers[constants.MOZART]: mozart_data = np.load(file=constants.dataset_path+'mozart.npy')
	if composers[constants.SCHUBERT]: schubert_data = np.load(file=constants.dataset_path+'schubert.npy')
	if composers[constants.SCHUMANN]: schumann_data = np.load(file=constants.dataset_path+'schumann.npy')
	chopin_train, chopin_dev, chopin_test = None, None, None
	beethoven_train, beethoven_dev, beethoven_test = None, None, None
	mozart_train, mozart_dev, mozart_test = None, None, None
	schubert_train, schubert_dev, schubert_test = None, None, None
	schumann_train, schumann_dev, schumann_test = None, None, None

	if split:
		if composers[constants.CHOPIN]: chopin_train, chopin_dev, chopin_test = split_dataset(dataset=chopin_data, split=split_idx)
		if composers[constants.BEETHOVEN]: beethoven_train, beethoven_dev, beethoven_test = split_dataset(dataset=beethoven_data, split=split_idx)
		if composers[constants.MOZART]: mozart_train, mozart_dev, mozart_test = split_dataset(dataset=mozart_data, split=split_idx)
		if composers[constants.SCHUBERT]: schubert_train, schubert_dev, schubert_test = split_dataset(dataset=schubert_data, split=split_idx)
		if composers[constants.SCHUMANN]: schumann_train, schumann_dev, schumann_test = split_dataset(dataset=schumann_data, split=split_idx)

		train_map = create_map(composers, chopin_train, beethoven_train, mozart_train, schubert_train, schumann_train)
		dev_map = create_map(composers, chopin_dev, beethoven_dev, mozart_dev, schubert_dev, schumann_dev)
		test_map = create_map(composers, chopin_test, beethoven_test, mozart_test, schubert_test, schumann_test)

		return train_map, dev_map, test_map

	else:
		return create_map(composers, chopin_data, beethoven_data, mozart_data, schubert_data, schumann_data)

def split_dataset(dataset, split=[0.8,0.9]):
	train, dev, test = np.split(dataset,(int(split[0]*len(dataset)),int(split[1]*len(dataset))))
	return train, dev, test

def create_map(composers, chopin_data=None, beethoven_data=None, mozart_data=None, schubert_data=None, schumann_data=None):
	result_map = {
		constants.CHOPIN: chopin_data,
		constants.BEETHOVEN: beethoven_data,
		constants.MOZART: mozart_data,
		constants.SCHUBERT: schubert_data,
		constants.SCHUMANN: schumann_data
	}
	for composer, include in composers.items():
		if not include:
			del result_map[composer]
	return result_map

def get_composer_data(composer_path):
	i=0
	for file in os.scandir(composer_path):
		state = np.array(midiToNoteStateMatrix(file.path), dtype='uint8')
		if i==0:
			data = state
		else:
			data = np.vstack((data,state))
		i+=1
	return data


def get_size(data_map):
	count = 0
	for key, value in data_map.items():
		count += value.shape[0]
	return count
