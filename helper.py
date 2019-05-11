import os
import numpy as np
from midi_to_statematrix import midiToNoteStateMatrix
import constants
from keras.utils import to_categorical


composer_token_map = {
	0: constants.CHOPIN,
	1: constants.BEETHOVEN,
	2: constants.MOZART,
	3: constants.SCHUBERT,
	4: constants.SCHUMANN
}

def randomized_generator(data_map, batch_size=64, timesteps=50, n_classes=5):
	# x_batch - shape: (batch_size, timesteps, input_dim)
	x_batch = []
	y_batch = []
	while True:
		i=0
		for batch_count in range(batch_size):
			# Randomly choose a composer dataset
			composer_data, composer_token = select_composer(data_map)
			# Randomly choose start index of timesteps
			start_idx = np.random.randint(len(composer_data)-timesteps)
			# Pick timesteps from start idx. This will be one state input
			end_idx = start_idx + timesteps
			state_input = np.expand_dims(composer_data[start_idx:end_idx], axis=0)

			# Add to x_batch, y_batch
			if i==0:
				x_batch = state_input
				y_batch = np.array(composer_token)
			else:
				x_batch = np.vstack([x_batch, state_input])
				y_batch = np.vstack([y_batch, np.array(composer_token)])
			i+=1

		#x_batch = np.array(x_batch)
		y_batch = to_categorical(y_batch, num_classes=n_classes)
		yield(x_batch, y_batch)


def select_composer(data_map):
	n_composers = len(data_map)
	composer_token = np.random.randint(n_composers)
	composer_data = data_map[composer_token_map[composer_token]]
	return composer_data, composer_token


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


def get_datamap(split=True, split_idx=[0.8,0.9]):
	chopin_data = np.load(file=constants.dataset_path+'chopin.npy')
	beethoven_data = np.load(file=constants.dataset_path+'beethoven.npy')
	mozart_data = np.load(file=constants.dataset_path+'mozart.npy')
	schubert_data = np.load(file=constants.dataset_path+'schubert.npy')
	schumann_data = np.load(file=constants.dataset_path+'schumann.npy')

	if split:
		chopin_train, chopin_dev, chopin_test = split_dataset(dataset=chopin_data, split=split_idx)
		beethoven_train, beethoven_dev, beethoven_test = split_dataset(dataset=beethoven_data, split=split_idx)
		mozart_train, mozart_dev, mozart_test = split_dataset(dataset=mozart_data, split=split_idx)
		schubert_train, schubert_dev, schubert_test = split_dataset(dataset=schubert_data, split=split_idx)
		schumann_train, schumann_dev, schumann_test = split_dataset(dataset=schumann_data, split=split_idx)

		train_map = create_map(chopin_train, beethoven_train, mozart_train, schubert_train, schumann_train)
		dev_map = create_map(chopin_dev, beethoven_dev, mozart_dev, schubert_dev, schumann_dev)
		test_map = create_map(chopin_test, beethoven_test, mozart_test, schubert_test, schumann_test)

		return train_map, dev_map, test_map

	else:
		return create_map(chopin_data, beethoven_data, mozart_data, schubert_data, schumann_data)

def split_dataset(dataset, split=[0.8,0.9]):
	train, dev, test = np.split(dataset,(int(split[0]*len(dataset)),int(split[1]*len(dataset))))
	return train, dev, test

def create_map(chopin_data, beethoven_data, mozart_data, schubert_data, schumann_data):
	return {
		constants.CHOPIN: chopin_data,
		constants.BEETHOVEN: beethoven_data,
		constants.MOZART: mozart_data,
		constants.SCHUBERT: schubert_data,
		constants.SCHUMANN: schumann_data
	}

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
