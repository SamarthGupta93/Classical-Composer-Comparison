import midi
from midi_to_statematrix import midiToNoteStateMatrix
import numpy as np
import os
from helper import create_dataset, randomized_generator, get_datamap, get_size
import constants
from gru_model import Composer_Classifier


n_classes = 5
n_epochs = 30
batch_size = 32
timesteps = 50
def run():
	
	train_map, dev_map, test_map = get_datamap(split=True)
	train_size = get_size(train_map)
	dev_size = get_size(dev_map)
	print(train_size, dev_size)
	# Create Train and Dev generators
	train_gen = randomized_generator(train_map, batch_size=batch_size, timesteps=timesteps)
	dev_gen = randomized_generator(dev_map, batch_size=batch_size, timesteps=timesteps)
	
	train_iterations = int(train_size/(10*batch_size))
	dev_iterations = int(dev_size/(10*batch_size))
	print(train_iterations, dev_iterations)

	
	# Train Model
	composer_clf = Composer_Classifier(n_classes=5)
	composer_clf.create_model()
	composer_clf.train(train_generator=train_gen, dev_generator=dev_gen, 
		steps_per_epoch=train_iterations, validation_steps=dev_iterations, epochs=n_epochs)
	
		
	
if __name__ == "__main__":
	#create_dataset()
	run()