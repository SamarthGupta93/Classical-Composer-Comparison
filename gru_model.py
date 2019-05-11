from keras.layers import GRU, Input, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Model

class Composer_Classifier():

	def __init__(self, gru_units=64, n_classes=5, input_dim=78, dense_units=32):

		self.n_classes = n_classes
		self.input = Input(shape=(None,input_dim), name="input_layer")
		self.gru = GRU(units=gru_units, dropout=0.3, return_state=True, name="gru_layer")
		self.dense = Dense(units=dense_units, activation='relu', name="FC_1")
		self.dropout = Dropout(0.3, name="dropout_layer")
		self.output = Dense(units=n_classes, activation='softmax', name="output_layer")
		self.callbacks = None

	def create_model(self):

		input_notes = self.input
		X, state_h = self.gru(input_notes)
		X = self.dense(X)
		X = self.dropout(X)
		output = self.output(X)

		optimizer = Adam(lr=0.0001)

		self.model = Model(inputs=input_notes, outputs=output)
		self.model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

		return self.model

	def train(self, train_generator, dev_generator, steps_per_epoch, validation_steps, epochs=10):
		# Fit the model with the generator
		history = self.model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, 
			callbacks=self.callbacks, validation_data=dev_generator, validation_steps=validation_steps, verbose=1)

		return history
