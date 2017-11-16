import numpy as np
import random
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers import Dense,Dropout,Activation
from keras.layers.advanced_activations import LeakyReLU


# from keras.callbacks import TensorBoard

# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
#                           write_graph=True, write_images=False)
# # define model
# model.fit(X_train, Y_train,
#           batch_size=batch_size,
#           epochs=nb_epoch,
#           validation_data=(X_test, Y_test),
#           shuffle=True,
#           callbacks=[tensorboard])

class NNQ:
	def __init__(self, inputs, outputs, discountFactor, learningRate):
		"""
		Parameters:
			- inputs: input size
			- outputs: output size
			- memorySize: size of the memory that will store each state
			- discountFactor: the discount factor (gamma)
			- learnStart: steps to happen before for learning. Set to 128
		"""
		self.input_size = inputs
		self.output_size = outputs
		self.discountFactor = discountFactor
		self.learningRate = learningRate

	def initNetworks(self, hiddenLayers):
		model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
		self.model = model


	def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
		model = Sequential()
		if len(hiddenLayers) == 0: 
			model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
			model.add(Activation("linear"))
		else :
			model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
			if (activationType == "LeakyReLU") :
				model.add(LeakyReLU(alpha=0.01))
			else :
				model.add(Activation(activationType))
			
			for index in range(1, len(hiddenLayers)):
				# print("adding layer "+str(index))
				layerSize = hiddenLayers[index]
				model.add(Dense(layerSize, init='lecun_uniform'))
				if (activationType == "LeakyReLU") :
					model.add(LeakyReLU(alpha=0.01))
				else :
					model.add(Activation(activationType))
			model.add(Dense(self.output_size, init='lecun_uniform'))
			model.add(Activation("linear"))
		optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
		model.compile(loss="mse", optimizer=optimizer,metrics=['accuracy'])
		model.summary()
		return model

	def getQValues(self, state):
		# predicted = self.model.predict(state.reshape(1,len(state)))
		predicted = self.model.predict(state.reshape(1,len(state)))
		return predicted[0]

	def selectAction(self, qValues, explorationRate):
		rand = random.random()
		if rand < explorationRate :
			action = np.random.randint(0, self.output_size)
		else :
			action = self.getMaxIndex(qValues)
		return action

	def saveModel(self, path):
		self.model.save(path)

	def loadWeights(self, path):
		self.model.set_weights(load_model(path).get_weights())

	def getMaxQ(self, qValues):
		return np.max(qValues)

	def getMaxIndex(self, qValues):
		return np.argmax(qValues)


	def calculateTarget(self, qValuesNewState, reward, isFinal):
		"""
		target = reward(s,a) + gamma * max(Q(s')
		"""
		if isFinal:
			return reward
		else : 
			return reward + self.discountFactor * self.getMaxQ(qValuesNewState)


	def learn_on_one_example(self, state, action, reward, nextState, isFinal, mini_batch_size):
		X_batch = np.empty((0,self.input_size), dtype = np.float64)
		Y_batch = np.empty((0,self.output_size), dtype = np.float64)
		
		X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
		qValuesNewState = self.getQValues(nextState)
		targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

		qValues = self.getQValues(state)
		Y_sample = qValues.copy()
		Y_sample[action] = targetValue
		Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)

		self.model.fit(X_batch, Y_batch, batch_size = mini_batch_size, nb_epoch=1, verbose = 1)