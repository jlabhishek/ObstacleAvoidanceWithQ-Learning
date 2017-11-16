import qEnvironment
import deepQ
import time


'''
	Agents : Takes Action in the environment,sets up simulation(Episodes, step size)
	Environment : sets up the environment(v-rep simulation), returns rewards and next states to agent based on the actions it took
	Algorithm :  The Learning Algorithm
'''
class Agent:
	def __init__(self):
		self.epsilon = 0.25
		self.inputs = 3 #state
		self.outputs = 3 #action
		self.dicountFactor = 0.8
		self.learningRate = 0.25
		self.savePath='/home/kaizen/BTP/Python/NeuralNet/pioneer_qlear_deep/ep'
		self.network_layers = []


	def start(self):

		weights_path = '/home/kaizen/BTP/Python/NeuralNet/pioneer_qlear_deep/ep5.h5'
		deepQLearn = deepQ.NNQ(self.inputs,self.outputs,self.dicountFactor,self.learningRate)
		deepQLearn.initNetworks(self.network_layers)
		# deepQLearn.loadWeights(weights_path)

		env = qEnvironment.Environment()

		num_episodes = 500
		steps = 200
		start_time = time.time()

		for episode in range(num_episodes):
			state = env.reset()
			cumulated_reward = 0

			for step in range(steps):

				qValues = deepQLearn.getQValues(state)
				action = deepQLearn.selectAction(qValues, self.epsilon )

				nextState,reward,done,info = env.step(action)
				cumulated_reward += reward

				deepQLearn.learn_on_one_example(state,action,reward,nextState,done,mini_batch_size = 1)

				if not(done):
					state = nextState
				else:
					print('done')
					break
				print(step)
				time.sleep(5)
			if (episode+1) % 100 == 0:
				qlearn.saveModel(savePath+str(episode+1)+'.h5')

			m, s = divmod(int(time.time() - start_time), 60)
			h, m = divmod(m, 60)
			print ("EP "+str(episode+1)+" Reward: "+ str(cumulated_reward)  +" Time: %d:%02d:%02d" % (h, m, s))                   
			# time.sleep(0.5)



if __name__ == '__main__':
	agent = Agent()
	agent.start()