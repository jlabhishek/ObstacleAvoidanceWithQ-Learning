import qEnvironment
import deepQ
import time
import h5py
import os, os.path
import matplotlib.pyplot as plt
import pickle


'''
	Agents : Takes Action in the environment,sets up simulation(Episodes, step size)
	Environment : sets up the environment(v-rep simulation), returns rewards and next states to agent based on the actions it took
	Algorithm :  The Learning Algorithm
'''
class Agent:
	def __init__(self):
		self.epsilon = 0.1
		self.inputs = 3 #state
		self.outputs = 3 #action
		self.dicountFactor = 0.8
		self.learningRate = 0.25
		self.savePath='pioneer_qlearn_deep/ep'
		self.network_layers = [5]
		self.graphPath = 'Images/'
		self.rewardFile = 'Files/rewards.pickle'
		self.stepFile = 'Files/steps.pickle'

		self.dict = {}

	def start(self):
		

		file_count= len([name for name in os.listdir('pioneer_qlearn_deep') ])
		# print("FILE",file_count)
		weights_path = 'pioneer_qlearn_deep/ep'+str(file_count)+'.h5'
		deepQLearn = deepQ.NNQ(self.inputs,self.outputs,self.dicountFactor,self.learningRate)
		deepQLearn.initNetworks(self.network_layers)
		
		#deepQLearn.plotModel('/home/kaizen/BTP/Python/NeuralNet/Images/')		

		if file_count != 0:
			deepQLearn.loadWeights(weights_path)
			print("Weights Loaded\n")

		env = qEnvironment.Environment()


		num_episodes = 10000
		steps = 300
		start_time = time.time()

		#  for plotting,data per episode
		stepList = []
		rewardList = []
		index = 1

		for episode in range(num_episodes):
			deepQLearn.saveWeights(episode)
			state = env.reset()
			cumulated_reward = 0
			
			for step in range(steps):
				qValues = deepQLearn.getQValues(state)
				action = deepQLearn.selectAction(qValues, self.epsilon )

				self.dict[''.join(str(e) for e in state)] = action

				nextState,reward,done,info = env.step(action)
				cumulated_reward += reward

				deepQLearn.learn_on_one_example(state,action,reward,nextState,done,mini_batch_size = 1)

				if not(done):
					state = nextState
				else:
					print('done')
					break
				print("Step = ",step)

				#  Average time per step = 0.004s

				time.sleep(0.2)
		
		
			stepList.append(step)
			rewardList.append(cumulated_reward)


			m, s = divmod(int(time.time() - start_time), 60)
			h, m = divmod(m, 60)
			print ("\n\n\EP "+str(episode+1)+" Reward: "+ str(cumulated_reward)  +" Time: %d:%02d:%02d" % (h, m, s))                   
			# time.sleep(0.5)

			if (episode +1)%2 == 0:


				rewardList = pickle.load(open(self.rewardFile, 'rb'))  + rewardList
				stepList = pickle.load(open(self.stepFile, 'rb'))  + stepList 
				print(len(rewardList))
				#print(stepList)

				pickle.dump(rewardList, open(self.rewardFile, 'wb'))
				pickle.dump(stepList, open(self.stepFile, 'wb'))

				''' COMMMENT THESE IF TESTING ANYTHING TO PREVENT DATA DAMAGE'''

				deepQLearn.saveModel(self.savePath+str(file_count + index)+'.h5')
				index = index + 1
				
				deepQLearn.saveQValues(episode)

				stepList = []
				rewardList = []
				print(self.dict)
		
		



if __name__ == '__main__':
	agent = Agent()
	agent.start()
