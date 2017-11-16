#!/bin/sh
import vrep                  #V-rep library
import sys
import time                #used to keep track of time
import numpy as np   
#import matplotlib.pyplot as plt
import math 

import nn_q
import pickle
from functools import reduce


PI = math.pi

if __name__ == '__main__':
	# Establish Communication
	
	for ep in range(100):
		
		print('---------------------------------Iteration No. ',(ep+1),'---------------------------------')

		last_time_steps = np.ndarray(0)
		
		qlearn = NNQLearn()
						

		initial_epsilon = qlearn.epsilon

		epsilon_discount = 0.9986

		start_time = time.time()
		total_episodes = 300
		highest_reward = 0
		avg =0	
		steps = 200
		
		f = open('q_table.txt','a')
		f2 = open('q_table_list.pickle','ab')
		
		# Initial Write
		pickle.dump(qlearn.q, f2)

		for x in range(total_episodes):
			done = False

			cumulated_reward = 0 #Should going forward give more reward then L/R ?
			
			observation =env.stop_start()

			# if qlearn.epsilon > 0.05:
			# 	qlearn.epsilon *= epsilon_discount

			state = ''.join(map(str, observation))
			# print("State = ",state," observation = ",observation)
			for i in range(steps):

				# Pick an action based on the current state
				nextState,reward,action,done = qlearn.chooseAction(state)

				# Execute the action and get feedback
				
				cumulated_reward += reward
				avg += reward

				if highest_reward < cumulated_reward:
					highest_reward = cumulated_reward

				# nextState = ''.join(map(str, observation))

				# qlearn.learn(state, action, reward, nextState)

				# env.monitor.flush(force=True)
				# print(i," S= ", state, " A = ",action, 'observation = ',observation)
				if not(done):
					state = nextState
				else:
					last_time_steps = np.append(last_time_steps, [int(i + 1)])	
					print('done')
					break

				time.sleep(0.5)

			if (x+1)%100 == 0:
				print("Iteration : ",(ep+1),"EP: "+str(x+1)+" Avg Reward: ",avg/100,"\nQ Table\n",sorted(qlearn.q.items()),file = f)
				pickle.dump(qlearn.q, f2)
				avg =0
				print(qlearn.q)
					


			m, s = divmod(int(time.time() - start_time), 60)
			h, m = divmod(m, 60)
			print ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))
			
			
		#Github table content
		print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |")

		l = last_time_steps.tolist()
		l.sort()

		#print("Parameters: a="+str)
		print("Overall score: {:0.2f}".format(last_time_steps.mean()))
		print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

		f.close()
		f2.close()
		


