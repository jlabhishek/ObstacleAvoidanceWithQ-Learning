import matplotlib.pyplot as plt
import pickle 
rewardList = pickle.load(open('Files/rewards.pickle', 'rb')) 
stepList = pickle.load(open('Files/steps.pickle', 'rb')) 

fig = plt.figure(1)
plt.subplot(211) # 211 = numrows,numcolumns, curront plot, 2 rows , 1 columns - first row first plot ; 212 = second row first plot
# https://matplotlib.org/users/pyplot_tutorial.html
plt.axis([1, len(stepList)+10, 0, max(stepList)+10])
plt.plot(stepList)
plt.title('Steps VS Episode')
plt.ylabel('steps per episode')
plt.xlabel('episode')

plt.subplot(212)
plt.axis([1, len(rewardList)+10, -601, max(rewardList)+10])
plt.plot(rewardList)
plt.title('Reward VS Episode')
plt.ylabel('reward per episode')
plt.xlabel('episode')

plt.show()
#fig.savefig(self.graphPath+str(episode + 1)+'.png')
