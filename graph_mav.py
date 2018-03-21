import matplotlib.pyplot as plt
import pickle 
import numpy
rewardList = pickle.load(open('Files/rewards.pickle', 'rb')) 
stepList = pickle.load(open('Files/steps.pickle', 'rb')) 


def movingaverage(interval, window_size):
    window = numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')


sum  = 0
print("Reward Average over 50 = ")

for i in range(len(rewardList)):
	sum += rewardList[i]
	if (i+1) % 50 == 0: 
		print(round(sum/50,5), end=" ")
		sum = 0

sum  = 0
print("\nStep Average over 50 = ")
fail = False
for i in range(len(stepList)):
	if stepList[i] > 195:
		fail = True
	sum += stepList[i]
	if (i+1) % 50 == 0: 
		print(sum/50.0, end=" ")
		sum = 0

step_mav = movingaverage(stepList,window_size=10)
print(step_mav)
print("\nFail = ",fail)
range = 3
offset = 1
fig = plt.figure(1)
plt.subplot(211) # 211 = numrows,numcolumns, curront plot, 2 rows , 1 columns - first row first plot ; 212 = second row first plot
# https://matplotlib.org/users/pyplot_tutorial.html
plt.axis([1, len(stepList)+10, 0, max(stepList)+10])
plt.plot(stepList)
plt.title('Steps VS Episode')
plt.ylabel('steps per episode')
plt.xlabel('episode')

offset = 1
plt.subplot(212)
plt.axis([1, len(stepList)+10, 0, max(stepList)+10])
plt.plot(step_mav)
plt.title('Mav_Steps VS Episode')
plt.ylabel('Moving Average')
plt.xlabel('episode')

plt.show()
#fig.savefig(self.graphPath+str(episode + 1)+'.png')
