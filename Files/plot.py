import matplotlib.pyplot as plt
import pickle

rewards =pickle.load(open('rewards.pickle','rb'))
plt.plot(rewards)
plt.show()

