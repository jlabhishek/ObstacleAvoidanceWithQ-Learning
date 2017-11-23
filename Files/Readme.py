#dumpy empty value in these pickle files

import pickle
pickle.dump([], open('rewards.pickle', 'wb'))
pickle.dump([], open('steps.pickle', 'wb'))



#pickle.dump(rewardList, open(self.rewardFile, 'wb'))
			#	pickle.dump(stepList, open(self.stepFile, 'wb'))
