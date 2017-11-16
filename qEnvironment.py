import vrep
import numpy as np
import random 
import time

class Environment:

	def __init__(self):
		# Launch the simulation with the given launchfile name
		self.action_space = [i for i in range(3)] #F,L,R
		self.reward_range = (-np.inf, np.inf)

		self.sensors = 3
		
		vrep.simxFinish(-1) # just in case, close all opened connections
		self.clientID=vrep.simxStart('127.0.0.1',19998,True,True,5000,5)


		if self.clientID!=-1:  #check if client connection successful
			print ('Connected to remote API server, clientID',self.clientID)

		else:
			print( 'Connection not successful')
			sys.exit('Could not connect')

		# errorCode,sensor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_ultrasonicSensor1',vrep.simx_opmode_oneshot_wait)
		# print(errorCode,sensor_handle)
		# errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(self.clientID,sensor_handle,vrep.simx_opmode_streaming)
		# print(errorCode,detectionState,detectedPoint,sensor_handle," done")

		# time.sleep(2)	
	#  RETURN THE NEW STATE OF THE AGENT based on the action taken by the agent
	def step(self, action):

		#  set velocity of left and right motor

		v=0.5	#forward velocity

		# can make this part more accurate
		kp=0.5	#steering gain
		steer = 0.5
		if action == 0: #FORWARD
			vl = v
			vr = v
		elif action == 1: #LEFT
			vl=v-kp*steer
			vr=v+kp*steer
			
		elif action == 2: #RIGHT
			vl=v+kp*steer
			vr=v-kp*steer
		# elif action == 3: #BACK
		# 	vl=-1*v
		# 	vr=-1*v

		errorCode=vrep.simxSetJointTargetVelocity(self.clientID,self.left_motor_handle,vl, vrep.simx_opmode_streaming)
		errorCode=vrep.simxSetJointTargetVelocity(self.clientID,self.right_motor_handle,vr, vrep.simx_opmode_streaming)

		state,done = self.get_observation() 	# Returns Observation as an Array

		if not done:
			if action == 0:
				reward = 5
			else:
				reward = 1
		else:
			reward = -200

		return state, reward, done, {}


	def observe_environment_state(self):
		time.sleep(1)
		print("handle", self.sensor_h)

		sensor_val=np.array([])  
		for x in range(1,self.sensors + 1):
			errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(self.clientID,self.sensor_h[x-1],vrep.simx_opmode_buffer)                
			if detectionState == True :
				sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
			else:
				sensor_val=np.append(sensor_val,np.inf)
				
		return sensor_val

	def get_observation(self):

		distance = self.observe_environment_state()
		discretized_ranges = np.empty((0,3), int)
		min_range = 0.10
		done = False
	   
		#  convert sensor values to descrete values for state encoding
		for i, item in enumerate(distance):
			if item < 0.25:
				discretized_ranges= np.append(discretized_ranges, 0)
			elif(item < 0.5):
				discretized_ranges= np.append(discretized_ranges, 1)
			elif(item < 0.75):
				discretized_ranges= np.append(discretized_ranges, 2)
			elif(item >=0.75 or np.isnan(item)):
				discretized_ranges= np.append(discretized_ranges, 3)


			if (min_range > item > 0):
				print('Done = True , Sensor = ',i, 'val = ',item)
				done = True

		return discretized_ranges,done

	def reinitialize_robot(self):

		#  make three models here with different orientation, and choose randomly out of those three for different state states

		_,robot = vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',vrep.simx_opmode_oneshot_wait)
		ret_val = vrep.simxSetObjectOrientation(self.clientID,robot ,robot, [0,0,random.choice([-0.15,0,0.15])],vrep.simx_opmode_oneshot)
		
		errorCode,self.left_motor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
		errorCode,self.right_motor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)

		self.sensor_h=[] #empty list for handles
		sensor_val=[]
		#for loop to retrieve sensor arrays and initiate sensors
		for x in range(1,3+1):
			errorCode,sensor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
			self.sensor_h.append(sensor_handle) #keep list of handles        
			
	def reset(self):
		# vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_blocking)
		# time.sleep(1)
		# ret_val = vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_blocking)


		# clientID=vrep.simxStart('127.0.0.1',19998,True,True,5000,5)
		errorCode,sensor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_ultrasonicSensor1',vrep.simx_opmode_oneshot_wait)
		errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(self.clientID,sensor_handle,vrep.simx_opmode_streaming)                
		 
		print(self.clientID,errorCode,detectionState,detectedPoint,detectedObjectHandle)
		time.sleep(1)



		self.reinitialize_robot()
		
		start_state,done = self.get_observation()

		return start_state




if __name__ == '__main__':
	env=Environment()