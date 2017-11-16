#!/bin/sh
import random
import vrep                  #V-rep library
import sys
import time                #used to keep track of time
import numpy as np   
#import matplotlib.pyplot as plt

class Environment():

	def __init__(self):
		# Launch the simulation with the given launchfile name
		self.action_space = [i for i in range(3)] #F,L,R
		self.reward_range = (-np.inf, np.inf)

		
		vrep.simxFinish(-1) # just in case, close all opened connections
		self.clientID=vrep.simxStart('127.0.0.1',19998,True,True,5000,5)


		if self.clientID!=-1:  #check if client connection successful
			print ('Connected to remote API server')

		else:
			print( 'Connection not successful')
			sys.exit('Could not connect')


		
	  
		# errorCode,self.left_motor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
		# errorCode,self.right_motor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)

		# # Preallocaation
		# self.sensor_h=[] #empty list for handles
		# sensor_val=np.array([]) #empty array for sensor measurements
   
		# #for loop to retrieve sensor arrays and initiate sensors
		# for x in range(1,16+1):
		# 	errorCode,sensor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
		# 	self.sensor_h.append(sensor_handle) #keep list of handles        
		# 	errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(self.clientID,sensor_handle,vrep.simx_opmode_streaming)                
		# 	sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
		

	#  let new ranges be lenght of state encodind that is number of sensors
	# make data a list of sensor values
	def discretize_observation(self,new_ranges):

		sensor_val=np.array([])  
		for x in range(1,3+1):
			errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(self.clientID,self.sensor_h[x-1],vrep.simx_opmode_buffer)                
			if detectionState == 1 :
				sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
			else:
				sensor_val=np.append(sensor_val,np.inf)

		# print("SENSORS",sensor_val)

		#controller specific ( only 3 sensors used- making state space smaller)
		sensor_sq=sensor_val[0:3] 
		
		# sensor_sq = sensor_sq[0:3]
		# sensor_sq = np.append(sensor_sq,sensor_val[7]*sensor_val[7])
		
		# print('sensor_sq = ',sensor_sq)
		data = sensor_sq
		discretized_ranges = []
		min_range = 0.13
		done = False
	   
		#  convert sensor values to descrete values for state encoding
		for i, item in enumerate(data):
			if item < 0.25:
				discretized_ranges.append(0)
			elif(item < 0.5):
				discretized_ranges.append(1)
			elif(item < 0.75):
				discretized_ranges.append(2)
			elif(item >=0.75 or np.isnan(item)):
				discretized_ranges.append(3)


			if (min_range > item > 0):
				print('Sensor = ',i, 'val = ',item)
				done = True

		return discretized_ranges,done

	
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

		state,done = self.discretize_observation(6)

		if not done:
			if action == 0:
				reward = 5
			else:
				reward = 1
		else:
			reward = -200

		return state, reward, done, {}

	
	def reset(self):

		_,robot = vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',vrep.simx_opmode_oneshot_wait)
		# vrep.simxResetDynamicObject(self.clientID,robot)
		vrep.simxRemoveModel(self.clientID,robot,vrep.simx_opmode_oneshot_wait)
		# vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_oneshot)
		
		# a,b=vrep.simxLoadModel(self.clientID, '/home/rip/Downloads/V-REP_PRO_EDU_V3_4_0_Linux/models/Custom_model/Pioneer_p3dx.ttm',0, vrep.simx_opmode_blocking)
		a,b=vrep.simxLoadModel(self.clientID, '/home/kaizen/BTP/Custom_model/Poineer_p3dx_3_sensors_size.ttm',0, vrep.simx_opmode_blocking)

		_,robot = vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',vrep.simx_opmode_oneshot_wait)
		a=vrep.simxSetObjectOrientation(self.clientID,robot ,robot, [0,0,random.choice([-0.15,0,0.15])],vrep.simx_opmode_oneshot)
		# _,prop = vrep.simxSetModelProperty(self.clientID,robot, vrep.sim_modelproperty_scripts_inactive,vrep.simx_opmode_oneshot)
		# 
		# errorCode=vrep.simxSetJointTargetVelocity(self.clientID,self.left_motor_handle,0, vrep.simx_opmode_streaming)
		# errorCode=vrep.simxSetJointTargetVelocity(self.clientID,self.right_motor_handle,0, vrep.simx_opmode_streaming)
		# time.sleep(4)
		errorCode,self.left_motor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
		errorCode,self.right_motor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)

		# Preallocaation
		self.sensor_h=[] #empty list for handles
		sensor_val=np.array([]) #empty array for sensor measurements
   
		#for loop to retrieve sensor arrays and initiate sensors
		for x in range(1,3+1):
			errorCode,sensor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
			self.sensor_h.append(sensor_handle) #keep list of handles        
			errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(self.clientID,sensor_handle,vrep.simx_opmode_streaming)                
			sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values

		# print(sensor_val)	
		# vrep.simxSetObjectPosition(self.clientID,robot,-1,[-1.4945,0.77500,0.13879],vrep.simx_opmode_oneshot_wait)
		# vrep.simxSetJointPosition(self.clientID,self.left_motor_handle,-1,vrep.simx_opmode_oneshot)
		# vrep.simxSetJointPosition(self.clientID,self.right_motor_handle,-1,vrep.simx_opmode_oneshot)

		#time.sleep(0.2)


		state,done = self.discretize_observation(3)

		return state

	def stop_start(self):
		vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_oneshot)
		time.sleep(1)
		a=vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_oneshot)
		# print("a=",a)


		_,robot = vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',vrep.simx_opmode_oneshot_wait)
		a=vrep.simxSetObjectOrientation(self.clientID,robot ,robot, [0,0,random.choice([-0.15,0,0.15])],vrep.simx_opmode_oneshot)
		
		errorCode,self.left_motor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
		errorCode,self.right_motor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)

		self.sensor_h=[] #empty list for handles
		sensor_val=np.array([]) #empty array for sensor measurements
   
		#for loop to retrieve sensor arrays and initiate sensors
		for x in range(1,3+1):
			errorCode,sensor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
			self.sensor_h.append(sensor_handle) #keep list of handles        
			errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(self.clientID,sensor_handle,vrep.simx_opmode_streaming)                
			sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
		state,done = self.discretize_observation(3)

		return state