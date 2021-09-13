from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import pandas as pd

malData=pd.read_csv('datasets/malicious_data_generated.csv')
malData=malData.to_numpy()
print(malData[1,:])
print(type(malData))
print(malData[:1].shape)

class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up, next variable
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Box(np.asarray([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100]),np.asarray([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]))
        # Set start temp
        self.state = malData[random.randint(0,499),:]
        # Set shower length
        self.shower_length = 60
        
    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature 
        self.state += action -1 
        # Reduce shower length by 1 second
        self.shower_length -= 1 
        
        # Calculate reward
        if self.state >=37 and self.state <=39: 
            reward =1 
        else: 
            reward = -1 
        
        # Check if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = malData[random.randint(0,499),:]
        # Reset shower time
        self.shower_length = 60 
        return self.state
    
env = ShowerEnv()