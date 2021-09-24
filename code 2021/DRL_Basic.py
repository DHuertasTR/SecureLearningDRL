from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

data=pd.read_csv('datasets/malicious_data_generated.csv')
npdata=data.to_numpy()
malData=np.copy(npdata)
print(type(malData[1,:]))
print(malData[:1].shape)

class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take, decrease, increse, none
        self.action_space = Discrete(21)
        # max-min array
        self.observation_space = Box(np.asarray([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100]),np.asarray([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]))
        # Set start 
        self.state = np.asarray(malData[random.randint(0,499),:])
        
        # Set time 
        self.length = 60
        
    def step(self, action):
        # Apply action for each state
        if(action<10):
            if(action==0):
                self.state[0]=self.state[0]+1
            elif(action==1):
                 self.state[1]=self.state[1]+1
            elif(action==2):
                 self.state[2]=self.state[2]+1
            elif(action==3):
                 self.state[3]=self.state[3]+1
            elif(action==4):
                 self.state[4]=self.state[4]+1
            elif(action==5):
                 self.state[5]=self.state[5]+1
            elif(action==6):
                 self.state[6]=self.state[6]+1
            elif(action==7):
                 self.state[7]=self.state[7]+1
            elif(action==8):
                 self.state[8]=self.state[8]+1
            else:
                 self.state[9]=self.state[9]+1                     
        else:
            if(action==10):
                self.state[0]+=self.state[0]-1
            elif(action==11):
                 self.state[1]+=self.state[1]-1
            elif(action==12):
                 self.state[2]+=self.state[2]-1
            elif(action==13):
                 self.state[3]+=self.state[3]-1
            elif(action==14):
                 self.state[4]+=self.state[4]-1
            elif(action==15):
                 self.state[5]+=self.state[5]-1
            elif(action==16):
                 self.state[6]+=self.state[6]-1
            elif(action==17):
                 self.state[7]+=self.state[7]-1
            elif(action==18):
                 self.state[8]+=self.state[8]-1
            else:
                 self.state[9]+=self.state[9]-1  
        
        self.length -= 1 
            
        
        # Calculate reward in ranges
        rewardMulti=0 
        
        if(self.state[0]>=-0.290698 and self.state[0]<=-133.441860):
                 rewardMulti=rewardMulti+1
        elif(self.state[1]>=0 and self.state[1]<=1184):
                 rewardMulti=rewardMulti+1
        elif(self.state[2]>=-0.666667 and self.state[2]<=10.666667):
                 rewardMulti=rewardMulti+1
        elif(self.state[3]>=-0.312383 and self.state[3]<=109.259173):
                 rewardMulti=rewardMulti+1
        elif(self.state[4]>=0 and self.state[4]<=30):
                 rewardMulti=rewardMulti+1
        elif(self.state[5]>=-0.322 and self.state[5]<=127.488889):
                 rewardMulti=rewardMulti+1
        elif(self.state[6]>=-0.282353 and self.state[6]<=147.976471):
                 rewardMulti=rewardMulti+1
        elif(self.state[7]>=-0.164688 and self.state[7]<=715.616633):
                 rewardMulti=rewardMulti+1
        elif(self.state[8]>=-0.324081 and self.state[8]<=106.407677):
                 rewardMulti=rewardMulti+1
        elif(self.state[9]>=-0.750000 and self.state[9]<=227.5):
                 rewardMulti=rewardMulti+1
                 
        if(rewardMulti==0):
            reward=-1
        else:
            reward=rewardMulti
                 
        
        # Check if is done
        if self.length <= 0: 
            done = True
        else:
            done = False
     
        
        
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        pass
    
    def reset(self):
        # Reset 
        self.state = malData[random.randint(0,499),:]
        # Reset time
        self.length = 60 
        return self.state
    
env = ShowerEnv()

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    
def build_model(states, actions):
    model = tensorflow.keras.Sequential()   
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

states = env.observation_space.shape
actions = env.action_space.n

model = build_model(states, actions)

print(model.summary())


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
#dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#dqn.fit(env, nb_steps=10000, visualize=False, verbose=1)