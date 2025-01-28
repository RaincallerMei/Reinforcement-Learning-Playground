# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:07:09 2024

@author: mhyra
"""

import gymnasium as gym
import time
import numpy as np

env = gym.make("CartPole-v1", render_mode = "human")
state, info = env.reset(seed=42)
#states include cart position, velocity, pole angle, angular velocity

env.render()

#env.close()

#push cart to left
env.step(0)

env.observation_space

env.observation_space.high

env.observation_space.low
#the 
env.spec



#------------------------------------------

