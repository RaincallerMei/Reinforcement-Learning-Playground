# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:32:39 2022

Demonstration of the OpenAI Gym Library 
and the Fronzen Lake reinforcement learning environment

@author: Aleksandar Haber 

Website accompanying this code with background information
and theoretical explanations is given here:

https://aleksandarhaber.com/introduction-to-state-transition-probabilities-actions-and-rewards-with-openai-gym-reinforcement-learning-tutorial/

"""
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

env=gym.make("FrozenLake-v1", render_mode="human")
env.reset()
env.render()
env.close()


#transition probabilities
#p(s'|s,a) probability of going to state s' 
#          starting from the state s and by applying the action a

# env.P[state][action]
env.P[0][1] 
# output is a list having the following entries
# (transition probability, next state, reward, Is terminal state?)

discount = 0.9
valueFunctionVector = np.zeros(env.observation_space.n)

maxIterations = 1000

convergenceDelta = 10**(-6)

convergenceTrack = []


for iter in range (maxIterations): #calculates v_{k+1} from v_k
     #Tracks the "magnitude" of the current value function v to track convergence over iterations.
    convergenceTrack.append(np.linalg.norm(valueFunctionVector,2))
    #create empty Function Vector v_{k+1} to later compare with v_k
    valueFunctionVectorNextIter = np.zeros(env.observation_space.n)
    for state in env.P:
        outerSum = 0
        for action in env.P[state]:
            innerSum = 0
            for prob, nextState, reward, isTermi, in env.P[state][action]:
                innerSum += prob*(reward + discount*valueFunctionVector[nextState])
            outerSum += 0.25*innerSum #the probability of each step in our policy pi
        valueFunctionVectorNextIter[state] = outerSum
    if (np.max(np.abs(valueFunctionVectorNextIter - valueFunctionVector)) < convergenceDelta):
        valueFunctionVector = valueFunctionVectorNextIter # update
        print("converged!")
        break
    valueFunctionVector = valueFunctionVectorNextIter # update

# visualize the state values
def grid_print(valueFunction,reshapeDim):
    ax = sns.heatmap(valueFunction.reshape(4,4),
                     annot=True, square=True,
                     cbar=False, cmap='Blues',
                     xticklabels=False, yticklabels=False)
    plt.savefig('valueFunctionGrid.png',dpi=600)
    plt.show()
     
grid_print(valueFunctionVector,4)

plt.plot(convergenceTrack)
plt.xlabel('steps')
plt.ylabel('Norm of the value function vector')
plt.savefig('convergence.png',dpi=600)
plt.show()
        
        
        
        