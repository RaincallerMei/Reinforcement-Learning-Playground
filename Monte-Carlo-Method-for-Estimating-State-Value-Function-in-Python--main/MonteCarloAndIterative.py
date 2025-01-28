# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:36:32 2024
Credit: https://aleksandarhaber.com/monte-carlo-method-for-learning-state-value-functions-first-visit-method-reinforcement-learning-tutorial/
@author: mhyra
"""


def MonteCarloLearnStateValueFunction(env,stateNumber,numberOfEpisodes,discountRate):
    import numpy as np
    
    #sum of returns for every state
    sumReturnEveryState = np.zeros(stateNumber)
    #number of visits of every state
    numVisitsEveryState = np.zeros(stateNumber)
    
    #estimate state's value function vector: The Final Result
    valueFunctionEstimate = np.zeros(stateNumber)
    
    ###########################################################################
    # START - episode simulation
    for indEpisode in range(numberOfEpisodes):
        # this list stores visited states in the current episode
        visitedStatesInEpisode=[]
        # this list stores the return in every visited state in the current episode
        rewardInVisitedState=[]
        
        (currentState,prob)=env.reset()
        visitedStatesInEpisode.append(currentState)
        print("Simulating episode {}".format(indEpisode))
        
        # when the terminal state is reached, this loop breaks
        while True:
            # select a random action
            randomAction= env.action_space.sample() #action_space.sample() -> taking random actions
            # here we step and return the state, reward, and boolean denoting if the state is a terminal state
            (currentState, currentReward, terminalState,_,_) = env.step(randomAction)
            
            rewardInVisitedState.append(currentReward)
            if terminalState:
                break
            visitedStatesInEpisode.append(currentState)
        ###########################################################################
        # END - single episode simulation
        ###########################################################################
        
        # how many states we visited in an episode    
        numberOfVisitedStates = len(visitedStatesInEpisode)
        
        # this is Gt=R_{t+1}+\gamma R_{t+2} + \gamma^2 R_{t+3} + ...
        Gt=0
        #Note that we are starting from the BACK to calculate Gt. Check note if you can't figure out why
        for indexCurrentState in range(numberOfVisitedStates-1, -1, -1):
            #compute the return for every state in the sequence
            stateTmp = visitedStatesInEpisode[indexCurrentState] 
            returnTmp = rewardInVisitedState[indexCurrentState]
            
            # this is an elegant way of summing the returns backwards
            Gt=discountRate*Gt+returnTmp
            
            # first visit implementation 
            if stateTmp not in visitedStatesInEpisode[0:indexCurrentState]:
                # note that this state is visited in the episode
                numVisitsEveryState[stateTmp] = numVisitsEveryState[stateTmp]+1
                # add the sum for that state to the total sum for that state
                sumReturnEveryState[stateTmp] += Gt
        
    ###########################################################################
    # END - episode simulation
    ###########################################################################
    
    #compute the final estimate of the state value function for each state
    for indexSum in range(stateNumber):
        if numVisitsEveryState[indexSum] !=0:
            valueFunctionEstimate[indexSum] = sumReturnEveryState[indexSum]/numVisitsEveryState[indexSum]
    return valueFunctionEstimate


##################
# this function computes the state value function by using the iterative policy evaluation algorithm
##################
def evaluatePolicy(env,valueFunctionVector,policy,discountRate,maxNumberOfIterations,convergenceTolerance):
    import numpy as np
    convergenceTrack=[]
    for iterations in range(maxNumberOfIterations):
        convergenceTrack.append(np.linalg.norm(valueFunctionVector,2))
        valueFunctionVectorNextIteration=np.zeros(env.observation_space.n)
        for state in env.P:
            outerSum=0
            for action in env.P[state]:
                innerSum=0
                for probability, nextState, reward, isTerminalState in env.P[state][action]:
                    #print(probability, nextState, reward, isTerminalState)
                    innerSum=innerSum+ probability*(reward+discountRate*valueFunctionVector[nextState])
                outerSum=outerSum+policy[state,action]*innerSum
            valueFunctionVectorNextIteration[state]=outerSum
        if(np.max(np.abs(valueFunctionVectorNextIteration-valueFunctionVector))<convergenceTolerance):
            valueFunctionVector=valueFunctionVectorNextIteration
            print('Iterative policy evaluation algorithm converged!')
            break
        valueFunctionVector=valueFunctionVectorNextIteration       
    return valueFunctionVector

##################

def grid_print(valueFunction,reshapeDim,fileNameToSave):
    import seaborn as sns
    import matplotlib.pyplot as plt  
    ax = sns.heatmap(valueFunction.reshape(reshapeDim,reshapeDim),
                     annot=True, square=True,
                     cbar=False, cmap='Blues',
                     xticklabels=False, yticklabels=False)
    plt.savefig(fileNameToSave,dpi=600)
    plt.show()