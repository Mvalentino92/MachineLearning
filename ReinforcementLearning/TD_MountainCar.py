import gym
import numpy as np
import matplotlib.pyplot as plt
import time as t

# Lets create enviroment and grab lower and upper bounds for the observation space
env = gym.make('MountainCar-v0')

# Lets construct the functions to return proper values for all observations, 1 and 3 being the ones with some meat.
def round0(val):
    return np.around(val,decimals=2)

def round1(val):
    return np.around(val,decimals=2)

# Wrap all these in one function
def discretize(obs):
    return [round0(obs[0]),round1(obs[1])]

def TD_genPolicy(P,Q,K,clips,epsilon,epsilon_cap,alpha,discount):

    # Set T_avg, the average number time steps lasted for an episode.
    T_avg = evalPolicy(P)

    # For every episode k
    for k in range(1,K+1):

        # Find the value h, for which to scale epsilon by
        h = (epsilon_cap/epsilon)**(1/T_avg) - 1

        # Clone epsilon for this episode
        ep = epsilon

        # Initialize the counter this episode
        counter = 0

        # Just print where we're at
        #if k % 1000 == 0:
         #   print('On episode:',k)

        # Reset and observe the first environment
        obs = env.reset()

        # Initialize done to false to track termination of episode
        done = False

        # While this episode is not finished
        while not done:

            # Increment the counter
            counter += 1
            
            # Update epsilon
            ep *= (1 + h)

            # Do the proper conversions for all values 
            obs = discretize(obs)

            # Convert this obs array into a tuple (so it's hashable)
            state = tuple(obs)

            # Attempt to add this state to P, since it has been encountered (with a random action policy for now)
            # Otherwise if it's there, don't do anything
            if np.isnan(P.get(state,np.nan)):
                P[state] = env.action_space.sample()
            
            # Record the next feedback from the environement when taking a step according to the soft policy
            # NOTE: If no policy existed (first time encountering), it is now random from above
            action = P.get(state)
            if np.random.rand() < ep:
                act_suboptimal = env.action_space.sample()
                while action == act_suboptimal:
                    act_suboptimal = env.action_space.sample()
                action = act_suboptimal

            # Attempt to add this state action pair to Q for policy improvement.
            # Try to grab the inner dictionary (meaning we've seen this state)
            innerDict = Q.get(state,False)

            # If it's False, we haven't seen this state, add the dictionary with key as action and action value as 0
            # Otherwise, check if this action has been encountered, if yes do nothing, otherwise set to 0
            if not innerDict:
                Q[state] = {action : 0}
            else:
                if np.isnan(innerDict.get(action,np.nan)):
                    innerDict[action] = 0

            # Take a step with this action, and record feedback from environment
            obs,reward,done,_ = env.step(action)

            # Update the Q(s,a) by grabbing Q'(s,a). Default value for Q(s',a') is 0
            state_prime = tuple(discretize(obs))
            q_sa_prime = 0
            innerDictPrime = Q.get(state_prime,False)

            # If there was no entry for state_prime, do nothing and use default 0
            # Otherwise, grab what action a' the policy says to use for s'.
            # Finally grab the aciton value Q(s',a'). Which should be present, but if its not use default 0.
            if innerDictPrime:
                action_prime = P.get(state_prime,np.nan)
                q_sa_prime = innerDictPrime.get(action_prime,0)

            # Now that we have q_sa_prime, we can update Q(s,a)
            expected = Q[state][action]
            Q[state][action] = expected + alpha*(reward + discount*q_sa_prime - expected)


        # Change T_Avg accordingly since episode has ended
        T_avg += (1/(k+1))*(counter - T_avg)

        # Finally, policy improvement. If it's after the proper amount of episode clips
        if k % clips == 0:

            # Iterate all the states in P and update their policy greedily with respect to Q
            for state in P.keys():

                # Grab the inner dictionary for this state from Q
                A = Q.get(state,np.nan)

                # Choose the optimal action (key), based on the action_value (value)
                mx = -np.Inf
                opt_act = np.nan
                for action in A.keys():
                    if A[action] > mx:
                        mx = A[action]
                        opt_act = action
                
                # Replace the policy for this state in P with this new optimal action
                P[state] = opt_act
    return T_avg        

def runPolicy(P,s):
    obs = env.reset()
    done = False
    counter = 0
    r_counter = 0
    info = 0
    while not done:
        env.render()
        t.sleep(s)
        counter += 1
        obs = discretize(obs)
        state = tuple(obs)
        action = P.get(state,np.nan)
        if np.isnan(action):
            r_counter += 1
            action = env.action_space.sample()

        obs,_,done,info = env.step(action)

    print('Total steps:',counter,'\tRandom steps:',r_counter)
    print(obs)
    env.close()
    return counter

def evalPolicy(P):
    obs = env.reset()
    done = False
    counter = 0
    while not done:
        counter += 1
        obs = discretize(obs)
        state = tuple(obs)
        action = P.get(state,np.nan)
        if np.isnan(action):
            action = env.action_space.sample()
        obs,_,done,_ = env.step(action)

    env.close()
    return counter
