import gym
import numpy as np
import matplotlib.pyplot as plt
import time as t

# Lets create enviroment and grab lower and upper bounds for the observation space
env = gym.make('CartPole-v1')

# Lets construct the functions to return proper values for all observations, 1 and 3 being the ones with some meat.
def round0(val):
    return np.around(val,decimals=1)

def round1(val):

    # Checking all different possibilities
    if val >= -0.75 and val <= 0.75:
        return np.around(val,decimals=2)
    elif (val > -2.5 and val < -0.75) or (val < 2.5 and val > 0.75):
        return np.around(val,decimals=1)
    else:
        return np.around(val,decimals=0)

def round2(val):
    return np.around(val,decimals=2)

def round3(val):

    # Checking all different possibilities
    if val >= -0.5 and val <= 0.5:
        return np.around(val,decimals=2)
    elif (val > -3.0 and val < -0.5) or (val < 3.0 and val > 0.5):
        return np.around(val,decimals=1)
    else:
        return np.around(val,decimals=0)


# Wrap all these in one function
def discretize(obs):
    return [round0(obs[0]),round1(obs[1]),round2(obs[2]),round3(obs[3])]

def genPolicy(P,Q,K,clips,epsilon,epsilon_cap,alpha):

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

        # Create the two arrays for tracking first time state action pairs encountered, and rewards obtained
        SA = []
        R = []

        # Create the set for tracking if state action pairs encountered (first visit MC)
        SA_set = set()

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

            # If it's nan, we haven't seen this state, add the dictionary with key as action and action value as 0
            # Otherwise, check if this action has been encountered, if yes do nothing, otherwise set to 0
            if not innerDict:
                Q[state] = {action : 0}
            else:
                if np.isnan(innerDict.get(action,np.nan)):
                    innerDict[action] = 0

            # Take a step with this action, and record feedback from environment
            obs,reward,done,_ = env.step(action)

            # Construct a state action pair tuple (for hashing)
            state_action = tuple(list(state) + [action])

            # Check if this state_action pair is in the set for this episode
            if state_action in SA_set:
                # If it was, then add False to SA instead of this pair
                # Do nothing else 
                SA.append(False)
            else:
                # Otherwise we need to add it to the set, and add it the array SA
                SA_set.add(state_action)
                SA.append(state_action)

            # Add reward to the R vector regardless
            R.append(reward)

        # Change T_Avg accordingly
        T_avg += (1/(k+1))*(counter - T_avg)

        # Now that the episode has ended, we must do policy evaluation.
        # The idea is to iterate the SA vector backward and keep a running from R 
        running_reward = 0
        for i in range(len(R)-1,-1,-1):

            # Add to running reward
            running_reward += R[i]

            # Get the state_action tuple
            state_action = SA[i]

            # Check if it's False, if not we need to update this action value in Q
            if state_action:

                # We need to split the tuple by state and action
                state = state_action[0:-1]
                action = state_action[-1]
        
                # Now let's update the action value function, where the observed reward is running total
                expected = Q.get(state,np.nan)[action]
                Q.get(state,np.nan)[action] = expected + alpha*(running_reward - expected)
            
        
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
