import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque

# Create the environement
env = gym.make('MountainCar-v0')

# Get relevant information for the policy network
input_ = env.observation_space.shape[0]
hidden1 = 150
hidden2 = 50
output_ = env.action_space.n

# Create the class for the model
class Model(nn.Module):
    
    # Layers and sizes for initialization
    def __init__(self):
        super(Model,self).__init__()
        self.input_ = nn.Linear(input_,hidden1)
        self.hidden1 = nn.Linear(hidden1,hidden2)
        self.hidden2 = nn.Linear(hidden2,output_)

    # Override the forward method (get back for free!)
    def forward(self,x):

        # Functionally chain the layers together (linear combination -> activation
        x = F.relu(self.input_(x))
        x = F.leaky_relu(self.hidden1(x))
        x = F.softmax(self.hidden2(x))
        return x

# Create the model
model = Model()

# Pick a learning rate (default for Adam is 0.001, and create optimizer
lr = 0.0009
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

# Set the discount factor
discount = 0.99

# ************** Information and explanation of the algorithm **********
# Rewards will be calculated monte carlo style, after each episode
# The loss function, will be -R*ln(prob(a|S))
# Where the R is the total discounted total rewards for taking action a in state S
# Also written as -> r_t + r_t+1*discount^1 + r_t+2*discount^2 + ... + r_t+2*discount^n
# And prob(a|S) is the probability of taken action a given state S. 
# Note, this is the actual action that we took during the episode. 
# The number of episodes is also equal to the number of epochs, since updates only happen in batches (for all rewards) at the end of the episode.


# Define the loss function
def loss_fn(discounted_returns,probabilities):
    return -1*torch.sum(discounted_returns*torch.log(probabilities))

# Gets the discounted rewards
def get_discounted_returns(rewards,discount):

    # Sum in reverse order, and keep discounting
    R = 0
    returns = []
    for i in range(len(rewards)-1,-1,-1):
        R = rewards[i] + discount*R
        returns.append(R)
    
    # Wrap in tensor and return
    returns = torch.Tensor(returns[::-1])
    return returns#/returns.max()

# Get the probabilities of the selected actions
def get_probabilities(states,actions):

    # Get the batch of probalities from the states and compress to single dimension
    prob_batch = model(torch.Tensor(states)).view(-1)

    # Scale up all actions to access correct elements
    actions = [i*output_ + actions[i] for i in range(0,len(actions))]

    # Return probabilities for actions taken
    return prob_batch[actions]

# Standardize rewards
def standardize_returns(returns):

    # Get mean and std
    mean = returns.mean()
    std = returns.std()

    # Return new
    return (returns - mean)/std
    
# Set number of epochs and clips
epochs = 1000
clips = 100

# Keeps a high probablity deque to train on during each episode. (states,actions,rewards)
deque_len = 32
deque_iter = epochs + 1
high_prob_trans = deque(maxlen=deque_len)
hp_scale = 0.1

lasted = []

# Start training
for epoch in range(epochs):

    # First let's reset the environment and grab the state
    state = env.reset()

    # We need a couple arrays to store the rewards, and probablities of actions that were selected
    rewards = []
    actions = []
    states = []

    # Set done equal to False so we continue the episode until termination
    done = False

    # Tracking turn for deque training
    turn = 0

    # Train on high_prob, if applicable
    if epoch % deque_iter == 0 and len(high_prob_trans) == deque_len:

        # Convert high_probs to list, and grab all the pieces (standardize returns)
        transitions = list(high_prob_trans)
        hp_states = [s for (s,a,r) in transitions]
        hp_actions = [a for (s,a,r) in transitions]
        hp_returns = standardize_returns(torch.Tensor([r for (s,a,r) in transitions]))

        # Get the probablities (already have returns)
        hp_probabilities = get_probabilities(hp_states,hp_actions)

        # Get mean of these
        scale = torch.mean(hp_probabilities)
        
        # Train on these
        loss = loss_fn(scale*hp_returns,hp_probabilities)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Begin this episode
    while not done:

        # Render the environment
        if epoch % clips == 0:
            env.render()

        # Increment turn
        turn += 1

        # Run the model to get a probablity distribution for actions (dont need grad, need to recompute anyway)
        with torch.no_grad():
            act_probs = model(torch.from_numpy(state).float())

        # Choose an action randomly using probabilities (the index)
        action = np.random.choice(np.arange(0,output_),p=act_probs.data.numpy())

        # Add this state and action
        states.append(state)
        actions.append(action)

        # Take this step and observe (don't need state anymore)
        state,reward,done,_ = env.step(action) 

        # Record the reward
        rewards.append(reward)

    lasted.append(len(rewards))

    # Episode has ended, get the discounted total returns
    returns = get_discounted_returns(rewards,discount)

    # Get the probabilities
    probabilities = get_probabilities(states,actions)

    # Add the highest probablity transitions to the queue
    hp_index = torch.argmax(probabilities)
    high_prob_trans.append((states[hp_index],actions[hp_index],returns[hp_index]))

    # Standardize rewards
    returns = standardize_returns(returns)

    # Calulate the loss
    loss = loss_fn(returns,probabilities)

    # Zero out gradient, backprop, and step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(probabilities)

    # Print the loss and close environement
    if epoch % clips == 0:
        env.close()

plt.plot(np.arange(0,epochs),np.array(lasted))
plt.show()
