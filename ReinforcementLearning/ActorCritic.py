# All necessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import gym

# First create the enviroment, so we can use info for create networks
env = gym.make('BipedalWalker-v3')

# ********* Create the Actor class and build model for network ***********

# Set of layers sizes for network
actor_input = env.observation_space.shape[0]
actor_hidden1 = 60
actor_hidden2 = 40
actor_hidden3 = 60
actor_output = env.action_space.shape[0]

# Define the class, just init and forward
class Actor(nn.Module):

    # Initialize, call super then build layers (named by input layer)
    def __init__(self):
        super(Actor,self).__init__()
        self.input_ = nn.Linear(actor_input,actor_hidden1)
        self.hidden1 = nn.Linear(actor_hidden1,actor_hidden2)
        self.hidden2 = nn.Linear(actor_hidden2,actor_hidden3)
        self.hidden3 = nn.Linear(actor_hidden3,actor_output)

    # Define forward (get backprop for free!)
    # Last activation function is softsign, because it's between -1 and 1
    def forward(self,x):
        x = F.relu(self.input_(x))
        x = F.leaky_relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        return x

# Create the model and define the optimizer
actor_model = Actor()
actor_lr = 0.00009
actor_optimizer = torch.optim.Adam(actor_model.parameters(),lr=actor_lr)

# ********* Create the Critic class and build model for network ***********

# Set of layers sizes for network
critic_input = env.observation_space.shape[0]
critic_hidden1 = 64
critic_hidden2 = 32
critic_output = 1

# Define the class, just init and forward
class Critic(nn.Module):

    # Initialize, call super then build layers (named by input layer)
    def __init__(self):
        super(Critic,self).__init__()
        self.input_ = nn.Linear(critic_input,critic_hidden1)
        self.hidden1 = nn.Linear(critic_hidden1,critic_hidden2)
        self.hidden2 = nn.Linear(critic_hidden2,critic_output)

    # Define forward (get backprop for free!)
    # Last activation function is just simply linear (all values)
    def forward(self,x):
        x = F.relu(self.input_(x))
        x = F.relu(self.hidden1(x))
        x = self.hidden2(x)
        return x

# Create the model and define the optimizer
critic_model = Critic()
critic_lr = 0.0009
critic_optimizer = torch.optim.Adam(critic_model.parameters(),lr=critic_lr)

# Loss function for actor
def actor_loss_fn(means,actions,returns,values):
    
    # Get difference between means and actions squared
    diff_sqrd = torch.abs(means - actions)

    # Sum along columns
    sum_vector = torch.sum(diff_sqrd,axis=1).reshape(-1,1)

    # Get advantage
    advantage = returns - values

    # Return sum of multiplication
    return torch.sum(sum_vector*advantage)

# Loss function for critic
critic_loss_fn = nn.MSELoss()

# Accepts means and stds, and returns actions from gaussian 
def get_actions(means,stds):
    
    # Simply pass to function and turn to np.array (get within bounds though!)
    actions = np.array(random.gauss(means,stds))
    actions = np.maximum(actions,env.action_space.low)
    actions = np.minimum(actions,env.action_space.high)
    return actions

# Gets the variable n-step returns for all states
def get_returns(rewards,state_prime_value,gamma,min_steps):

    # Set R what it needs to be before even hitting min_steps
    R = np.sum(rewards[-min_steps:] * gamma**np.arange(min_steps)) + state_prime_value * gamma**min_steps
    returns = [R]
    
    # Go in reverse order from min_steps, and keep adding to R and returns
    for i in range(len(rewards)-min_steps-1,-1,-1):

        # Add to R 
        R = rewards[i] + gamma*R

        # Append new R to returns
        returns.append(R)

    # Return as list
    return returns[::-1]

# Standardize returns
def standardize_returns(returns):

    # Get mean and STD
    mean = returns.mean()
    std = returns.std()

    # Return standardized
    return (returns - mean)/std


# Define variables for running episodes
epochs = 2500
stds = np.full(env.action_space.shape[0],1.0)
min_stds = np.full(stds.size,0.05)
deltas = (stds - min_stds)/epochs
gamma = 0.99
clips = 50
critic_minimizer = 1/3
max_steps = 73
min_steps = 10
upper_bounds_steps = max_steps
lower_bounds_steps = min_steps
avg_episode_length = 0
train_times = 10

# Begin to train on epochs
for epoch in range(epochs):

    # Reset get state
    state = env.reset()

    # Set done to False
    done = False

    # Need lists for states,rewards, and chosen actions
    states = []
    rewards = []
    actions = []

    # Tracks the number of samples in batch
    num_samples = 0

    # Reset the batch_size each time
    batch_size = max_steps - min_steps + 1

    # Tracks number of turns the episode lasted to adjust max_steps
    episode_length = 0

    # Track times trained this epoch
    trained = 0

    # While we havent terminated this episode
    while not done:

        # Increase episode length
        episode_length += 1

        # Render if applicable
        if epoch % clips == 0:
            env.render()

        # If we have reached the max number of samples, do batch training
        if num_samples == max_steps:

            # Change trained to true
            trained += 1

            # Grab componenets we need for loss function (will only have max_steps - min_steps + 1 amount of them)
            mean_matrix = actor_model(torch.Tensor(states[0:batch_size]).float())
            action_matrix = torch.Tensor(actions[0:batch_size]).float()
            with torch.no_grad(): # No gradients for critic, only update with respect to actor
                state_value_vector = critic_model(torch.Tensor(states[0:batch_size]).float()).reshape(-1,1)
                state_prime_value = critic_model(torch.Tensor([state_prime]).float())

            # Calculate the total returns and convert into Tensor and reshape
            returns_vector = standardize_returns(torch.Tensor(get_returns(rewards,state_prime_value,gamma,min_steps)).reshape(-1,1))

            # Perform all actions for actor
            actor_optimizer.zero_grad()
            actor_loss = actor_loss_fn(mean_matrix,action_matrix,returns_vector,state_value_vector)
            actor_loss.backward()
            actor_optimizer.step()

            # Perform all actions for critic (recompute state_value_vector with gradients)
            state_value_vector = critic_model(torch.Tensor(states[0:batch_size]).float()).reshape(-1,1)
            critic_optimizer.zero_grad()
            critic_loss = critic_minimizer*critic_loss_fn(state_value_vector,returns_vector)
            critic_loss.backward()
            critic_optimizer.step()

            # Clip all the vectors, at min_steps and reset num_samples
            states = states[-min_steps:]
            rewards = rewards[-min_steps:]
            actions = actions[-min_steps:]
            num_samples = min_steps

        # Run the state through the Actor model to get vector of means
        # (No gradient)
        with torch.no_grad():
            means = actor_model(torch.from_numpy(state).float())

        # Get the actions by running these means through a normal distribution sample
        actions_ = get_actions(means,stds)

        # Take these actions and observe
        state_prime,reward,done,_ = env.step(actions_)

        # Add these all to their respective lists
        states.append(state)
        rewards.append(reward)
        actions.append(actions_)

        # Set state as state_prime
        state = state_prime

        # Increase number of samples
        num_samples += 1

    # *** Episode has ended *** 
    # Decrease stds by deltas
    stds -= deltas

    # Change max_steps based on episode length
    avg_episode_length += int(0.5*(episode_length - avg_episode_length))
    max_steps = np.minimum(np.maximum(int((avg_episode_length - min_steps)/train_times + min_steps),lower_bounds_steps),upper_bounds_steps)

    # Print if trained this epoch
    print('Epoch:',epoch,'\tTimes trained:',trained,'\tBatch size:',batch_size,'\tstd:',stds[0],'\tMeans:',means)

    # Close if was rendered
    if epoch % clips == 0:
        print('Episode:',epoch)
        env.close()
