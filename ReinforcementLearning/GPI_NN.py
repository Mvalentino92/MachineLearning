import gym
import torch
import numpy as np

# Create the environment
env = gym.make('CartPole-v0')

# ********Create the policy network********

# Dimensions
policy_input_ = env.observation_space.shape[0]
policy_hidden1 = 128
policy_hidden2 = 64
policy_hidden3 = 32
policy_hidden4 = 16
policy_output_ = env.action_space.n

# Create the model
policy_model = torch.nn.Sequential(
        torch.nn.Linear(policy_input_,policy_hidden1),
        torch.nn.ReLU(),
        torch.nn.Linear(policy_hidden1,policy_hidden2),
        torch.nn.ReLU(),
        torch.nn.Linear(policy_hidden2,policy_hidden3),
        torch.nn.ReLU(),
        torch.nn.Linear(policy_hidden3,policy_hidden4),
        torch.nn.ReLU(),
        torch.nn.Linear(policy_hidden4,policy_output_),
        torch.nn.Softsign()
        )
     
# Optimizer for model
policy_optimizer = torch.optim.Adam(policy_model.parameters())

# *******Create the action value network*****

# Dimensions (keep the this network small, so changing actions impact don't
# diminish through network.
AV_input_ = policy_input_ + policy_output_
AV_hidden1 = 128
AV_hidden2 = 64
AV_output_ = 1

# Create the model
AV_model = torch.nn.Sequential(
        torch.nn.Linear(AV_input_,AV_hidden1),
        torch.nn.ReLU(),
        torch.nn.Linear(AV_hidden1,AV_hidden2),
        torch.nn.ReLU(),
        torch.nn.Linear(AV_hidden2,AV_output_)
        )

# Optimizer for the model
AV_optimizer = torch.optim.Adam(AV_model.parameters())

# ****** Begin to set up for training for the model *****

# Number of epochs
epochs = 1000

# Number of states observed (regardless of episodes) before updating the policy netowrk
# Total number of evals so far, and empty array to hold all the values
update_quota = 32
num_evals = 0
states = np.empty(update_quota,dtype=object)

# Learning rate for fining optimal actions using AV network
alpha = 1

# Discount for training AV model
discount = 0.95

# Loss function for both
loss_fn = torch.nn.MSELoss()

# Define clips for rendering
clips = 10

# Number of actions
num_action = env.action_space.n

# Epsilon for random actions
eps = 0.777

# Function for updating policy
def update_policy(alpha,states,policy_model,AV_model,policy_iters=100,max_iters=100):

    # Calc action just to have it
    num_action = env.action_space.n

    # List for appending features for X, and list for appending labels for y
    X = []
    y = []
    
    # For each state
    for state in states:

        # Grab the state and grab the action (detach to be safe)
        state = state.detach()
        action = torch.from_numpy(np.array([env.action_space.sample()]))

        # Combine both and set requires grad to true
        input_ = torch.cat((state,action))
        input_.requires_grad = True

        # Set val1 and val2 (want gradient for val2)
        val1 = -np.inf
        val2 = AV_model(input_)

        # Set iters
        iters = 0

        # Use gradient descent to find an optimal action (maximize)
        while val2 > val1 and iters < max_iters:

            # Compute gradient with backprop
            val2.backward()

            # Perform one iteration of gradient descent (for MAXIMIZING, and no grad!)
            with torch.no_grad():
                input_[-num_action:] += alpha*input_.grad[-num_action:]
                input_[-num_action:] = np.minimum(input_[-num_action:].detach(),env.action_space.high)
                input_[-num_action:] = np.maximum(input_[-num_action:].detach(),env.action_space.low)

            # Zero out the grad
            input_.grad.zero_()

            # Set val1 as val2
            val1 = val2

            # Recalculate val2
            val2 = AV_model(input_)

            # Increment iters
            iters += 1

        # Add the state to X (for feature matrix), and action to y (for label)
        X.append(input_[:-num_action].detach().tolist())
        y.append(input_[-num_action:].detach().tolist())

    # Create Tensors for training the policy model
    X = torch.Tensor(X)
    y = torch.Tensor(y)

    # Begin to train the policy model for policy iters
    for i in range(policy_iters):

        # Get predictions and loss
        y_predict = policy_model(X)
        loss = loss_fn(y_predict,y)

        # Zero out the gradients
        policy_model.zero_grad()

        # Backprop and step the optimizer
        loss.backward()
        policy_optimizer.step()

    

# *********Start training for the model ****************

# Number of episodes = epochs
for epoch in range(epochs):

    # Decrease epsilon
    eps *= 0.99

    # Reset the environment and get observations
    state = torch.from_numpy(env.reset()).float()

    # Choose an action using the state and policy model (don't track gradients)
    with torch.no_grad():
        action = policy_model(state)

    # Set done equal to False, (continue actions while episode has not terminated)
    done = False

    # Count how many iterations the episode lasted
    count = 0

    # Hold if ready to view
    if epoch % clips == 0:
        input('Ready to view')

    # Begin episode
    while not done:

        # If at certain point, render
        if epoch % clips == 0:
            env.render()

        # Increment count
        count += 1

        # Add this state action pair to the list and increment num_evals
        states[num_evals] = state
        num_evals += 1

        # If num_evals equals update_quota (we hit it), then reset it to 0, and update policy
        # To update we need all state action pairs, and BOTH models
        if num_evals == update_quota:
            num_evals = 0
            update_policy(alpha,states,policy_model,AV_model)

        # Random chance to sample action
        if np.random.rand() < eps:
            action = torch.from_numpy(np.array([env.action_space.sample()]))

        # Take this action, and observe the new state, reward and if done
        with torch.no_grad():
            state2, reward, done, _ = env.step(action)

        # Convert this state to a tensor
        state2 = torch.from_numpy(state2).float()

        # Select action for this state using policy model (don't track gradients
        with torch.no_grad():
            action2 = policy_model(state2)

        # Get the action value for the current state (want gradient)
        action_value = AV_model(torch.cat((state,action)))

        # Get the target value (reward + discount*action_value next state, with no grad)
        # Detach just to be safe
        with torch.no_grad():
            target = reward + discount*(torch.Tensor([0]) if done else AV_model(torch.cat((state2,action2))).detach())

        # Compute the loss
        loss = loss_fn(action_value,target)

        # Zero the gradients
        AV_model.zero_grad()

        # Back Prop
        loss.backward()

        # Step the optimizer
        AV_optimizer.step()

        # Update state and actions
        state = state2
        action = action2

    # Print how long the episode lasted
    print('Lasted',count,'iterations on epoch',epoch)

    # Close if was rendered
    #if epoch % clips == 0:
     #   env.close()


