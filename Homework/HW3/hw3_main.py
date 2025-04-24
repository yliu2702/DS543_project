#%%
import os
# %matplotlib inline
# pip install "gymnasium[mujoco]"
#%%
import gymnasium as gym
from typing import Optional, Tuple, Union
from gymnasium import logger, spaces

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time

#%%
# %cd /projectnb/vkolagrp/yiliu/hrandomw
file_path = "./DS543_HW3/HalfCheetah_expert_data.pkl"

with open(file_path, "rb") as f:
    expert_data = pickle.load(f)[0]
print(expert_data.keys())
print("number of data:", len(expert_data['observation']))

# Extract expert states and actions
states = torch.tensor(expert_data["observation"], dtype=torch.float32)
actions = torch.tensor(expert_data["action"], dtype=torch.float32)
#%%
def plot_loss(num_epochs, loss_list, algo_name):
    plt.title("Training loss vs. Epochs")
    plt.plot(range(num_epochs), loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.show()
    plt.savefig(f"./DS543_HW3/{algo_name}_training_loss.png")
#%%
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize policy network
env = gym.make("HalfCheetah-v5")
state_dim = states.shape[1]
action_dim = actions.shape[1]
policy_bc = PolicyNet(state_dim, action_dim)

# Define the evaluate_policy function
def evaluate_policy(policy, env, episodes=10):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        for i in range(1000):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = policy(state_tensor).detach().numpy()[0]
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            if done:
              break
        total_rewards.append(episode_reward)
    print(f"Evaluation Results: Mean Reward = {np.mean(total_rewards):.2f}, Std Reward = {np.std(total_rewards):.2f}")
    return np.mean(total_rewards), np.std(total_rewards)

mean_reward, std_reward = evaluate_policy(policy_bc, env)

#%%
# Implement the Behavior cloning algorithm to train policy_bc to imitate on the given expert data

# Define loss function and optimizer
criterion = nn.MSELoss() 
optimizer = optim.Adam(policy_bc.parameters(), lr=1e-3)

# num_epochs = 50
# batch_size = 64
#%%
def train_bc(states, actions, policy_bc, num_epochs = 50, batch_size =64):
    dataset = torch.utils.data.TensorDataset(states, actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_list = []
    for epoch in tqdm(range(num_epochs), desc = "Training Epochs"):
        epoch_loss = 0
        for obs_batch, act_batch in dataloader:
            optimizer.zero_grad()
            predicted_actions = policy_bc(obs_batch)  # Forward pass
            loss = criterion(predicted_actions, act_batch)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss/len(dataloader)
        loss_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Behavior Cloning Training Completed!")
    return loss_list

# loss_list = train_bc(states, actions, policy_bc, num_epochs = 70, batch_size =64)
# plot_loss(70, loss_list, "bc")
#%%
# Save trained policy
# mean_reward, std_reward = evaluate_policy(policy_bc, env)
# policy_path = "./DS543_HW3/bc_policy.pth"
# torch.save(policy_bc.state_dict(), policy_path)
# print(f"Trained policy saved at {policy_path}")
#%%
# Implement the dagger algorithm that train policy_dagger to imitate on the policy_bc you just trained above.
# Initialize policy_dagger
policy_dagger = PolicyNet(state_dim, action_dim)
# Define loss function and optimizer
optimizer = optim.Adam(policy_dagger.parameters(), lr=1e-3)
criterion = nn.MSELoss() 

def train_dagger(env, policy_bc, policy_dagger, num_dagger_iters=10, batch_size = 64, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_bc.to(device)
    policy_dagger.to(device)

    loss_history = []
    start_time = time.time()
    # copy policy_bc weights so that dagger starts from bc's initial knowledge
    policy_dagger.load_state_dict(policy_bc.state_dict())
    time_1 = time.time()
    print(f"Copy policy_bc parameters to policy dagger in {(time_1-start_time):.2f} seconds.")

    for i in tqdm(range(num_dagger_iters), desc = "Dagger iterations"):
        start_iteration_time = time.time()
        print(f"Dagger Iteration {i+1}/{num_dagger_iters}")
        states, expert_actions = [], []
        state = env.reset()[0]
        done = False
        step = 0
        max_steps = 500

        
        while not done and step < max_steps:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = policy_dagger(state_tensor).detach().cpu().numpy().squeeze()
            # time_2 = time.time()
            # print(f"Agent selected action in {(time_2 - start_iteration_time):.2f} seconds.")
            states.append(state)
            
            expert_action = policy_bc(state_tensor).detach().cpu().numpy().squeeze()
            expert_actions.append(expert_action) 
            # time_3 = time.time()
            # print(f"Query expert action in {(time_3 - time_2):.2f} seconds.")
            # take action in the env
            state, reward, done, _, _ = env.step(action)
            step += 1
        # convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        expert_actions = torch.tensor(np.array(expert_actions), dtype=torch.float32).to(device)
        time_done = time.time()
        print(f"Finally done in {(time_done - start_iteration_time):.2f} seconds.")

        if i==0:
            dataset_states, dataset_actions = states, expert_actions
        else:
            dataset_states = torch.cat((dataset_states, states), dim=0)
            dataset_actions = torch.cat((dataset_actions, expert_actions), dim=0)
        time_4 = time.time()
        print(f"Successfully create datasets in {(time_4 - start_time):.2f} seconds.")

        # Train dagger policy on the aggregated dataset
        dataset = torch.utils.data.TensorDataset(dataset_states, dataset_actions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # store loss per epoch
        loss_epoch = []
        for epoch in tqdm(range(num_epochs), desc = "Training Epochs for Dagger"):
            start_epoch_time = time.time()
            epoch_loss = 0
            for state_batch, act_batch in dataloader:
                state_batch = state_batch.to(device)
                act_batch = act_batch.to(device)

                optimizer.zero_grad()
                predicted_actions = policy_dagger(state_batch)  # Forward pass
                loss = criterion(predicted_actions, act_batch)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update parameters
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss/len(dataloader)
            loss_epoch.append(avg_loss)
            end_epoch_time = time.time()
            print(f"Getting loss for one epoche in {(end_epoch_time- start_epoch_time):.2f} seconds. ")
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        loss_history.append(loss_epoch[-1])

    print("Dagger Training Completed!")
    return loss_history
#%%
loss_history = train_dagger(env, policy_bc, policy_dagger,num_dagger_iters=10, batch_size = 64, num_epochs=50)
plot_loss(50, loss_history, "dagger")
# %%
mean_reward, std_reward = evaluate_policy(policy_dagger, env)
policy_path = "./DS543_HW3/dagger_policy.pth"
torch.save(policy_dagger.state_dict(), policy_path)
print(f"Trained policy saved at {policy_path}")
#%%