#%%
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
import time
import random

import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from itertools import product
from tqdm import tqdm
import json
#%%
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed()
#%%
# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device {device}.")
#%%
class NaiveBlackjackEnv(gym.Env):
    def __init__(self):
      super(NaiveBlackjackEnv, self).__init__()
      self.observation_space = spaces.Tuple(
                                (spaces.Discrete(32), # P: total value of the player’s hand # 32?
                                  spaces.Discrete(2), # A: binary variable, whether player has an active Ace
                                  spaces.Discrete(11)  # D: Dealer's Visible card (1 to 10, where 1 is Ace)
                                  ))
      self.action_space = gym.spaces.Discrete(2) # 0: Stand, 1: Hit 
      # Card deck (ignoring suits)
      self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  # Ace card represents as 1
      self.reset()

    def reset(self):
        # super().reset(seed=42)
        card1 = np.random.choice(self.deck)
        self.player_total = card1
        self.player_ace = int(card1 == 1)  # Check if an Ace is counted as 11
        self.dealer_card = np.random.choice(self.deck)
        self.done = False
        return self._get_obs()

    def _get_obs(self):
      return (self.player_total, self.player_ace, self.dealer_card)

    def step(self, action):
      if self.done:
        return self._get_obs(), 0, True, {}
      if action == 1:
        new_card = np.random.choice(self.deck)
        if new_card == 1 and self.player_ace ==0 and self.player_total <= 10 :
          self.player_total += 11
          self.player_ace = 1
        else:
          self.player_total += new_card

        if self.player_total > 21 and self.player_ace == 1:
          self.player_total -= 10
          self.player_ace = 0
        if self.player_total > 21 and self.player_ace == 0:
          self.done = True
          # return new state, reward, terminal status
          # if player bust, return =-1
          return self._get_obs(), -1, self.done, {} 
        # continue the game
        return self._get_obs(), 0, self.done, {}
      else:
        dealer_total = self.dealer_card + np.random.choice(self.deck)
        while dealer_total < 17:
          dealer_total += np.random.choice(self.deck)
          if dealer_total > 21:
            # Dealer bust, player didn't player earn reward 1
            return self._get_obs(), 1, True, {} 
        self.done = True
        # neither bust
        if self.player_total > dealer_total:
          return self._get_obs(), 1, self.done, {}
        else:
          return self._get_obs(), -1, self.done, {}
#%%
env = NaiveBlackjackEnv()
#%%
# n_episodes = 1000
# total_reward = 0

# for _ in range(n_episodes):
#     state = env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()  # Random action
#         state, reward, done, _ = env.step(action)
#         total_reward += reward

# avg_reward = total_reward / n_episodes
# print(f"Average Reward over {n_episodes} episodes: {avg_reward}")
#%%
# Multi-step Return DQN
NUM_LAYERS = [2, 3, 4]  # Number of layers in the network
NUM_NEURONS = [64, 128, 256]  # Number of neurons per layer
BATCH_SIZE = [128, 256, 512]  # Batch size
LEARNING_RATE = [1e-3, 1e-4, 1e-5]  # Learning rate
DISCOUNT_FACTOR = [0.9, 0.95, 0.99]  # Discount factor
MULTI_STEP = [3, 5, 10]  # Number of steps for multi-step returns
EPSILON_DECAY = [0.99, 0.995, 0.999]  # Exploration decay rate

# Define Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_layers, num_neurons):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_size
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, num_neurons))
            layers.append(nn.ReLU())
            input_dim = num_neurons
        layers.append(nn.Linear(input_dim, action_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.model(state)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, num_layers, num_neurons, lr, gamma, batch_size, target_update, epsilon_decay, multi_step):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.target_update = target_update
        self.multi_step = multi_step
        self.memory = deque(maxlen=10000)
        self.n_step_buffer = deque(maxlen=multi_step)
        
        self.q_network = QNetwork(state_size, action_size, num_layers, num_neurons).to(device)
        self.target_network = QNetwork(state_size, action_size, num_layers, num_neurons).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.q_network(state)).item()
    
    def store_transition(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) == self.multi_step:
            state, action, reward, next_state, done = self.n_step_buffer[0]
            for i in range(1, self.multi_step):
                reward += (self.gamma ** i) * self.n_step_buffer[i][2]
                next_state = self.n_step_buffer[i][3]
                done = self.n_step_buffer[i][4]
            self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.multi_step) * max_next_q_values # multi-step return
        
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Train and Evaluate
# Simple Moving Average
def moving_average(data, window_size=100):
    if len(data) < window_size:
        return np.mean(data)
    return np.mean(data[-window_size:])

def train_dqn(agent, env, episodes):
    rewards = []
    smoothed_rewards = []
    for episode in range(episodes):
        state = flatten_state(env.reset())
        total_reward = 0
        step_count = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor).squeeze(0)
            if np.random.rand() < agent.epsilon:
                slate = random.sample(range(env.num_candidates), env.slate_size)
            else:
                slate = torch.topk(q_values, k=env.slate_size).indices.tolist()

            next_state_dict, reward, done, _ = env.step(slate)
            next_state = flatten_state(next_state_dict)

            # For training, we can pick the top-1 as a representative action (approximation)
            main_action = slate[0]
            agent.store_transition((state, main_action, reward, next_state, done))
            agent.train()

            state = next_state
            total_reward += reward
            step_count += 1

        per_step_avg_reward = total_reward / max(1, step_count)

        if episode % agent.target_update == 0:
            agent.update_target_network()

        rewards.append(per_step_avg_reward)
        smoothed_rewards.append(moving_average(rewards, window_size=100))

        if episode % 100 == 0:
            print(f"Episode {episode}, Avg Step Reward: {per_step_avg_reward:.4f}, Moving Avg Reward: {smoothed_rewards[-1]:.4f}, Epsilon: {agent.epsilon:.4f}")

    return rewards, smoothed_rewards


# Initialize hyperparameter
best_params = {
    "num_layers": random.choice(NUM_LAYERS),
    "num_neurons": random.choice(NUM_NEURONS),
    "batch_size": random.choice(BATCH_SIZE),
    "lr": random.choice(LEARNING_RATE),
    "gamma": random.choice(DISCOUNT_FACTOR),
    "multi_step": random.choice(MULTI_STEP),
    "epsilon_decay": random.choice(EPSILON_DECAY),
}

# Store best results
# best_params = None
best_agent = None
best_auc = -np.inf
best_reward = -np.inf

env = NaiveBlackjackEnv()
state_size = len(env.observation_space)
action_size = env.action_space.n

num_iterations = 5
for iteration in tqdm(range(num_iterations), desc = "Coordinate Descent Hyperparameters tuning", total = num_iterations):
   print(f"\n===== Coordinate Descent Iteration {iteration + 1} =====")
   for param_name, param_values in tqdm([
        ("num_layers", NUM_LAYERS),
        ("num_neurons", NUM_NEURONS),
        ("batch_size", BATCH_SIZE),
        ("lr", LEARNING_RATE),
        ("gamma", DISCOUNT_FACTOR),
        ("multi_step", MULTI_STEP),
        ("epsilon_decay", EPSILON_DECAY),
    ], desc = "Optimizing Hyperparameter", total = 7):
        print(f"Optimizing {param_name}...")

        for param_value in tqdm(param_values, desc=f"Tuning {param_name}"):
            # Test current params
            temp_params = best_params.copy()
            temp_params[param_name] = param_value

            # Train new model
            agent = DQNAgent(
                state_size, action_size, 
                temp_params["num_layers"], temp_params["num_neurons"], 
                temp_params["lr"], temp_params["gamma"], temp_params["batch_size"], 
                10, temp_params["epsilon_decay"], temp_params["multi_step"]
            )
            rewards, smoothed_rewards = train_dqn(agent, env, 10000)  
            auc = np.trapz(rewards)

            if auc > best_auc:
                best_agent = agent
                best_rewards = rewards
                best_smoothed_rewards = smoothed_rewards
                best_params[param_name] = param_value
                best_auc = auc
                print(f"Updated best {param_name} -> {param_value} (AUC: {best_auc})")

# Plot rewards
save_figure_path = "./HW_figure"
save_result_path = "./results"

plt.plot(best_smoothed_rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Multistep Return DQN Best Episodes reward')

plt.savefig(os.path.join(save_figure_path,'Multistep_return_best_dqn_performance.png'), dpi=300, bbox_inches='tight')
plt.show()

results_dict = {
    "best_avg_step_reward": np.mean(best_reward),
    "best_avg_smoothed_reward": np.mean(best_smoothed_rewards),
    "reward_list": best_reward,  
    "smoothed_reward_list": best_smoothed_rewards,  
    "auc": best_auc,
    "best_parameters": best_params
}
print("Save element in json!")

result_file = os.path.join(save_result_path,"multistep_return_results.txt")
with open(result_file, "w") as f:
    json.dump(results_dict, f, indent = 4)
print(f"Results saved in {result_file}")

# with open(os.path.join(save_result_path,"multistep_return_results.txt"), "a") as f:
#     f.write("="*30 + "\n")  
#     f.write(f"Best Avg step reward: {np.mean(best_reward)}\n")
#     f.write(f"Best Avg smoothed reward: {np.mean(best_smoothed_rewards)}\n")
#     f.write(f"AUC: {best_auc}\n")
#     f.write(f"Best parameters: {best_params}\n\n")