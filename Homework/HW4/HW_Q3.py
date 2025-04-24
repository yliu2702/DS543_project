import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque

#%%
print("Question 1")
print("------------------------")
class CombLockMDP(gym.Env):
    """
    Methods:
        reset(): Resets the environment to the starting state.
        step(action): Takes a step in the environment.
    """

    def __init__(self, H=10, A=10, S=3, R=10, seed=0):
        ## Fill in here
        ## use the seed input as the random seed when generating the good actions.

        self.H=H
        self.A=A
        self.S=S
        self.R=R
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # state = {h, i}, initialize as (0,0) s_{1,1}
        self.state = (0,0)
        self.current_step = 0
        self.good_actions = self.rng.integers(low=0, high=self.A, size=self.H)

    # get one-hot encoding for state {h,i}
    def _get_obs(self):
        obs = np.zeros(self.H * self.S)
        h, i = self.state
        index = h * self.S + i
        obs[index] = 1
        return obs

    def reset(self):
        ## Fill in here
        self.current_step = 0
        self.state = (0,0)
        return self._get_obs()

    def step(self, action):
        ## Fill in here
        h,i = self.state
        done = False
        reward = 0

        if h >= self.H:
          done = True
          return self.state, reward, done, {}
        good_action = self.good_actions[h]

        if i == 2:
          # bad state
          next_i = 2
        elif action == good_action:
          next_i = self.rng.choice([0, 1])  # s_{h+1,1} or s_{h+1,2}
        else:
          # bad action, into bad state
          next_i = 2

        self.current_step += 1
        next_h = self.current_step

        # Final step
        if next_h == self.H and next_i in [0,1]:
          reward = self.R
          done = True
        elif next_h == self.H:
          done = True

        self.state = (next_h, next_i)
        if done: 
            return np.zeros(self.H * self.S), reward, done, {}
        else:
            return self._get_obs(), reward, done, {}
        
print("Question 3")
print("------------------------")
# Question 3
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device {device}.")
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_layers, num_neurons):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_size
        for _ in range(num_layers):
            # print("Layer dimension: ",(input_dim, num_neurons))
            layers.append(nn.Linear(input_dim, num_neurons))
            layers.append(nn.ReLU())
            input_dim = num_neurons
        layers.append(nn.Linear(input_dim, action_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, state):
        # print("Inside QNetwork.forward, state.shape =", state.shape)
        return self.model(state)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, num_layers, num_neurons, lr, gamma, batch_size, target_update, epsilon_decay):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.target_update = target_update
        
        self.memory = deque(maxlen=10000)
        
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
        self.memory.append(transition)  # 直接存储 transition，不用 multi-step buffer

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(device)
        # print("State shape: ", states.shape)
        # print("State: ", states)
        actions = torch.LongTensor(actions).to(device)
        # print("Action: ", actions.shape)
        rewards = torch.FloatTensor(rewards).to(device)
        # print("reward: ", rewards.shape)
        next_states = torch.FloatTensor(next_states).to(device)
        # print("Next state shape: ", next_states.shape)
        dones = torch.FloatTensor(dones).to(device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values  # 1-step return

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Train and Evaluate
def moving_average(data, window_size=100):
    if len(data) < window_size:
        return np.mean(data)
    return np.mean(data[-window_size:])

def train_dqn(agent, env, episodes):
    rewards = []
    smoothed_rewards = []
    for episode in tqdm(range(episodes), desc = "Training DQN"):
        state = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition((state, action, reward, next_state, done))
            agent.train()
            state = next_state
            total_reward += reward
            step_count += 1
        per_step_avg_reward = total_reward / step_count
        
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        rewards.append(per_step_avg_reward)
        smoothed_rewards.append(moving_average(rewards, window_size=100))

        if episode % 5000 == 0:
            print(f"Episode {episode}, Avg Step Reward: {per_step_avg_reward}, Moving Avg Reward: {smoothed_rewards[-1]:.4f}, Epsilon: {agent.epsilon:.4f}")
    return rewards, smoothed_rewards

H_values= [5,10,20]
EPSILON_DECAY = [0.995, 0.99]

for H in tqdm(H_values, desc = "H value"):
    print(f"\n=== Training on CombLockMDP with H = {H} ===")
    env = CombLockMDP(H=H)
    state_size = len(env.reset())
    # print("State_size: ", state_size, env.S)
    action_size = env.A
    # print("Action_size: ", action_size)
    
    best_auc = -np.inf
    best_params = {
        "num_layers": 4,
        "num_neurons": 256,
        "batch_size": 256,
        "lr": 1e-4,
        "gamma": 0.95,
        "epsilon_decay": random.choice(EPSILON_DECAY),
    }
    for epsilon_decay in EPSILON_DECAY:
        print(f"Testing epsilon_decay = {epsilon_decay}")
        agent = DQNAgent(
            state_size, action_size,
            best_params["num_layers"], best_params["num_neurons"],
            best_params["lr"], best_params["gamma"], best_params["batch_size"],
            target_update=10,
            epsilon_decay=epsilon_decay
        )
        rewards, smoothed_rewards = train_dqn(agent, env, episodes=100000)
        print("Smoothed reward shape:", len(smoothed_rewards))

        auc = np.trapz(smoothed_rewards)
        if auc > best_auc:
            best_auc = auc
            best_rewards = rewards
            best_smoothexszd_rewards = smoothed_rewards
            best_decay = epsilon_decay

    plt.plot(smoothed_rewards, label=f"H={H}, ε-decay={best_decay}")

plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward (smoothed)")
plt.title("DQN + ε-greedy on Combination Lock MDP")
plt.legend()
plt.grid(True)
plt.savefig("./results/Q3_comb_lock_dqn_learning_curve.png", dpi=300)
plt.show()