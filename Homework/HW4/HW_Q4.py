import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque

print("Question 1")
print("------------------------")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

print("Question 4")
print("------------------------")
# Hyperparameter: alpha, the replay buffer size, learning rate
class NN(torch.nn.Module):
    def __init__(self,in_dim,out_dim,n_hid):
        super(NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid = n_hid
        
        self.fc1 = torch.nn.Linear(in_dim,n_hid,'linear')
        self.fc2 = torch.nn.Linear(n_hid,n_hid,'linear')
        self.fc3 = torch.nn.Linear(n_hid,out_dim,'linear')
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        #y = self.softmax(y)
        return y

class RND:
    def __init__(self,in_dim,out_dim,n_hid):
        self.target = NN(in_dim,out_dim,n_hid).to(device)
        self.model = NN(in_dim,out_dim,n_hid).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)
        
    def get_reward(self,x):
        x = x.to(device)
        y_true = self.target(x).detach()
        y_pred = self.model(x)
        reward = torch.pow(y_pred - y_true,2).sum()
        return reward
    
    def update(self,Ri):
        Ri.sum().backward()
        self.optimizer.step()


class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, epsilon=0.1, gamma=0.99, lr=1e-3):
        self.q_net = NN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = NN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.epsilon = epsilon
        self.gamma = gamma
        self.action_dim = action_dim
        self.update_freq = 50  
        self.step_counter = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.q_net(state)
                return q_values.argmax().item()

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter % self.update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.ptr = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

ALPHA = [0.01, 0.05, 0.1]
LEARNING_RATE = [1e-2, 1e-3, 1e-4]
CAPACITY = [5000, 8000]

best_params = {
    "alpha": random.choice(ALPHA),
    "lr": random.choice(LEARNING_RATE),
    "buffer_size_capacity": random.choice(CAPACITY),
}
param_spaces = {
    "alpha": ALPHA,
    "lr": LEARNING_RATE,
    "capacity": CAPACITY,
}
best_score = -float("inf")
best_accumulative_reward = None

for param_name, param_values in tqdm(param_spaces.items(), desc="Optimizing Hyperparameters"):
    print(f"Optimizing {param_name}...")
    for param_value in tqdm(param_values, desc=f"Tuning {param_name}"):
        temp_params = best_params.copy()
        temp_params[param_name] = param_value
# for alpha, lr, capacity in itertools.product(ALPHA, LEARNING_RATE, CAPACITY):
        env = CombLockMDP(H=10, S=3, A=10, R=10)
        in_dim = env.H * env.S
        rnd = RND(in_dim=in_dim, out_dim=16, n_hid=64)  
        ## TODO: hyperparameter: lr
        agent = DQNAgent(state_dim=env.H * env.S, action_dim=env.A, hidden_dim=64, epsilon=0.1, gamma=0.99, lr=temp_params['lr'])
        ## TODO: hyperparameter: replay_buffer_size
        replay_buffer = ReplayBuffer(capacity=temp_params['buffer_size_capacity'])
        
        all_cumulative_rewards = []
        # num_episodes = 1000
        num_episodes = 50000

        for episode in tqdm(range(num_episodes), desc = "Num of episodes"):
            obs = env.reset()
            done = False
            cumulative_reward = 0

            while not done:
                state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = agent.select_action(obs)  
                next_obs, env_reward, done, _ = env.step(action)

                rnd_reward = rnd.get_reward(state_tensor).item()
                ## TODO: hyperparameter: alpha
                total_reward = env_reward + temp_params['alpha'] * rnd_reward
                cumulative_reward += env_reward  

                replay_buffer.push(obs, action, total_reward, next_obs, done)
                rnd_input = torch.FloatTensor(obs).unsqueeze(0).to(device)
                Ri = rnd.get_reward(rnd_input)
                rnd.optimizer.zero_grad()
                rnd.update(Ri)  
                # update QNetwork
                agent.update(replay_buffer, batch_size=64)
                obs = next_obs

            all_cumulative_rewards.append(cumulative_reward)
        avg_reward = np.mean(all_cumulative_rewards[-100:])
        print(f"alpha={temp_params['alpha']}, lr={temp_params['lr']}, capacity={temp_params['buffer_size_capacity']}, avg_reward={avg_reward:.2f}")

        if avg_reward > best_score:
            best_score = avg_reward
            best_params[param_name] = param_value
            best_accumulative_reward = all_cumulative_rewards

print("Best param: ",best_params, "whose last_100_avg_reward: ", best_score)
plt.plot(best_accumulative_reward)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("DQN + RND on CombLockMDP")
plt.savefig("./results/Q4_DQN_plus_RND_on_CombLockMDP.png")
plt.show()
