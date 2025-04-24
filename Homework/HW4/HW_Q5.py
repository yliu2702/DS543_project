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

print("Question 5")
print("------------------------")
H_values = [20]
num_episodes = 1000  
results_dict = {H: {"UCBVI": [], "DQN+ε-greedy": [], "DQN+RND": []} for H in H_values}

# ------------------ Utils ------------------
def decode_state(state_vec, H, S):
    index = int(np.argmax(state_vec))
    h = index // S
    i = index % S
    return (h, i)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------ UCBVI ------------------
class UCBVI:
    def __init__(self, H=10, A=10, alpha=1.0):
        self.H, self.A, self.alpha = H, A, alpha
        self.N_sa = defaultdict(int)
        self.N_sas = defaultdict(int)
        self.R_hat = defaultdict(float)
        self.P_hat = defaultdict(lambda: defaultdict(float))
        self.Q = defaultdict(lambda: np.zeros(A))
        self.V = defaultdict(float)

    def update(self, state, action, next_state, reward, done):
        h, s = decode_state(state, self.H, 3)
        _, s_next = decode_state(next_state, self.H, 3)
        self.N_sa[(h, s, action)] += 1
        self.N_sas[(h, s, action, s_next)] += 1
        self.R_hat[(h, s, action)] = reward
        for s_prime in range(3):
            self.P_hat[(h, s, action)][s_prime] = self.N_sas[(h, s, action, s_prime)] / self.N_sa[(h, s, action)]
        for h_iter in reversed(range(self.H)):
            for s_iter in range(3):
                for a_iter in range(self.A):
                    n_sa = max(1, self.N_sa[(h_iter, s_iter, a_iter)])
                    bonus = self.alpha * np.sqrt(1 / n_sa)
                    expected_V = sum(self.P_hat[(h_iter, s_iter, a_iter)].get(s_p, 0) * self.V.get((h_iter + 1, s_p), 0) for s_p in range(3))
                    self.Q[(h_iter, s_iter)][a_iter] = min(self.R_hat[(h_iter, s_iter, a_iter)] + bonus + expected_V, self.H)
                self.V[(h_iter, s_iter)] = np.max(self.Q[(h_iter, s_iter)])

    def action(self, state):
        h, s = decode_state(state, self.H, 3)
        return int(np.argmax(self.Q[(h, s)]))

# ------------------ DQN + ε-greedy ------------------
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_size)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size, epsilon_decay=0.99):
        self.q_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 256
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.target_net.load_state_dict(self.q_net.state_dict())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.q_net.net[-1].out_features - 1)
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(state).argmax().item()

    def store(self, transition):
        self.memory.append(transition)

    def train(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, d = zip(*batch)
        s, a, r, s_, d = map(lambda x: torch.tensor(x).float().to(device), (s, a, r, s_, d))
        a = a.long()
        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            max_q_next = self.target_net(s_).max(1)[0]
            target = r + (1 - d) * self.gamma * max_q_next
        loss = self.criterion(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# ------------------ DQN + RND ------------------
class NN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class RND:
    def __init__(self, in_dim, out_dim=16):
        self.target = NN(in_dim, out_dim)
        self.model = NN(in_dim, out_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
    def get_reward(self, x):
        with torch.no_grad(): y_true = self.target(x)
        y_pred = self.model(x)
        return ((y_pred - y_true) ** 2).sum()
    def update(self, x):
        reward = self.get_reward(x)
        self.optimizer.zero_grad()
        reward.backward()
        self.optimizer.step()

class ReplayBuffer:
    def __init__(self, cap=10000):
        self.buffer = deque(maxlen=cap)
    def push(self, *args):
        self.buffer.append(args)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
    def __len__(self):
        return len(self.buffer)

# ------------------ main ------------------
def smooth(x, k=5000):
    return np.convolve(x, np.ones(k)/k, mode='valid')

H_list = [5, 10, 20]

for H in H_list:
    env = CombLockMDP(H=H)
    state_size, action_size = len(env.reset()), env.A

    # --- UCBVI ---
    ucbvi = UCBVI(H=H, A=action_size, alpha=1.0)
    ucbvi_rewards = []
    for ep in tqdm(range(2000), desc=f"UCBVI H={H}"):
        s, total = env.reset(), 0
        for _ in range(H):
            a = ucbvi.action(s)
            s_, r, d, _ = env.step(a)
            total += r
            ucbvi.update(s, a, s_, r, d)
            s = s_
            if d: break
        ucbvi_rewards.append(total)

    # --- DQN ---
    dqn = DQNAgent(state_size, action_size, epsilon_decay=0.99)
    dqn_rewards = []
    for ep in tqdm(range(100000), desc=f"DQN H={H}"):
        s, total = env.reset(), 0
        while True:
            a = dqn.act(s)
            s_, r, d, _ = env.step(a)
            dqn.store((s, a, r, s_, float(d)))
            dqn.train()
            s = s_; total += r
            if d: break
        if ep % 10 == 0: dqn.update_target()
        dqn_rewards.append(total)

    # --- DQN + RND ---
    rnd_agent = DQNAgent(state_size, action_size)
    rnd = RND(state_size)
    buffer = ReplayBuffer()
    rnd_rewards = []
    for ep in tqdm(range(100000), desc=f"DQN+RND H={H}"):
        s, total = env.reset(), 0
        while True:
            s_tensor = torch.FloatTensor(s).unsqueeze(0)
            a = rnd_agent.act(s)
            s_, r, d, _ = env.step(a)
            intrinsic = rnd.get_reward(s_tensor).item()
            total_reward = r + 0.1 * intrinsic
            buffer.push(s, a, total_reward, s_, float(d))
            rnd.update(s_tensor)
            rnd_agent.train()
            s = s_; total += r
            if d: break
        rnd_rewards.append(total)

    # --- Plot ---
    plt.figure(figsize=(10,6))
    plt.plot(smooth(ucbvi_rewards), label='UCBVI')
    plt.plot(smooth(dqn_rewards), label='DQN + ε-greedy')
    plt.plot(smooth(rnd_rewards), label='DQN + RND')
    plt.title(f'Cumulative Reward Comparison (H = {H})')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./results/Q5_compare_all_methods_H_{H}_1.png", dpi=300)
    plt.show()