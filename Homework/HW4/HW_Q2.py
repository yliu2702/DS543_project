#%%
# %cd /projectnb/vkolagrp/yiliu/hrandomw/HW4
#%%
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
# %%
print("Question 2")
print("------------------------")
## Problem 2 (20 Points)
# Current MDP use one-hot state -> decode to (h,i)
def decode_state(state_vec, H, S):
    """将 one-hot 状态向量解码为 (h, i) 的 tuple"""
    index = int(np.argmax(state_vec))
    h = index // S
    i = index % S
    return (h, i)

class UCBVI(object):
    def __init__(self, H=10, A=10, alpha=1):
        ## fill in other local parameters here
        self.H = H
        self.A = A
        self.alpha = alpha
        self.N_sa = defaultdict(int)
        self.N_sas = defaultdict(int)
        self.R_hat = defaultdict(float)
        # 状态转移估计
        self.P_hat = defaultdict(lambda: defaultdict(float)) 
        self.Q = defaultdict(lambda: np.zeros(A))  # Q[h][s][a]
        self.V = defaultdict(float)  # V[h][s]

    def update(self, state, action, next_state, reward, done):
        ## fill in the local updates of UCBVI, including counts, \hat{P} and reward bonus and value iteration,
        ## for which you can keep track of a Q table Q_h(s,a).
        h, s = decode_state(state, self.H, 3)
        h_next, s_next = decode_state(next_state, self.H, 3)
        self.N_sa[(h, s, action)] += 1
        self.N_sas[(h, s, action, s_next)] += 1
        self.R_hat[(h, s, action)] = reward

        # 更新状态转移估计
        for s_prime in range(3):
          self.P_hat[(h, s, action)][s_prime] = self.N_sas[(h, s, action, s_prime)] / self.N_sa[(h, s, action)]
        
        # Value iteration, update Q and V
        for h_iter in reversed(range(self.H)):
            for s_iter in range(3):  # 状态编号
                for a_iter in range(self.A):
                  n_sa = max(1, self.N_sa[(h_iter, s_iter, a_iter)])
                  bonus = self.alpha * np.sqrt(1 / n_sa)
                  expected_V = 0
                  for s_prime in range(3):
                      prob = self.P_hat[(h_iter, s_iter, a_iter)].get(s_prime, 0)
                      expected_V += prob * self.V.get((h_iter + 1, s_prime), 0)

                  q_val = self.R_hat[(h_iter, s_iter, a_iter)] + bonus + expected_V
                  self.Q[(h_iter, s_iter)][a_iter] = min(q_val, self.H)  # clip 到最大值 H

                # 更新 V
                self.V[(h_iter, s_iter)] = np.max(self.Q[(h_iter, s_iter)])

    def action(self, state):
        h, s = decode_state(state, self.H, 3)
        q_vals = self.Q[(h, s)]
        return int(np.argmax(q_vals))

num_episodes = 1000
H_values = [5, 10, 20]
alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0] 
best_results = {}

plt.figure(figsize=(10,6))

for H in H_values:
  best_total_reward = None
  best_alpha = None
  for alpha in alpha_values:
      env = CombLockMDP(H=H, seed=42)
      agent = UCBVI(H=H, alpha=alpha)
      total_rewards = np.zeros(num_episodes)

      for i_episode in tqdm(range(num_episodes), desc = f"Traning for H={H}"):
          state = env.reset()
          for h in range(env.H):
              action = agent.action(state)
              next_state, reward, done, _ = env.step(action)
              total_rewards[i_episode] += reward
              agent.update(state,action,next_state,reward, done)
              state = next_state
              if done:
                  break
      if best_total_reward is None or total_rewards.sum() > best_total_rewards.sum():
          best_total_rewards = total_rewards
          best_alpha = alpha

      # cumulative_rewards = np.cumsum(total_rewards)
      smoothed = np.convolve(best_total_rewards, np.ones(5000)/5000, mode='valid')
      plt.plot(smoothed, label=f'H = {H}, alpha = {alpha}')

plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.title('Training Performance of UCBVI')
plt.legend()
plt.grid(True)
plt.savefig("./results/Q2_Training_Performance_of_UCBVI_cumulative.png")
plt.show()    
#%%

