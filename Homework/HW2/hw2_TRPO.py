#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import json
import gymnasium as gym
from gymnasium import spaces

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

    def reset(self, seed =None):
        super().reset(seed=seed)
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
env.reset(seed = 42)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(3, 128)
        self.affine2 = nn.Linear(128, 2)
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to('cuda:3')
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

def conjugate_gradient(Ax, b, tol=1e-10, max_steps=10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    print("p: ", p)
    for i in range(max_steps):
        Ap = Ax(p)
        alpha = rdotr / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

def fisher_vector_product(policy, states, p):
    kl = compute_kl(policy, states)
    kl_grad = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
    klp = (kl_grad_vector * p).sum()
    klp_grad = torch.autograd.grad(klp, policy.parameters())
    return torch.cat([grad.contiguous().view(-1) for grad in klp_grad])

def compute_kl(policy, states):
    old_probs = policy.forward(states).detach()
    new_probs = policy.forward(states)
    kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(dim=1).mean()
    return kl

def trpo_update(policy, optimizer, states, actions, returns, damping=0.1, max_kl=0.01):
    probs = policy.forward(states)
    m = Categorical(probs)
    log_probs = m.log_prob(actions)
    
    loss = -torch.mean(log_probs * returns)
    
    grads = torch.autograd.grad(loss, policy.parameters(), create_graph=True)
    grads_vector = torch.cat([grad.view(-1) for grad in grads])
    
    def Ax(p):
        return fisher_vector_product(policy, states, p) + damping * p
    
    stepdir = conjugate_gradient(Ax, grads_vector)
    
    shs = 0.5 * torch.dot(stepdir, Ax(stepdir).clone())
    step_size = torch.sqrt(2 * max_kl / (shs + 1e-8))
    
    new_params = torch.cat([p.view(-1) for p in policy.parameters()]) + step_size * stepdir
    
    offset = 0
    with torch.no_grad():
        for param in policy.parameters():
            param.copy_(new_params[offset:offset + param.numel()].view(param.size()))
            offset += param.numel()

def training_loop(lr, gamma):
    policy = Policy().to('cuda:3')
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    smoothed_reward = 0
    ep_rewards = []
    smoothed_rewards = []
    
    for i_episode in range(10000):
        state = env.reset(seed=42)
        state = torch.tensor(state, dtype=torch.float32, device='cuda:3')

        ep_reward = 0
        done = False
        states, actions, rewards = [], [], []
        
        while not done:
            action = policy.select_action(state.cpu().numpy())
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(torch.tensor(action, dtype=torch.int64, device='cuda:3'))
            rewards.append(torch.tensor(reward, dtype=torch.float32, device='cuda:3'))
            
            state = torch.tensor(next_state, dtype=torch.float32, device='cuda:3')
            ep_reward += reward
            
        returns = deque()
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns, dtype=torch.float32, device='cuda:3')
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        
        trpo_update(policy, optimizer, states, actions, returns)
        
        ep_rewards.append(ep_reward)
        smoothed_reward = 0.05 * ep_reward + (1 - 0.05) * smoothed_reward
        smoothed_rewards.append(smoothed_reward)
    
    return smoothed_rewards


#%%
LEARNING_RATE = [1e-3, 1e-4, 1e-5]
BATCH_SIZE = [128, 256, 512]
DISCOUNT_FACTOR = [0.9, 0.95, 0.99]

smoothed_rewards = training_loop(1e-3, 0.9)
# %%
