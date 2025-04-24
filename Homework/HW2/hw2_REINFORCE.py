#%%
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import random
from itertools import count
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple, deque
from tqdm import tqdm
from sklearn.metrics import auc
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
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else
#     "mps" if torch.backends.mps.is_available() else
#     "cpu"
# )
# print(f"Using device {device}.")
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

# REINFORCE
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.embedding1 = nn.Embedding(32, 16)  # First element: 32 possible values → 16-dim embedding
        self.embedding2 = nn.Embedding(5, 8)    # Second element: 5 possible values → 8-dim embedding
        self.embedding3 = nn.Embedding(32, 16) 

        self.affine1 = nn.Linear(3, 128)
        # self.affine1 = nn.Linear(3, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x1 = self.affine1(x)
        x2 = self.dropout(x1)
        x3 = F.relu(x2)

        action_scores = self.affine2(x3)
        return F.softmax(action_scores, dim=1)

    def select_action(self, state):
        # state 传入policy network 得到action prob 分布
        state = torch.from_numpy(state).float().unsqueeze(0).to('cuda:1')
        probs = self.forward(state)
        entropy_bonus = -0.01 * (probs * probs.log()).sum()
        m = Categorical(probs)
        # sample one action
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self, policy, optimizer, gamma, eps = np.finfo(np.float32).eps.item()):
        R = 0
        policy_loss = []
        returns = deque()
        # print("policy.rewards before calling finish_episode:", policy.rewards)
        for r in policy.rewards[::-1]:  # policy_rewards[::-1]:
            R = r + gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns, dtype = torch.float32, device = 'cuda:1')
        
        if returns.std() == 0 or torch.isnan(returns.std()):
            returns = returns - returns.mean()
        else: 
            # returns = (returns - returns.mean()) / (returns.std() + eps)
            returns = (returns - returns.mean()) / (returns.std().clamp(min=eps))
        
        entropy_bonus = 0.0
        for log_prob, R in zip(policy.saved_log_probs, returns):
            entropy_bonus += 0.01 * (-log_prob.exp() * log_prob).sum()
            # print(f"log_prob: {log_prob.item()}, R: {R.item()}, log_prob * R: {(-log_prob * R).item()}")
            policy_loss.append(-log_prob * R + entropy_bonus)

        # print(f"policy_loss length: ", len(policy_loss))
        optimizer.zero_grad()
        if len(policy_loss) > 0:
            policy_loss = torch.cat(policy_loss).sum() * 10
        else:
            policy_loss = torch.tensor(0.0, device='cuda:1')
        # print(f"Policy loss: {policy_loss.item()}")
        policy_loss.backward()
        

        # for name, param in policy.named_parameters():
        #   if param.grad is not None:
        #       print(f"{name} gradient mean: {param.grad.mean().item()}")
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        del policy.rewards[:]
        del policy.saved_log_probs[:]
        # return np.mean(param.grad.mean().item())
        

env = NaiveBlackjackEnv()
env.reset(seed = 42)


def training_loop(lr, gamma):
    policy = Policy().to('cuda:1')
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    smoothed_reward = 0
    ep_rewards = []
    smoothed_rewards = []
    # param_grads = []

    for i_episode in tqdm(range(10000), desc = "Processing episode"):
        state = env.reset(seed=42)
        state = torch.tensor(state, dtype=torch.float32, device='cuda:1')

        ep_reward = 0
        done = False
        while not done:  # Don't infinite loop while learning
            action = policy.select_action(state.cpu().numpy())
            state, reward, done, _ = env.step(action)
            state = torch.tensor(state, dtype=torch.float32, device='cuda:1')
            reward = torch.tensor(reward, dtype=torch.float32, device='cuda:1')
            # env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        # print("Reward list ", policy.rewards)
        # print("Ep_reward for one round: ", ep_reward)
        ep_rewards.append(ep_reward)
        smoothed_reward = 0.05 * ep_reward + (1-0.05) * smoothed_reward
        # print("Smoothed_reward: ", smoothed_reward)
        smoothed_rewards.append(smoothed_reward)
        # print("Reward list ", policy.rewards)
        policy.finish_episode(policy, optimizer, gamma)
        # param_grads.append(param_grad)

    # smoothed_rewards = smoothed_rewards.cpu().numpy() 
    smoothed_rewards = [item.cpu().numpy() if isinstance(item, torch.Tensor) else item for item in smoothed_rewards]
    # check device for smoothed_rewards
    for i, item in enumerate(smoothed_rewards):
      if isinstance(item, torch.Tensor):
        if item.device.type != 'cpu':
          print(f"Element {i} is a Tensor on {item.device}")
          break

    AUC = auc(range(len(smoothed_rewards)), smoothed_rewards)
    print("The avg smoothed_reward per episod is ", round(np.mean(smoothed_rewards),2), " and the AUC is ", round(AUC,2))
    # plot_smoothed_reward_espisod(smoothed_rewards)
    
    return AUC, smoothed_rewards

#%%
# Check param gradient 
AUC, smoothed_rewards = training_loop(1e-5, 0.9)
#%%


#%%
LEARNING_RATE = [1e-3, 1e-4, 1e-5]
BATCH_SIZE = [128, 256, 512]
DISCOUNT_FACTOR = [0.9, 0.95, 0.99]

best_params = {
    "batch_size": random.choice(BATCH_SIZE),
    "lr": random.choice(LEARNING_RATE),
    "gamma": random.choice(DISCOUNT_FACTOR),
}
best_auc = -np.inf

num_iterations = 5
for iteration in tqdm(range(num_iterations), desc = "Coordinate Descent Hyperparameters tuning", total = num_iterations):
   print(f"\n===== Coordinate Descent Iteration {iteration + 1} =====")
   for param_name, param_values in tqdm([
        ("batch_size", BATCH_SIZE),
        ("lr", LEARNING_RATE),
        ("gamma", DISCOUNT_FACTOR),
    ], desc = "Optimizing Hyperparameter", total = 6):
        print(f"Optimizing {param_name}...")

        for param_value in tqdm(param_values, desc=f"Tuning {param_name}"):
            # Test current params
            temp_params = best_params.copy()
            temp_params[param_name] = param_value

            if param_name == "batch_size":
                temp_params["lr"] = best_params["lr"] * (param_value / best_params["batch_size"])

            AUC, smoothed_rewards = training_loop(temp_params['lr'], temp_params['gamma'])

            if AUC > best_auc: 
                smoothed_rewards_list = smoothed_rewards
                best_params[param_name] = param_value
                best_auc = AUC
                print(f"Updated best {param_name} -> {param_value} (AUC: {best_auc})")

save_figure_path = "./HW_figure"
save_result_path = "./results"

smoothed_rewards_list = [float(r) for r in smoothed_rewards_list]
plt.plot(smoothed_rewards_list)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('REINFORCE Episodes reward')

plt.savefig(os.path.join(save_figure_path,'REINFORCE_performance.png'), dpi=300, bbox_inches='tight')
# plt.show()

results_dict = {
    "best_avg_smoothed_reward": round(np.mean(smoothed_rewards_list),4),
    "smoothed_reward_list": smoothed_rewards_list,
    "auc": round(float(best_auc),4), 
    "best_parameters": best_params
}
print("Save element in json!")
result_file = os.path.join(save_result_path,"REINFORCE_results.txt")
with open(result_file, "w") as f:
    json.dump(results_dict, f, indent = 4)
print(f"Results saved in {result_file}")

# plt.figure(figsize=(8, 5))
# plt.plot(range(len(param_grads)), param_grads, marker='o', linestyle='-', color='b', alpha=0.6, label="Gradient Mean")

# plt.xlabel("Episode")
# plt.ylabel("Mean Gradient Value")
# plt.title("Gradient Mean per Episode")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(save_figure_path,'REINFORCE_grad_episode.png'), dpi=300, bbox_inches='tight')
# # plt.show()





# with open(os.path.join(save_result_path,"reinforce_results.txt"), "a") as f:
#     f.write("="*30 + "\n")  
#     f.write(f"Best Avg smoothed reward: {np.mean(best_smoothed_rewards)}\n")
#     f.write(f"AUC: {best_auc}\n")
#     f.write(f"Best parameters: {best_params}\n\n")

