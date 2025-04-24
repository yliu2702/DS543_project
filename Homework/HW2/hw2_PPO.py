#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import json
import random
from tqdm import tqdm

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
n_actions = env.action_space.n
obs_dim = len(env.observation_space.spaces)
#%%
def moving_average(data, window_size=100):
    if len(data) < window_size:
        return np.mean(data)
    return np.mean(data[-window_size:])

def compute_auc(rewards):
    return np.trapz(rewards, dx=1)


#%%
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    """创建多层感知机 (MLP)"""
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

# 策略网络 (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = mlp([obs_dim, 64, 64, n_actions], activation=nn.Tanh)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, obs):
        logits = self.net(obs)
        return self.softmax(logits)

# 价值网络 (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = mlp([obs_dim, 64, 64, 1], activation=nn.Tanh)
    
    def forward(self, obs):
        return self.net(obs).squeeze(-1)

# 计算优势估计 (GAE)
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * last_advantage
    return advantages

# PPO 训练循环
def train_ppo(env, policy_net, value_net, policy_opt, value_opt, epochs=300, batch_size=256, gamma=0.99, clip_epsilon=0.2):
    reward_history = []
    smoothed_reward_histroy = []
    # policy_grad_history = []
    for epoch in range(epochs):
        obs_list, action_list, reward_list, log_prob_list, value_list = [], [], [], [], []
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device = 'cuda:0')
            action_probs = policy_net(obs_tensor)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[action].to('cuda:0'))
            value = value_net(obs_tensor).item()
            next_obs, reward, done, _ = env.step(action)
            
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            log_prob_list.append(log_prob)
            value_list.append(value)
            episode_reward += reward
            obs = next_obs
        
        reward_history.append(episode_reward)
        smoothed_reward = moving_average(reward_history)
        smoothed_reward_histroy.append(smoothed_reward)

        obs_list = np.array(obs_list)
        action_list = np.array(action_list)
        rewards = np.array(reward_list)
        log_prob_list = torch.tensor(log_prob_list, dtype=torch.float32, device = "cuda:0")
        value_list.append(0)  # 终止状态的 V 值设为 0
        value_list = np.array(value_list)
        
        # 计算优势估计和目标值
        advantages = compute_advantages(rewards, value_list, gamma)
        returns = advantages + value_list[:-1]
        advantages = torch.tensor(advantages, dtype=torch.float32, device = "cuda:0")
        returns = torch.tensor(returns, dtype=torch.float32, device = "cuda:0")
        
        # 更新策略网络
        for _ in range(10000):
            new_log_probs = torch.log(policy_net(torch.tensor(obs_list, dtype=torch.float32, device = "cuda:0"))[range(len(action_list)), action_list])
            ratio = torch.exp(new_log_probs - log_prob_list)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            policy_opt.zero_grad()
            policy_loss.backward()
            # policy_grad = torch.cat([p.grad.view(-1) for p in policy_net.parameters() if p.grad is not None]).norm().item()
            # policy_grad_history.append(policy_grad)
            policy_opt.step()
        
        # 更新价值网络
        value_preds = value_net(torch.tensor(obs_list, dtype=torch.float32, device = 'cuda:0'))
        value_loss = (value_preds - returns).pow(2).mean()
        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}: Loss {policy_loss.item():.4f}, Value Loss {value_loss.item():.4f}, Smoothed Reward {smoothed_reward:.2f}") #, Policy Grad {policy_grad:.4f}")

    auc_score = compute_auc(reward_history)
    print(f"Final AUC Score: {auc_score:.2f}")
    
    return smoothed_reward_histroy, auc_score
#%%
# policy_net = PolicyNetwork(obs_dim, n_actions)
#             value_net = ValueNetwork(obs_dim)
#             policy_opt = optim.Adam(policy_net.parameters(), lr=1e-3)
#             value_opt = optim.Adam(value_net.parameters(), lr=1e-3)
# train_ppo(env, policy_net, value_net, policy_opt, value_opt, epochs=1000, batch_size=256, gamma=0.99, clip_epsilon=0.2)


#%%
CLIP_EPSILON = [0.2, 0.3, 0.4]
LEARNING_RATE = [1e-3, 1e-4, 1e-5]
BATCH_SIZE = [128, 256, 512]
DISCOUNT_FACTOR = [0.9, 0.95, 0.99]

best_params = {
    "clip_epsilon": random.choice(CLIP_EPSILON),
    "batch_size": random.choice(BATCH_SIZE),
    "lr": random.choice(LEARNING_RATE),
    "gamma": random.choice(DISCOUNT_FACTOR),
}
best_auc = -np.inf

num_iterations = 5
for iteration in tqdm(range(num_iterations), desc = "Coordinate Descent Hyperparameters tuning", total = num_iterations):
   print(f"\n===== Coordinate Descent Iteration {iteration + 1} =====")
   for param_name, param_values in tqdm([
        ("clip_epsilon", CLIP_EPSILON),
        ("batch_size", BATCH_SIZE),
        ("lr", LEARNING_RATE),
        ("gamma", DISCOUNT_FACTOR),
    ], desc = "Optimizing Hyperparameter", total = 6):
        print(f"Optimizing {param_name}...")

        for param_value in tqdm(param_values, desc=f"Tuning {param_name}"):
            # Test current params
            temp_params = best_params.copy()
            temp_params[param_name] = param_value

            # 初始化 PPO 训练
            policy_net = PolicyNetwork(obs_dim, n_actions).to('cuda:0')
            value_net = ValueNetwork(obs_dim).to('cuda:0')
            policy_opt = optim.Adam(policy_net.parameters(), lr=temp_params['lr'])
            value_opt = optim.Adam(value_net.parameters(), lr=temp_params['lr'])

            smoothed_reward_histroy, auc_score = train_ppo(env, policy_net, value_net, policy_opt, value_opt, epochs=1000, batch_size=temp_params['batch_size'], gamma=temp_params['gamma'], clip_epsilon=temp_params['clip_epsilon'])

            if auc_score > best_auc: 
                smoothed_rewards = smoothed_reward_histroy
                best_params[param_name] = param_value
                best_auc = auc_score
                print(f"Updated best {param_name} -> {param_value} (AUC: {best_auc})")

save_figure_path = "./HW_figure"
save_result_path = "./results"
smoothed_rewards_list = [float(r) for r in smoothed_rewards]
plt.plot(smoothed_rewards_list)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('PPO Episodes reward')

plt.savefig(os.path.join(save_figure_path,'PPO_performance.png'), dpi=300, bbox_inches='tight')
# plt.show()

results_dict = {
    "best_avg_smoothed_reward": np.mean(smoothed_rewards_list),
    "smoothed_reward_list": smoothed_rewards_list,
    "auc": best_auc, 
    "best_parameters": best_params
}
print("Save element in json!")
result_file = os.path.join(save_result_path,"PPO_results.txt")
with open(result_file, "w") as f:
    json.dump(results_dict, f, indent = 4)
print(f"Results saved in {result_file}")




#%%