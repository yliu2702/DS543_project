import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
from Env_definition import SlateRecSimEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("================Testing Multistep DQN.================")
print("Using device: ",device)

NUM_LAYERS = [2, 3, 4]
NUM_NEURONS = [64, 128, 256]
BATCH_SIZE = [128, 256, 512]
LEARNING_RATE = [1e-3, 1e-4, 1e-5]
DISCOUNT_FACTOR = [0.9, 0.95, 0.99]
MULTI_STEP = [3, 5, 10]
EPSILON_DECAY = [0.99, 0.995, 0.999]

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
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.multi_step) * max_next_q_values

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def flatten_state(state):
    flat = [
        state["user_interest"],
        np.array([state["user_age"] / 100.0]),
        np.array([state["user_sex"]]),
        state["user_personality"],
        state["document_qualities"],
        state["document_lengths"] / 10.0,
        state["document_popularity"],
        state["document_ratings"]
    ]
    return np.concatenate(flat)

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

            for a in slate:
                agent.store_transition((state, a, reward / len(slate), next_state, done))
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

best_params = {
    "num_layers": random.choice(NUM_LAYERS),
    "num_neurons": random.choice(NUM_NEURONS),
    "batch_size": random.choice(BATCH_SIZE),
    "lr": random.choice(LEARNING_RATE),
    "gamma": random.choice(DISCOUNT_FACTOR),
    "multi_step": random.choice(MULTI_STEP),
    "epsilon_decay": random.choice(EPSILON_DECAY),
}

best_agent = None
best_auc = -np.inf
best_reward = -np.inf

env = SlateRecSimEnv(alpha=1.0, beta=0.1)
state_size = flatten_state(env.reset()).shape[0]
action_size = env.num_candidates

num_iterations = 5
for iteration in tqdm(range(num_iterations), desc="Coordinate Descent Hyperparameters tuning"):
    print(f"\n===== Coordinate Descent Iteration {iteration + 1} =====")
    for param_name, param_values in tqdm([
        ("num_layers", NUM_LAYERS),
        ("num_neurons", NUM_NEURONS),
        ("batch_size", BATCH_SIZE),
        ("lr", LEARNING_RATE),
        ("gamma", DISCOUNT_FACTOR),
        ("multi_step", MULTI_STEP),
        ("epsilon_decay", EPSILON_DECAY),
    ], desc="Optimizing Hyperparameter"):
        print(f"Optimizing {param_name}...")

        for param_value in tqdm(param_values, desc=f"Tuning {param_name}"):
            temp_params = best_params.copy()
            temp_params[param_name] = param_value

            agent = DQNAgent(
                state_size, action_size,
                temp_params["num_layers"], temp_params["num_neurons"],
                temp_params["lr"], temp_params["gamma"], temp_params["batch_size"],
                10, temp_params["epsilon_decay"], temp_params["multi_step"]
            )
            rewards, smoothed_rewards = train_dqn(agent, env, 5000)
            auc = np.trapz(rewards)

            if auc > best_auc:
                best_agent = agent
                best_rewards = rewards
                best_smoothed_rewards = smoothed_rewards
                best_params[param_name] = param_value
                best_auc = auc
                best_reward = best_rewards
                print(f"Updated best {param_name} -> {param_value} (AUC: {best_auc})")

save_figure_path = "./HW_figure"
save_result_path = "./results"
os.makedirs(save_figure_path, exist_ok=True)
os.makedirs(save_result_path, exist_ok=True)

plt.plot(best_smoothed_rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Multistep Return DQN Best Episodes Reward')
plt.savefig(os.path.join(save_figure_path, 'Multistep_return_best_dqn_performance.png'), dpi=300, bbox_inches='tight')
plt.show()

results_dict = {
    "best_avg_step_reward": np.mean(best_reward),
    "best_avg_smoothed_reward": np.mean(best_smoothed_rewards),
    "reward_list": best_reward,
    "smoothed_reward_list": best_smoothed_rewards,
    "auc": best_auc,
    "best_parameters": best_params
}

result_file = os.path.join(save_result_path, "multistep_return_results.txt")
with open(result_file, "w") as f:
    json.dump(results_dict, f, indent=4)
print(f"Results saved in {result_file}")
