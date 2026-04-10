import kagglehub
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Download dataset
# ---------------------------------------------------

path = kagglehub.dataset_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs")

stock_files = glob.glob(os.path.join(path, "Stocks/*.txt"))

# ---------------------------------------------------
# Load one stock for training
# ---------------------------------------------------

def load_stock(file):

    df = pd.read_csv(file)

    prices = df["Close"].values

    returns = np.diff(prices) / prices[:-1]

    return prices, returns


prices, returns = load_stock(stock_files[0])

# ---------------------------------------------------
# RL Environment
# ---------------------------------------------------

class StockEnv:

    def __init__(self, prices, window=10):

        self.prices = prices
        self.window = window

        self.reset()

    def reset(self):

        self.t = self.window
        self.position = 0

        return self._state()

    def _state(self):

        window_prices = self.prices[self.t-self.window:self.t]

        returns = np.diff(window_prices) / window_prices[:-1]

        state = np.concatenate([returns, [self.position]])

        return state.astype(np.float32)

    def step(self, action):

        # actions
        # 0 short
        # 1 flat
        # 2 long

        if action == 0:
            self.position = -1
        elif action == 1:
            self.position = 0
        else:
            self.position = 1

        price_change = self.prices[self.t+1] - self.prices[self.t]

        reward = self.position * price_change

        self.t += 1

        done = self.t >= len(self.prices)-2

        return self._state(), reward, done


env = StockEnv(prices)

state_dim = env.reset().shape[0]
action_dim = 3

# ---------------------------------------------------
# DQN Model
# ---------------------------------------------------

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,action_dim)
        )

    def forward(self,x):

        return self.net(x)


model = DQN(state_dim,action_dim)

optimizer = optim.Adam(model.parameters(),lr=1e-3)

gamma = 0.99

# ---------------------------------------------------
# Replay Buffer
# ---------------------------------------------------

buffer = []

def store(exp):

    buffer.append(exp)

    if len(buffer) > 100000:
        buffer.pop(0)


def sample(batch=64):

    idx = np.random.choice(len(buffer),batch)

    return [buffer[i] for i in idx]


# ---------------------------------------------------
# Training
# ---------------------------------------------------

episodes = 200

epsilon = 1.0

for ep in range(episodes):

    s = env.reset()

    total_reward = 0

    while True:

        if random.random() < epsilon:
            a = random.randint(0,2)
        else:
            with torch.no_grad():
                q = model(torch.tensor(s))
                a = torch.argmax(q).item()

        s2,r,done = env.step(a)

        store((s,a,r,s2,done))

        s = s2

        total_reward += r

        if len(buffer) > 1000:

            batch = sample()

            states = torch.tensor([b[0] for b in batch])
            actions = torch.tensor([b[1] for b in batch])
            rewards = torch.tensor([b[2] for b in batch])
            next_states = torch.tensor([b[3] for b in batch])
            dones = torch.tensor([b[4] for b in batch])

            q = model(states)

            q = q.gather(1,actions.unsqueeze(1)).squeeze()

            with torch.no_grad():
                q_next = model(next_states).max(1)[0]

            target = rewards + gamma*q_next*(1-dones)

            loss = ((q-target)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon *= 0.995

    print("Episode",ep,"Reward",total_reward)

# ---------------------------------------------------
# Test policy
# ---------------------------------------------------

s = env.reset()

positions = []

while True:

    with torch.no_grad():

        a = torch.argmax(model(torch.tensor(s))).item()

    positions.append(a-1)

    s,r,done = env.step(a)

    if done:
        break


plt.plot(prices[:len(positions)])
plt.plot(np.cumsum(positions))
plt.show()
