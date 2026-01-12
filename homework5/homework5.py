import gymnasium
import flappy_bird_gymnasium
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pygame

from collections import deque


ENV_NAME = "FlappyBird-v0"
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
BUFFER_SIZE = 100_000
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 1000
NUM_EPISODES = 1500
FRAME_STACK = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(frame):
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    frame = cv2.resize(frame, (84, 84))
    frame = frame / 255.0
    return frame.astype(np.float32)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def train():
    env = gymnasium.make(ENV_NAME, render_mode="rgb_array")

    n_actions = env.action_space.n
    policy_net = DQN(n_actions).to(DEVICE)
    target_net = DQN(n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPSILON_START
    steps = 0

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        frame = preprocess(obs)

        frame_stack = deque([frame] * FRAME_STACK, maxlen=FRAME_STACK)
        state = np.stack(frame_stack)

        episode_reward = 0

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state).unsqueeze(0).to(DEVICE)
                    action = policy_net(state_tensor).argmax().item()

            next_obs, reward, done, _, _ = env.step(action)
            next_frame = preprocess(next_obs)
            frame_stack.append(next_frame)
            next_state = np.stack(frame_stack)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1

            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(DEVICE)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
                dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
                q_values = policy_net(states).gather(1, actions).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + GAMMA * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
        print(f"Episode {episode} reward: {episode_reward:.2f} epsilon: {epsilon:.3f}")

    torch.save(policy_net.state_dict(), "flappy_dqn.pth")
    env.close()

def play():
    import time

    env = gymnasium.make(ENV_NAME, render_mode="human")
    time.sleep(3.0)

    n_actions = env.action_space.n
    model = DQN(n_actions).to(DEVICE)
    model.load_state_dict(torch.load("flappy_dqn.pth", map_location=DEVICE))
    model.eval()

    obs, _ = env.reset()
    frame = preprocess(obs)
    frame_stack = deque([frame] * FRAME_STACK, maxlen=FRAME_STACK)

    while True:
        state = np.stack(frame_stack)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action = model(state_tensor).argmax().item()

        obs, _, done, _, _ = env.step(action)
        frame_stack.append(preprocess(obs))

        time.sleep(0.03)

        if done:
         input("Enter pt inchidere fereastra...")
         break

    env.close()

if __name__ == "__main__":
    train()
    play()
