import os
import torch
import torch.nn as nn
import numpy as np

from nes_py.wrappers import JoypadSpace
from torch.distributions import Categorical
from gym.spaces import Box
import gym

import matplotlib.pyplot as plt
import gym_super_mario_bros
from torchvision import transforms
from gym.wrappers import FrameStack

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    
    def step(self, action):
        total_reward = 0.0
        done = False

        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward

            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)

    def observation(self, observation):
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(observation, (2,0,1)).copy(), dtype=torch.float))


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]

        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transformations = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        return transformations(observation).squeeze(0)





env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, [["right"], ["right", "A"]])
# env = JoypadSpace(env, COMPLEX_MOVEMENT)

env = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, skip=4)), shape=84), num_stack=4)
seed = 30
env.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
np.random.seed(42)


class MarioAgent:
    def __init__(self, learning_rate):
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=4, 
                    out_channels=32,
                    kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, 
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
            nn.Softmax(dim=-1)
        ).cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate, eps=1e-4)
        self.reset()


        self.gamma = 0.95
        

    def forward(self, x):
        return self.model(x)


    def reset(self):
        self.episode_actions = torch.tensor([], requires_grad = True).cuda()
        self.episode_rewards = []


    def save_checkpoints(self, directory, episode):
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, "checkpoint_{}.pth".format(episode))
        torch.save(self.model.state_dict(), f = filename)
        print("checkpoint saved to \'{}\'". format(filename))

    def load_checkpoint(self, directory, filename):
        self.model.load_state_dict(torch.load(os.path.join(directory, filename)))
        print("resuming training from checkpoint \'{}\'".format(filename))
        return int(filename[11:-4])

    def backward(self):
        future_reward = 0
        rewards = []

        for r in self.episode_rewards[::-1]:
            future_reward += r + self.gamma * future_reward
            rewards.append(future_reward)

        eps = 0.0001

        rewards = torch.tensor(rewards[::-1], dtype=torch.float32).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + + np.finfo(np.float32).eps)

        loss = torch.sum(torch.mul(self.episode_actions, rewards).mul(-1))
        
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.reset()



load_filename = None
save_directory = "./mario_pg"

batch_rewards = []
episode = 0
batch_size = 10

model = MarioAgent(learning_rate=0.0003)

if load_filename is not None:
    episode = model.load_checkpoint(save_directory, load_filename)

all_episode_rewards = []
all_mean_rewards = []


while True:
    observation = env.reset()
    done = False

    while not done:
        env.render()

        observation = torch.tensor(observation.__array__()).cuda().unsqueeze(0)
        temp = model.forward(observation)
        distribution = Categorical(temp)
        
        action = distribution.sample()
        observation, reward, done, _ = env.step(action.item())

        model.episode_actions = torch.cat([model.episode_actions, distribution.log_prob(action.reshape(1))])
        model.episode_rewards.append(reward)

        if done:
            all_episode_rewards.append(np.sum(model.episode_rewards))
            batch_rewards.append(np.sum(model.episode_rewards))
            model.backward()
            episode += 1

            if episode % batch_size == 0:
                print("Batch: {}, average reward: {}".format(episode // batch_size, np.array(batch_rewards).mean()))
                batch_rewards = []

                all_mean_rewards.append(np.mean(all_episode_rewards[-batch_size:]))
                
                plt.plot(all_mean_rewards)
                plt.clf()

            if episode % 50 == 0 and save_directory is not None:
                model.save_checkpoints(save_directory, episode)



