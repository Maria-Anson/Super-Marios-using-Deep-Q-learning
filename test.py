import gym_super_mario_bros as sm
import numpy as np
import random
from matplotlib import pyplot as plt

env = sm.make("SuperMarioBros-1-1-v0")
# print(env.observation_space.shape)
# print(env.action_space.n)


class QAgent:
    def __init__(self, env):
        self.alpha = 0.7
        self.gamma = 0.618
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decay = 0.01

        self.train_episodes = 2000
        self.test_episodes = 200
        self.max_steps = 100

        self.env = env

        obs = np.array(env.reset().__array__())
        print(obs)
        plt.imshow(obs)
        plt.show()
        print(env.observation_space.shape)
        self.Q = np.zeros((env.observation_space.shape, env.action_space.n)) 


    def train(self):
        training_rewards = []
        epsilons = []
        epsilon = 1

        for episode in range(self.train_episodes):
            state = self.env.reset()
            total_training_rewards = 0

            for step in range(100):
                exp_exp_tradeoff = random.uniform(0,1)
                if exp_exp_tradeoff > epsilon:
                    action = np.argmax(self.Q[state, ])
                else:
                    action = self.env.action_space.sample()


                next_state, reward, done, info = self.env.step(action)

                TD = reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action]
                self.Q[state, action] += self.alpha * TD

                total_training_rewards += reward
                state = next_state

                if done:
                    break

            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * episode)

            training_rewards.append(total_training_rewards)
            epsilons.append(epsilon)

        print("Training score over time : ", str(sum(training_rewards)/self.train_episodes))

        return training_rewards, epsilons


agent = QAgent(env)

training_rewards, epsilons = agent.train()


x = range(train_epochs)
plt.plot(x, training_rewards)
plt.xlabel("Episode")
plt.ylabel("Training total reward")
plt.title("Total rewards over all episodes")
plt.show()


plt.plot(epsilons)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon for Episode")
plt.show()




        

                