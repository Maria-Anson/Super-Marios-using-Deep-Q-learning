import gym
import numpy as np
import random
from matplotlib import pyplot as plt 

env = gym.make("Taxi-v3").env
env.reset()
env.render()

print(env.action_space)
print(env.observation_space)


# STEP 1: Condifigure hyper-parameters

alpha = 0.7
gamma = 0.618   
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01

train_epochs = 2000
test_epochs = 200
max_steps = 100


Q = np.zeros((env.observation_space.n, env.action_space.n))


training_rewards = []
epsilons = []

# STEP 2: Choosing an action

for epoch in range(train_epochs):
    state = env.reset()
    total_training_rewards = 0


    for step in range(100):
        # exploration exploitation tradeoff
        exp_exp_tradeoff = random.uniform(0,1)

        if exp_exp_tradeoff > epsilon:
            # exploitation 
            action = np.argmax(Q[state, ])

        else:
            # exploration
            action = env.action_space.sample()


# STEP 3: perform action and obtain reward
        next_state, reward, done, info = env.step(action)



# STEP 4: update the Q table

        TD = reward + gamma * np.max(Q[next_state, :])- Q[state, action]
        Q[state, action] += alpha * TD

        total_training_rewards += reward
        state = next_state

        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay*epoch)

    training_rewards.append(total_training_rewards)
    epsilons.append(epsilon)

print("Training score over time : ", str(sum(training_rewards)/train_epochs))


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