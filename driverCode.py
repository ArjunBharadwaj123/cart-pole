# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from functions import Q_Learning


# ------------------------------------------------------------
# Create the environment
# ------------------------------------------------------------
env = gym.make('CartPole-v1')
(state, _) = env.reset()

# ------------------------------------------------------------
# Define the parameters for state discretization
# ------------------------------------------------------------

# Copy high/low to avoid modifying Gym's internal ranges
upperBounds = env.observation_space.high.copy()
lowerBounds = env.observation_space.low.copy()

# Clamp velocity & angular velocity to reasonable ranges
upperBounds[1] = 100         # cart velocity max
lowerBounds[1] = -100        # cart velocity min

upperBounds[3] = 10        # pole angle velocity max
lowerBounds[3] = -10       # pole angle velocity min


# ------------------------------------------------------------
# Number of bins for discretization
# ------------------------------------------------------------
numberOfBins = [30, 30, 30, 30]   # position, velocity, angle, angle-velocity


# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------
alpha = 0.1 # learning rate
gamma = 0.99 # randomness discount factor
epsilon = 0.2  # random action probability
numberEpisodes = 15000 # number of episodes for training


# ------------------------------------------------------------
# Create Q-learning agent
# ------------------------------------------------------------
Q1 = Q_Learning(
    env,
    alpha,
    gamma,
    epsilon,
    numberEpisodes,
    numberOfBins,
    lowerBounds,
    upperBounds
)


# ------------------------------------------------------------
# Run training
# ------------------------------------------------------------
Q1.simulateEpisodes()


# ------------------------------------------------------------
# Run learned strategy ONCE
# ------------------------------------------------------------
obtainedRewardsOptimal, env1 = Q1.simulateLearnedStrategy()


# ------------------------------------------------------------
# Plot convergence
# ------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(Q1.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards in Episode')
plt.yscale('log')
plt.savefig('convergence.png')
plt.show()


# ------------------------------------------------------------
# Close environment
# ------------------------------------------------------------
env1.close()

print("Total reward with learned strategy:", np.sum(obtainedRewardsOptimal))


# ------------------------------------------------------------
# (Optional) Remove these unless simulateRandomStrategy exists
# ------------------------------------------------------------
# obtainedRewardsRandom, env2 = Q1.simulateRandomStrategy()
# plt.hist(obtainedRewardsRandom)
# plt.xlabel('Sum of rewards')
# plt.ylabel('Percentage')
# plt.savefig('random_histogram.png')
# plt.show()
