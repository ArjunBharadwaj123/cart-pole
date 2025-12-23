import numpy as np
import gymnasium as gym
import time


class Q_Learning:

    # ------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------
    def __init__(self, env, alpha, gamma, epsilon, numberEpisodes,
                 numberOfBins, lowerBounds, upperBounds):

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.actionNumber = env.action_space.n
        self.numberEpisodes = numberEpisodes
        self.numberOfBins = numberOfBins
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds

        self.sumRewardsEpisode = []

        # Initialize Q-table
        self.Qmatrix = np.random.uniform(
            low=0, high=1,
            size=(
                numberOfBins[0],
                numberOfBins[1],
                numberOfBins[2],
                numberOfBins[3],
                self.actionNumber
            )
        )

    # ------------------------------------------------------------
    # Discretize a continuous state
    # ------------------------------------------------------------
    def returnIndexState(self, state):

        cartPositionBin = np.linspace(
            self.lowerBounds[0], self.upperBounds[0], self.numberOfBins[0]
        )
        cartVelocityBin = np.linspace(
            self.lowerBounds[1], self.upperBounds[1], self.numberOfBins[1]
        )
        poleAngleBin = np.linspace(
            self.lowerBounds[2], self.upperBounds[2], self.numberOfBins[2]
        )
        poleAngleVelocityBin = np.linspace(
            self.lowerBounds[3], self.upperBounds[3], self.numberOfBins[3]
        )

        indexPosition = np.maximum(np.digitize(state[0], cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(state[1], cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(state[2], poleAngleBin) - 1, 0)
        indexAngularVelocity = np.maximum(
            np.digitize(state[3], poleAngleVelocityBin) - 1, 0
        )

        return (
            indexPosition,
            indexVelocity,
            indexAngle,
            indexAngularVelocity
        )

    # ------------------------------------------------------------
    # Epsilon-greedy action selection
    # ------------------------------------------------------------
    def selectAction(self, state, index):

        if index < 500:
            return np.random.choice(self.actionNumber)

        randomNumber = np.random.random()

        if index > 7000:
            self.epsilon *= 0.999

        if randomNumber < self.epsilon:
            return np.random.choice(self.actionNumber)

        stateIndex = self.returnIndexState(state)
        q_values = self.Qmatrix[stateIndex]

        best_actions = np.where(q_values == np.max(q_values))[0]
        return np.random.choice(best_actions)

    # ------------------------------------------------------------
    # Learn episodes using Q-learning updates
    # ------------------------------------------------------------
    def simulateEpisodes(self):

        for indexEpisode in range(self.numberEpisodes):

            rewardsEpisode = []

            stateS, _ = self.env.reset()
            stateS = list(stateS)

            print(f"Simulating episode {indexEpisode}")

            terminalState = False

            while not terminalState:

                stateSIndex = self.returnIndexState(stateS)
                actionA = self.selectAction(stateS, indexEpisode)

                nextState, reward, terminated, truncated, info = self.env.step(actionA)
                nextState = list(nextState)

                terminalState = terminated or truncated
                rewardsEpisode.append(reward)

                nextStateIndex = self.returnIndexState(nextState)
                QmaxPrime = np.max(self.Qmatrix[nextStateIndex])

                oldQ = self.Qmatrix[stateSIndex + (actionA,)]

                if not terminalState:
                    target = reward + self.gamma * QmaxPrime
                else:
                    target = reward

                error = target - oldQ
                self.Qmatrix[stateSIndex + (actionA,)] += self.alpha * error

                stateS = nextState

            episodeReward = np.sum(rewardsEpisode)
            print("Sum of rewards:", episodeReward)
            self.sumRewardsEpisode.append(episodeReward)

    # ------------------------------------------------------------
    # Run learned strategy (no exploration)
    # ------------------------------------------------------------
    def simulateLearnedStrategy(self):

        env1 = gym.make('CartPole-v1', render_mode='human')
        state, _ = env1.reset()
        state = list(state)

        obtainedRewards = []

        for t in range(1000):
            stateIndex = self.returnIndexState(state)
            q_values = self.Qmatrix[stateIndex]

            best_actions = np.where(q_values == np.max(q_values))[0]
            action = np.random.choice(best_actions)

            nextState, reward, terminated, truncated, info = env1.step(action)
            nextState = list(nextState)

            obtainedRewards.append(reward)
            time.sleep(0.05)

            if terminated or truncated:
                time.sleep(1)
                break

            state = nextState

        return obtainedRewards, env1
