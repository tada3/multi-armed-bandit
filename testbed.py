import matplotlib.pyplot as plt
import numpy as np

from agent import EpsilonGreedyAgent
from agent import GreedyAgent
from agent import RandomAgent
from bandit import Bandit


class Playground:
    def __init__(self, n):
        self.nBandits = n
        self.bandits = [Bandit() for _ in range(n)]
        return

    def run(self, agent, n):
        for i in range(n):
            r = agent.play(self.bandit)
            print(i, ': ', r)

    def run2(self, agent, n, time):
        avgRewards = np.zeros(time)
        bestActionRate = np.zeros(time)
        for i in range(n):
            agent.reset()
            for t in range(time):
                a, r = agent.play(self.bandits[i])
                avgRewards[t] += r
                if a == self.bandits[i].get_best_action():
                    bestActionRate[t] += 1

        # print('Before', avgRewards)
        avgRewards /= n
        bestActionRate /= n
        # print('After', avgRewards)
        return avgRewards, bestActionRate


def testRandom():
    pg = Playground()
    a1 = RandomAgent(k=10)
    pg.run(a1, 10)
    print('total: ', a1.getPoint())

def testGreedy():
    nBandits = 2000
    time = 1000
    testbed = Playground(nBandits)
    agent = GreedyAgent()
    avgRewards, bestActionRates = testbed.run2(agent, nBandits, time)

    plt.plot(avgRewards, label='epsilon = 0.1')
    plt.xlabel('Steps')
    plt.ylabel('average reward')
    plt.legend()


# Figure 2.2
def test_epsilon_greedy():
    nBandits = 2000
    time = 1000
    testbed = Playground(nBandits)

    f1 = plt.figure(1)
    f2 = plt.figure(2)
    ax1 = f1.subplots()
    ax2 = f2.subplots()

    for epsilon in [0, 0.01, 0.1]:
        agent = EpsilonGreedyAgent(e=epsilon)
        rewards, bestActions = testbed.run2(agent, nBandits, time)
        ax1.plot(rewards, label='epsilon = ' + str(epsilon))
        ax2.plot(bestActions, label='epsilon = ' + str(epsilon))

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('average reward')
    ax1.legend()

    ax2.set_xlabel('Steps')
    ax2.set_ylabel('optimal action')
    ax2.legend()

if __name__ == '__main__':
    #testRandom()
    #testGreedy()
    test_epsilon_greedy()
    plt.show()
