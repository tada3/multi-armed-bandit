
import numpy as np

# Always select 0
class Agent:

    def __init__(self, name='Anonymous'):
        self.name = name
        self.point = 0

    def play(self, bandit):
        a = self.chooseAction()
        r = self.takeAction(bandit, a)
        return r

    def getPoint(self):
        return self.point

    def chooseAction(self):
        return 0

    def takeAction(self, bandit, a):
        r = bandit.pull(a)
        self.point += r
        return r


# Select randomly
class RandomAgent(Agent):

    def __init__(self, k=10):
        super().__init__(name='Random')
        self.kArms = k

    def chooseAction(self):
        return np.random.randint(self.kArms)
        r = bandit.pull(a)
        self.point += r
        return r

    def takeAction(self, bandit, a):
        r = bandit.pull(a)
        self.point += r
        return r


class GreedyAgent(Agent):

    def __init__(self, k=10):
        super().__init__(name='Greedy')
        self.kArms = k
        self.qEst = np.zeros(k)
        self.action_count = np.zeros(k, int)

    def play(self, bandit):
        a = np.argmax(self.qEst)
        r = bandit.pull(a)
        self.action_count[a] += 1
        self.qEst[a] += (r - self.qEst[a])/self.action_count[a]
        self.point += r
        return r

class EpsilonGreedyAgent(Agent):

    def __init__(self, k=10, e=0.1):
        super().__init__(name='E-Greedy')
        self.kArms = k
        self.qEst = np.zeros(k)
        self.action_count = np.zeros(k, int)

        self.epsilon = e

    def play(self, bandit):
        explore = np.random.binomial(1, self.epsilon)
        if explore == 1:
            a = np.random.randint(self.kArms)
        else:
            a = np.argmax(self.qEst)

        r = bandit.pull(a)
        self.action_count[a] += 1
        self.qEst[a] += (r - self.qEst[a])/self.action_count[a]
        self.point += r
        return r
