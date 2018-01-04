
import numpy as np

# Always select 0
class Agent:
    def __init__(self, name='Anonymous', k=10):
        self.name = name
        self.kArms = k

    def play(self, bandit):
        a = self.chooseAction()
        r = bandit.pull(a)
        self.postAction(a, r)
        return a, r

    def getPoint(self):
        return self.point

    def reset(self):
        pass

    def chooseAction(self):
        return 0

    def postAction(self, a, r):
        pass


# Select randomly
class RandomAgent(Agent):

    def __init__(self, k=10):
        super().__init__(name='Random', k=k)


    def chooseAction(self):
        return np.random.randint(self.kArms)


class GreedyAgent(Agent):

    def __init__(self, k=10):
        super().__init__(name='Greedy', k=k)
        self.reset()
        print('XXX action_count', self.action_count)
        print('qEst', self.qEst)

    def reset(self):
        self.qEst = np.zeros(self.kArms)
        self.action_count = np.zeros(self.kArms, int)

    def chooseAction(self):
        return np.argmax(self.qEst)

    def postAction(self, a, r):
        self.action_count[a] += 1
        self.qEst[a] += (r - self.qEst[a])/self.action_count[a]
        # print('qEst', self.qEst)


class EpsilonGreedyAgent(Agent):

    def __init__(self, k=10, e=0.1):
        super().__init__(name='E-Greedy', k=k)
        self.epsilon = e
        self.reset()

    def reset(self):
        self.qEst = np.zeros(self.kArms)
        self.action_count = np.zeros(self.kArms, int)

    def chooseAction(self):
        explore = np.random.binomial(1, self.epsilon)
        if explore == 1:
            a = np.random.randint(self.kArms)
        else:
            a = np.argmax(self.qEst)
        return a

    def postAction(self, a, r):
        self.action_count[a] += 1
        self.qEst[a] += (r - self.qEst[a])/self.action_count[a]
