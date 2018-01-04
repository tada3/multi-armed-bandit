
from bandit import Bandit
from agent import Agent
from agent import RandomAgent
from agent import GreedyAgent


class Playground:

    def __init__(self):
        self.bandit = Bandit()
        return

    def run(self, agent, n):
        for i in range(n):
            r = agent.play(self.bandit)
            print(i, ': ', r)



def testRandom():
    pg = Playground()
    a1 = RandomAgent(k=10)
    pg.run(a1, 10)
    print('total: ', a1.getPoint())

def testGreedy():
    pg = Playground()
    a1 = GreedyAgent(k=10)
    pg.run(a1, 10)
    print('total: ', a1.getPoint())


if __name__ == '__main__':
    testRandom()
    #testGreedy()

