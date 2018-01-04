import numpy as np

class Bandit:
  
  def __init__(self, k=10):
    self.k = k
    self.q_true = []

    # initialize real rewards with N(0,1) distribution and estimations with desired initial value
    for i in range(0, self.k):
      self.q_true.append(np.random.randn())

    self.best_action = np.argmax(self.q_true)

    print('qTrue', self.q_true)


  def pull(self, action):
      # print('XXXX action=', action)
      # print('XXXX', self.q_true)
      qt_a = self.q_true[action]
      r = np.random.normal(qt_a, 1)
      return r

  def get_best_action(self):
      return self.best_action
