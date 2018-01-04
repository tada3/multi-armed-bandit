import numpy as np

class Bandit:
  
  def __init__(self, k=10):
    self.k = k
    self.q_true = []

    # initialize real rewards with N(0,1) distribution and estimations with desired initial value
    for i in range(0, self.k):
      self.q_true.append(np.random.randn())
      print(i, ': ', self.q_true[i])


  def pull(self, action):
      qt_a = self.q_true[action]
      r = np.random.normal(qt_a, 1)
      return r


