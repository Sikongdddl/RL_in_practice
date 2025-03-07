import numpy as np
import matplotlib.pyplot as plt

# bernoulli bandit model
# for each action: reward is a bernoulli distributtion
# which is totally random
class BernoulliBandit:
    def __init__(self, K):
        #random initialize rewards list for bandits
        self.probs = np.random.uniform(size=K)
        #gain best performance bandit id by probs metrix
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K
    
    def step(self, curStep):
        #do action based on bandit with id 'curStep'
        if (np.random.rand() < self.probs[curStep]):
            return 1
        else:
            return 0

# np.random.seed(114514)
# bandit_10_bernoulli = BernoulliBandit(10)
# print("now you have a bandit machine with arms:", 10)
# print("the best performance bandit id is: ", bandit_10_bernoulli.best_idx)
# print("the best probs is: ",bandit_10_bernoulli.best_prob)
