import bandit
import numpy as np
import matplotlib.pyplot as plt

class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        # attempt times for each bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = []
        self.regrets = []

    #regret for attempt k compared with best-performance
    #work correctly only when function input k was called by sequence 
    def update_regret(self,k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
    
    def run_one_step(self):
        #decide which bandit to use
        return 0
        

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    #less than epsilon:explore
    #more than epsilon:use the best one
    def __init__(self,bandit,epsilon=0.01,init_prob=1.0):
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.ones(self.bandit.K)
        
    def run_one_step(self):
        if(np.random.random() < self.epsilon):
            k = np.random.randint(0,self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        #maintain estimates metrix by reward r this time
        #attention: use average reward as estimates
        #in this problem, bandit prob is a const value so reward is a bernoulli distribution
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class DecayEpsilonGreedy(Solver):
    #epsilon decay from 1 to 0 each step
    #explore first then use the best one
    #some kind of policy iteration
    def __init__(self,bandit,init_prob=1.0):
        super(DecayEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.ones(self.bandit.K)
        self.total_count = 0
    
    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.savefig(str(solver_names) + '.png')
    plt.close()

np.random.seed(11451)
bandit_10 = bandit.BernoulliBandit(10)
solver = Solver(bandit_10)

epsilon_greedy_solver = EpsilonGreedy(bandit_10,epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

decay_epsilon_greedy_solver = DecayEpsilonGreedy(bandit_10)
decay_epsilon_greedy_solver.run(5000)
print('epsilon-decay-贪婪算法的累积懊悔为：', decay_epsilon_greedy_solver.regret)
plot_results([decay_epsilon_greedy_solver], ["DecayEpsilonGreedy"])
