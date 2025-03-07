#在这里我们开始model-free的方法复现
#model-free的经典方法是蒙特卡洛采样 在采样得到轨迹后 使用这条轨迹的G(t)来更新这个路径上结点的V function
#这会带来两个问题：首先是蒙特卡洛虽然是无偏估计 可以收敛到正确的Value iteration结果上，但是它的随机性过强 方差太大
#为了解决这个误差 我们引入了重要性估计 但是另外的，蒙特卡罗方法的高训练开销无法解决

#高训练开销的本质是每次得到一条完整的序列后才能进行一次更新，这极大增加了训练开销
#时序差分算法正是诞生于这种背景下，它每次只走一步，根据这一步的R和终点的V function来更新起点的V function
#不难想象，这会带来巨大的bias，但它的训练开销小了一个量级 因此是一个值得纪念的算法

#在这里我们实现了两种时序差分算法，分别是on-policy的SARSA和off-policy的Q-learning
#他们最主要的区别是抵达终点后，执行第二个action开始下一次迭代时，使用的是更新前的策略（on-policy）还是更新后的策略(Q-learning)
#这里有一个翻译梗 on-policy常常被称作在线学习 在线的意思是它不能开挂 必须使用当前策略选择第二个动作 就像你玩原神不能改自己账户的原石数量一样 因为你联网了
#而off-policy常常被称作离线学习 这与它的行为不谋而合 Q-learning本质上是对Q的价值迭代 可以称作Q-iteration，他在选择动作时选择的是事实上让Q增长最大的，而不是他的策略里认为应该在这个state选择的
#你看 前辈们管这个随机应变听调不听宣的叫“离线学习”是不是很合理
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm  # tqdm是显示循环进度条的库


class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 2 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

class SARSA:
    def __init__(self,n_col,n_row,alpha,epsilon,gamma,n_action = 4):
        self.Q_table = np.zeros([n_row*n_col, n_action])
        self.n_action = n_action
        self.alpha = alpha#学习率
        self.gamma = gamma#折扣因子
        self.epsilon = epsilon#epsilon贪婪策略 小于某临界时使用随即策略 大于某临界时使用贪婪策略

    def take_action(self,state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def update(self,s0,a0,r0,s1,a1):
        td_error = r0 + self.gamma* self.Q_table[s1,a1] - self.Q_table[s0,a0]
        self.Q_table[s0,a0] += self.alpha * td_error

class Q_learning:
    def __init__(self,n_col,n_row,alpha,epsilon,gamma,n_action = 4):
        self.Q_table = np.zeros([n_row * n_col,n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            #遍历这个state的下一个state 找出最大的Q对应的action
            action = np.argmax(self.Q_table[state])
        return action
    def update(self,s0,a0,r0,s1):
        Q_sota = self.Q_table[s1].max()
        td_error = r0 + self.gamma * Q_sota- self.Q_table[s0,a0]
        self.Q_table[s0,a0] += self.alpha * td_error
ncol = 12
nrow = 4
env = CliffWalkingEnv(ncol, nrow)
np.random.seed(time.localtime())
epsilon = 0.1
alpha = 0.1
gamma = 0.9
agent = Q_learning(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500  # 智能体在环境中运行的序列的数量

return_list = []  # 记录每一条序列的回报
for i in range(10):  # 显示10个进度条
    # tqdm的进度条功能
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
            episode_return = 0
            state = env.reset()
            action = agent.take_action(state)
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done = env.step(action)
                episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                agent.update(state, action, reward, next_state)
                state = next_state
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Q_learning on {}'.format('Cliff Walking'))
plt.savefig('./1.png')

def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                Q_max = np.max(agent.Q_table[i*env.ncol + j])
                a = [0 for _ in range(agent.n_action)]
                for k in range(agent.n_action):
                    if agent.Q_table[i * env.ncol + j,k] == Q_max:
                        a[k] = 1
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


action_meaning = ['^', 'v', '<', '>']
print('Q_learning算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning, list(range(25, 35)), [35])