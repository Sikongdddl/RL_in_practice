#策略梯度算法来自于一个简单的思想：好的策略应该达到value function更大的地方
#因此策略的优化方向应该是value function的增长方向。
#由于value function是Q(s,a)在策略（a|s）上的期望 而Q function与网络无关
#所以value function的导数变成了策略网络的导数 也就是所谓的策略梯度

#在策略梯度的获得过程中 Q function是一个重要难点 往往会采用构造一个策略项的方式巧妙地将策略梯度化为两项
#第一项变成了Value function，因此可以通过各种方式来得到
#第二项变成了网络的log形式的导数 由于log函数的平滑和单调 这几乎没有影响计算量和网络表现
#这里使用的REINFORCE算法获得value function的方式十分直接，就是使用蒙特卡罗方法用轨迹的G逼近value function
#这种方法的好处是无偏 但是由于我们每一条轨迹都参与了神经网络的优化 这个方法的方差会大到不可思议的程度
#可以看到 即使在训练的最后时刻依然会有极度低的return出现
#后续所有策略相关的方法都会首先解决这个训练方差的问题

import os
import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils

class PolicyNet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(PolicyNet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,action_dim)

    def forward(self,state):
        state = F.relu(self.fc1(state))
        return F.softmax(self.fc2(state),dim=1)
    

class REINFORCE:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,device):
        self.policy_net = PolicyNet(state_dim,hidden_dim,action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),lr = learning_rate)

        self.gamma = gamma
        self.device = device
    
    def take_action(self,state):
        state = torch.tensor([state],dtype=torch.float).to(self.device)
        action_softmax = self.policy_net(state)
        action_dist = torch.distributions.Categorical(action_softmax)
        action = action_dist.sample()
        return action.item()
    
    def update(self,transition_dict):
        #deal with a trajectory each time
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1,1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1,action))
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()


learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
os.environ["CUDA_VISIBLE_DEVICES"]="7"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = "CartPole-v0"
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                  device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
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
plt.title('REINFORCE on {}'.format(env_name))
plt.savefig("raw.png")

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.savefig("smooth.png")