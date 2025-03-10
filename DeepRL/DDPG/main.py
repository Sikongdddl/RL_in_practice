# 走到DDPG这一步，强化学习的基础算法基本已经全面覆盖到了每一个方面
# 令我感慨的是 这基本覆盖了wyb的课堂内容 而应该在2023年完成的这件事我竟然一直拖了两年
# 白盒环境中使用的，基于动态规划的Value iteration和policy iteration奠定了强化学习的范式
# 而离开白盒环境后，最直观的算法就是基于蒙特卡罗方法的model free方法 也就是我们前面提到的时序差分算法（Q learning和SARSA）
# 这里就引入了强化学习的第一个重要维度：model free和model based
# model free方法的代表就是SARSA和Q-learning（分别是on-policy算法和off-policy算法），而model based方法的代表就是DynaQ
# model based方法在训练的时候会先用一个模型来拟合环境，然后再用这个模型来训练agent
# 而model free方法则是直接用agent和环境交互来训练agent

# 在这之后，我们开始引入深度强化学习，也就是用神经网络来拟合Q函数或者策略函数
# 这里的代表就是DQN和Policy Gradient
# DQN是基于Q learning的，而Policy Gradient是基于SARSA的 他们都是model free方法
# 在深度强化学习中，我们有一些方法在学习价值函数（DQN），有一些方法在学习策略函数（策略梯度）
# 集大成者就是actor critic方法，他们同时学习价值函数和策略函数
# 基于actor-critic方法诞生了很多算法，比如A3C，TRPO，PPO等等 PPO走在这个领域的顶点
# 他在理论上美丽，在实现上又足够高效和工程化，因此成为了目前最流行的强化学习算法之一
# 而在这个过程中，我们也引入了第二个重要维度：on-policy和off-policy
# on-policy方法的代表就是TRPO，off-policy方法的代表就是这里要做的DDPG 和后面要学习到的 也就是我接触的第一款强化学习算法SAC
# on-policy方法和off-policy方法的区别在于：on-policy方法在更新策略的时候，只能用当前策略采集到的数据
# 而off-policy方法则可以用历史数据来更新策略
# 这两种方法各有优劣，on-policy方法的优势在于他的稳定性，而off-policy方法的优势在于他的效率

import random
import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import os

# same as DQN, DDPG use target network and soft update
# so for critic network, what DDPG do is absolutely same as DQN
# for actor network, we have a determined policy and a target policy
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma,actor_lr,critic_lr,tau,gamma,device):
        self.device = device
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.actor_target = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.sigma = sigma
        self.tau = tau #soft update parameter
        self.gamma = gamma
        self.action_dim = action_dim

    def take_action(self,state):
        state = torch.tensor([state],dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        action_with_noise = action + self.sigma*np.random.randn(self.action_dim)
        return action_with_noise

    def soft_update(self,net,target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],dtype=torch.float).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)

        # update critic network like DQN(TD loss)
        next_q = self.critic_target(next_states,self.actor_target(next_states))
        q_target = rewards + (1-dones) * self.gamma * next_q
        critic_loss = torch.mean(F.mse_loss(self.critic(states,actions),q_target))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #update actor network like policy gradient
        actor_loss = -torch.mean(self.critic(states,self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #update target network
        self.soft_update(self.actor,self.actor_target)
        self.soft_update(self.critic,self.critic_target)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
actor_lr = 3e-4
critic_lr = 3e-3
num_episodes = 200
hidden_dim = 64
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma = 0.01  # 高斯噪声标准差
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'Pendulum-v0'
env = gym.make(env_name)
random.seed(11)
np.random.seed(4)
env.seed(5)
torch.manual_seed(14)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.savefig('DDPG.png')
plt.close()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.savefig('DDPG_moving_average.png')
plt.close()