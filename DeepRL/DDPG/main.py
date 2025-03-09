# 走到DDPG这一步，强化学习的基础算法基本已经全面覆盖到了每一个方面
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
