#basic implementations of markov process with S,P,R,gamma
import numpy as np

np.random.seed(77)
#all 6 states here
P = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])
R = [-1, -2, -2, 10, 1, 0]
gamma = 0.5

#start_idx: start state idx
#chain: a series sampled from buffer:[1,2,3,6]
#which means start from state 1 and end at state 6
#gamma: die down parameter
def compute_G(start_idx, chain, gamma):
    G = 0
    for i in reversed(range(start_idx, len(chain))):
        G *= gamma
        G += R[chain[i]-1]
    return G

print("测试序列1236回报为: ", compute_G(0,[1,2,3,6],gamma))

#V is a estimation of G, given a start state s
#due to P, a certain s will end with different series
#which means a certain s will gain different G
#So V is a estimation, and V is relatively hard to compute
def compute_V(P, R, gamma, states_num):
    R = np.array(R).reshape((-1, 1))  #将rewards写成列向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P),
                   R)
    return value

print("测试转移矩阵的V为: ",compute_V(P,R,gamma,6))
