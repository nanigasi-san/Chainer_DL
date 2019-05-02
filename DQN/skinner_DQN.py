# In[]:
import numpy as np
from chainer import Chain,optimizers
import chainer.functions as F
import chainer.links as L
import chainerrl
import matplotlib.pyplot as plt

# In[]:
class QFunction(Chain):
    def __init__(self, obj_size, n_actions, n_hidden_channels=2):
        super().__init__(
            l1=L.Linear(obj_size, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_hidden_channels),
            l3=L.Linear(n_hidden_channels, n_actions)
        )

    def __call__(self, x, test=False):
        h1 = F.tanh(self.l1(x))
        h2 = F.tanh(self.l2(h1))
        y = chainerrl.action_value.DiscreteActionValue(self.l3(h2))

        return y

def random_action():
    return np.random.choice([0,1])

def step(state,action):
    reward = 0
    if state == 0:
        if action == 0:
            state = 1
        else:
            state = 0
    else:
        if action == 0:
            state = 0
        else:
            state = 1
            reward = 1
    return np.array([state]),reward


gamma = 0.9
alpha = 0.5
max_number_of_steps = 5 #1試行のstep数
num_episodes = 100 #総試行回数

q_func = QFunction(1,2)
optimizer = optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0,end_epsilon=0.1,decay_steps=num_episodes,random_action_func=random_action)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)
phi = lambda x: x.astype(np.float32,copy=False)
agent = chainerrl.agents.DQN(
    q_func,optimizer,replay_buffer,gamma,explorer,
    replay_start_size=500,update_interval=1,target_update_interval=100,phi=phi
    )

# In[]:
# agent.load("DQN/skinner_agent")
Rlist = []
for episode in range(num_episodes):
    state = np.array([0])
    R = 0
    reward = 0
    done = True

    for t in range(max_number_of_steps):
        action = agent.act_and_train(state,reward)
        next_state,reward = step(state,action)
        # print(state,action,reward)
        R += reward
        state = next_state

    agent.stop_episode_and_train(state,reward,done)
    # print("episode : {0} ,total reward : {1}".format(episode,R))
    Rlist.append(R)
agent.save("DQN/skinner_agent")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.array(Rlist))
plt.savefig("DQN/image/100.png",dpi=1000)
