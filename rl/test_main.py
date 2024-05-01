from os import stat
import numpy as np
from matplotlib import pyplot as plt
from numpy.random.mtrand import random
import torch
from pathlib import Path
from copy import deepcopy

from flexible_beam import *
from ddpg import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = FlexbleBeamFixLin()
torch.manual_seed(0)
np.random.seed(0)

#Environment action ans states
obs_window = 1
state_dim = 4*obs_window
action_dim = 1
max_action = float(2)

# Create a DDPG instance
agent = DDPG(state_dim, action_dim, device)

agent.load('models/best_')
# agent.load('models/')

rewards = []
state = env.reset(random=False)
all_w = [state[0]]
all_phi = [state[1]]
all_torques = []
# observation window
state_window = []
for i in range(obs_window):
    state_window.append(state)
next_state_window=deepcopy(state_window)
while True:
    action = agent.select_action(np.array(state_window).flatten())
    next_state,reward,done = env.step(action)
    all_torques.append(action[0])
    all_w.append(next_state[0])
    all_phi.append(next_state[1])

    next_state_window.pop(0)
    next_state_window.append(next_state)

    rewards.append(reward)

    # state = next_state
    state_window = deepcopy(next_state_window)
    if done:
        break

plt.plot(all_phi,label='phi')
plt.plot(all_w,label='deviation w')
plt.plot(rewards,label='instant reward')
plt.legend()
plt.title("DDPG Agent Performance")
plt.show()

plt.plot(all_torques,label='torque')
plt.legend()
plt.title("DDPG Agent Action")
plt.show()

print("DDPG Rewards:",np.sum(rewards))