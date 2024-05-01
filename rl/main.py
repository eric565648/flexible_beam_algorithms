from os import stat
import numpy as np
from numpy.lib.utils import deprecate
import torch
from pathlib import Path
from copy import deepcopy

from flexible_beam import *
from ddpg import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define different parameters for training the agent
max_episode=20000
ep_r = 0
total_step = 0

env = FlexbleBeamFixLin()
torch.manual_seed(0)
np.random.seed(0)
#Environment action ans states
obs_window = 1
state_dim = 4*obs_window
action_dim = 1
max_action = float(2)
# Exploration Noise
exploration_noise=0.1
exploration_noise=0.1 * max_action
ou_noise = OU_Noise(action_dim,0,mu=0,sigma=0.4)

# Create a DDPG instance
agent = DDPG(state_dim, action_dim, device)

# create save folder
Path('models/').mkdir(exist_ok=True)
Path('results/').mkdir(exist_ok=True)

# Train the agent for max_episodes
import time
st = time.time()
ready_to_train = False
episode_G=[]
max_episode_G=-999999999
for ep in range(max_episode):
    rewards = []
    state = env.reset()
    # observation window
    state_window = []
    for i in range(obs_window):
        state_window.append(state)
    next_state_window=deepcopy(state_window)
    while True:
        action = agent.select_action(np.array(state_window).flatten())
        # Add Gaussian noise to actions for exploration
        action = (action + np.random.normal(0, 0.2, size=action_dim)).clip(-max_action, max_action)
        # action += ou_noise.sample()
        next_state,reward,done = env.step(action)
        next_state_window.pop(0)
        next_state_window.append(next_state)

        rewards.append(reward)
        agent.replay_buffer.push((np.array(state_window).flatten(), np.array(next_state_window).flatten(), \
            action, reward, float(done)))
        
        # update
        if ready_to_train:
            agent.update()
        else:
            if len(agent.replay_buffer.storage)>agent.batch_size:
                print("Ready to Train")
                ready_to_train=True

        # state = next_state
        state_window = deepcopy(next_state_window)
        if done:
            break
    episode_G.append(np.sum(rewards))
    # agent.update()
    if ep%10==1:
        agent.save('models/')
        status_str="Episode: "+str(ep)+', Rewards: '+str(np.mean(episode_G[-10:]))
        print(status_str)
    if episode_G[-1]>max_episode_G:
        print("Save Best! Episode G:",episode_G[-1])
        agent.save('models/best_')
        max_episode_G=episode_G[-1]
    np.save('results/episode_G',episode_G)
print(time.time()-st)