import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from model import *
from utils import *

#Set Hyperparameters
# Hyperparameters adapted for performance from
#https://ai.stackexchange.com/questions/22945/ddpg-doesnt-converge-for-mountaincarcontinuous-v0-gym-environment
capacity=1000000
batch_size=64
update_iteration=200
tau=0.001 # tau for soft updating
gamma=0.99 # discount factor
directory = 'models/'
hidden1=20 # hidden layer for actor
hidden2=64 #hiiden laye for critic

class DDPG(object):
    def __init__(self, state_dim, action_dim,device):
        """
        Initializes the DDPG agent. 
        Takes three arguments:
               state_dim which is the dimensionality of the state space, 
               action_dim which is the dimensionality of the action space, and 
               max_action which is the maximum value an action can take. 
        
        Creates a replay buffer, an actor-critic  networks and their corresponding target networks. 
        It also initializes the optimizer for both actor and critic networks alog with 
        counters to track the number of training iterations.
        """
        self.device=device

        self.replay_buffer = Replay_buffer()
        
        self.actor = Actor(state_dim, action_dim, hidden1).to(self.device)
        self.actor_target = Actor(state_dim, action_dim,  hidden1).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim,  hidden2).to(self.device)
        self.critic_target = Critic(state_dim, action_dim,  hidden2).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5*1e-3)
        # learning rate

        

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.batch_size = batch_size

    def select_action(self, state):
        """
        takes the current state as input and returns an action to take in that state. 
        It uses the actor network to map the state to an action.
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()


    def update(self):
        """
        updates the actor and critic networks using a batch of samples from the replay buffer. 
        For each sample in the batch, it computes the target Q value using the target critic network and the target actor network. 
        It then computes the current Q value 
        using the critic network and the action taken by the actor network. 
        
        It computes the critic loss as the mean squared error between the target Q value and the current Q value, and 
        updates the critic network using gradient descent. 
        
        It then computes the actor loss as the negative mean Q value using the critic network and the actor network, and 
        updates the actor network using gradient ascent. 
        
        Finally, it updates the target networks using 
        soft updates, where a small fraction of the actor and critic network weights are transferred to their target counterparts. 
        This process is repeated for a fixed number of iterations.
        """

        # for it in range(update_iteration):
        # For each Sample in replay buffer batch
        state, next_state, action, reward, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(1-done).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (done * gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss as the negative mean Q value using the critic network and the actor network
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        
        """
        Update the frozen target models using 
        soft updates, where 
        tau,a small fraction of the actor and critic network weights are transferred to their target counterparts. 
        """
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
           
            # self.num_actor_update_iteration += 1
            # self.num_critic_update_iteration += 1
    def save(self):
        """
        Saves the state dictionaries of the actor and critic networks to files
        """
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        torch.save(self.actor_target.state_dict(), directory + 'actor_target.pth')
        torch.save(self.critic_target.state_dict(), directory + 'critic_target.pth')

    def load(self,model_directory):
        """
        Loads the state dictionaries of the actor and critic networks to files
        """
        self.actor.load_state_dict(torch.load(model_directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(model_directory + 'critic.pth'))