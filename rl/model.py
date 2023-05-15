import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    The Actor model takes in a state observation as input and 
    outputs an action, which is a continuous value.
    
    It consists of four fully connected linear layers with ReLU activation functions and 
    a final output layer selects one single optimized action for the state
    """
    def __init__(self, n_states, action_dim, hidden1):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden1), 
            nn.ReLU(), 
            nn.Linear(hidden1, hidden1), 
            nn.ReLU(), 
            nn.Linear(hidden1, hidden1), 
            nn.ReLU(), 
            nn.Linear(hidden1, 1)
        )
        
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    """
    The Critic model takes in both a state observation and an action as input and 
    outputs a Q-value, which estimates the expected total reward for the current state-action pair. 
    
    It consists of four linear layers with ReLU activation functions, 
    State and action inputs are concatenated before being fed into the first linear layer. 
    
    The output layer has a single output, representing the Q-value
    """
    def __init__(self, n_states, action_dim, hidden2):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states + action_dim, hidden2), 
            nn.ReLU(), 
            nn.Linear(hidden2, hidden2), 
            nn.ReLU(), 
            nn.Linear(hidden2, hidden2), 
            nn.ReLU(), 
            nn.Linear(hidden2, action_dim)
        )
        
    def forward(self, state, action):
        return self.net(torch.cat((state, action), 1))