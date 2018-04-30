import numpy as np
import matplotlib.pyplot as plt

# Network
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Optimizer
import torch.optim as optim

# Game setup
num_types = 3    # Number of item types
max_item = 5     # Maximum number of each item in a pool
max_utility = 10 # Maximum utility value for agents

# Linguistic channel
num_vocab = 10   # Symbol vocabulary size for linguistic channel
len_message = 6  # Linguistic message length

# Appendix
lambda1 = 0.05  # Entropy regularizer for pi_term, pi_prop
lambda2 = 0.001 # Entropy regularizer for pi_utt
smoothing_const = 0.7 # Smoothing constant for the exponential moving average baseline

# Sample an item pool for a game
def create_item_pool(num_types, max_item):
    # Possible to have zero items?
    pool = np.random.randint(0, max_item+1, num_types)
    
    return pool
        
# Sample agent utility
def create_agent_utility(num_types, max_utility):
    utility = np.zeros(num_types) # Initialize zero vector
    
    while np.sum(utility) == 0:   # At least one item has non-zero utility
        utility = np.random.randint(0, max_utility+1, num_types)
        
    return utility

# Calculate reward
def reward(share, utility):
    return np.dot(utility, share)
    
    class combined_policy(nn.Module):
    def __init__(self, embedding_dim = 100, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False):
        super(combined_policy, self).__init__()
        
        # Numerical encoder
        self.encoder1 = nn.Embedding(max_utility, embedding_dim)
        # Linguistic encoder
        self.encoder2 = nn.Embedding(num_vocab, embedding_dim)
        
        # Item context LSTM
        self.lstm1 = nn.LSTM(embedding_dim, embedding_dim, num_layers, bias, batch_first, dropout, bidirectional)
        # Linguistic LSTM
        self.lstm2 = nn.LSTM(embedding_dim, embedding_dim, num_layers, bias, batch_first, dropout, bidirectional)
        # Proposal LSTM
        self.lstm3 = nn.LSTM(embedding_dim, embedding_dim, num_layers, bias, batch_first, dropout, bidirectional)
        
        # Feed-forward
        self.ff = nn.Linear(embedding_dim, embedding_dim)
        
        # Termination policy
        self.policy_term = nn.Linear(embedding_dim, 1)
        # Linguistic policy
        self.policy_ling = nn.LSTM(embedding_dim, embedding_dim, num_layers, bias, batch_first, dropout, bidirectional)
        # Proposal policies
        self.policy_prop = []
        for i in range(num_types):
            ff = nn.Linear(embedding_dim, 1)
            self.policy_prop.append(ff)
        
    def forward(self, x):
        # Item context
        x1 = x[0]
        # Previous linguistic message
        x2 = x[1]
        # Previous proposal
        x3 = x[2]

        # Initial embedding
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x3 = self.encoder1(x1)
        
        # LSTM
        hidden = torch.zeros(1,embedding_dim) # Initial hidden
        
        x1, hidden = self.lstm1(x1,hidden)
        x2, hidden = self.lstm2(x2,hidden)
        x3, hidden = self.lstm3(x3,hidden)
        
        # Concatenate
        x = np.hstack([x1,x2,x3])
        
        # Feedforward
        h = self.ff(x)
        h = F.relu(h)
        
        # Termination probability
        p_term = F.sigmoid(self.policy_term(h))
        # Linguistic construction
        hidden = torch.zeros(1,num_vocab)
        
        
        
        # Proposal probability
        p_prop = torch.zeros(1,num_types)
        for i in range(num_types):
            p_prop[i] = F.sigmoid(self.policy_prop[i](h))