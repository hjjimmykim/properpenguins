import numpy as np
import sys
import torch

# Sample number of turns for each game
def truncated_poisson_sampling(lam, min_N, max_N, num_games):
    # lam = poisson parameter
    # min_N = lower cutoff
    # max_N = upper cutoff
    # num_games = batch size
    # output = longtensor of shape num_games x 1
    
    # Truncated Poisson sampling
    N = np.random.poisson(lam,num_games)
    N = np.minimum(N,max_N)
    N = np.maximum(N,min_N)
    N = torch.from_numpy(N).view(num_games,1).long()
    
    return N

# Sample an item pool for each game
def create_item_pool(num_types, max_item, batch_size):
    # num_types = number of item types (int)
    # max_item = maximum number of each item in the pool (int)
    # batch_size = number of pools to generate (int)
    # output = item pools for each batch (longtensor of shape batch_size x num_types)
    
    # Note: possible to have zero items?
    pool = np.random.randint(0, max_item+1, (batch_size,num_types))
    return torch.from_numpy(pool).long()
        
# Sample agent utility for each game
def create_agent_utility(num_types, max_utility, batch_size):
    # num_types = number of item types (int)
    # max_utility = maximum utility of each item (int)
    # batch_size = number of pools to generate (int)
    # output = utility values for each batch (longtensor of shape batch_size x num_types)
    
    utility = np.zeros((batch_size,num_types)) # Initialize zero vector
    
    while 0 in np.sum(utility,1): # At least one item has to have non-zero utility
        utility = np.random.randint(0, max_utility+1, [batch_size, num_types])

    return torch.from_numpy(utility).long()

# Calculate reward (self-interested)
def rewards_func(share, utility, pool, log_p, baseline):
    # share = agent's share of the pool (longtensor of shape batch_size x num_types)
    # utility = agent's utility values for each item (")
    # pool = item pool (")
    # log_p = summed log likelihoods of chosen actions in the policy (float? tensor of shape batch_size x 1)
    # baseline = number
    
    # Note : When share > pool, reward should be 0 (see paper)
    
    # Dot product (for each batch) of utility & share, divided by maximum possible reward for normalization between [0,1]
    # sys.float_info.min to ensure no division by zero (pytorch seems to be prone to crashing in such scenarios)
    # Note: When max. possible reward = 0 (actual reward = 0 necessarily), above prescription will lead to zero reward, which is bad (compared to baseline); but the agents didn't really have any freedom of action so should they still be penalized?
    reward = torch.sum(utility*share,1).numpy()/(torch.sum(utility*pool,1).numpy()+sys.float_info.min)
    reward = torch.from_numpy(reward).view(-1,1) # Change shape to batch_size x 1
    reward = reward.float() # Convert to float tensor

    reward_loss = -log_p * (reward - baseline) # REINFORCE algorithm with baseline

    reward_loss = reward_loss.mean() # Average over batches
    
    return reward, reward_loss
