import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import sampling
from importlib import *
import time
import nets
import alive_sieve
import calc_rewards

class State(object):
    def __init__(self, N, pool, utilities):
        batch_size = N.size()[0]
        self.N = N
        self.pool = pool
        self.utilities = torch.zeros(batch_size, 2, 3).long()
        self.utilities[:, 0] = utilities[0]
        self.utilities[:, 1] = utilities[1]

        self.last_proposal = torch.zeros(batch_size, 3).long()
        self.m_prev = torch.zeros(batch_size, 6).long()

    def cuda(self):
        self.N = self.N.cuda()
        self.pool = self.pool.cuda()
        self.utilities = self.utilities.cuda()
        self.last_proposal = self.last_proposal.cuda()
        self.m_prev = self.m_prev.cuda()

    def sieve_(self, still_alive_idxes):
        self.N = self.N[still_alive_idxes]
        self.pool = self.pool[still_alive_idxes]
        self.utilities = self.utilities[still_alive_idxes]
        self.last_proposal = self.last_proposal[still_alive_idxes]
        self.m_prev = self.m_prev[still_alive_idxes]
        

def run_episode(
        batch,
        enable_cuda,
        enable_comms,
        enable_proposal,
        prosocial,
        agent_models,
        # batch_size,
        testing):
    """
    turning testing on means, we disable stochasticity: always pick the argmax
    """

    type_constr = torch.cuda if enable_cuda else torch
    batch_size = batch['N'].size()[0]
    s = State(**batch)
    if enable_cuda:
        s.cuda()

    sieve = alive_sieve.AliveSieve(batch_size=batch_size, enable_cuda=enable_cuda)
    actions_by_timestep = []
    alive_masks = []

    # next two tensofrs wont be sieved, they will stay same size throughout
    # entire batch, we will update them using sieve.out_idxes[...]
    rewards = type_constr.FloatTensor(batch_size, 3).fill_(0)
    num_steps = type_constr.LongTensor(batch_size).fill_(10) #128
    term_matches_argmax_count = 0
    utt_matches_argmax_count = 0
    utt_stochastic_draws = 0
    num_policy_runs = 0
    prop_matches_argmax_count = 0
    prop_stochastic_draws = 0
    entropy_loss_by_agent = [
        Variable(type_constr.FloatTensor(1).fill_(0)),
        Variable(type_constr.FloatTensor(1).fill_(0))
    ]
#     if render:
#         print('  ')
    term_probss = []
    message0 = []
    message1 = []
    message = [message0,message1]
    for t in range(10):
        agent = t % 2

        agent_model = agent_models[agent]
        if enable_comms:
            _prev_message = s.m_prev
        else:
            # we dont strictly need to blank them, since they'll be all zeros anyway,
            # but defense in depth and all that :)
            _prev_message = type_constr.LongTensor(sieve.batch_size, 6).fill_(0)
        if enable_proposal:
            _prev_proposal = s.last_proposal
        else:
            # we do need to blank this one though :)
            _prev_proposal = type_constr.LongTensor(sieve.batch_size, 3).fill_(0)
        
 #       print(_prev_message)
 #       print(_prev_proposal)
        nodes, term_a, s.m_prev, this_proposal, _entropy_loss, term_probs = agent_model(
            pool=Variable(s.pool),
            utility=Variable(s.utilities[:, agent]),
            m_prev=Variable(_prev_message),
            prev_proposal=Variable(_prev_proposal),
            testing=testing
        )
        entropy_loss_by_agent[agent] += _entropy_loss
        
        actions_by_timestep.append(nodes)
        message[agent].append(s.m_prev.clone())
        
        new_rewards = calc_rewards.calc_rewards(
            t=t,
            s=s,
            term=term_a
        )
        rewards[sieve.out_idxes] = new_rewards
        s.last_proposal = this_proposal
#         print(term_a.view(-1).nonzero().long().view(-1))
        sieve.mark_dead(term_a)
        sieve.mark_dead(t + 1 >= s.N)
#         print(1, sieve.alive_mask)
        alive_masks.append(sieve.alive_mask.clone())
#         print(2, sieve.alive_mask)
        sieve.set_dead_global(num_steps, t + 1) 
#         print(3, sieve.alive_mask)
        if sieve.all_dead():
            break
#        last_state = {"t":t, "N":s.N, "pool":s.pool, "utilities":s.utilities, "proposal":s.last_proposal, "m":s.m_prev}
        s.sieve_(sieve.alive_idxes)
#         print(4, sieve.alive_mask)
        sieve.self_sieve_()
#         print(5/, sieve.alive_mask)
#         print(term_a)
#     if render:
#         print('  r: %.2f' % rewards[0].mean())
#         print('  ')
    return actions_by_timestep, rewards, num_steps, alive_masks, entropy_loss_by_agent, term_probss, message