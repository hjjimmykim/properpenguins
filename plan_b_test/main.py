import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import sampling
from importlib import *
import nets
import utils
import calc_rewards
import alive_sieve
import numpy as np

        
test_seed = None
batch_size = 128
enable_cuda = True
enable_comms = True
enable_proposal = False
term_entropy_reg = 0.05
utterance_entropy_reg = 0.0001
proposal_entropy_reg = 0.005
train_r = np.random
prosocial = False

test_r = np.random.RandomState(test_seed)
test_batches = sampling.generate_test_batches(batch_size=batch_size, num_batches=5, random_state=test_r)
test_hashes = sampling.hash_batches(test_batches)

agent_models = []
agent_opts = []
for i in range(2):
    model = nets.AgentModel(
        enable_comms=enable_comms,
        enable_proposal=enable_proposal,
        term_entropy_reg=term_entropy_reg,
        utterance_entropy_reg=utterance_entropy_reg,
        proposal_entropy_reg=proposal_entropy_reg
    )
    if enable_cuda:
        model = model.cuda()
    agent_models.append(model)
    agent_opts.append(optim.Adam(params=agent_models[i].parameters()))
    
type_constr = torch.cuda if enable_cuda else torch

rewards_sum = type_constr.FloatTensor(3).fill_(0)
steps_sum = 0
count_sum = 0

baseline = type_constr.FloatTensor(3).fill_(0)
t_reward = []
t_reward1 = []
t_reward2 = []
for epoch in range(100000):
    batch = sampling.generate_training_batch(batch_size=batch_size, test_hashes=test_hashes, random_state=train_r)
    actions, rewards, steps, alive_masks, entropy_loss_by_agent, \
                term_probss, message   = utils.run_episode(
                batch=batch,
                enable_cuda=enable_cuda,
                enable_comms=enable_comms,
                enable_proposal=enable_proposal,
                agent_models=agent_models,
                prosocial=prosocial,
                testing=False)
    
    testing = False
    if not testing:
        for i in range(2):
            agent_opts[i].zero_grad()
        reward_loss_by_agent = [0, 0]
        baselined_rewards = rewards - baseline
        rewards_by_agent = []
        for i in range(2):
            if prosocial:
                rewards_by_agent.append(baselined_rewards[:, 2])
            else:
                rewards_by_agent.append(baselined_rewards[:, i])
        sieve_playback = alive_sieve.SievePlayback(alive_masks, enable_cuda=enable_cuda)
        for t, global_idxes in sieve_playback:
            agent = t % 2
            if len(actions[t]) > 0:
                for action in actions[t]:
                    _rewards = rewards_by_agent[agent]
                    _reward = _rewards[global_idxes].float().contiguous().view(
                        sieve_playback.batch_size, 1)
                    _reward_loss = - (action * Variable(_reward))
                    _reward_loss = _reward_loss.mean()
                    reward_loss_by_agent[agent] += _reward_loss
        for i in range(2):
            loss = entropy_loss_by_agent[i] + reward_loss_by_agent[i]
            #print(entropy_loss_by_agent[i])
            loss.backward()
            agent_opts[i].step()
            
    rewards_sum += rewards.sum(0)
    steps_sum += steps.sum()
    baseline = 0.7 * baseline + 0.3 * rewards.mean(0)
    count_sum += batch_size
    if epoch% 1 == 0:
        test_rewards_sum = 0
        test_rewards_sum1 = 0
        test_rewards_sum2 = 0
        for test_batch in test_batches:
            actions, test_rewards, steps, alive_masks, entropy_loss_by_agent, \
                 term_probss, message  = utils.run_episode(
                batch=test_batch,
                enable_cuda=enable_cuda,
                enable_comms=enable_comms,
                enable_proposal=enable_proposal,
                agent_models=agent_models,
                prosocial=prosocial,
                testing=True)
            test_rewards_sum += float(test_rewards[:, 2].mean())
            test_rewards_sum1 += float(test_rewards[:, 0].mean())
            test_rewards_sum2 += float(test_rewards[:, 1].mean())

    #     print('test reward=%.3f' % (test_rewards_sum / len(test_batches)))
        t_reward.append(test_rewards_sum / len(test_batches))
        t_reward1.append(test_rewards_sum1 / len(test_batches))
        t_reward2.append(test_rewards_sum2 / len(test_batches))
        np.savetxt("./test_rewards_TFT2_sum.csv", np.array(t_reward))
        np.savetxt("./test_rewards_TFT2_sum1.csv", np.array(t_reward1))
        np.savetxt("./test_rewards_TFT2_sum2.csv", np.array(t_reward2))
        for a in range(len(agent_models)):
            torch.save(agent_models[a].state_dict(), "./saved_models_TFT2_0500105_agent%i"%a) 
