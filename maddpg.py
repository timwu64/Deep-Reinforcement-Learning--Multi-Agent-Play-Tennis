# main code that contains the neural network setup
# policy + critic updates
# see ddpg_agent.py for other details in the network


import numpy as np
import random
import copy
from collections import namedtuple, deque

from mamodel import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

class MADDPG:
    def __init__(self, config):
        super(MADDPG, self).__init__()
        
        self.config = config
        # Replay memory
        self.memory = ReplayBuffer(self.config.buffer_size, self.config.batch_size,
                                   self.config.random_seed,self.config.num_agents)
        
        print("[AGENT INFO] MADDPG constructor initialized parameters:\n num_agents={} \n state_size={} \n action_size={} \n random_seed={} \n actor_fc1_units={} \n actor_fc2_units={} \n critic_fcs1_units={} \n critic_fc2_units={} \n buffer_size={} \n batch_size={} \n gamma={} \n tau={} \n lr_actor={} \n lr_critic={} \n weight_decay={} \n ou_mu={}\n ou_theta={}\n ou_sigma={}\n update_every_t_steps={}\n".format(self.config.num_agents, self.config.state_size, 
        self.config.action_size, self.config.random_seed, self.config.actor_fc1_units, self.config.actor_fc2_units, self.config.critic_fcs1_units, self.config.critic_fc2_units, self.config.buffer_size, self.config.batch_size, 
        self.config.gamma, self.config.tau, self.config.lr_actor, self.config.lr_critic, self.config.weight_decay, self.config.ou_mu, self.config.ou_theta, self.config.ou_sigma, self.config.update_every_t_steps))
        
        self.maddpg_agent = [Agent(self.config) for _ in range(self.config.num_agents)]
    
    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()
            
    def act(self, states, add_noise=True):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(state, add_noise=True) for agent, state in zip(self.maddpg_agent, states)]
        return np.array(actions).reshape(1, -1)
    
    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.config.batch_size and timestep % self.config.update_every_t_steps == 0:
            for agent in self.maddpg_agent:
                experiences = self.memory.sample()
                agent.learn(experiences, self.config.gamma)

    def save_agents(self):
        # save models for local actor and critic of each agent
        for i, agent in enumerate(self.maddpg_agent):
            torch.save(agent.actor_local.state_dict(),  f"checkpoint_actor_agent_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"checkpoint_critic_agent_{i}.pth")

