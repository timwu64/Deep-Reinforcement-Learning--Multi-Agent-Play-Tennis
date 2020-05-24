# Deep Reinforcement Learning Nanodegree -- Project 3 "Collaboration and Competition"

[//]: # (Image References)

[image1]: https://github.com/timwu64/Deep-Reinforcement-Learning-Multi-Agent-Play-Tennis/blob/master/images/Tennis_Results.png "Tennis Result"

[image2]: https://github.com/timwu64/Deep-Reinforcement-Learning-Multi-Agent-Play-Tennis/blob/master/images/MADDPG.png "MADDPG"

### Tim Wu

### May 24th, 2020

## Introduction

In this project, I built the Multi Agent Reinforcement Learning (RL) to solve Tennis environment.
I created two angents that collaboratd in the spaces to play Tennis.  Each agent has its own observation, and those two agents collaboratively learn to play tennis.

In the following sessions of the report, I summarized how to build,
implement, fine tune and improve the learning of MADDPG.

Tennis

-   I am able to solve the Tennis environment in 2420 episodes with average score 0.5 using MADDPG
    agents

### Code Location

-   The Agent Class implemented in the (maddpg\_agent.py)

-   The MADDPG network implemented in the file (maddpg_model.py)

-   The model training and etc. implemented in the
    (Tennis.ipynb)

-   The input and hyperparameters save in the file (config.py)

-   The model weights are saved in the (MADDPG\_ckpt) folder

### Deep Deterministic Policy Gradient (DDPG) Implementation**

The solution is based on MADDPG architecture

The (Actor Critic) Network Architecture and Agent Hyperparameters

\[MODEL INFO\] Actor initialized with parameters:
state\_size=24
action\_size=2
seed=123
fc1\_units=128
fc2\_units=128

\[MODEL INFO\] CRITIC initialized with parameters:
state\_size=24
action\_size=2
seed=123
fcs1\_units=128
fc2\_units=128

The Agent Hyperparameters:

\[AGENT INFO\] DDPG constructor initialized parameters:
num\_agents = 2 
state\_size = 24 
action\_size = 2
seed = 123 \# random seed number
n\_episodes\_max = 10000 \# number of training episodes
max\_t = 1000 \# number of timesteps per episode
actor\_fc1\_units = 128 \# actor network hidden layer \#1 number of unit
actor\_fc2\_units = 128 \# actor network hidden layer \#2 number of unit
critic\_fcs1\_units = 128 \# critic network hidden layer \#1 number of
unit
critic\_fc2\_units = 128 \# critic network hidden layer \#2 number of
unit\
BUFFER\_SIZE = int(1e6) \# replay buffer size
BATCH\_SIZE = 128 \# minibatch size
GAMMA = 0.99 \# discount factor
TAU = 0.2 \# for soft update of target parameters
LR\_ACTOR = 2e-4 \# learning rate of the actor
LR\_CRITIC = 2e-4 \# learning rate of the critic
WEIGHT\_DECAY = 0.00 \# L2 weight decay
OU\_MU = 0.0 \# OUNoise mu
OU\_THETA = 0.15 \# OUNoise theta
OU\_SIGMA = 0.1 \# OUNoise sigma
UPDATE\_EVERY\_T\_STEPS = 2 \# timesteps between updates
NUM\_OF\_UPDATES = 1 \# num of update passes when updating

### Results

The best performing agent can solve the environment in 2420 episodes. The
file with the saved model weights of the best agent saved in the
checkpoint folder MADDPG\_ckpt

### The Best Agent Reacher Result:

Environment solved in 2420 episodes! Average Score: 0.50

![Tennis Result][image1]

### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
I refer to this paper - Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments, by Lowe and Wu, along with other researchers from OpenAI, UC Berkeley, and McGill University. 
![MADDPG][image2]

### Future Improvements

1.  Extensive hyperparameter optimization, fine tune the experience
    replay feeding buffer size and update frequency

2.  Add prioritized experience replay

3.  Apply more advance model like Twin Delayed DDPG (TD3)

4.  Exploration vs Exploitation - add EPS_START, EPS_EP_END, EPS_FINAL (OUNoise is not enough for strong exploration. I might want to add 100% random actions for the first 1000 episodes for improvement)

5.  Adjust Learning Interval

6.  Apply Gradient Clipping, such as torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)


### Reference
1.  <https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf> - Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (Lowe and Wu et al)

2.  <https://ai.atamai.biz/> - Reinforcement Learning with Pytorch course slides

3.  <https://www.superdatascience.com/pages/drl-2-resources-page> - Deep
    Reinforcement Learning 2.0

4.  Scott Fujimoto, Herke van Hoof, David Meger Addressing Function
    Approximation Error in Actor-Critic Methods
    <https://arxiv.org/pdf/1802.09477.pdf> arXiv:1802.09477v3 \[cs.AI\]
    22 Oct 2018
