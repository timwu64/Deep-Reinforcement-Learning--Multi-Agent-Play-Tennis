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

### The solution is based on Multi-Agent Deep Deterministic Policy Gradient (MADDPG) architecture
This project involve interaction between 2 agents, where emergent behavior and complexity arise from agents co-evolving together.  As stateed by "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" paper, traditional reinforcement learning approaches such as Q-Learning or policy gradient
are poorly suited to multi-agent environments. One issue is that each agent’s policy is changing
as training progresses, and the environment becomes non-stationary from the perspective of any
individual agent (in a way that is not explainable by changes in the agent’s own policy). This presents
learning stability challenges and prevents the straightforward use of past experience replay.  "MADDPG general-purpose multi-agent learning algorithm that: (1) leads to learned
policies that only use local information (i.e. their own observations) at execution time, (2) does
not assume a differentiable model of the environment dynamics or any particular structure on the
communication method between agents, and (3) is applicable not only to cooperative interaction
but to competitive or mixed interaction involving both physical and communicative behavior. The
ability to act in mixed cooperative-competitive environments may be critical for intelligent agents;
while competitive training provides a natural curriculum for learning, agents must also exhibit
cooperative behavior (e.g. with humans) at execution time" [1]

### What approraches have been use?

#### Experience Replay
One of the problems listed in Deepmind’s paper is that the agent sometimes face highly correlated state and actions and it makes hard converge.  Experience replay allows the RL agent to learn from past experience and give 2 major advantages. 
•	More efficient use of previous experience by learning with its multiple times. This is key when gaining real-world experience is costly, we can get full use of it. 
•	Better convergence behavior when training a function approximator.

Each experience is stored in a replay buffer as the agent interacts with the environment. The replay buffer contains a collection of experience tuples with the state, action, reward, and next state (s, a, r, s'). The agent then samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive learning algorithm could otherwise become biased by correlations between sequential experience tuples.

Therefore, I used the experience replay buffer that allows the RL agent to learn from past experience.  The implementation of the replay buffer can be found in the class Agent(): in the `maddpg_agent.py` file of the source code.  Both agents sharing same experience buffer, the experiences utilized by the central critic, which allow both of the agents to learn from each others' experiences.

#### Exploration vs Exploitation
One challenge with the Q-function is choosing which action to take while the agent is still learning the optimal policy.  Should the agent go for best decision vs. more information?  This is known as the exploration vs. exploitation dilemma.

- Ornstein-Uhlenbeck Noise: 
I added the Ornstein-Uhlenbeck noise to the agents for each time step as suggested in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING" paper.  Ornstein-Uhlenbeck process correlated to previous noise and therefore tends to stay in the same direction for longer durations.  I reduced the hyperparameters sigma: the volatility parameter to give the reasonable result.  Please check the detail code implemenation of the Ornstein-Uhlenbeck noise in the `maddpg_agent.py` file of the source code.

- Epsilon Greedy Algorithm:
I did not implement the Epsilon Greedy Algorithm in this project.  However, this fucntion can easily be added as future improvement.


#### Why choose the particular hyperparameters
I started the hyperparameters from using the paper "Human-level control through deep reinforcement learning" [Nature. 2015], "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING", and my previous Udacity Deep Reinforcement Learning projects. 

After experimenting with different numbers of hidden layers from [400,300] to [256, 256], [128, 128].  I concluded that 2 standard feed-forward 128 units layers with ReLu activation with Batch normalization give good results. With state space dimension = 24, this problem does not need high numbers of hidden layers and high number of units within the layers.

#### Actor/Critic model architecture
Another problem is that Deep Q-Networks can overestimate Q-values.  To solve this issue, we need to apply two neural networks, one neural network calculates the Target value and the other neural network chooses the best action.  In addition, besides high variance of gradients, another problem with policy gradients occurs trajectories have a cumulative reward of 0. The essence of policy gradient is increasing the probabilities for “good” actions and decreasing those of “bad” actions in the policy distributionhese issues contribute to the instability and slow convergence of vanilla policy gradient methods.  The combine the advantage of value base and policy based algorithms and imporve all the instabliites that mentationed above, I choose Actor/Critic for this project.

Using Actor/Critic 
- The “Critic” estimates the value function. This could be the action-value (the Q value) or state-value (the V value).
- The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients).

The Actor/Critic implementation can be found in the class Actor() & Critic() in the `maddpg_model.py` file of the source code

After experimenting with different numbers of hidden layers for the Actor/Critic model. I concluded that 2 standard feed-forward 128 units layers with ReLu activation give good results. With state space dimension of 24 does not need high numbers of hidden layers and high number of units within the layers.

The (Actor Critic) Network Architecture and Agent Hyperparameters shown below:

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
In this project, I used my previous work vanilla (DDPG) and expanded it with the Multi-Agent approach -refer to this paper "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", by Lowe and Wu, along with other researchers from OpenAI, UC Berkeley, and McGill University. 
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
