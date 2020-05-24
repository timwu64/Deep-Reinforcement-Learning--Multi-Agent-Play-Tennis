# Deep-Reinforcement-Learning-Multi-Agent-Collaboration-and-Competition

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

## Introduction

For this project, I trained the Multi-Agent RL to play tennis (Shown blow)

![Trained Agent][image1]

In this Unity ML-Agents Tennis Environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


## Project Starter Code

The original Udacity repo for this project can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

## Getting Started
### Descriptions
- `README.md` describe the environment, along with how to install the requirements.

### Code
- `Tennis.ipynb` control and train the Multi-Agent Deep Deterministic Policy Gradients (DDPG)
- `maddpg_agent.py` defines the Multi-Agent Deep Deep Deterministic Policy Gradients (DDPG) agent
- `maddpg_model.py` defines the Multi-Agent Deep Deep Deterministic Policy Gradients (DDPG) network architecture
- `config.py` defines the Multi-Agent Deep Deep Deterministic Policy Gradients (DDPG) input and hyperparameters

### Report
- `Report.md` describe the learning algorithm and the details of the implementation, along with ideas for future work.

### The Trained Agent
- `checkpoint_actor_0.pth` & `checkpoint_actor_1.pth` contains the weights for actor network for agent 0 and agent 1
- `checkpoint_critic_0.pth` & & `checkpoint_critic_1.pth` contains the weights for critic network for agent 0 and agent 1 for the Multi-Agent Deep Deterministic Policy Gradients (DDPG) implementation
- Please check `Report.md` for details

### Installation
- `python` folder contains the installation dependencies

  Follow the instructions below to explore the environment on your own machine! You will also learn how to use the Python API to control your agent.

  The Step by Step installation example shown below:

1. Clone the DRLND Repository

    If you haven't already, please follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

    (For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

2. Download the Unity Environment

    For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

      - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
      - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
      - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
      - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
      Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

    (For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

3. Create and activate a new environment with Python 3.6
    
     ###### Linux or Mac:
     
      `conda create --name drlnd python=3.6`
      
      `source activate drlnd`

     ###### Windows:

      `conda create --name drlnd python=3.6`
      
      `activate drlnd`

4. Clone the following repository 

    `git clone https://github.com/udacity/deep-reinforcement-learning.git`

    and start to run the first cell of the  to install all the dependencies

    or you can install the dependencies as below (optional)
   
   `cd ./python`
   
   `pip install .`


