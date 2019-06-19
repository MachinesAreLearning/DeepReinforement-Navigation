# Deep Reinforcement Learning : Navigation

Repository for project Navigation for the Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

## Project Objective
The objective of this project is to train an agent to navigate a virtual world and collect as many yellow bananas as possible while avoiding blue bananas. The environment contents both yellow and blue banana as depicted in the animated gif below. or this project we have to train an agent to navigate a large square world and collect yellow bananas. The world contains both yellow and blue banana as depicted in the animated gif below.
![In Project 1, train an agent to navigate a large world.](images/Banana.gif)

### Rewards:
1. The agent is given a reward of +1 for collecting a yellow banana
1. Reward of -1 for collecting a blue banana.

### State Space 
Has 37 dimensions and the contains the agents velocity, along with ray-based precpetion of objects around the agents foward direction.

### Actions 
Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.


A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas. The task is episodic, and in order to solve the environment, In order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.


### Approach
The following approach is used in building an agent that solves this environment.
1.	Evaluate the state and action space.
2.	Establish baseline using a random action policy.
3.	Implement learning algorithm.
4.	Run experiments to measure agent performance.
5.	Select best performing agent and capture video of it navigating the environment.


## Baseline Model

Before building an agent that learns, I started by testing an agent that selects actions (uniformly) at random at each time step.


env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break

print("Score: {}".format(score))


Running this agent a few times resulted in scores from -2 to 2. Obviously, if the agent needs to achieve an average score of 13 over 100 consecutive episodes, then choosing actions at random won't work.

### Algorithm used to train the agent

![Algorithm.](images/DQN.png)
This algorithm screenshot is taken from the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)


#### Train a agent
There are 2 options for training the Agent:
1. Execute the provided notebook within this Nanodegree Udacity Online Workspace for "project #1  Navigation".
1. Or build your own local environment and make necessary adjustements for the path to the UnityEnvironment in the code.

Note: that the Workspace does not allow you to see the simulator of the environment; so, if you want to watch the agent while it is training, you should train locally.

#### Files and Folders
Navigation.ipynb: the main routine for the basic-banana project. It is the high-level calls to construct the environment, agent, and train.
dqn_agent.py: the class definitions of the agent implementing a DQN algorithm
model.py: the deep neural network model class is defined in this file.
replay_buffer: the class definition for the replay buffer used in DQN and double-DQN algorithms.
train.py: training routine
navigate.py: a routine to be called after the agent was trained to visualize its behaviour.
folder images: contains the figures generated for this project.
Report.pdf : Report file
