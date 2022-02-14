import gym
from gym import spaces
import numpy as np


def generateAction():
    return np.random.uniform(-1,1,4)


env = gym.make('CarRacing-v0')

observation = env.reset()

for episode in range(1):
    # reset the environment after each episode
    observation = env.reset()

    done = False

    # each episode has timesteps
    timestep = 0
    while not done:
        env.render()

        print(observation)

        # get a random sample from action space
        action = generateAction()
        # action = env.action_space.sample()

        # step returns four values: observation, reward, done, info
        observation, reward, done, info = env.step(action)

        timestep += 1

        if done:
            print(f"Episode finished after {timestep+1} timesteps")
            break

env.close()
