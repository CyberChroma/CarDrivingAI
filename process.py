import gym
import cv2

from canny import cannyImageProcess
from imageCompression import imageCompressionProcess

def genomeProcess(processID, net, car, fitness, draw):
    env = gym.make('CarRacing-v0')
    observation = env.reset()

    car.fitness = 0
    timestep = 0
    done = False

    while not done:
        if draw:
            env.render()

        obervationCut = observation[0:84,:,:]
        grayImage = cv2.cvtColor(obervationCut, cv2.COLOR_RGB2GRAY)

        #inputs = cannyImageProcess(grayImage)
        #if draw and timestep == 50:
        inputs = imageCompressionProcess(grayImage)

        action = net.activate(tuple(inputs))
        action[0] = (action[0] - 0.5) * 2

        observation, reward, done, info = env.step(action)
        car.fitness += reward

        if draw:
            if timestep % 10 == 0:
                print("Car No: " + str(processID) + ", Timestep: " + str(timestep) + ", Reward: " + str(car.fitness) + ", Action: " + str(action)) #+ ", Inputs: " + str(inputs))

        # Go to the next step
        timestep += 1

    # close the environments
    env.close()
    fitness.value = car.fitness
