import neat
import time
import gym
import cv2

from canny import cannyImageProcess
from logger import *


def generation(genomes = [], config = []):
    genStartTime = time.time()

    nets = []
    cars = []
    envs = []
    observations = []

    # Setup and run processes
    for i in range(len(genomes)):
        net = neat.nn.FeedForwardNetwork.create(genomes[i][1], config)
        nets.append(net)
        cars.append(genomes[i][1])
        cars[i].fitness = 0
        genomes[i][1].fitness = 0
        envs.append(gym.make('CarRacing-v0'))
        observations.append(envs[i].reset())
    
    timestep = 0
    done = False
    while not done:
        # envs[0].render()

        for i in range(len(cars)):
            obervationCut = observations[i][0:84,:,:]
            grayImage = cv2.cvtColor(obervationCut, cv2.COLOR_RGB2GRAY)

            inputs = cannyImageProcess(grayImage)

            action = net.activate(tuple(inputs))
            action[0] = (action[0] - 0.5) * 2

            observation, reward, done, info = envs[i].step(action)
            cars[i].fitness += reward

            if i == 0 and timestep % 10 == 0:
                print("Timestep: " + str(timestep) + ", Reward: " + str(cars[0].fitness) + ", Action: " + str(action) + ", Inputs: " + str(inputs))

        # Go to the next step
        timestep += 1

    # close the environments
    for i in range(len(envs)):
        envs[i].close()
    
    genEndTime = time.time()
    genTotalTime = genEndTime - genStartTime
    logTime(genTotalTime)

    # get best fitness
    bestFitness = cars[0].fitness
    averageFitness = 0
    for car in cars:
        averageFitness += car.fitness
        if car.fitness > bestFitness:
            bestFitness = car.fitness
        
    averageFitness /= len(cars)
    logBestFitness(bestFitness)
    logAverageFitness(averageFitness)
    