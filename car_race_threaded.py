from __future__ import print_function
from concurrent.futures import process
import os
import neat

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
# import threading
from multiprocessing import Process, Pipe, Value
import cv2

population = 0

def setup():
    generations = 100
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward-car")

    # NEAT function
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    global population
    population = len(p.population)

    p.add_reporter(neat.StdOutReporter(True))
    stats =neat.StatisticsReporter()
    p.add_reporter(stats)

    # for i in range(population):
    #     envs.append(gym.make('CarRacing-v0'))
    #     envs[i].reset()

    winner = p.run(runGeneration, generations)


def runGeneration(genomes = [], config = []):
    processes = []
    fitnesses = []

    #Setup and run processes
    for i in range(population):
        fitnesses.append(Value('d', 0.0))
        net = neat.nn.FeedForwardNetwork.create(genomes[i][1], config)
        #singleGenerationProcess(i, net, genomes[i][1], config)
        draw = False
        if i == 0:
            draw = True
        processes.append(Process(target=singleGenerationProcess, args=(i, net, genomes[i][1], fitnesses[i], config, draw)))
        processes[i].start()
    
    # join the threads
    for i in range(population):
        processes[i].join()
        # print(f"Passed fitness: {fitnesses[i].value} from child: {i}")
        genomes[i][1].fitness = fitnesses[i].value


def singleGenerationProcess(processID, net, car, fitness, config, draw):
    # env = gym.make('CarRacing-v0')
    # env.reset()
    #env = envs[processID]
    env = gym.make('CarRacing-v0')
    observation = env.reset()

    # for genome_id, genome in genomes:
        # genome.fitness = 4.0
    car.fitness = 0

    done = False

    # each episode has timesteps
    timestep = 0

    fig, ax = plt.subplots()

    while not done:
        # render each environment
        # for i in range(population):
        #     envs[i].render()
        if draw:
            env.render()
        
        # for col in range(0, 96, 4):
        #     numGreens = 0
        #     for row in range(0, 96, 4):
        #         if (observation[row][col][1] > observation[row][col][0] and observation[row][col][1] > observation[row][col][2]): # If pixel is green
        #             numGreens += 1
        #     inputs.append(numGreens / 24)

        # for row in range(len(observation)):
        #     for col in range(len(observation[row])):
        #         inputs.append(0)
                #inputs.append(((observation[row][col][0] + observation[row][col][1] + observation[row][col][2]) / 3) / 255)
        

        image = np.zeros((84, 96))
        for row in range(0, 84, 2):
            imageRow = []
            for col in range(0, 96, 2):
                if observation[row][col][1] > observation[row][col][0] and observation[row][col][1] > observation[row][col][2]:
                    image[row][col] = -1
                    #imageRow.append(-1)
                else:
                    image[row][col] = 1
                    #imageRow.append(1)
            #image.append(imageRow)

        inputs = [48, 64, 48]
        carPos = [66, 48]
        carPixelColor = image[carPos[0]][carPos[1]]
        # print("Car Pixel Color: ", carPixelColor)

        # # if pixel is 1, car is on road
        # inputs[0] = carPixelColor

        curPos = carPos.copy()
        curPixelColor = carPixelColor
        lastPixelColor = curPixelColor
        
        # check left
        for i in range(0, 46, 2):
            curPos[1] -= 2
            curPixelColor = image[curPos[0]][curPos[1]]
            if curPixelColor != lastPixelColor:
                inputs[0] = carPos[1] - curPos[1]
                break
            lastPixelColor = curPixelColor

        curPos = carPos.copy()
        curPixelColor = carPixelColor
        lastPixelColor = curPixelColor

        # check up
        for i in range(0, 64, 2):
            curPos[0] -= 2
            curPixelColor = image[curPos[0]][curPos[1]]
            if curPixelColor != lastPixelColor:
                inputs[1] = carPos[0] - curPos[0]
                break
            lastPixelColor = curPixelColor

        curPos = carPos.copy()
        curPixelColor = carPixelColor
        lastPixelColor = curPixelColor

        # check right
        for i in range(0, 46, 2):
            curPos[1] += 2
            curPixelColor = image[curPos[0]][curPos[1]]
            if curPixelColor != lastPixelColor:
                inputs[2] = curPos[1] - carPos[1]
                break
            lastPixelColor = curPixelColor


        if carPixelColor == -1:
            inputs[0] *= -1
            inputs[1] *= -1
            inputs[2] *= -1

        # image = np.asarray(image)



        # image = cv2.Canny(observation,150,250)
        # image = cv2.resize(image, (5, 4))

        #inputs = np.reshape(image, (-1, 1))

        # print(inputs.shape)

        action = net.activate(tuple(inputs))
        action[1] = (action[1] / 2) + 0.5
        action[2] = (action[2] / 2) + 0.5

        observation, reward, done, info = env.step(action)
        car.fitness += reward        

        if action[1] > 0.8 and action[2] < 0.2 and inputs[1] == 64:
            car.fitness += 1

        if processID == 0:
            # if timestep == 50:
                # cv2.imshow('Canny Edges', image)
                # cv2.waitKey(0)

                # plt.imshow(image, cmap='gray')
                # plt.show()
            # img = ax.imshow(image, cmap='gray')
            # fig.canvas.draw()
            # plt.show()
            print("Timestep: " + str(timestep) + ", Reward: " + str(car.fitness) + ", Action: " + str(action), "Inputs: " + str(inputs))

        # Go to the next step
        timestep += 1

        if done or timestep > 1000:
            # print(f"Episode finished after {timestep+1} timesteps")
            break

    # close the environments
    env.close()
    fitness.value = car.fitness

if __name__ == '__main__':
    setup()