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
import pickle

bestFromLastGen = 0
p = None

def setup(isRestore = False):
    generations = 370
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward-car")

    # NEAT function
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # for i in range(population):
    #     envs.append(gym.make('CarRacing-v0'))
    #     envs[i].reset()
    global p
    winner = None

    if isRestore:
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-369')
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))
        winner = p.run(runGeneration, generations - p.generation)
    else:
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))
        winner = p.run(runGeneration, generations)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
    f.close()



def replay_genome(config_path="config-feedforward-car", genome_path="winner.pkl"):
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    # Convert loaded genome into required data structure
    genomes = [(1, genome)]
    # Call game with only the loaded genome
    runGeneration(genomes, config)


def runGeneration(genomes = [], config = []):
    global bestFromLastGen
    processes = []
    fitnesses = []

    #Setup and run processes
    for i in range(len(genomes)):
        fitnesses.append(Value('d', 0.0))
        net = neat.nn.FeedForwardNetwork.create(genomes[i][1], config)
        #singleGenerationProcess(i, net, genomes[i][1], config)
        draw = False
        if i == bestFromLastGen:
            draw = True
        processes.append(Process(target=singleGenerationProcess, args=(i, net, genomes[i][1], fitnesses[i], draw)))
        processes[i].start()
    
    curMaxFitness = fitnesses[0].value
    curMaxI = 0

    # join the threads
    for i in range(len(genomes)):
        processes[i].join()
        # print(f"Passed fitness: {fitnesses[i].value} from child: {i}")
        if curMaxFitness < fitnesses[i].value:
            curMaxFitness = fitnesses[i].value
            curMaxI = i
        genomes[i][1].fitness = fitnesses[i].value

    bestFromLastGen = curMaxI

def singleGenerationProcess(processID, net, car, fitness, draw):
    env = gym.make('CarRacing-v0')
    observation = env.reset()

    car.fitness = 0

    done = False

    # each episode has timesteps
    timestep = 0

    while not done:
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

        # image = np.zeros((84, 96))
        # for row in range(0, 84, 2):
        #     imageRow = []
        #     for col in range(0, 96, 2):
        #         if observation[row][col][1] > observation[row][col][0] and observation[row][col][1] > observation[row][col][2]:
        #             image[row][col] = -1
        #             #imageRow.append(-1)
        #         else:
        #             image[row][col] = 1
        #             #imageRow.append(1)
        #     #image.append(imageRow)


        obervationCut = observation[0:84,:,:]
        gray_image = cv2.cvtColor(obervationCut, cv2.COLOR_RGB2GRAY)
        edgeImage = cv2.Canny(gray_image, 80, 150, apertureSize=3)
        kernel = np.ones((3,3), np.uint8)
        thickLineImage = cv2.dilate(edgeImage, kernel, iterations = 1)

        inputs = lineDirections(thickLineImage)

        # if inputs[0] == 0 or inputs[4] == 0:
        #     car.fitness -= 1

        action = net.activate(tuple(inputs))
        action[0] = (action[0] - 0.5) * 2

        observation, reward, done, info = env.step(action)
        car.fitness += reward

        if draw:
            # if timestep % 10 == 0:
            if timestep == 50:
                obervationCut = observation[0:84,:,:]
                gray_image = cv2.cvtColor(obervationCut, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray_image, 80, 150, apertureSize=3)
                cv2.imshow('Original', obervationCut)
                cv2.imshow('Canny Edges', edges)
                cv2.waitKey(0)

                # plt.imshow(edges, cmap='gray')
                # plt.show()
            # img = ax.imshow(image, cmap='gray')
            # fig.canvas.draw()
            # plt.show()
                print("Car No: " + str(processID) + ", Timestep: " + str(timestep) + ", Reward: " + str(car.fitness) + ", Action: " + str(action) + ", Inputs: " + str(inputs))

        # Go to the next step
        timestep += 1

        if done:
            # print(f"Episode finished after {timestep+1} timesteps")
            break

    # close the environments
    env.close()
    fitness.value = car.fitness


def lineDirections(image):
    inputs = [0, 0, 1, 0, 0]
    carPosRow = 62
    carPosCol = 48
    curPos = [carPosRow, carPosCol]

    # check left
    for i in range(0, 40):
        curPos[1] -= 1
        if image[carPosRow][curPos[1]] == 255:
            inputs[0] = 1 - (abs(carPosCol - curPos[1]) / 40)
            break

    curPos = [carPosRow, carPosCol]
    
    # check up-left
    for i in range(0, 40):
        curPos[0] -= 1
        curPos[1] -= 1
        if image[curPos[0]][curPos[1]] == 255:
            inputs[1] = 1 - (abs(carPosCol - curPos[1]) / 40)
            break

    curPos = [carPosRow, carPosCol]

    # check up
    for i in range(0, 40):
        curPos[0] -= 1
        if image[curPos[0]][carPosCol] == 255:
            inputs[2] = (abs(curPos[0] - carPosRow) / 40)
            break

    curPos = [carPosRow, carPosCol]

    # check up-right
    for i in range(0, 40):
        curPos[0] -= 1
        curPos[1] += 1
        if image[curPos[0]][curPos[1]] == 255:
            inputs[3] = 1 - (abs(curPos[1] - carPosCol) / 40)
            break
        
    curPos = [carPosRow, carPosCol]
    
    # check right
    for i in range(0, 40):
        curPos[1] += 1
        if image[carPosRow][curPos[1]] == 255:
            inputs[4] = 1 - (abs(curPos[1] - carPosCol) / 40)
            break

    return inputs


def promote_center(image):
    inputs = [-1, 1]
    carPosRow = 62
    carPosCol = 48
    curPos = [carPosRow, carPosCol]
    curPixelColor = 0
        

    leftDis = -1
    # check left
    for i in range(0, 46):
        curPos[1] -= 1
        curPixelColor = image[carPosRow][curPos[1]]
        if curPixelColor == 255:
            leftDis = (carPosCol - curPos[1]) / 48
            #inputs[0] = 48 - (carPos[1] - curPos[1])
            break

    curPos = [carPosRow, carPosCol]
    curPixelColor = 0

    rightDis = -1
    # check right
    for i in range(0, 46):
        curPos[1] += 1
        curPixelColor = image[carPosRow][curPos[1]]
        if curPixelColor == 255:
            rightDis = (curPos[1] - carPosCol) / 48
            #inputs[2] = 48 - (curPos[1] - carPos[1])
            break

    curPos = [carPosRow, carPosCol]
    curPixelColor = 0

    if leftDis != -1 and rightDis != -1:
        inputs[0] = 1-abs(leftDis-rightDis)

    # check up
    for i in range(0, 20):
        curPos[0] -= 1
        curPixelColor = image[curPos[0]][carPosCol]
        if curPixelColor == 255:
            inputs[1] = 1 - (curPos[0] - carPosRow) / 20
            break
    
    return inputs

if __name__ == '__main__':
    setup(isRestore=False)
    # replay_genome()