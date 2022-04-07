import neat
import time
from multiprocessing import Process, Value

from process import genomeProcess
from logger import *

bestFromLastGen = 0

def generation(genomes = [], config = []):
    genStartTime = time.time()

    global bestFromLastGen
    processes = []
    fitnesses = []

    #Setup and run processes
    for i in range(len(genomes)):
        fitnesses.append(Value('d', 0.0))
        net = neat.nn.FeedForwardNetwork.create(genomes[i][1], config)
        draw = False
        if i == bestFromLastGen:
            draw = True
        processes.append(Process(target=genomeProcess, args=(i, net, genomes[i][1], fitnesses[i], draw)))
        processes[i].start()
    
    curMaxFitness = fitnesses[0].value
    curMaxI = 0

    # join the processes
    for i in range(len(genomes)):
        processes[i].join()
        if curMaxFitness < fitnesses[i].value:
            curMaxFitness = fitnesses[i].value
            curMaxI = i
        genomes[i][1].fitness = fitnesses[i].value

    bestFromLastGen = curMaxI

    genEndTime = time.time()
    genTotalTime = genEndTime - genStartTime
    logTime(genTotalTime)

    # get best fitness
    bestFitness = fitnesses[0].value
    averageFitness = 0
    for fitness in fitnesses:
        averageFitness += fitness.value
        if fitness.value > bestFitness:
            bestFitness = fitness.value
        
    averageFitness /= len(fitnesses)
    logBestFitness(bestFitness)
    logAverageFitness(averageFitness)