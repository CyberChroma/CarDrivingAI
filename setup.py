import os
import neat
import pickle

from multiProcessGeneration import generation
from logger import *

def setup(isRestore = False):
    p = None
    generations = 1365
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward-car")

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    winner = None
    
    if isRestore:
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-1364')
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))
        winner = p.run(generation, generations - p.generation)
        stats.save()
    
    else:
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))
        winner = p.run(generation, generations)
        stats.save()

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
    f.close()

if __name__ == '__main__':
    setup(isRestore=False)
