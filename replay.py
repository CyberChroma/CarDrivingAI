import neat
import pickle

from generation import generation

def replay_genome(config_path="config-feedforward-car", genome_path="winner.pkl"):
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    # Convert loaded genome into required data structure
    genomes = [(1, genome)]
    # Call game with only the loaded genome
    generation(genomes, config)

if __name__ == '__main__':
    replay_genome()