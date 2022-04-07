from __future__ import print_function
import os
import neat

import gym
from gym import spaces
import numpy as np

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]


# def eval_genomes(genomes, config):
    


# def run(config_file):
#     # Load configuration.
#     config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                          config_file)

#     # Create the population, which is the top-level object for a NEAT run.
#     p = neat.Population(config)

#     # Add a stdout reporter to show progress in the terminal.
#     p.add_reporter(neat.StdOutReporter(True))
#     stats = neat.StatisticsReporter()
#     p.add_reporter(stats)
#     p.add_reporter(neat.Checkpointer(5))

#     # Run for up to 300 generations.
#     winner = p.run(eval_genomes, 300)

#     # Display the winning genome.
#     print('\nBest genome:\n{!s}'.format(winner))

#     # Show output of the most fit genome against training data.
#     print('\nOutput:')
#     winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
#     for xi, xo in zip(xor_inputs, xor_outputs):
#         output = winner_net.activate(xi)
#         print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

#     node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
#     # visualize.draw_net(config, winner, True, node_names=node_names)
#     # visualize.plot_stats(stats, ylog=False, view=True)
#     # visualize.plot_species(stats, view=True)

#     p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
#     p.run(eval_genomes, 10)


# def main2():
#     if __name__ == '__main__':
#         # Determine path to configuration file. This path manipulation is
#         # here so that the script will run successfully regardless of the
#         # current working directory.
#         local_dir = os.path.dirname(__file__)
#         config_path = os.path.join(local_dir, 'config-feedforward-car')
#         run(config_path)









# def generateAction(observation):
#     inputLayers = []

#     hiddenLayers = []

#     # Do calculations

#     outputLayers = [-1, 1, 0]
#     return outputLayers
#     #return np.random.uniform(-1,1,4)




population = 0
envs = []
observations = []

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

    for i in range(population):
        envs.append(gym.make('CarRacing-v0'))
        observations.append(envs[i].reset())

    winner = p.run(main, generations)


def main(genomes = [], config = []):
    actions = []
    rewards = []

    nets = []
    cars = []

    # if __name__ == "__main__":
    for i in range(population):
        observations[i] = envs[i].reset()
    # observations[0] = envs[0].reset()

    for genome_id, genome in genomes:
        # genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(genome)
        genome.fitness = 0
        
        # # for xi, xo in zip(xor_inputs, xor_outputs):
        # for observation in observations:
        #     action = net.activate(observation)

        #     actions.append(action)
        #     rewards.append(0)
        #     ge.append(genome)
        #     observations[genome_id], rewards[genome_id], done, info = envs[genome_id].step(actions[genome_id])

        #     # genome.fitness -= (output[0] - xo[0]) ** 2
        #     genome.fitness = rewards[genome_id]

        # reset the environment after each episode


    done = False

    # each episode has timesteps
    timestep = 0
    fail = [False] * len(envs)
    allFailed = False

    while not done:
        # render each environment
        # for i in range(population):
        #     envs[i].render()
        envs[0].render()

        actions = []
        rewards = []
        
        for i in range(len(envs)):
            rewards.append(0)
            if not fail[i]:
                inputs = []
                for col in range(0, 96, 4):
                    numGreens = 0
                    for row in range(0, 96, 4):
                        if (observations[i][row][col][1] > observations[i][row][col][0] and observations[i][row][col][1] > observations[i][row][col][2]):
                            numGreens += 1
                    inputs.append(numGreens / 24)

                # for x in range(len(observations[i])):
                #     for y in range(len(observations[i][x])):
                #         #for c in range(len(observations[i][x][y])):
                #         inputs.append(((observations[i][x][y][0] + observations[i][x][y][1] + observations[i][x][y][2]) / 3) / 255)

                action = nets[i].activate(tuple(inputs))
                action[1] = (action[1] / 2) + 0.5
                action[2] = (action[2] / 2) + 0.5

                actions.append(action)

                observations[i], rewards[i], done, info = envs[i].step(actions[i])
                cars[i].fitness += rewards[i]

                if cars[i].fitness < -10:
                    fail[i] = True
                    allFailed = True
                    for j in range(len(fail)):
                        if not fail[j]:
                            allFailed = False

                if i == 0:
                    print("Timestep: " + str(timestep) + ", Reward: "+str(cars[0].fitness) + ", Action: " + str(actions[0]))
            
            else:
                actions.append([0, 0, 0])
                if i == 0:
                    print("Timestep: " + str(timestep) + ", Car 0 has failed!")

            #for i in range(population):
                # generate action for each agent
                #actions.append(generateAction(observations[i]))
                # take action and get new observation and reward

        # Go to the next step
        timestep += 1

        if done or timestep > 1000 or allFailed:
            print(f"Episode finished after {timestep+1} timesteps")
            break

    # close the environments
    for i in range(population):
        envs[i].close()

setup()