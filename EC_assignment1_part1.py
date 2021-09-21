# Evolutionary Computing Standard Assigment 1 - Part I
# Using NEAT with default fitness function (from EvoMan)

# imports framework
import sys, os
import neat
import numpy as np
from neat import config

sys.path.insert(0, 'evoman')
from environment import Environment
from EC1_controller import player_controller

class Environment_1(Environment):
    def fitness_single(self):
        return 0.9*(100 - self.get_enemylife()) + 0.1*self.get_playerlife() - np.log(self.get_time())

experiment_name = 'EC_assignment1_part1'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# if headless = True : Will not run visuals (hence be faster)
headless = False
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# initializes environment for ai player against one static enermy
env = Environment_1(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(),
                  # only against 1st enemy (out of 8)
                  enemies=[3],
                  # possible: "normal" or "fastest"
                  speed="fastest",
                  enemymode="static",
                  # must be 2 according to assignment
                  level=2)

# genome : current population

def simulate(env, x):
    # f = fitness, p = player life, e = enemy life, t = game run time
    [f, p, e, t] = env.play(pcont=x)
    return f


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # creates a neural network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # computes the fitness for each genome
        genome.fitness = simulate(env, net)


def run(config_path):
    # building from the configuration path
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)

    # builds the population of genomes
    p1 = neat.Population(config)
    p2 = neat.Population(config)
    p3 = neat.Population(config)
    p4 = neat.Population(config)

    # statistics during run
    p1.add_reporter(neat.StdOutReporter(True))
    p1.add_reporter(neat.StatisticsReporter())
    p2.add_reporter(neat.StdOutReporter(True))
    p2.add_reporter(neat.StatisticsReporter())
    p3.add_reporter(neat.StdOutReporter(True))
    p3.add_reporter(neat.StatisticsReporter())
    p4.add_reporter(neat.StdOutReporter(True))
    p4.add_reporter(neat.StatisticsReporter())

    # the amount of generations it is run for
    amount_generations = 100
    # winner =  winning genomes
    for i in range(amount_generations):
        winner_p1 = p1.run(fitness_function=eval_genomes, n=1)
        pop1 = p1.population.values()
        winner_p2 = p2.run(fitness_function=eval_genomes, n=1)
        pop2 = p1.population.values()
        winner_p3 = p3.run(fitness_function=eval_genomes, n=1)
        pop3 = p1.population.values()
        winner_p4 = p4.run(fitness_function=eval_genomes, n=1)
        pop4 = p1.population.values()

if __name__ == "__main__":
    # gives us the path to the directory we are in 
    local_dir = os.path.dirname(__file__)
    # finds the absolute path of config-EC.txt
    config_path = os.path.join(local_dir, "config-EC1.txt")
    run(config_path)
