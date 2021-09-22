# Evolutionary Computing Standard Assigment 1 - Part I
# Using NEAT with default fitness function (from EvoMan)

# imports framework
import sys, os
import neat
import numpy as np
import random
import copy
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
headless = True
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


def migration(populations, n_migrations):
    candidates = []
    # takes random genomes out of a population and stores them in candidates
    for i in range(len(populations)):
        candidates.append(random.sample(populations[i].population.items(), n_migrations))
        # takes the genomes in a list instead of tuples in a list
        candidates[i] = [candidate[1] for candidate in candidates[i]]

    for j in range(len(populations)):
        for _ in range(n_migrations):
            a = list(range(len(populations)))
            a.remove(j)
            pop = random.choice(a)
            key = random.choice(list(populations[j].population.keys()))
            candidate = random.sample(candidates[pop], 1)[0]
            candidates[pop].remove(candidate)
            populations[j].population[key] = copy.deepcopy(candidate)
            populations[j].population[key].key = key
    return populations


def run(config_path):
    # To specify how many islands to use
    number_of_islands = 2
    # the amount of generations it is run for
    amount_generations = 10 
    # After how many generations an individual migrates
    migration_interval = 1 # for testing
    # How many migrations should be performed each epoch
    number_of_migrations = 3
    # building from the configuration path
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)

    populations = []

    # creating populations
    for j in range(number_of_islands):
        populations.append(neat.Population(config))
        populations[j].add_reporter(neat.StdOutReporter(True))
        populations[j].add_reporter(neat.StatisticsReporter())

    # let generations play and migrate
    for i in range(int(amount_generations/migration_interval)):
        for j in range(number_of_islands):
            populations[j].run(fitness_function=eval_genomes, n=migration_interval)
        populations = migration(populations, number_of_migrations)


if __name__ == "__main__":
    # gives us the path to the directory we are in 
    local_dir = os.path.dirname(__file__)
    # finds the absolute path of config-EC.txt
    config_path = os.path.join(local_dir, "config-EC1.txt")
    run(config_path)
