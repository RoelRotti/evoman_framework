# Evolutionary Computing Standard Assigment 1 - Part I
# Using NEAT with default fitness function (from EvoMan)

# imports framework
import sys, os
import neat
import numpy as np
import random
import copy
from neat import config
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt # for data visualization
import pickle

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
                  enemies=[8],
                  # possible: "normal" or "fastest"
                  speed="fastest",
                  enemymode="static",
                  randomini = "no",
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
    # modifies populations by migrating genomes to other populations
    for j in range(len(populations)):
        for _ in range(n_migrations):
            # chooses candidates from genomes that are picked out from other populations
            chosen_population = j+1
            if j+1 >= len(populations):
                chosen_population = 0
            candidate = random.sample(candidates[chosen_population], 1)[0]
            candidates[chosen_population].remove(candidate)
            # determines where to insert migrated genome (NOTE 3 genomes are deleted because of this, could be improved)
            key = random.choice(list(populations[j].population.keys()))
            # without making a deep copy, unwanted genomes are modified
            candidate = copy.deepcopy(candidate)
            # whole genome cannot be transferred directly so key,nodes and connections are seperately
            populations[j].population[key].connections = candidate.connections
            populations[j].population[key].nodes = candidate.nodes
            populations[j].population[key].key = key
    return populations


# To specify how many islands to use
number_of_islands = 1
number_of_runs = 1

def run(config_path):#, df, n_run):
    # the amount of generations it is run for
    amount_generations = 20#20
    # After how many generations an individual migrates
    migration_interval = 6 #6 # for testing
    # How many migrations should be performed each epoch
    number_of_migrations = 3
    # building from the configuration path
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)

    populations = []
    stats = []

    # creating populations
    for j in range(number_of_islands):
        populations.append(neat.Population(config))
        populations[j].add_reporter(neat.StdOutReporter(True))
        stats_single = neat.StatisticsReporter()
        stats.append(stats_single)
        #populations[j].add_reporter(neat.StatisticsReporter())
        populations[j].add_reporter(stats_single)

    # let generations play and migrate
    for i in range(int(amount_generations/migration_interval+amount_generations%migration_interval)):
        for j in range(number_of_islands):
            env.print_logs(f"Island:{j+1}")
            populations[j].run(fitness_function=eval_genomes, n=migration_interval)
        populations = migration(populations, number_of_migrations)

    # Gather fittest genome for testing repeatedly 5 times (for boxplot)
    best_genomes = []
    for j in range(number_of_islands):
        best_genomes.append( (stats[j].best_genome(), stats[j].best_genome().fitness) )
    #print('best_genomes     =   ', best_genomes)
    # Retrieve individual with best fitness
    best_genome = max(best_genomes,key=lambda item:item[1])[0]
    #[c.fitness for c in stats[i].most_fit_genomes]
    #print('best_genome     =   ', best_genome)
    #print('best_genome.fitness     =   ', best_genome.fitness)

    # Test 5 times:
    best_genome_fitness = []
    for test_round in range(5):
        config_path_single = os.path.join(local_dir, "config-single.txt")
        config_single = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path_single)
        # # creates a neural network
        net = neat.nn.FeedForwardNetwork.create(best_genome, config_single)
        [f, p, e, t] = env.play(pcont=net)
        best_genome_fitness.append(f)
        #print("best_genome_fitness  =   ", best_genome_fitness)

    # Calculate the mean of the fitnesses of 5 test runs of best genome after 1 run
    mean_best_fitness = np.mean(best_genome_fitness)

    #print("best_genome_fitness  =   ", best_genome_fitness)

    return mean_best_fitness


if __name__ == "__main__":
    # gives us the path to the directory we are in 
    local_dir = os.path.dirname(__file__)
    # finds the absolute path of config-EC.txt
    config_path = os.path.join(local_dir, "config-EC1.txt")

    # Deze miste volgens mij nog?
    # number_of_islands = 2

    mean_fitnesses_boxplot = []
    for i in range(number_of_runs):
        mean_fit_best_genome = run(config_path)
        mean_fitnesses_boxplot.append(mean_fit_best_genome)
        # open both files
        with open('EC_assignment1_part1/evoman_logs.txt', 'r+') as firstfile, open(f'EC_assignment1_part1/enemy{env.enemies[0]}_{number_of_islands}islands_run{i+1}.txt', 'a') as secondfile:
            for line in firstfile:
                secondfile.write(line)
            firstfile.truncate(0)

    ax = sns.boxplot(y=mean_fitnesses_boxplot).set_title("Fitness of best genome from {} runs".format(number_of_runs)) 
    plt.show()

    # Open the file for writing
    F = open('EC_assignment1_part1/enemy{}_{}islands_data_boxplots_.txt'.format(env.enemies[0], number_of_islands), 'w')
    # Use a list comprehension to convert the 
    # numbers to strings then join all strings 
    # using a new line
    F.write("\n".join([str(x) for x in mean_fitnesses_boxplot]))
    # Close the file
    F.close()

