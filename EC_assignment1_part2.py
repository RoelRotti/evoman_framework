# Evolutionary Computing Standard Assigment 1 - Part I
# Using NEAT with default fitness function (from EvoMan)

# imports framework
import sys, os
import numpy as np

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

experiment_name = 'EC_assignment1_part1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# if headless = True : Will not run visuals (hence be faster)
headless = False
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 1

# initializes environment for ai player against one static enermy
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  # only against 1st enemy (out of 8)
                  enemies=[1],
                  # possible: "normal" or "fastest"
                  speed="fastest",
                  enemymode="static",
                  # must be 2 according to assignment
                  level=2)


from deap import base, creator
creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1))
creator.create("Individual", list, fitness=creator.FitnessMax)

import random
from deap import tools

IND_SIZE = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

toolbox = base.Toolbox()
toolbox.register("attribute", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def simulate(env, x):
    # f = fitness
    # p = player life
    # e = enemy life
    # t = game run time
    [f, p, e, t] = env.play(pcont=x)
    return f


def evaluate(genome):
    # computes the fitness for each genome
    genome.fitness = simulate(env, np.array(genome))

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    pop = toolbox.population(n=10)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 10
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return pop

if __name__ == "__main__":
    main()
