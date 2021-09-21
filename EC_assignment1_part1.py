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


def run(config_path):

    # To specify how many islands to use
    number_of_islands = 2
    # the amount of generations it is run for
    amount_generations = 10 
    # After how many generations an individual migrates
    migration_interval = 1 # for testing

    #TODO: specify population size here, not in config (this is only for if-statement in migration part, not the config yet)
    population_size = 10

    # building from the configuration path
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)


    if number_of_islands >= 1:
        # builds the population of genomes
        p1 = neat.Population(config)
        # statistics during run
        p1.add_reporter(neat.StdOutReporter(True))
        p1.add_reporter(neat.StatisticsReporter())
    if number_of_islands >= 2:
        p2 = neat.Population(config)
        p2.add_reporter(neat.StdOutReporter(True))
        p2.add_reporter(neat.StatisticsReporter())
    if number_of_islands >= 3:
        p3 = neat.Population(config)
        p3.add_reporter(neat.StdOutReporter(True))
        p3.add_reporter(neat.StatisticsReporter())
    if number_of_islands >= 4:
        p4 = neat.Population(config)
        p4.add_reporter(neat.StdOutReporter(True))
        p4.add_reporter(neat.StatisticsReporter())
    if number_of_islands >= 5:
        p5 = neat.Population(config)
        p5.add_reporter(neat.StdOutReporter(True))
        p5.add_reporter(neat.StatisticsReporter())

    # Alle if statements wel heel lelijk, maar efficienter dan for loop denk ik

    print("p1.population  =  ", p1.population)

    # winner =  winning genomes
    for generation in range(1, amount_generations):
        if number_of_islands >= 1:
            winner_p1 = p1.run(fitness_function=eval_genomes, n=1)
            pop1 = p1.population.values()
            print('Length pop1  =  ', len(p1.population))
            print('winner_p1  =  ', winner_p1)
            print("p1.population  =  ", p1.population)
        if number_of_islands >= 2:
            winner_p2 = p2.run(fitness_function=eval_genomes, n=1)
            pop2 = p2.population.values()
            print('Length pop2  =  ', len(p2.population))
            print('winner_p2  =  ', winner_p2)
            print("p2.population  =  ", p2.population)
        if number_of_islands >= 3:
            winner_p3 = p3.run(fitness_function=eval_genomes, n=1)
            pop3 = p3.population.values()
        if number_of_islands >= 4:
            winner_p4 = p4.run(fitness_function=eval_genomes, n=1)
            pop4 = p4.population.values()
        if number_of_islands >= 5:
            winner_p5 = p5.run(fitness_function=eval_genomes, n=1)
            pop5 = p5.population.values()

        print("p1.population  =  ", p1.population)


        # MIGRATION INTERVAL
        if generation % migration_interval == 0:
            print("\nINTERVAL")
            #TODO: migreren we altijd de winner?
            #TODO: waar plaatsen we de winnaar? nu altijd op plaats 1 (maar zou ook winnaar van die populatie kunnen zijn)
            #TODO: MIGRATION SIZE ( nu 1 )

            # Generation begint op 1
            # Populatie begint ook op 1 (1e generatie is bvb 1-10, niet 0-9)
            # Na elke generatie verschuift de index mee met de grootte van de populatie (bvb 2e generatie : 11-20)
            if number_of_islands == 2:
                p1.population[(generation*population_size)+1] = winner_p2 
                p2.population[(generation*population_size)+1] = winner_p1 
            if number_of_islands == 3:
                p1.population[(generation*population_size)+1] = winner_p3 
                p2.population[(generation*population_size)+1] = winner_p1
                p3.population[(generation*population_size)+1] = winner_p2
            if number_of_islands == 4:
                p1.population[(generation*population_size)+1] = winner_p4 
                p2.population[(generation*population_size)+1] = winner_p1
                p3.population[(generation*population_size)+1] = winner_p2
                p4.population[(generation*population_size)+1] = winner_p3
            if number_of_islands == 5:
                p1.population[(generation*population_size)+1] = winner_p5
                p2.population[(generation*population_size)+1] = winner_p1
                p3.population[(generation*population_size)+1] = winner_p2
                p4.population[(generation*population_size)+1] = winner_p3
                p5.population[(generation*population_size)+1] = winner_p4
            
            

if __name__ == "__main__":
    # gives us the path to the directory we are in 
    local_dir = os.path.dirname(__file__)
    # finds the absolute path of config-EC.txt
    config_path = os.path.join(local_dir, "config-EC1.txt")
    run(config_path)
