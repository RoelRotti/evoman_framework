# Evolutionary Computing Standard Assigment 1 - Part I
# Using NEAT with default fitness function (from EvoMan)

# imports framework
import sys, os
import neat
from neat import config
sys.path.insert(0, 'evoman') 
from environment import Environment
from EC1_controller import player_controller

experiment_name = 'EC_assignment1_part1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# if headless = True : Will not run visuals (hence be faster)
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# initializes environment for ai player against one static enermy
env = Environment(experiment_name = experiment_name,
				  playermode = "ai",
				  player_controller = player_controller(),
                  # only against 1st enemy (out of 8)
                  enemies = [1],
                  # possible: "normal" or "fastest"
			  	  speed = "fastest",
				  enemymode = "static",
                  #must be 2 according to assignment
				  level = 2 )

#genome : current population

def simulate(env, x):
    #f = fitness
    #p = player life 
    #e = enemy life
    #t = game run time
    [f, p, e, t] = env.play(pcont = x)
    return f

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        #creates a neural network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # computes the fitness for each genome
        genome.fitness = simulate(env, net)

def run(config_path):
    # building from the configuration path
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path )
    # builds the population of genomes
    p = neat.Population(config)
    
    #statistics during run
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    
    # the amount of generations it is run for
    amount_generations = 100
    # winner =  winning genomes
    winner = p.run(fitness_function = eval_genomes, n = amount_generations)  
    # here you could print winnning genomes
if __name__ == "__main__":
    # gives us the path to the directory we are in 
    local_dir = os.path.dirname(__file__) 
    #finds the absolute path of config-EC.txt
    config_path = os.path.join(local_dir, "config-EC1.txt")
    run(config_path)