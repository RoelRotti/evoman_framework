# imports framework
import sys, os, random
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from matplotlib import pyplot
from scipy.optimize import differential_evolution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

experiment_name = 'EC_assignment1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 10

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[7],
                  multiplemode="no",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
C = 10

# Create a custom environment
def create_env(group):
    if len(group) > 1: multim = "yes" 
    else: multim = "no" 
    # initializes environment with ai player using random controller, playing against static enemy
    env = Environment(experiment_name=experiment_name,
                    enemies=group,
                    multiplemode=multim,
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest")
    return env

# define objective function
def obj(x, group):
    # create env
    envv = create_env(group)
    # f = fitness, p = player life, e = enemy life, t = game run time
    [f, p, e, t] = envv.play(pcont=x)
    return f


# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [np.clip(mutated[i], bounds[0], bounds[1]) for i in range(len(mutated))]
    return mutated_bound


# define crossover operation
def crossover(mutated, target, cr, method):
    if method == 'bin':
        # generate a uniform random value for every dimension
        p = np.random.rand(len(mutated))
        # generate trial vector by binomial crossover
        trial = [mutated[i] if p[i] < cr else target[i] for i in range(len(mutated))]
    else:
        # generate and sort two points in between the mutant vector is expressed
        two_points = np.random.randint(0, len(mutated)-1, 2)
        two_points = sorted(two_points)#two_points.sort()
        # generate trial vector by exponential crossover
        trial = [mutated[i] if two_points[0] <= i <= two_points[1] else target[i] for i in range(len(mutated))]
    return trial


# DE with 2 randomly chosen candidates and binomial crossover scheme
def differential_evolution(pop_size, bounds, n_generations, F, cr, group):

    # for plotting
    max_fitness_gen = []
    mean_fitness_gen = []

    # initialise population of candidate solutions randomly within the specified bounds
    pop = np.random.uniform(-1, 1, (pop_size, n_vars))
    # evaluate initial population of candidate solutions
    obj_all = [obj(ind, group) for ind in pop]
    # find the best performing vector of initial population
    best_vector = pop[np.argmax(obj_all)]
    best_obj = max(obj_all)
    prev_obj = best_obj
    # initialise list to store the objective function value at each iteration
    obj_iter = list()
    # run iterations of the algorithm
    for i in range(n_generations):
        # iterate over all candidate solutions
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], F)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = np.array(crossover(mutated, pop[j], cr, 'exp'))
            # compute objective function value for target vector
            obj_target = obj(pop[j], group)
            # compute objective function value for trial vector
            obj_trial = obj(trial, group)
            # perform selection
            if obj_trial >= obj_target:
                # replace the target vector with the trial vector
                pop[j] = trial
                # store the new objective function value
                obj_all[j] = obj_trial
##############################################################################################
            # simulated annealing
            else:
                global C
                C = C - C/100
                p_accept_trial = np.exp((obj_target-obj_trial)/C)
                if p_accept_trial > np.random.uniform(0, 1, 1):
                    # replace the target vector with the trial vector
                    pop[j] = trial
                    # store the new objective function value
                    obj_all[j] = obj_trial
###############################################################################################
        # find the best performing vector at each iteration
        best_obj = max(obj_all)
        # for plotting
        max_fitness_gen.append(best_obj)
        mean_fitness_gen.append(np.mean(obj_all))
        # store the highest objective function value
        if best_obj > prev_obj:
            best_vector = pop[np.argmin(obj_all)]
            prev_obj = best_obj
            obj_iter.append(best_obj)
            # report progress at each iteration
            print('Iteration: %d f([%s]) = %.5f' % (i, np.around(best_vector, decimals=5), best_obj))
    return [best_vector, best_obj, obj_iter, max_fitness_gen, mean_fitness_gen]


# define population size
pop_size = 5#0
# define lower and upper bounds for every dimension
bounds = [-1.0, 1.0]
# define number of iterations
n_generations = 2#0
# define scale factor for mutation
F = 0.5
# define crossover rate for recombination
cr = 0.7

def make_plots_save_data(max_fitness_generations, mean_f, group):
    ### LINE PLOTS ###
    # add mean
    mean_f["mean"] = mean_f.mean(axis = 1)
    # get sd
    mean_f["sd"] = mean_f.std(axis=1)
    # get upper bound
    mean_f["ub"] = mean_f["mean"] + mean_f["sd"]
    # get lower bound
    mean_f["lb"] = mean_f["mean"] - mean_f["sd"]
    # get mean of maxes
    mean_f["max"] = max_f.mean(axis = 1)
    print(mean_f)
    # plot
    plt.plot(mean_f["mean"])
    plt.plot(mean_f["max"])
    plt.fill_between(mean_f.index, mean_f["ub"], mean_f["lb"], facecolor='blue', alpha=0.5,
                 interpolate=True)
    plt.title(f"Mean Agent Fitness Against Group {group}")
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.grid()
    plt.xlim(0,n_generations-1)
    plt.legend(labels=["Mean", "Max"])
    plt.xticks(np.arange(0, n_generations, 1.0))
    plt.savefig(f"EC_assignment2/group{group}_lineplot.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    # save pandas dataframe
    mean_f.to_csv(f'EC_assignment2/DF_mean_fitness_group{group}.txt', sep='\t')


number_of_runs = 2

best_vectors = []
best_fitnesses_boxplot = []
obj_iterations = []
max_f = pd.DataFrame()
mean_f = pd.DataFrame()

groups = [[2,5,8]]#, [1,2,3]]

EAs = [1]

for group in groups:

    for EA in EAs:

        for i in range(number_of_runs):

            # perform differential evolution
            solution = differential_evolution(pop_size, bounds, n_generations, F, cr, group)
            print('\nSolution: f([%s]) = %.5f' % (np.around(solution[0], decimals=5), solution[1]))

            best_vectors.append(solution[0])
            best_fitnesses_boxplot.append(solution[1])
            obj_iterations.append(solution[2])

            max_f["Run_"+str(i)] = solution[3]
            mean_f["Run_"+str(i)] = solution[4]

        make_plots_save_data(max_f, mean_f, group)

        





