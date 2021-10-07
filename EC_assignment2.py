# imports framework
import sys, os, random
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from matplotlib import pyplot
from scipy.optimize import differential_evolution
import numpy as np

experiment_name = 'EC_assignment1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 10

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[7,8],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# define objective function
def obj(x):
    # f = fitness, p = player life, e = enemy life, t = game run time
    [f, p, e, t] = env.play(pcont=x)
    return f


# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [np.clip(mutated[i], bounds[0], bounds[1]) for i in range(len(mutated))]
    return mutated_bound


# define crossover operation
def crossover(mutated, target, cr):
    # generate a uniform random value for every dimension
    p = np.random.rand(len(mutated))
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(len(mutated))]
    return trial


def differential_evolution(pop_size, bounds, n_generations, F, cr):
    # initialise population of candidate solutions randomly within the specified bounds
    pop = np.random.uniform(-1, 1, (pop_size, n_vars))
    # evaluate initial population of candidate solutions
    obj_all = [obj(ind) for ind in pop]
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
            trial = np.array(crossover(mutated, pop[j], cr))
            # compute objective function value for target vector
            obj_target = obj(pop[j])
            # compute objective function value for trial vector
            obj_trial = obj(trial)
            # perform selection
            if obj_trial > obj_target:
                # replace the target vector with the trial vector
                pop[j] = trial
                # store the new objective function value
                obj_all[j] = obj_trial
        # find the best performing vector at each iteration
        best_obj = max(obj_all)
        # store the highest objective function value
        if best_obj > prev_obj:
            best_vector = pop[np.argmin(obj_all)]
            prev_obj = best_obj
            obj_iter.append(best_obj)
            # report progress at each iteration
            print('Iteration: %d f([%s]) = %.5f' % (i, np.around(best_vector, decimals=5), best_obj))
    return [best_vector, best_obj, obj_iter]


# define population size
pop_size = 10
# define lower and upper bounds for every dimension
bounds = [-1.0, 1.0]
# define number of iterations
n_generations = 10
# define scale factor for mutation
F = 0.5
# define crossover rate for recombination
cr = 0.7

# perform differential evolution
solution = differential_evolution(pop_size, bounds, n_generations, F, cr)
print('\nSolution: f([%s]) = %.5f' % (np.around(solution[0], decimals=5), solution[1]))

# line plot of best objective function values
pyplot.plot(solution[2], '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()