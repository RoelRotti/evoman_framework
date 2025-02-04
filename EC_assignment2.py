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
import seaborn as sns
from scipy.stats import cauchy

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



# Create a custom environment
def create_env(group, randomm = "yes"):
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
                    speed="fastest",
                    randomini=randomm)
    return env

# define objective function
def obj(x, group, return_all=False, randomm = "yes"):
    # create env
    envv = create_env(group, randomm)
    # f = fitness, p = player life, e = enemy life, t = game run time
    [f, p, e, t] = envv.play(pcont=x)
    if return_all: return [f, p, e, t]
    else:    return f


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
def differential_evolution(pop_size, bounds, n_generations, group, EA):
    # assign variables of this specific EA
    T = EA["T"]
    F_init = EA["F_init"]
    CR_init = EA["CR_init"]
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
    F_matrix = [F_init]*pop_size
    CR_matrix = [CR_init]*pop_size
    for i in range(n_generations):
        # geometric annealing constant k
        k = 0.9
        # lower the temperature 
        T = T * k
        # index for F_ and CR_memory
        F_memory = []
        CR_memory = []
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], F_matrix[j])
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = np.array(crossover(mutated, pop[j], CR_matrix[j], 'exp'))
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
                # store F and CR 
                F_memory.append(F_matrix[j])
                CR_memory.append(CR_matrix[j])
##############################################################################################
            # simulated annealing: will not run if T = 0
            if (obj_trial < obj_target) & (T > 0):
                p_accept_trial = np.exp(-(obj_target-obj_trial)/T)
                if p_accept_trial > np.random.uniform(0, 1, 1):
                    # replace the target vector with the trial vector
                    pop[j] = trial
                    # store the new objective function value
                    obj_all[j] = obj_trial
###############################################################################################
        # find the best performing vector at each iteration
        # compute F_avr is F_memory is NOT empty
        if F_memory:
            F_avr = np.mean(F_memory)
            CR_avr = np.mean(CR_memory)
            for j in range(pop_size):
                F_matrix[j] = cauchy.rvs(loc = 0, scale = 0.1, size = 1) + F_avr
                CR_matrix[j] = cauchy.rvs(loc = 0, scale = 0.1, size = 1) + CR_avr    
            # F and CR are adjusted adjusted to bounds
            F_matrix = check_bounds(F_matrix, [0.1,1])
            CR_matrix = check_bounds(CR_matrix, [0,1])
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

def make_plots_save_data(max_f, mean_f, best_vectors, group, show):

    ### LINE PLOTS ###
    fig_number = 0
    if len(groups) > 1:
        if group == groups[1]:
            fig_number = 2
    plt.figure(fig_number)

    colors = ["red", "orange", "blue", "green"]

    for i in range(len(max_f)):
        # add mean
        mean_f[i]["mean"] = mean_f[i].mean(axis = 1)
        # get sd
        mean_f[i]["sd"] = mean_f[i].std(axis=1)
        # get upper bound
        mean_f[i]["ub"] = mean_f[i]["mean"] + mean_f[i]["sd"]
        # get lower bound
        mean_f[i]["lb"] = mean_f[i]["mean"] - mean_f[i]["sd"]
        # get mean of maxes
        mean_f[i]["max"] = max_f[i].mean(axis = 1)
        print(mean_f[i])
        # plot
        c_i = i*2
        plt.plot(mean_f[i]["max"], color = colors[c_i])
        plt.plot(mean_f[i]["mean"], color = colors[c_i+1]) 
        plt.fill_between(mean_f[i].index, mean_f[i]["ub"], mean_f[i]["lb"], facecolor=colors[c_i+1], alpha=0.5,
                    interpolate=True)
        # save pandas dataframe
        mean_f[i].to_csv(f'EC_assignment2/DF_mean_fitness_lineplot_EA{EAs[i]}_group{group}.txt', sep='\t')
    plt.title(f"Mean Agent Fitness against Group {group}. {number_of_runs} runs, {n_generations} gens, pop size {pop_size}")
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.grid()
    plt.xlim(0,n_generations-1)
    # adjustable legend
    if len(EAs) > 1:        plt.legend(labels=[f"Max_{EAs[0]}", f"Mean_{EAs[0]}",f"Max_{EAs[1]}", f"Mean_{EAs[1]}"]) 
    else:   plt.legend(labels=[f"Max_{EAs[0]}", f"Mean_{EAs[0]}"])
    plt.xticks(np.arange(0, n_generations, 1.0))
    plt.savefig(f"EC_assignment2/group{group}_lineplot{EAs}.pdf", dpi=300, bbox_inches='tight')
    if show:    plt.show()

    ### BOX PLOTS ###
    plt.figure(fig_number+1)
    all_enemies = [[1],[2],[3],[4],[5],[6],[7],[8]]
    mean_gain = []
    p_life = []
    e_life = []
    for i in range(len(best_vectors)):
        mean_gain.append(pd.DataFrame())
        p_life.append(pd.DataFrame())
        e_life.append(pd.DataFrame())
        #mean_gain[i]["enemies"] = all_enemies

    # test best vectors of each EA
    for i in range(len(best_vectors)):
        # test the best vector of each run
        for j in range(len(best_vectors[i])):
            # test against each enemy
            player_life = []
            enemy_life = []
            enemy_gain = []
            for enemy in all_enemies:
                # test 5 times
                play_life = []
                en_life = []
                gains = []
                for k in range(5):
                    # play agains enemy
                    stats = obj(best_vectors[i][j], enemy, return_all=True, randomm="no")
                    # gain = player_life - enemy_life
                    gain = stats[1]-stats[2] 
                    play_life.append(stats[1])
                    en_life.append(stats[2])
                    gains.append(gain)
                player_life.append(sum(play_life)/len(play_life))
                enemy_life.append(sum(en_life)/len(en_life))
                enemy_gain.append(sum(gains)/len(gains))
            p_life[i][j] = player_life
            e_life[i][j] = enemy_life
            mean_gain[i][j] = enemy_gain



    max_gain = -np.inf

    # calculate mean of each 
    for i in range(len(best_vectors)):

        #TODO: do column mean, not row mean

        #mean_gain[i]["mean"] = mean_gain[i].mean(axis = 1)
        #mean_gain[i].rename(columns = {'mean':'mean'+str(i)}, inplace = True)
        mean_gain[i].loc['mean'] = mean_gain[i].mean()

        max_column = mean_gain[i].loc['mean'].idxmax()
        max_g = mean_gain[i][max_column].loc['mean']
        if max_g > max_gain:
            max_gain = max_g
            max_vector = max_column
            ea_i = i

    real_best_vector = best_vectors[ea_i][max_vector]
    np.savetxt(f'EC_assignment2/real_best_vector{group}_EA{EAs}.txt',real_best_vector)

    np.savetxt(f'EC_assignment2/best_gain{group}_EA{EAs}.txt',mean_gain[ea_i][max_column])
    np.savetxt(f'EC_assignment2/best_player_life{group}_EA{EAs}.txt',p_life[ea_i][max_column])
    np.savetxt(f'EC_assignment2/best_enemy_life{group}_EA{EAs}.txt',e_life[ea_i][max_column])

    print("ea_i = ", ea_i)
    print("max_column = ", max_column)

    # TODO: Save best best vector

    for i in range(len(best_vectors)):
        mean_gain[i].to_csv(f'EC_assignment2/DF_mean_gain_boxplot_group{group}_EA{EAs[i]}.txt', sep='\t')

    if len(best_vectors) > 1 :    
        for i in range(len(best_vectors)):
            mean_gain[i].rename(index = {'mean':'mean'+str(i)}, inplace = True)
        dat = pd.concat([mean_gain[0].loc['mean0'], mean_gain[1].loc["mean1"]], axis=1)
        dat = pd.melt(dat)
        sns.boxplot(data=dat, x="variable", y = "value").set_title(f"Gain of best genome. Group {group}, {number_of_runs} runs, {n_generations} gens, pop size {pop_size}") 
        plt.xlabel("EA's")
        plt.xticks([0, 1], [EAs[0], EAs[1]])
    else:   
        dat = mean_gain[0].loc['mean']
        sns.boxplot(data=dat).set_title(f"Gain of best genome. Group {group}, {number_of_runs} runs, {n_generations} gens, pop size {pop_size}") 
        plt.xlabel("EA")
        plt.xticks([0], [EAs[0]])
    # plot boxplot
    plt.ylabel("Gain")
    plt.savefig(f"EC_assignment2/group{group}_boxplot_EAs_{EAs}.pdf", dpi=300, bbox_inches='tight')
    
    if show: plt.show()

    # Add a table containing the average (of 5 repetitions) energy points of player and enemy (for each of all enemies) 
    # for your VERY best solution (only one of the[10+10]*2). This is the solution you will submit to the competition.


## VARIABLES

# Overall
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
# define population size
pop_size = 30
# define lower and upper bounds for every dimension
bounds = [-1.0, 1.0]
# define number of iterations
n_generations = 25

number_of_runs = 10

groups = [[2,5,8],[4,6,7]]

EAs = [{"T": 10**4, "F_init": 0.5, "CR_init" : 0.9}, {"T": 0, "F_init": 0.5, "CR_init" : 0.9}]
# assign empty lists
best_vectors = []
best_fitnesses_boxplot = []
obj_iterations = []
max_f = []
mean_f = []
# fill lists according to amount of EA's
for i in range(len(EAs)):
    best_vectors.append([])
    best_fitnesses_boxplot.append([])
    obj_iterations.append([])
    max_f.append(pd.DataFrame())
    mean_f.append(pd.DataFrame())

for group in groups:

    for i, EA in enumerate(EAs):

        for j in range(number_of_runs):

            # perform differential evolution
            solution = differential_evolution(pop_size, bounds, n_generations, group, EA)
            print('\nSolution: f([%s]) = %.5f' % (np.around(solution[0], decimals=5), solution[1]))

            best_vectors[i].append(solution[0])
            best_fitnesses_boxplot[i].append(solution[1])
            obj_iterations[i].append(solution[2])

            max_f[i]["Run_"+str(j)] = solution[3]
            mean_f[i]["Run_"+str(j)] = solution[4]

    make_plots_save_data(max_f, mean_f, best_vectors, group, show=False)

