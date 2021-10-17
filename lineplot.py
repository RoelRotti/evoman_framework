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

### LINE PLOTS ###
mean_f = []
mean_f.append(pd.read_csv("EC_assignment2/DF_mean_fitness_lineplot_EA{'T': 10000, 'F_init': 0.5, 'CR_init': 0.9}_group[4, 6, 7].txt", sep="\t"))#, index_col=[0])
mean_f.append(pd.read_csv("EC_assignment2/DF_mean_fitness_lineplot_EA{'T': 0, 'F_init': 0.5, 'CR_init': 0.9}_group[4, 6, 7].txt", sep="\t"))#, index_col=[0])

group = [4,6,7]

# Overall
#n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
# define population size
pop_size = 30
# define lower and upper bounds for every dimension
bounds = [-1.0, 1.0]
# define number of iterations
n_generations = 100

number_of_runs = 10

EAs = [{"T": 10**4, "F_init": 0.5, "CR_init" : 0.9}, {"T": 0, "F_init": 0.5, "CR_init" : 0.9}]

plt.figure(1)

colors = ["red", "orange", "blue", "green"]

for i in range(len(mean_f)):
    # plot
    c_i = i*2
    plt.plot(mean_f[i]["max"], color = colors[c_i])
    plt.plot(mean_f[i]["mean"], color = colors[c_i+1]) 
    plt.fill_between(mean_f[i].index, mean_f[i]["ub"], mean_f[i]["lb"], facecolor=colors[c_i+1], alpha=0.5,
                interpolate=True)
    # save pandas dataframe



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
plt.show()