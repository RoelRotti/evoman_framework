# imports framework
import sys, os, random
sys.path.insert(0, 'evoman')
# from environment import Environment
# from demo_controller import player_controller
from matplotlib import pyplot
from scipy.optimize import differential_evolution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import cauchy

### LINE PLOTS ###
mean_gain = []
# group 4,6,7
mean_gain.append(pd.read_csv("EC_assignment2/DF_mean_gain_boxplot_group[4, 6, 7]_EA{'T': 10000, 'F_init': 0.5, 'CR_init': 0.9}.txt", sep="\t", index_col=[0]))
mean_gain.append(pd.read_csv("EC_assignment2/DF_mean_gain_boxplot_group[4, 6, 7]_EA{'T': 0, 'F_init': 0.5, 'CR_init': 0.9}.txt", sep="\t", index_col=[0]))

# group 2,5,8
# mean_f.append(pd.read_csv("EC_assignment2/DF_mean_fitness_lineplot_EA{'T': 10000, 'F_init': 0.5, 'CR_init': 0.9}_group[2, 5, 8].txt", sep="\t", index_col=[0]))
# mean_f.append(pd.read_csv("EC_assignment2/DF_mean_fitness_lineplot_EA{'T': 0, 'F_init': 0.5, 'CR_init': 0.9}_group[2, 5, 8].txt", sep="\t", index_col=[0]))


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
for i in range(len(mean_gain)):
    mean_gain[i].rename(index = {'mean':'mean'+str(i)}, inplace = True)
dat = pd.concat([mean_gain[0].loc['mean0'], mean_gain[1].loc["mean1"]], axis=1)
dat = pd.melt(dat)
sns.boxplot(data=dat, x="variable", y = "value").set_title(f"Gain of best genome. Group {group}, {number_of_runs} runs, {n_generations} gens, pop size {pop_size}") 
plt.xlabel("EA's")
plt.xticks([0, 1], [EAs[0], EAs[1]])

# plot boxplot
plt.ylabel("Gain")
plt.savefig(f"EC_assignment2/group{group}_boxplot_EAs_{EAs}.pdf", dpi=300, bbox_inches='tight')

plt.show()