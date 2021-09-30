import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from EC_assignment1_part1 import number_of_islands, number_of_runs

enemy = 8
population_size = 22
number_of_islands = 4

# ############################# 4 ISLANDS ############################## #

results_frame = []
for i in range(4):
    results_frame.append(pd.DataFrame(columns=['Enemy', 'Fitness', 'Player life', 'Enemy life', 'Time']))

df_means = []
for i in range(number_of_runs):
    file = f'EC_assignment1_part1/enemy{enemy}_4islands_run{i+1}.txt'
    with open(file) as f:
        numbers = r"[-+]?\d*\.\d+|\d+"
        for line in f:
            if line.startswith('Island:'):
                island = int(re.findall(numbers, line)[0])
            if line.startswith('RUN:'):
                results = re.findall(numbers, line)
                results = [float(i) for i in results]
                result_series = pd.Series(data=results, index=results_frame[island-1].columns)
                results_frame[island-1] = results_frame[island-1].append(result_series, ignore_index=True)
    f.close()
    df_concat = pd.DataFrame()
    for j in range(number_of_islands):
        df_concat = pd.concat((df_concat, results_frame[j]))
        results_frame[j] = pd.DataFrame(columns=['Enemy', 'Fitness', 'Player life', 'Enemy life', 'Time'])
    by_row_index = df_concat.groupby(df_concat.index)
    df_means.append(by_row_index.mean())
    del by_row_index, df_concat

df_concat_means = pd.DataFrame()
for k in range(number_of_runs):
    df_concat_means = pd.concat((df_concat_means, df_means[k]))
by_row_index_means = df_concat_means.groupby(df_concat_means.index)

analysis_frame = pd.DataFrame()
analysis_frame["average"] = by_row_index_means.mean()["Fitness"]
analysis_frame["sd"] = by_row_index_means.std()["Fitness"]
analysis_frame["lb"] = analysis_frame["average"] - analysis_frame["sd"]
analysis_frame["ub"] = analysis_frame["average"] + analysis_frame["sd"]
analysis_frame = analysis_frame.groupby(np.arange(len(analysis_frame))//population_size).mean()


# ############################# NO ISLANDS ############################## #

results_frame = pd.DataFrame(columns=['Enemy', 'Fitness', 'Player life', 'Enemy life', 'Time'])
all_results = []
df_means = []
for i in range(number_of_runs):
    file = f'EC_assignment1_part1/enemy{enemy}_1islands_run{i+1}.txt'
    with open(file) as f:
        for line in f:
            if line.startswith('RUN:'):
                results = re.findall(numbers, line)
                results = [float(i) for i in results]
                result_series = pd.Series(data=results, index=results_frame.columns)
                results_frame = results_frame.append(result_series, ignore_index=True)
        all_results.append(results_frame)
        results_frame = pd.DataFrame(columns=['Enemy', 'Fitness', 'Player life', 'Enemy life', 'Time'])
    f.close()
df_concat_means = pd.DataFrame()
for k in range(number_of_runs):
    df_concat_means = pd.concat((df_concat_means, all_results[k]))
    all_results[k] = pd.DataFrame(columns=['Enemy', 'Fitness', 'Player life', 'Enemy life', 'Time'])
by_row_index_means = df_concat_means.groupby(df_concat_means.index)

analysis_frame_no_islands = pd.DataFrame()
analysis_frame_no_islands["average"] = by_row_index_means.mean()["Fitness"]
analysis_frame_no_islands["sd"] = by_row_index_means.std()["Fitness"]
analysis_frame_no_islands["lb"] = analysis_frame_no_islands["average"] - analysis_frame_no_islands["sd"]
analysis_frame_no_islands["ub"] = analysis_frame_no_islands["average"] + analysis_frame_no_islands["sd"]
analysis_frame_no_islands = analysis_frame_no_islands.groupby(np.arange(len(analysis_frame_no_islands))//(population_size*4)).mean()
plt.rcParams['font.size'] = '13'
plt.plot(analysis_frame["average"])
plt.fill_between(analysis_frame.index, analysis_frame["ub"], analysis_frame["lb"], facecolor='blue', alpha=0.5,
                 interpolate=True)

plt.plot(analysis_frame_no_islands["average"])
plt.fill_between(analysis_frame_no_islands.index, analysis_frame_no_islands["ub"], analysis_frame_no_islands["lb"],
                 facecolor='red', alpha=0.5, interpolate=True)

plt.title(f"Mean Agent Fitness Against Enemy {enemy}")
plt.ylabel("Fitness")
plt.xlabel("Generation")
plt.grid()
plt.legend(labels=[f"4 islands", "no islands"])
plt.xlim(0,20)
plt.xticks(np.arange(0, 21, 1.0))
plt.savefig(f"EC_assignment1_part1/enemy{enemy}_lineplot.pdf", dpi=300, bbox_inches='tight')
plt.show()