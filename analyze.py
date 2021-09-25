import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt

results_frame = pd.DataFrame(columns=['Enemy', 'Fitness', 'Player life', 'Enemy life', 'Time'])

file = 'EC_assignment1_part1/evoman_logs.txt'

with open(file) as f:
    numbers = r"[-+]?\d*\.\d+|\d+"
    for line in f:
        if line.startswith('Island:'):
            island = re.findall(numbers, line)
        if line.startswith('RUN:'):
            results = re.findall(numbers, line)
            results = [float(i) for i in results]
            result_series = pd.Series(results, index=results_frame.columns)
            result_series['island'] = island
            results_frame = results_frame.append(result_series, ignore_index=True)

analysis_frame = pd.DataFrame()
analysis_frame["average"] = results_frame["Fitness"].groupby(results_frame["Fitness"].index//25).mean()
analysis_frame["sd"] = results_frame["Fitness"].groupby(results_frame["Fitness"].index//25).std()
analysis_frame["lb"] = analysis_frame["average"] - analysis_frame["sd"]
analysis_frame["ub"] = analysis_frame["average"] + analysis_frame["sd"]

plt.plot(analysis_frame["average"])
plt.fill_between(analysis_frame.index, analysis_frame["ub"], analysis_frame["lb"], facecolor='red', alpha=0.5, interpolate=True)
plt.title("Evolving fitness over time")
plt.ylabel("Fitness")
plt.xlabel("Generation")
plt.grid()
plt.show()
plt.savefig('results.png', dpi=300, bbox_inches='tight')
