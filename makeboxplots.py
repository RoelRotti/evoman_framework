from pandas.core.frame import DataFrame
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

enemies = [5,8]
enemies_list = [5]*20 + [8]*20 #+[2]*20
islands = ["1 island"]*10 + ["4 islands"]*10 + ["1 island"]*10 + ["4 islands"]*10 #+ ["1 island"]*10 + ["4 islands"]*10
boxplot_values_content= []
file_in = open("/Users/maike/Documents/MasterAI/EvolutionaryComputing/StandardAssignment1/evoman_framework/EC_assignment1_part1/data_boxplots_full.txt", "r")
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
for i in range(2):
    file_in = open(f"/Users/maike/Documents/MasterAI/EvolutionaryComputing/StandardAssignment1/evoman_framework/EC_assignment1_part1/enemy{enemies[i]}_1islands_data_boxplots.txt", "r")
    for y in file_in.read().split('\n'):
        if is_number(y):
            boxplot_values_content.append(float(y))
    file_in.close()
    file_in = open(f"/Users/maike/Documents/MasterAI/EvolutionaryComputing/StandardAssignment1/evoman_framework/EC_assignment1_part1/enemy{enemies[i]}_4islands_data_boxplots.txt", "r")
    for y in file_in.read().split('\n'):
        if is_number(y):
            boxplot_values_content.append(float(y))
    file_in.close()
data = {"number of islands" : islands}
df = pd.DataFrame(data)
df["average fitness"] = pd.DataFrame(boxplot_values_content)
df["enemy"] = pd.DataFrame(enemies_list)
sns.set(font_scale = 1.5)
all_boxplots_img = sns.boxplot(x = "enemy", y = "average fitness", hue = "number of islands", data = df)
all_boxplots_img.set_title("Comparison of the Best Agents")
plt.show()
