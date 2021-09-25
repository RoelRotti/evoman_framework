# In run:

#TODO: mean and max appear to be the same
Gather data for plotting
mean_fitness = []
max_fitness = []
for i in range(number_of_islands):
    #print("stats[i]     =   ", stats[i])
    mean_fitness.append(stats[i].get_fitness_mean())
    max_fitness.append( [c.fitness for c in stats[i].most_fit_genomes] )

# Add all mean and all max values to the dataframe
mean_max = ['mean', 'max']
for type in mean_max:
    for j in range(number_of_islands):
        temp_df = pd.DataFrame(columns=['Run', 'Generation', 'Island', 'mean_max', 'Fitness' ])

        temp_df['Fitness'] = mean_fitness[j] # add all fitnesses, the height of df is now set
        temp_df['Run'] = n_run
        temp_df['Generation'] = range(1,amount_generations+1) # This height should be identical
        temp_df['Island'] = j
        temp_df['mean_max'] = str(type)

        df = df.append(temp_df)



# In main:
# Create dataframe for storing mean/max values
    df = pd.DataFrame(columns=['Run', 'Generation', 'Island', 'mean_max', 'Fitness' ])
    # Standard definition for testing
    number_runs = 2

    # Run for number of runs
    for n_run in range(1, number_runs+1): # +1 to use inclusive numbers : 1,2,..,9,10 instead of stopping at 9
        df = run(config_path, df, n_run)
    df = df.reset_index() # Reset index since we have been appending everything
    print(df)

    # Average of all islands
    sns.lineplot(data=df, x="Generation", y="Fitness", hue="mean_max").set_title('Max and Mean Fitness over generations for {} runs'.format(number_runs))
    plt.show()

    # All different Islands in one plot
    sns.lineplot(data=df, x="Generation", y="Fitness", hue="mean_max", style="Island").set_title('Max and Mean Fitness over generations for {} runs'.format(number_runs))
    plt.show()

    # Different plots for different Islands
    sns.relplot(
    data=df, x="Generation", y="Fitness", col="Island", hue="mean_max", style="mean_max", kind="line" )
    plt.show()

    # Om een matrix van te maken:
    # flights_wide = flights.pivot("Run", "Generation", "fitness", "mean") ???????

    # Mean:
    #       Gen 1   Gen 2   Gen 3   ...
    # Run 1 x       x       x       ...
    # Run 2 x       x       x       ...
    # ...

    # Max:
    #       Gen 1   Gen 2   Gen 3   ...
    # Run 1 x       x       x       ...
    # Run 2 x       x       x       ...
    # ...
