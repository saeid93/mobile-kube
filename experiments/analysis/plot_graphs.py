# TODO fix if needed
# import seaborn as sns
# import pandas as pd
# import os
# import sys
# import matplotlib.pyplot as plt
# import numpy as np
# from constants import RESULTS_PATH
#
#
# def reader(dir2load, type_of):
#     bin_size = 50
#     repeats = os.listdir(dir2load)
#     repeats.remove('graphs')
#
#     dfs = []
#     for repeat in repeats:
#         df = pd.read_csv(f"{dir2load}/{repeat}/csv_results/{type_of}.csv")
#         dfs.append(df)
#
#     if len(dfs)>1:
#         df = pd.concat(dfs)
#     else:
#         df = dfs[0]
#     df['steps'] = df['step']//bin_size+1
#     return(df)
#
#
# def plot_graphs(dir2load):
#
#     dir2save = f'{dir2load}/graphs'
#     os.makedirs(dir2save, exist_ok=True)
#
#     # plot episode_rewards
#     plt.figure()
#     type_of = "episode_reward"
#     df = reader(dir2load, type_of)
#     sns_plot = sns.lineplot(x="steps", y="value", data=df, legend=False)
#     plt.legend(title='Reward Value', loc=4)
#     plt.ylabel('Reward Value')
#     plt.xlabel('Steps      .5*10^2')
#     sns_plot.get_figure().savefig(os.path.join(dir2save, 'rewards.png'), dpi=300)
#
#     # plot fraction_of_latency
#     plt.figure()
#     type_of = "fraction_of_latency"
#     df = reader(dir2load, type_of)
#     sns_plot = sns.lineplot(x="steps", y="value", data=df, legend=False)
#     plt.legend(title='Fraction of Latency', loc=4)
#     plt.ylabel('Fraction of Latency')
#     plt.xlabel('Steps      .5*10^2')
#     sns_plot.get_figure().savefig(os.path.join(dir2save, 'latency.png'), dpi=300)
#
#     # plot num_of_consolidated
#     plt.figure()
#     type_of = "num_of_consolidated"
#     df = reader(dir2load, type_of)
#     sns_plot = sns.lineplot(x="steps", y="value", data=df, legend=False)
#     plt.legend(title='Number of Consolidated', loc=4)
#     plt.ylabel('Number of Consolidated')
#     plt.xlabel('Steps      .5*10^2')
#     sns_plot.get_figure().savefig(os.path.join(dir2save, 'consolidation.png'), dpi=300)