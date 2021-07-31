# TODO fix based on our new needs
# import seaborn as sns
# import pandas as pd
# import os
# import sys
# import matplotlib.pyplot as plt
# import numpy as np
# from constants import RESULTS_PATH
# from constants import MODELS_PATH
#
#
# def reader(dir2load, label):
#     types_of = ["episode_reward", "fraction_of_latency",
#                 "num_of_consolidated"]
#     all_results = {}
#     for type_of in types_of:
#         bin_size = 50
#         repeats = os.listdir(dir2load)
#         repeats.remove('graphs')
#
#         dfs = []
#         for repeat in repeats:
#             df = pd.read_csv(f"{dir2load}/{repeat}/csv_results/{type_of}.csv")
#             dfs.append(df)
#
#         if len(dfs)>1:
#             df = pd.concat(dfs)
#         else:
#             df = dfs[0]
#         df['steps'] = df['step']//bin_size+1
#         df['label'] = label
#         all_results[type_of] = df
#     return all_results
#
# config = {"dataset":2,
#           "type_env":[2,2,2],
#           "sessions":[1,2,3],
#           "session_labels":["high","medium","low"]}
#
# num_sessions = len(config['type_env'])
# episode_reward = []
# fraction_of_latency = []
# num_of_consolidated = []
# for i in range(num_sessions):
#     dir2load = os.path.join(MODELS_PATH, str(config["dataset"]),
#                  f'v{config["type_env"][i]}', str(config["sessions"][i]))
#     all_results = reader(dir2load, config["session_labels"][i])
#     episode_reward.append(all_results["episode_reward"])
#     fraction_of_latency.append(all_results["fraction_of_latency"])
#     num_of_consolidated.append(all_results["num_of_consolidated"])
#
# os.makedirs(f'{RESULTS_PATH}/experiments-results', exist_ok=True)
#
# # plot rewards
# df = pd.concat(episode_reward)
# plt.figure()
# sns_plot = sns.lineplot(x="steps", y="value", style='label', hue='label', data=df, legend=False)
# plt.legend(title='Experiments', loc=4, labels=config['session_labels'])
# plt.ylabel('Reward Value')
# plt.xlabel('Steps      .5*10^2')
# sns_plot.get_figure().savefig(f'{RESULTS_PATH}/experiments-results/rewards.pdf', dpi=300)
#
# # plot fraction of latencies
# df = pd.concat(fraction_of_latency)
# plt.figure()
# sns_plot = sns.lineplot(x="steps", y="value", style='label', hue='label', data=df, legend=False)
# plt.legend(title='Experiments', loc=4, labels=config['session_labels'])
# plt.ylabel('Fraction of latency')
# plt.xlabel('Steps      .5*10^2')
# sns_plot.get_figure().savefig(f'{RESULTS_PATH}/experiments-results/fraction_of_latency.pdf', dpi=300)
#
# # plot consolidations
# df = pd.concat(num_of_consolidated)
# plt.figure()
# sns_plot = sns.lineplot(x="steps", y="value", style='label', hue='label', data=df, legend=False)
# plt.legend(title='Experiments', loc=4, labels=config['session_labels'])
# plt.ylabel('Num of consolidated')
# plt.xlabel('Steps      .5*10^2')
# sns_plot.get_figure().savefig(f'{RESULTS_PATH}/experiments-results/num_of_consolidated.pdf', dpi=300)