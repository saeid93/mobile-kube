import os
import sys
from ray.tune import Analysis

import ray
from ray import tune
from ray.rllib.utils.framework import try_import_torch
import pprint

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import RESULTS_PATH

series = 6
type_env = 2
dataset_id = 0
workload_id = 0
experiment_id = 0
algorithm = 'PPO'
trial = 'PPO_CloudSimV2_847c5_00000_0_fcnet_hiddens_0=20,fcnet_hiddens_1=20_2021-02-14_22-46-46'
experiment_folder_path = os.path.join(
    RESULTS_PATH,
    "series",      str(series),
    "envs",        str(type_env),
    "datasets",    str(dataset_id),
    "workloads",   str(workload_id),
    "experiments", str(experiment_id),
    str(algorithm), trial)

analysis = Analysis(experiment_folder_path)
df = analysis.trial_dataframes[experiment_folder_path]
selected_stats = ['episode_reward_mean', 'episodes_this_iter',
                  'timesteps_total', 'episodes_total',
                  'experiment_id',
                  'custom_metrics/num_moves_mean',
                  'custom_metrics/num_consolidated_mean',
                  'custom_metrics/num_overloaded_mean']
df = df[selected_stats]
# each row of the df is the training iteration
a = 1
