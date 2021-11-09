"""
Testing phase of the experiments on the test data
"""
import os
import sys
import pickle
import click
from typing import Dict, Any
import json
from ray.rllib.utils.framework import try_import_torch
import pprint
import gym
import pandas as pd
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    TRAIN_RESULTS_PATH,
    TESTS_RESULTS_PATH,
    ENVSMAP
)
from experiments.utils import (
    add_path_to_config_edge,
    make_env_class
)

torch, nn = try_import_torch()


def run_experiments(
    *, test_series: int, train_series: int, type_env: str, dataset_id: int,
    workload_id: int, network_id: int, trace_id: int,
    experiment_id: int, episode_length):
    """
    """
    path_env = type_env if type_env != 'kube-edge' else 'sim-edge'    
    experiments_config_folder = os.path.join(
        TRAIN_RESULTS_PATH,
        "series",      str(train_series),
        "envs",        path_env,
        "datasets",    str(dataset_id),
        "workloads",   str(workload_id),
        "networks",    str(network_id),
        "traces",      str(trace_id),
        "experiments", str(experiment_id),
        "experiment_config.json")

    with open(experiments_config_folder) as cf:
        config = json.loads(cf.read())

    pp = pprint.PrettyPrinter(indent=4)
    print('start experiments with the following config:\n')
    pp.pprint(config)

    # extract differnt parts of the input_config
    learn_config = config['learn_config']
    algorithm = config["run_or_experiment"]
    env_config_base = config['env_config_base']

    # add evn_config_base updates
    env_config_base.update({
        'episode_length': episode_length,
        'no_action_on_overloaded': True
    })

    # add the additional nencessary arguments to the edge config
    env_config = add_path_to_config_edge(
        config=env_config_base,
        dataset_id=dataset_id,
        workload_id=workload_id,
        network_id=network_id,
        trace_id=trace_id
    )

    # trained ray agent should always be simulation
    # however the agent outside it can be kuber agent or
    # other types of agent
    if type_env not in ['CartPole-v0', 'Pendulum-v0']:
        env = gym.make(ENVSMAP[type_env], config=env_config)
        ray_config = {"env": make_env_class('sim-edge'),
                    "env_config": env_config}
        ray_config.update(learn_config)
    else:
        ray_config = {"env": type_env}
        ray_config.update(learn_config)

    path_env = type_env if type_env != 'kube-edge' else 'sim-edge'
    # experiments_folder = os.path.join(TRAIN_RESULTS_PATH,
    #                                   "series",      str(series),
    #                                   "envs",        path_env,
    #                                   "datasets",    str(dataset_id),
    #                                   "workloads",   str(workload_id),
    #                                   "networks",     str(network_id),
    #                                   "traces",       str(trace_id),
    #                                   "experiments", str(experiment_id),
    #                                   algorithm)

    episode_reward = 0
    done = False
    states = []
    _ = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        state = flatten(env.raw_observation, action, reward, info)
        states.append(state)
        episode_reward += reward
    states = pd.DataFrame(states)
    print(f"episode reward: {episode_reward}")
    info = {
        'type_env': type_env,
        'dataset_id': dataset_id,
        'workload_id': workload_id,
        'network_id': network_id,
        'trace_id': trace_id,
        'episode_length': episode_length,
        'algorithm': algorithm
    }
    # make the new experiment folder
    test_series_path = os.path.join(
        TESTS_RESULTS_PATH,
        'series', str(test_series),
        'tests')
    if not os.path.isdir(test_series_path):
        os.makedirs(test_series_path)
    content = os.listdir(test_series_path)
    new_test = len(content)
    this_test_folder = os.path.join(test_series_path,
                                    str(new_test))
    os.makedirs(this_test_folder)

    # save the necesarry information
    with open(os.path.join(this_test_folder, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
    with open(os.path.join(
        this_test_folder, 'episodes.pickle'), 'wb') as out_pickle:
        pickle.dump([states], out_pickle)

def flatten(raw_obs, action, reward, info):
    return {
        'action': action,
        'services_nodes': raw_obs['services_nodes'],
        'users_stations': raw_obs['users_stations'],
        'num_consolidated': info['num_consolidated'],
        'num_moves': info['num_moves'],
        'num_overloaded': info['num_overloaded'],
        'users_distances': info['users_distances'],
        'reward_latency': info['rewards']['reward_latency'],
        'reward_move': info['rewards']['reward_move'],
        'reward_illegal': info['rewards']['reward_illegal'],
        'reward_variance': info['rewards']['reward_variance'],
        'reward': reward
    }


@click.command()
@click.option('--local-mode', type=bool, default=True)
@click.option('--test-series', required=True, type=int, default=1)
@click.option('--train-series', required=True, type=int, default=11)
@click.option('--type-env', required=True,
              type=click.Choice(['sim-edge']),
              default='sim-edge')
@click.option('--dataset-id', required=True, type=int, default=6)
@click.option('--workload-id', required=True, type=int, default=0)
@click.option('--network-id', required=False, type=int, default=7)
@click.option('--trace-id', required=False, type=int, default=0)
@click.option('--experiment-id', required=True, type=int, default=0)
@click.option('--checkpoint', required=False, type=int, default=10000)
@click.option('--episode-length', required=False, type=int, default=50)
@click.option('--num-episodes', required=False, type=int, default=10)
def main(local_mode: bool, test_series: int, train_series: int, type_env: str,
         dataset_id: int, workload_id: int, network_id: int,
         trace_id: int, experiment_id: int, checkpoint: int,
         num_episodes: int, episode_length: int):
    """[summary]

    Args:
        local_mode (bool): run in local mode for having the 
        config_file (str): name of the config folder (only used in real mode)
        use_callback (bool): whether to use callbacks or storing and visualising
        checkpoint_freq (int): checkpoint the ml model at each (n-th) step
        series (int): to gather a series of datasets in a folder
        type_env (str): the type of the used environment
        dataset_id (int): used cluster dataset
        workload_id (int): the workload used in that dataset
        network_id (int): edge network of some dataset
        trace_id (int): user movement traces
    """

    run_experiments(
        test_series=test_series,
        train_series=train_series, type_env=type_env,
        dataset_id=dataset_id, workload_id=workload_id,
        network_id=network_id, trace_id=trace_id,
        experiment_id=experiment_id,
        episode_length=episode_length)


if __name__ == "__main__":
    main()
