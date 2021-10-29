"""
scripts to check a learned agent
based-on:
https://github.com/ray-project/ray/issues/9123
https://github.com/ray-project/ray/issues/7983
"""
import os
import sys
import pickle
import click
from typing import Dict, Any
import json

import ray
from ray.rllib.utils.framework import try_import_torch
import pprint
import gym
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.pg as pg
import ray.rllib.agents.dqn as dqn
import pandas as pd
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    CONFIGS_PATH,
    EXPERIMENTS_PATH,
    ENVSMAP
)
from experiments.utils import (
    add_path_to_config_edge,
    make_env_class
)

torch, nn = try_import_torch()


def learner(*, config: Dict[str, Any],
            type_env: str, dataset_id: int,
            workload_id: int, network_id: int, trace_id: int,
	    episode_length: int):
    """
    """
    # extract differnt parts of the input_config
    learn_config = config['learn_config']

    algorithm = {
	    'sim-binpacking': 'binpacking',
        'sim-greedy': 'greedy'
    }[type_env]
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
    env = gym.make(ENVSMAP[type_env], config=env_config)

    # generate the ray_config
    # make the learning config based on the type of the environment
    ray_config = {"env": make_env_class(type_env),
                  "env_config": env_config}
    ray_config.update(learn_config)

    episode_reward = 0
    done = False
    states = []
    obs = env.reset()
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
    content = os.listdir(EXPERIMENTS_PATH)
    new_experiment = len(content)
    this_experiment_folder = os.path.join(EXPERIMENTS_PATH,
                                          str(new_experiment))
    os.mkdir(this_experiment_folder)

    # save the necesarry information
    with open(
        os.path.join(
            this_experiment_folder, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
    with open(
        os.path.join(
            this_experiment_folder, 'episodes.pickle'), 'wb') as out_pickle:
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
@click.option('--config-file', type=str, default='experimental')
@click.option('--type-env', required=True,
              type=click.Choice(['sim-binpacking', 'sim-greedy']),
              default='sim-binpacking')
@click.option('--dataset-id', required=True, type=int, default=6)
@click.option('--workload-id', required=True, type=int, default=0)
@click.option('--network-id', required=False, type=int, default=0)
@click.option('--trace-id', required=False, type=int, default=0)
@click.option('--episode-length', required=False, type=int, default=50)
def main(config_file: str, type_env: str, dataset_id: int,
         workload_id: int, network_id: int, trace_id: int,
         episode_length: int):
    """[summary]

    Args:
        config_file (str): name of the config folder (only used in real mode)
        type_env (str): the type of the used environment
        dataset_id (int): used cluster dataset
        workload_id (int): the workload used in that dataset
        network_id (int): edge network of some dataset
        trace_id (int): user movement traces
    """
    config_file_path = os.path.join(
        CONFIGS_PATH, 'train', f"{config_file}.json")
    with open(config_file_path) as cf:
        config = json.loads(cf.read())

    pp = pprint.PrettyPrinter(indent=4)
    print('start experiments with the following config:\n')
    pp.pprint(config)

    learner(config=config,
            type_env=type_env, dataset_id=dataset_id,
            workload_id=workload_id, network_id=network_id,
            trace_id=trace_id, episode_length=episode_length)


if __name__ == "__main__":
    main()
