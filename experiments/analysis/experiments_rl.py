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
    RESULTS_PATH,
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
            series: int, type_env: str, dataset_id: int,
            workload_id: int, network_id: int, trace_id: int,
            checkpoint: int, experiment_id: int,
            num_episodes: int, episode_length: int,
            local_mode: bool):
    """
    """
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
    env = gym.make(ENVSMAP[type_env], config=env_config)

    # generate the ray_config
    # make the learning config based on the type of the environment
    ray_config = {"env": make_env_class(type_env),
                  "env_config": env_config}
    ray_config.update(learn_config)

    # generate the path
    # folder formats: <environmet>/datasets/<dataset>/workloads/<workload>
    # example:        env1/dataset/1/workloads/3
    experiments_folder = os.path.join(RESULTS_PATH,
                                      "series",      str(series),
                                      "envs",        str(type_env),
                                      "datasets",    str(dataset_id),
                                      "workloads",   str(workload_id),
                                      "experiments", str(experiment_id),
                                      algorithm)
    checkpoint_path = os.path.join(
        experiments_folder,
        os.listdir(experiments_folder)[0],
        f"checkpoint_{checkpoint}",
        f"checkpoint-{checkpoint}"
    )

    ray.init(local_mode=local_mode)

    if algorithm == 'PPO':
        agent = ppo.PPOTrainer(
            config=ray_config,
            env=make_env_class(type_env))
    elif algorithm == 'A3C':
        agent = a3c.A3CTrainer(
            config=ray_config,
            env=make_env_class(type_env))
    elif algorithm == 'PG':
        agent = pg.PGTrainer(
            config=ray_config,
            env=make_env_class(type_env))
    elif algorithm == 'DQN':
        agent = dqn.DQNTrainer(
            config=ray_config,
            env=make_env_class(type_env))

    episodes = []
    for i in range(0, num_episodes):
        print(f"episode: {i}")
        agent.restore(checkpoint_path=checkpoint_path)
        episode_reward = 0
        done = False
        states = []
        obs = env.reset()
        while not done:
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            state = flatten(env.raw_observation, action, reward, info)
            states.append(state)
            episode_reward += reward
        states = pd.DataFrame(states)
        print(f"episode reward: {episode_reward}")
        episodes.append(states)
    info = {
        'series': series,
        'type_env': type_env,
        'dataset_id': dataset_id,
        'workload_id': workload_id,
        'network_id': network_id,
        'trace_id': trace_id,
        'checkpint': checkpoint,
        'experiments': experiment_id,
        'episode_length': episode_length,
        'num_episodes': num_episodes,
        'algorithm': algorithm
    }
    # make the new experiment folder
    content = os.listdir(EXPERIMENTS_PATH)
    new_experiment = len(content)
    this_experiment_folder = os.path.join(EXPERIMENTS_PATH,
                                          str(new_experiment))
    os.mkdir(this_experiment_folder)

    # save the necesarry information
    with open(os.path.join(this_experiment_folder, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
    with open(os.path.join(this_experiment_folder, 'episodes.pickle'), 'wb') as out_pickle:
        pickle.dump(episodes, out_pickle)


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
@click.option('--config-folder', type=str, default='experimental')
@click.option('--series', required=True, type=int, default=1)
@click.option('--type-env', required=True,
              type=click.Choice(['sim-edge']),
              default='sim-edge')
@click.option('--dataset-id', required=True, type=int, default=3)
@click.option('--workload-id', required=True, type=int, default=0)
@click.option('--network-id', required=False, type=int, default=0)
@click.option('--trace-id', required=False, type=int, default=0)
@click.option('--experiment_id', required=True, type=int, default=0)
@click.option('--checkpoint', required=False, type=int, default=100)
@click.option('--episode-length', required=False, type=int, default=10)
@click.option('--num-episodes', required=False, type=int, default=10)
def main(local_mode: bool, config_folder: str, series: int,
         type_env: str, dataset_id: int, workload_id: int, network_id: int,
         trace_id: int, experiment_id: int,
	     checkpoint: int, num_episodes: int, episode_length: int):
    """[summary]

    Args:
        local_mode (bool): run in local mode for having the 
        config_folder (str): name of the config folder (only used in real mode)
        use_callback (bool): whether to use callbacks or storing and visualising
        checkpoint_freq (int): checkpoint the ml model at each (n-th) step
        series (int): to gather a series of datasets in a folder
        type_env (str): the type of the used environment
        dataset_id (int): used cluster dataset
        workload_id (int): the workload used in that dataset
        network_id (int): edge network of some dataset
        trace_id (int): user movement traces
    """
    config_file_path = os.path.join(
        CONFIGS_PATH, 'train', config_folder,
        'run.json')
    with open(config_file_path) as cf:
        config = json.loads(cf.read())

    pp = pprint.PrettyPrinter(indent=4)
    print('start experiments with the following config:\n')
    pp.pprint(config)

    learner(config=config, series=series,
            type_env=type_env, dataset_id=dataset_id,
            workload_id=workload_id, network_id=network_id,
            trace_id=trace_id, experiment_id=experiment_id,
            checkpoint=checkpoint, num_episodes=num_episodes,
            episode_length=episode_length, local_mode=local_mode)


if __name__ == "__main__":
    main()
