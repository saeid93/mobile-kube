"""
Testing phase of the experiments on the test data
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
import copy

import ray
from ray.rllib.utils.framework import try_import_torch
import pprint
import gym
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.impala as impala
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
    *, test_series: int, train_series: int, type_env: str,
    dataset_id: int, workload_id: int, network_id: int,
    trace_id: int, experiment_id: int, local_mode: bool,
    episode_length, num_episodes: int, workload_id_test: int,
    trace_id_test: int, checkpoint_to_load: str):
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

    # fix the grid searches
    algorithm, env_configs, learn_configs, num_workers = fix_grid_searches(
        config=config,
        dataset_id=dataset_id,
        workload_id_test=workload_id_test,
        network_id=network_id,
        trace_id_test=trace_id_test,
        episode_length=episode_length
    )

    # pp = pprint.PrettyPrinter(indent=4)
    # print('start experiments with the following config:\n')
    # pp.pprint(config)

    # # extract differnt parts of the input_config
    # learn_config = config['learn_config']
    # algorithm = config["run_or_experiment"]
    # env_config_base = config['env_config_base']

    # # update the difffent part of the envs
    # env_config_base.update({
    #     'episode_length': episode_length,
    #     'no_action_on_overloaded': True,
    #     'timestep_reset': True,
    #     'placement_reset': True
    # })
    # num_workers = 4
    # learn_config.update({"num_workers": num_workers})

    # # add the additional nencessary arguments to the edge config
    # env_config = add_path_to_config_edge(
    #     config=env_config_base,
    #     dataset_id=dataset_id,
    #     workload_id=workload_id_test,
    #     network_id=network_id,
    #     trace_id=trace_id_test
    # )

    path_env = type_env if type_env != 'kube-edge' else 'sim-edge'
    experiments_folder = os.path.join(TRAIN_RESULTS_PATH,
                                      "series",       str(train_series),
                                      "envs",         path_env,
                                      "datasets",     str(dataset_id),
                                      "workloads",    str(workload_id),
                                      "networks",     str(network_id),
                                      "traces",       str(trace_id),
                                      "experiments",  str(experiment_id),
                                      algorithm)

    ray.init(local_mode=local_mode)
    experiments_str = []
    for item in os.listdir(experiments_folder):
        if 'json' not in item:
            experiments_str.append(item)

    experiments_str.sort()
    for experiment_str, env_config, learn_config in zip(
        experiments_str, env_configs, learn_configs):

    # trained ray agent should always be simulation
    # however the agent outside it can be kuber agent or
    # other types of agent
        if type_env not in ['CartPole-v0', 'Pendulum-v0']:
            env = gym.make(ENVSMAP[type_env], config=env_config)
            # reset the env at the beginning of each episode
            ray_config = {"env": make_env_class('sim-edge'),
                        "env_config": env_config}
            ray_config.update(learn_config)
        else:
            ray_config = {"env": type_env}
            ray_config.update(learn_config)
                # break

        # checkpoint_string = [
        #     s for s in filter (
        #         lambda x: 'checkpoint' in x, os.listdir(
        #             os.path.join(
        #                 experiments_folder, experiment_string)))][0]
        # checkpoint = int(checkpoint_string.replace('checkpoint_',''))
        # checkpoint_path = os.path.join(
        #     experiments_folder,
        #     experiment_string,
        #     # os.listdir(experiments_folder)[0],
        #     checkpoint_string,
        #     f"checkpoint-{checkpoint}"
        # )
        if checkpoint_to_load=='last':
            checkpoint_string = sorted([
                s for s in filter (
                    lambda x: 'checkpoint' in x, os.listdir(
                        os.path.join(
                            experiments_folder, experiment_str)))])[-1]
            checkpoint = int(checkpoint_string.replace('checkpoint_',''))
            checkpoint_path = os.path.join(
                experiments_folder,
                experiment_str,
                # os.listdir(experiments_folder)[0],
                checkpoint_string,
                f"checkpoint-{checkpoint}"
            )
            checkpoint_to_load_info = checkpoint
        else:
            checkpoint_path = os.path.join(
                experiments_folder,
                experiment_str,
                # os.listdir(experiments_folder)[0],
                f"checkpoint_{checkpoint_to_load}",
                f"checkpoint-{int(checkpoint_to_load)}"
            )
            checkpoint_to_load_info = int(checkpoint_to_load)

        alg_env = make_env_class(type_env)
        if algorithm == 'PPO':
            agent = ppo.PPOTrainer(
                config=ray_config,
                env=alg_env)
        if algorithm == 'IMPALA':
            agent = impala.ImpalaTrainer(
                config=ray_config,
                env=alg_env)
        elif algorithm == 'A3C' or algorithm == 'A2C':
            agent = a3c.A3CTrainer(
                config=ray_config,
                env=alg_env)
        elif algorithm == 'PG':
            agent = pg.PGTrainer(
                config=ray_config,
                env=alg_env)
        elif algorithm == 'DQN':
            agent = dqn.DQNTrainer(
                config=ray_config,
                env=alg_env)
        import time
        episodes = []
        for i in range(0, num_episodes):
            print(f"---- \nepisode: {i} ----\n")
            agent.restore(checkpoint_path=checkpoint_path)
            episode_reward = 0
            done = False
            states = []
            obs = env.reset()
            # print(f"observation: {env.env.raw_observation}")
            # start = time.time()
            while not done:
                # print(f"timestep: {env.env.timestep}")
                action = agent.compute_action(obs)
                # pp.pprint(f"action: {action}")
                obs, reward, done, info = env.step(action)
                # pp.pprint(f"observation: {env.env.raw_observation}")
                # print('\n'+50*'-'+'\n')
                state = flatten(env.raw_observation, action, reward, info)
                states.append(state)
                episode_reward += reward
            # print("time elapsed: {}".format(time.time() - start))
            states = pd.DataFrame(states)
            print(f"episode reward: {episode_reward}")
            episodes.append(states)
        info = {
            'series': train_series,
            'dataset_id': dataset_id,
            'workload_id': workload_id,
            'network_id': network_id,
            'trace_id': trace_id,
            'trace_id_test': trace_id_test,
            'checkpoint': checkpoint_to_load_info,
            'experiment_str': experiment_str,
            'experiments': experiment_id,
            'episode_length': episode_length,
            'num_episodes': num_episodes,
            'algorithm': algorithm,
            'penalty_latency': env_config['penalty_latency'],
            'penalty_consolidated': env_config['penalty_consolidated'],
            'num_workers': num_workers
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

def fix_grid_searches(
    config, dataset_id, workload_id_test, network_id,
    trace_id_test, episode_length):
    """
    This function is used to fix the grid searches.
    """
    num_workers = 4
    pp = pprint.PrettyPrinter(indent=4)
    print('start experiments with the following config:\n')
    pp.pprint(config)
    learn_configs = []
    env_configs = []

    # extract differnt parts of the input_config
    learn_config = config['learn_config']
    algorithm = config["run_or_experiment"]
    env_config_base = config['env_config_base']

    values = []
    for k, v in env_config_base.items():
        if type(v) == dict:
            if 'grid_search' in v:
                values = v['grid_search']
                break

    # for k, v in learn_config.items():
    #     if v is dict:
    #         values = v['grid_search']
    values.sort()
    if values != []:
        for value in values:
            # update the difffent part of the envs 
            env_config_base_copy = copy.deepcopy(env_config_base)
            env_config_base_copy.update({
                'episode_length': episode_length,
                'no_action_on_overloaded': True,
                'timestep_reset': True,
                'placement_reset': True,
                k: value
            })
            learn_config.update({"num_workers": num_workers})
            # add the additional nencessary arguments to the edge config
            env_config = add_path_to_config_edge(
                config=env_config_base_copy,
                dataset_id=dataset_id,
                workload_id=workload_id_test,
                network_id=network_id,
                trace_id=trace_id_test
            )
            learn_configs.append(learn_config)
            env_configs.append(env_config)
    else:
        # update the difffent part of the envs
        env_config_base.update({
            'episode_length': episode_length,
            'no_action_on_overloaded': True,
            'timestep_reset': True,
            'placement_reset': True,
        })
        learn_config.update({"num_workers": num_workers})
        # add the additional nencessary arguments to the edge config
        env_config = add_path_to_config_edge(
            config=env_config_base,
            dataset_id=dataset_id,
            workload_id=workload_id_test,
            network_id=network_id,
            trace_id=trace_id_test
        )
        learn_configs.append(learn_config)
        env_configs.append(env_config)
    return algorithm, env_configs, learn_configs, num_workers
    # fix the grid searches


@click.command()
@click.option('--local-mode', type=bool, default=True)
@click.option('--test-series', required=True, type=int, default=74)
@click.option('--train-series', required=True, type=int, default=74)
@click.option('--type-env', required=True,
              type=click.Choice(['sim-edge', 'kube-edge']),
              default='kube-edge')
@click.option('--dataset-id', required=True, type=int, default=6)
@click.option('--workload-id', required=True, type=int, default=0)
@click.option('--network-id', required=False, type=int, default=1)
@click.option('--trace-id', required=False, type=int, default=2)
@click.option('--experiment-id', required=True, type=int, default=0)
@click.option('--episode-length', required=False, type=int, default=3453)
@click.option('--num-episodes', required=False, type=int, default=1)
@click.option('--workload-id-test', required=False, type=int, default=0)
@click.option('--trace-id-test', required=False, type=int, default=0)
@click.option('--checkpoint-to-load', required=False, type=str, default='last')
def main(local_mode: bool, test_series: int, train_series: int,
         type_env: str, dataset_id: int, workload_id: int,
         network_id: int, trace_id: int, experiment_id: int,
         num_episodes: int, episode_length: int,
         workload_id_test: int, trace_id_test: int,
         checkpoint_to_load: str):
    """[summary]

    Args:
        local_mode (bool): run in local mode for having the 
        test-series (int): series of the tests
        train-series (int): series of the trainining phase
        type_env (str): the type of the used environment
        dataset_id (int): used cluster dataset
        workload_id (int): the workload used in that dataset
        network_id (int): edge network of some dataset
        trace_id (int): user movement traces
        checkpoint (int): training checkpoint to load
        experiment-id (int): the trained agent experiment id
        episode-length (int): number of steps in the test episode
    """

    run_experiments(
        test_series=test_series,
        train_series=train_series, type_env=type_env,
        dataset_id=dataset_id, workload_id=workload_id,
        network_id=network_id, trace_id=trace_id,
        experiment_id=experiment_id,
        num_episodes=num_episodes, episode_length=episode_length,
        local_mode=local_mode, workload_id_test=workload_id_test,
        trace_id_test=trace_id_test, checkpoint_to_load=checkpoint_to_load)


if __name__ == "__main__":
    main()
