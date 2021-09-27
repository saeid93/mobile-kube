"""
scripts to check a learned agent
based-on:
https://github.com/ray-project/ray/issues/9123
https://github.com/ray-project/ray/issues/7983
"""
import os
import sys
import shutil
import click
from typing import Dict, Any
import json

import ray
from ray import tune
from ray.rllib.utils.framework import try_import_torch
import pprint
import gym
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.pg as pg
import ray.rllib.agents.dqn as dqn
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    RESULTS_PATH,
    CONFIGS_PATH,
    ENVSMAP
)
from experiments.utils import (
    add_path_to_config_edge,
    make_env_class,
    CloudCallback
)

torch, nn = try_import_torch()


def learner(*, config_file_path: str, config: Dict[str, Any],
            series: int, type_env: str, dataset_id: int,
            workload_id: int, network_id: int, trace_id: int,
            use_callback: bool, checkpoint: int, experiment_id: int,
            local_mode: bool):
    """
    """
    # extract differnt parts of the input_config
    learn_config = config['learn_config']
    algorithm = config["run_or_experiment"]
    env_config_base = config['env_config_base']

    # add evn_config_base updates
    env_config_base.update({
        'episode_length': 1000,
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
    ray_config = {"env": make_env_class('sim-edge'),
                  "env_config": env_config}
    ray_config.update(learn_config)

    # generate the path
    # folder formats: <environmet>/datasets/<dataset>/workloads/<workload>
    # example:        env1/dataset/1/workloads/3
    experiments_folder = os.path.join(RESULTS_PATH,
                                      "series",      str(series),
                                      "envs",        'sim-edge',
                                      "datasets",    str(dataset_id),
                                      "workloads",   str(workload_id),
                                      "experiments", str(experiment_id),
                                      algorithm)
    for item in os.listdir(experiments_folder):
        if 'json' not in item:
            experiment_string = item
            break

    checkpoint_path = os.path.join(
        experiments_folder,
        experiment_string,
        # os.listdir(experiments_folder)[0],
        f"checkpoint_000{checkpoint}",
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

    agent.restore(checkpoint_path=checkpoint_path)
    episode_reward = 0
    done = False
    obs = env.reset()
    env.render()
    while not done:
        action = agent.compute_action(obs)
        print("\n\n--------action--------")
        print(action)
        obs, reward, done, info = env.step(action)
        env.render()
        print('info:')
        pp.pprint(info)
        episode_reward += reward
    print(f"episode reward: {episode_reward}")


@click.command()
@click.option('--local-mode', type=bool, default=True)
@click.option('--config-folder', type=str, default='PPO')
@click.option('--series', required=True, type=int, default=1)
@click.option('--type-env', required=True,
              type=click.Choice(['sim-edge', 'kube-edge']),
              default='sim-edge')
@click.option('--dataset-id', required=True, type=int, default=1)
@click.option('--workload-id', required=True, type=int, default=0)
@click.option('--network-id', required=False, type=int, default=0)
@click.option('--trace-id', required=False, type=int, default=0)
@click.option('--use-callback', required=True, type=bool, default=False)
@click.option('--experiment_id', required=True, type=int, default=1)
@click.option('--checkpoint', required=False, type=int, default=200)
def main(local_mode: bool, config_folder: str, series: int,
         type_env: str, dataset_id: int, workload_id: int, network_id: int,
         trace_id: int, use_callback: bool, experiment_id: int,
	 checkpoint: int):
    """[summary]
    Args:
        local_mode (bool): run in local mode for having the 
        config_folder (str): name of the config folder (only used in real mode)
        use_callback (bool): whether to use callbacks or storing and visualising
        checkpoint (int): selected checkpoint to test
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

    learner(config_file_path=config_file_path,
            config=config, series=series,
            type_env=type_env, dataset_id=dataset_id,
            workload_id=workload_id, network_id=network_id,
            trace_id=trace_id, use_callback=use_callback,
            experiment_id=experiment_id, checkpoint=checkpoint,
	    local_mode=local_mode)


if __name__ == "__main__":
    main()