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
            use_callback: bool, checkpoint_freq: int,
            local_mode: bool):
    """
    input_config: {"env_config_base": ...,
                    "run_or_experiment": ...,
                    "learn_config": ...,
                    "stop": ...}
    - is used to build:
    ray_config: {...
                learning paramters
                ...,
                env: <environment class>,
                env_config: <environment config read before>
                }

    - the results are saved into the concatenation of the following paths:
        - results path:
          data/results/
        - environment info:
          env/<env_id>/datasets/<dataset_id>/workloads/<workload_id>
          /experiments/<experiment_id>
        - rllib:
          <name_of_algorithm>/<trial>
    """
    # extract differnt parts of the input_config
    stop = config['stop']
    learn_config = config['learn_config']
    run_or_experiment = config["run_or_experiment"]
    env_config_base = config['env_config_base']
    # type_env = ENVSMAP[type_env]

    # add the additional nencessary arguments to the edge config
    env_config = add_path_to_config_edge(
        config=env_config_base,
        dataset_id=dataset_id,
        workload_id=workload_id,
        network_id=network_id,
        trace_id=trace_id
    )

    # generate the ray_config
    # make the learning config based on the type of the environment
    if type_env not in ['CartPole-v0', 'Pendulum-v0']:
        ray_config = {"env": make_env_class(type_env),
                    "env_config": env_config}
    else:
        ray_config = {"env": type_env}

    # generate the path
    # folder formats: <environmet>/datasets/<dataset>/workloads/<workload>
    # example:        env1/dataset/1/workloads/3
    experiments_folder = os.path.join(RESULTS_PATH,
                                      "series",     str(series),
                                      "envs",       str(type_env),
                                      "datasets",   str(dataset_id),
                                      "workloads",  str(workload_id),
                                      "experiments")
    # make the base path if it does not exists
    if not os.path.isdir(experiments_folder):
        os.makedirs(experiments_folder)
    # generate new experiment folder
    content = os.listdir(experiments_folder)
    new_experiment = len(content)
    this_experiment_folder = os.path.join(experiments_folder,
                                          str(new_experiment))
    # make the new experiment folder
    os.mkdir(this_experiment_folder)

    # copy our input json to the path a change
    # the name to a unified name
    shutil.copy(config_file_path, this_experiment_folder)
    source_file = os.path.join(this_experiment_folder,
                               os.path.split(config_file_path)[-1])
    dest_file = os.path.join(this_experiment_folder, 'experiment_config.json')
    os.rename(source_file, dest_file)

    # update the ray_config with learn_config
    ray_config.update(learn_config)

    # if callback is specified add it here
    if use_callback and\
        type_env not in ['CartPole-v0', 'Pendulum-v0']:
        ray_config.update({'callbacks': CloudCallback})

    ray.init(local_mode=local_mode)
    # run the ML after fixing the folders structres
    _ = tune.run(local_dir=this_experiment_folder,
                 run_or_experiment=run_or_experiment,
                 config=ray_config,
                 stop=stop,
                 checkpoint_freq=checkpoint_freq,
                 checkpoint_at_end=True)

    # delete the unnecessary big json file
    # TODO maybe of use in the analysis
    this_experiment_trials_folder = os.path.join(
        this_experiment_folder, run_or_experiment)
    this_experiment_trials_folder_contents = os.listdir(
        this_experiment_trials_folder)
    for item in this_experiment_trials_folder_contents:
        if 'json' in item:
            json_file_name = item
            break
    json_file_path = os.path.join(
        this_experiment_trials_folder,
        json_file_name)
    os.remove(json_file_path)


@click.command()
@click.option('--local-mode', type=bool, default=False)
@click.option('--config-file', type=str, default='PPO-search')
@click.option('--series', required=True, type=int, default=6)
@click.option('--type-env', required=True,
              type=click.Choice(['sim-edge', 'sim-binpacking', 'sim-edge-greedy',
                                 'CartPole-v0', 'Pendulum-v0']),
              default='sim-edge')
@click.option('--dataset-id', required=True, type=int, default=6)
@click.option('--workload-id', required=True, type=int, default=0)
@click.option('--network-id', required=False, type=int, default=0)
@click.option('--trace-id', required=False, type=int, default=0)
@click.option('--use-callback', required=True, type=bool, default=True)
@click.option('--checkpoint-freq', required=False, type=int, default=100)
def main(local_mode: bool, config_file: str, series: int,
         type_env: str, dataset_id: int, workload_id: int, network_id: int,
         trace_id: int, use_callback: bool, checkpoint_freq: int):
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
    config_file_path = os.path.join(
        CONFIGS_PATH, 'train', f"{config_file}.json")
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
            checkpoint_freq=checkpoint_freq, local_mode=local_mode)


if __name__ == "__main__":
    main()
