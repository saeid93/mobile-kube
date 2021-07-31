import os
import sys
import gym
import click
import numpy as np
from typing import Dict, Any
import time

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import ENVSMAP

from experiments.utils import (
    add_path_to_config_cloud,
    add_path_to_config_edge,
    config_reader,
    action_pretty_print,
    config_check_env_check
)




def check_env(*, config: Dict[str, Any], type_env: str,
              dataset_id: int, workload_id: int,
              network_id: int, trace_id: int):

    env_config_base = config["env_config_base"]
    if type_env == 'sim-edge' or type_env == 'kube-edge':
        env_config = add_path_to_config_edge(
            config=env_config_base,
            dataset_id=dataset_id,
            workload_id=workload_id,
            network_id=network_id,
            trace_id=trace_id
        )
    else:
        env_config = add_path_to_config_cloud(
            config=env_config_base,
            dataset_id=dataset_id,
            workload_id=workload_id
        )
    type_env = ENVSMAP[type_env]
    # make the approperiate env based on the type
    env = gym.make(type_env, config=env_config)

    i = 1
    _ = env.reset()
    while i < 10000:
        action = env.action_space.sample()
        print(f"action:\n <{action}>")
        time.sleep(1)
        _, reward, done, info = env.step(action)
        env.render()
        print(f"\niteration <{i}>:")           
        print(f"reward:\n <{reward}>")
        print(f"info:\n {info}")
        i += 1


@click.command()
@click.option('--mode', type=click.Choice(['experimental', 'real']),
              default='experimental')
@click.option('--config_folder', type=str, default='')
@click.option('--type-env', required=True,
              type=click.Choice(['sim-edge', 'sim-binpacking',
                                 'kube-edge', 'kube-binpacking']),
              default='sim-edge')
@click.option('--dataset-id', required=True, type=int, default=4)
@click.option('--workload-id', required=True, type=int, default=0)
@click.option('--network-id', required=False, type=int, default=0)
@click.option('--trace-id', required=False, type=int, default=0)
def main(mode: str, config_folder: str , type_env: str, dataset_id: int,
         workload_id: int, network_id: int, trace_id: int):
    """
    run it outside with
    python experiments false <-- name_of_the_config_folder -->
    e.g. for name_of_the_config_folder: A3C
    """
    config, _ = config_reader(mode, config_folder, 'check')
    config_check_env_check(config)
    check_env(config=config,
              type_env=type_env,
              dataset_id=dataset_id,
              workload_id=workload_id,
              network_id=network_id,
              trace_id=trace_id)


if __name__ == "__main__":
    main()
