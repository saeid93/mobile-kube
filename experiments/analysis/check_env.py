import os
import sys
import gym
import click
import matplotlib
import numpy as np
from typing import Dict, Any
import time
import json
from pprint import PrettyPrinter
import matplotlib.pyplot as plt
matplotlib.use("Agg")
pp = PrettyPrinter(indent=4)

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    ENVSMAP,
    CONFIGS_PATH,
    DATA_PATH
)
from experiments.utils import (
    add_path_to_config_edge,
    action_pretty_print,
    config_check_env_check
)


def check_env(*, config: Dict[str, Any], type_env: str,
              dataset_id: int, workload_id: int,
              network_id: int, trace_id: int):

    env_config_base = config["env_config_base"]
    env_config = add_path_to_config_edge(
        config=env_config_base,
        dataset_id=dataset_id,
        workload_id=workload_id,
        network_id=network_id,
        trace_id=trace_id
    )
    if type_env not in ['CartPole-v0', 'Pendulum-v0']:
        type_env = ENVSMAP[type_env]
        env = gym.make(type_env, config=env_config)
    else:
        env = gym.make(type_env)

    i = 1
    total_timesteps = 100
    _ = env.reset()

    reward_latency_1 = []
    reward_latency_2 = []
    reward_latency_3 = []
    reward_latency_4 = []
    reward_latency_5 = []
    reward_consolidation_1 = []
    reward_consolidation_2 = []
    reward_total = []
    users_distances = []

    while i < total_timesteps:
        action = env.action_space.sample()
        # print("\n\n--------action--------")
        print(f'timestamp {i}')
        print(env.raw_observation)
        print(action)
        # time.sleep(1)
        _, reward, done, info = env.step(action)
        # env.render()
        # env.edge_simulator.visualize_debug().savefig(
        #     os.path.join(DATA_PATH, 'plots', '3-5-2',f'{i}.png'))
        # print(f"\niteration <{i}>:")
        # print(f"reward:\n <{reward}>")
        reward_latency_1.append(
            info['rewards']['latency_rewards'][1])
        reward_latency_2.append(
            info['rewards']['latency_rewards'][2])
        reward_latency_3.append(
            info['rewards']['latency_rewards'][3])
        reward_latency_4.append(
            info['rewards']['latency_rewards'][4])
        reward_latency_5.append(
            info['rewards']['latency_rewards'][5])
        reward_consolidation_1.append(
            info['rewards']['consolidation_rewards'][1])
        reward_consolidation_2.append(
            info['rewards']['consolidation_rewards'][2])
        reward_total.append(reward)
        users_distances.append(
            info['users_distances'])
        # print('info:')
        # pp.pprint(info)
        i += 1
    x = np.arange(total_timesteps-1)
    # plt.plot(x, np.array(reward_latency_1), label = "L1")
    # plt.plot(x, np.array(reward_latency_2), label = "L2")
    # plt.plot(x, np.array(reward_latency_3), label = "L3")
    # plt.plot(x, np.array(reward_latency_4), label = "L4")
    plt.plot(x, np.array(reward_latency_5), label = "L5")
    # plt.plot(x, users_distances, label = "users_distances")
    # plt.plot(x, 1/np.array(users_distances), label = "1/users_distances")
    # plt.plot(x, reward_consolidation_1, label = "C1")
    plt.plot(x, reward_consolidation_2, label = "C2")
    plt.plot(x, reward_total, label = "reward")
    plt.legend()
    plt.grid()
    plt.savefig(f'pic_network_2')

@click.command()
@click.option('--type-env', required=True,
              type=click.Choice(['sim-edge', 'sim-binpacking', 'sim-greedy',
                                 'kube-edge', 'kube-binpacking', 'kube-greedy',
                                 'CartPole-v0', 'Pendulum-v0']),
              default='sim-edge')
@click.option('--dataset-id', required=True, type=int, default=6)
@click.option('--workload-id', required=True, type=int, default=0)
@click.option('--network-id', required=False, type=int, default=1)
@click.option('--trace-id', required=False, type=int, default=2)
def main(type_env: str, dataset_id: int,
         workload_id: int, network_id: int, trace_id: int):
    """[summary]

    Args:
        type_env (str): the type of the used environment
        dataset_id (int): used cluster dataset
        workload_id (int): the workload used in that dataset
        network_id (int): edge network of some dataset
        trace_id (int): user movement traces
    """
    config_file_path = os.path.join(
        CONFIGS_PATH, 'check',
        'check_env.json')
    with open(config_file_path) as cf:
        config = json.loads(cf.read())

    config_check_env_check(config)
    check_env(config=config,
              type_env=type_env,
              dataset_id=dataset_id,
              workload_id=workload_id,
              network_id=network_id,
              trace_id=trace_id)


if __name__ == "__main__":
    main()
