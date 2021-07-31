import gym
import os
import json
import pickle
import pytest

file_folder = os.path.dirname(os.path.abspath(__file__))
CONFIGS_PATH = os.path.join(file_folder, "sample_data", "config")
DATASETS_PATH = os.path.join(file_folder, "sample_data", "dataset")
dataset_id = 0
network_id = 0
workload_id = 0


@pytest.fixture
def env_config():
    """
    read the dataset, workload and network from the disk
    and their path
    """
    config_file_path = os.path.join(CONFIGS_PATH, "config_check_env.json")

    with open(config_file_path) as cf:
        config = json.loads(cf.read())
    env_config_base = config["env_config_base"]
    # dataset paths
    dataset_path = os.path.join(DATASETS_PATH, str(dataset_id))
    workload_path = os.path.join(dataset_path, "workloads", str(workload_id))
    # if network_id != None:
    #     network_path = os.path.join(dataset_path, "networks", str(network_id))
    network_path = os.path.join(dataset_path, "networks", str(network_id))

    # load dataset
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f'dataset {dataset_id} does not exists')
    with open(os.path.join(dataset_path, 'dataset.pickle'), 'rb') as in_pickle:
        dataset = pickle.load(in_pickle)

    # load workload
    if not os.path.isdir(workload_path):
        raise FileNotFoundError((f'workload {workload_id} for dataset '
                                 f'{dataset_id} does not exists'))
    with open(os.path.join(workload_path, 'workload.pickle'),
              'rb') as in_pickle:
        workload = pickle.load(in_pickle)

    # load network
    if not os.path.isdir(network_path):
        raise FileNotFoundError((f'network {network_id} for dataset '
                                f'{dataset_id} does not exists'))
    with open(os.path.join(network_path, 'network.pickle'),
            'rb') as in_pickle:
        network = pickle.load(in_pickle)

    # load dataset config
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f'dataset {dataset_id} does not exists')
    with open(os.path.join(dataset_path, 'info.json'), 'rb') as cf:
        dataset_info = json.loads(cf.read())
    nodes_cap_rng = dataset_info['nodes_cap_rng']
    services_cap_rng = dataset_info['services_cap_rng']

    # add additional necessary info to the config
    env_config = env_config_base

    env_config.update({'dataset': dataset,
                       'workload': workload,
                       'nodes_cap_rng': nodes_cap_rng,
                       'services_cap_rng': services_cap_rng,
                       'network': network})
    return env_config

@pytest.fixture
def sample_cloud_env(env_config):
    env_config.update({'reward_mode': 'cloud'})
    env = gym.make(f'CloudSim-v0', config=env_config)
    return env

@pytest.fixture
def sample_edge_env(env_config):
    env_config.update({'reward_mode': 'edge'})
    env = gym.make(f'CloudSim-v5', config=env_config)
    return env

@pytest.fixture
def sample_network(env_config):
    return env_config['network']
