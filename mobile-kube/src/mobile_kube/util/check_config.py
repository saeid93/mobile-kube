from typing import List
from typing import Dict, Any
from contextlib import suppress

def check_config(config: Dict[str, Any]):
    """
    check the structure of env_config_base_check
    """
    # check the for illegal items
    allowed_items = ['obs_elements', 'penalty_illegal', 'penalty_move',
                     'penalty_variance', 'penalty_latency',
                     'penalty_consolidated', 'mitigation_tries',
                     'workload_stop', 'episode_length', 'timestep_reset',
                     'placement_reset', 'reward_mode',
                     'compute_greedy_num_consolidated', 'seed', 'dataset',
                     'workload', 'nodes_cap_rng', 'services_request_rng',
                     'num_users', 'num_stations', 'network', 'normalise_latency',
                     'trace', 'from_dataset', 'edge_simulator_config',
                     'action_method', 'step_method', 'kube',
                     'dataset_path', 'workload_path', 'network_path', 'trace_path',
                     'no_action_on_overloaded']

    for key, _ in config.items():
        assert key in allowed_items, (f"<{key}> is not an allowed items for"
                                      " the environment config")
    # type checks
    ints = ['episode_length',
            'mitigation_tries', 'seed']
    for item in ints:
        assert type(config[item]) == int, f"<{item}> must be an integer"

    floats = ['workload_stop', 'penalty_illegal', 'penalty_illegal',
              'penalty_variance', 'penalty_consolidated',
              'penalty_latency', ]
    for item in floats:
        assert type(config[item])==float or type(config[item])==int,\
            f"[{item}] must be a float"

    bools = ['timestep_reset',  'placement_reset',
              'normalise_latency']
    for item in bools:
        assert type(config[item]) == bool, f"<{item}> must be a boolean"

    lists = ['obs_elements']
    for item in lists:
        assert type(config[item]) == list, f"<{item}> must be a list"

    strs = ['dataset_path', 'workload_path', 'network_path', 'trace_path']
    for item in strs:
        with suppress(KeyError):
            assert type(config[item]) == str, f"<{item}> must be a string"

    # observation checks
    all_obs_elements: List[str] = ["services_resources_usage",
                                   "nodes_resources_usage",
                                   "services_resources_usage_frac",
                                   "nodes_resources_usage_frac",
                                   "services_nodes",
                                   "auxiliary_resources_usage",
                                   "users_stations"]

    assert set(config['obs_elements']).issubset(
        set(all_obs_elements)), f"wrong input for the obs_element <{config['obs_elements']}>"

    # observation checks
    kube: List[str] = ["admin_config",
                       "service_image",
                       "namespace",
                       "clean_after_exit",
                       "services_nodes",
                       "using_auxiliary_server",
                       "utilization_image",
                       "workload_path",
                       "dataset_path"]

    assert set(config['kube']).issubset(
        set(kube)), "wrong input for the kube"


    # check the reward methods
    assert config['reward_mode'] in ['cloud', 'edge', 'both'],\
        f"Unkown reward option: <{config['reward_mode']}>"

    # check workload arguments
    if "workload_stop" in config:
        assert config['workload_stop'] <= 1, \
            "workload_stop is greater than 1"
        assert config['workload_stop'] >= 0, \
            "workload_stop is smaller than 0"

def check_config_edge(config: Dict[str, Any]):
    """check if it is a legal combination for the edge envs
    """
    assert config['reward_mode'] in ['edge', 'both'],\
        f"reward_mode <{config['reward_mode']}> is not compatible with edge env"

