from typing import List
from typing import Dict, Any

def config_check_env_check(config: Dict[str, Any]):
    """
    check the structure of the config_check_env_check
    """
    allowed_items = ['env_config_base']
    for key, _ in config.items():
        assert key in allowed_items, (f"<{key}> is not an allowed items for"
                                      " the environment config")
    env_config_base_check(config['env_config_base'])

def config_dataset_generation_check(config: Dict[str, Any]):
    """
    check the structure of the dataset generation
    """
    allowed_items = ['notes', 'nums', 'metrics', 'nodes_cap_rng',
                     'services_request_rng', 'cutoff', 'start_workload', 'seed']
    for key, _ in config.items():
        assert key in allowed_items, (f"<{key}> is not an allowed items for"
                                      " the environment config")
    
    config_nums = config['nums']
    allowed_items_in_nums = ['nodes', 'services', 'resources',
                             'services_types', 'services_types_map']
    for key, _ in config_nums.items():
        assert key in allowed_items_in_nums, (f"<{key}> is not an allowed"
                                              " items for the "
                                              "dataset generation config "
                                              "in nums variable")
    assert config_nums['services_types'] ==\
        len(config_nums['services_types_map']),\
            (f"services_types <{config_nums['services_types']}> is not"
             " equal to the length of the services_types_map"
             f" <{len(config_nums['services_types_map'])}>")
    assert config_nums['services'] ==\
        sum(config_nums['services_types_map']),\
            (f"number of services <{config_nums['services']}> is not"
             " equal to the sum of contaienrs_types_map"
             f" <{sum(config_nums['services_types_map'])}>")

def config_workload_generation_check(config: Dict[str, Any]):
    allowed_items = ['notes', 'dataset_id', 'timesteps', 'services_types',
                     'workloads_var', 'plot_smoothing', 'seed']
    for key, _ in config.items():
        assert key in allowed_items, (f"<{key}> is not an allowed items for"
                                      " the workload generation config")

def config_network_generation_check(config: Dict[str, Any]):
    allowed_items = ['notes', 'dataset_id', 'num_users', 'num_stations',
                     'width', 'length', 'speed_limit', 'from_dataset',
                     'users_services_distributions', 'dataset_metadata',
                     'nodes_stations_con', 'nodes_selection', 'nodes_list',
                     'seed', 'colocated']
    for key, _ in config.items():
        assert key in allowed_items, (f"<{key}> is not an allowed items for"
                                      " the network generation config")
    # type checks
    floats = ['width', 'length']
    for item in floats:
        assert type(config[item]) == float, f"<{item}> must be an float"
    ints = ['dataset_id', 'num_users', 'num_stations',
            'speed_limit', 'dataset_metadata',
            'nodes_stations_con', 'seed']
    for item in ints:
        assert type(config[item]) == int, f"<{item}> must be an integer"
    strs = ['notes', 'users_services_distributions', 'nodes_selection']
    for item in strs:
        assert type(config[item]) == str, f"<{item}> must be an string"
    bools = ['from_dataset', 'colocated']
    for item in bools:
        assert type(config[item]) == bool, f"<{item}> must be an boolean"
    lists = ['nodes_list']
    for item in lists:
        assert type(config[item]) == list, f"<{item}> must be an list"

    # option checks
    assert config['users_services_distributions'] in ['random', 'equal'],\
        ("Unkown users_services_distributions option:"
         f" <{config['users_services_distributions']}>")
    assert config['nodes_selection'] in ["ordered", "random", "node_list"],\
        ("Unkown node_selection option:"
         f" <{config['nodes_selection']}>")

def config_trace_generation_check(config: Dict[str, Any]):
    allowed_items = ['dataset_id', 'network_id', 'speed', 'timesteps',
                     'from_dataset', 'seed']
    for key, _ in config.items():
        assert key in allowed_items, (f"<{key}> is not an allowed items for"
                                      " the network generation config")
    # type checks
    ints = ['dataset_id', 'network_id', 'speed', 'timesteps', 'seed']
    for item in ints:
        assert type(config[item]) == int, f"<{item}> must be an integer"
    bools = ['from_dataset']
    for item in bools:
        assert type(config[item]) == bool, f"<{item}> must be an boolean"


def env_config_base_check(config: Dict[str, Any]):
    """
    check the structure of env_config_base_check
    """
    # check the items
    allowed_items = ['obs_elements', 'penalty_illegal', 'penalty_move',
                     'penalty_variance', 'penalty_latency',
                     'penalty_consolidated', 'mitigation_tries',
                     'episode_length', 'placement_reset',
                     'compute_greedy_num_consolidated', 'seed', 'dataset',
                     'workload', 'nodes_cap_rng', 'services_request_rng',
                     'num_users', 'num_stations', 'network',
                     'normalise_latency', 'trace', 'from_dataset',
                     'edge_simulator_config', 'action_method', 'step_method',
                     'kube', "no_action_on_overloaded", "latency_reward_option",
                     'latency_lower', 'latency_upper', 'consolidation_lower',
                     'consolidation_upper']

    for key, _ in config.items():
        assert key in allowed_items, (f"<{key}> is not an allowed items for"
                                      " the environment config")
    # type checks
    ints = ['episode_length', 'seed']
    for item in ints:
        assert type(config[item]) == int, f"<{item}> must be an integer"
    floats = ['penalty_illegal', 'penalty_illegal',
              'penalty_variance', 'penalty_consolidated',
              'penalty_latency', ]
    for item in floats:
        assert type(config[item])==float or type(config[item])==int,\
            f"[{item}] must be a float"
    # bools = ['placement_reset']
    # for item in bools:
    #     assert type(config[item]) == bool, f"<{item}> must be a boolean"
    # assert type(config['obs_elements']) == list,\
    #     "obs_elements' must be a list"

    # observation checks
    all_obs_elements: List[str] = ["services_resources_usage",
                                   "nodes_resources_usage",
                                   "services_resources_usage_frac",
                                   "nodes_resources_usage_frac",
                                   "services_nodes",
                                   "users_stations"]

    assert set(config['obs_elements']).issubset(
        set(all_obs_elements)),\
            f"wrong input for the obs_element <{config['obs_elements']}>"

    # observation checks
    kube: List[str] = ["admin_config",
                       "service_image",
                       "namespace",
                       "clean_after_exit",
                       "services_nodes",
                       "utilization_image",
                       "workload_path",
                       "dataset_path"]

    assert set(config['kube']).issubset(
        set(kube)), "wrong input for the kube"
