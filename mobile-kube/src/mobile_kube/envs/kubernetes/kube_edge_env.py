"""base class of edge enviornments
"""
import numpy as np
from typing import (
    Dict,
    Any
)

from gym.spaces import (
    Box,
    MultiDiscrete
)

from mobile_kube.util import (
    Preprocessor,
    override,
    check_config_edge,
    load_object
)
from mobile_kube.network import NetworkSimulator

from .kube_base_env import KubeBaseEnv

class KubeEdgeEnv(KubeBaseEnv):
    def __init__(self, config: Dict[str, Any]):
        # the network for edge simulators
        edge_simulator_config = load_object(config['network_path'])
        trace = load_object(config['trace_path'])

        self.edge_simulator = NetworkSimulator(
            trace=trace,
            **edge_simulator_config
            )

        self.users_stations = self.edge_simulator.users_stations
        self.num_users = self.edge_simulator.num_users
        self.num_stations = self.edge_simulator.num_stations
        self.normalise_latency = config['normalise_latency']
        # self.normalise_factor = self.edge_simulator.get_largest_station_node_path()
        check_config_edge(config)
        super().__init__(config)
        # TODO change here and remove the initialiser in the envs
        # TODO add trace here to the nuseretwork
        self.observation_space, self.action_space =\
            self._setup_space(config['action_method'])
        # _ = self.reset()

    @override(KubeBaseEnv)
    def _setup_space(self, action_space_type: str):
        """
        States:
            the whole or a subset of the following dictionary:
            observation = {
                    "services_resources_usage":
                        self.services_resources_usage,
                    "nodes_resources_usage":
                        self.nodes_resources_usage,
                    "services_resources_frac":
                        self.services_resources_frac,
                    "nodes_resources_frac":
                        self.nodes_resources_frac,
                    "services_nodes":
                        self.services_nodes,
                    "users_stations": --> always in the observation
                        self.users_stations
            }
        Actions:
                              nodes
            services [                   ]
        """
        # numuber of elements based on the obs in the observation space
        obs_size = 0
        for elm in self.obs_elements:
            if elm == "services_resources_usage":
                obs_size += self.num_services * self.num_resources
            elif elm == "nodes_resources_usage":
                obs_size += self.num_nodes * self.num_resources
            elif elm == "services_resources_usage_frac":
                obs_size += self.num_services * self.num_resources
            elif elm == "nodes_resources_usage_frac":
                obs_size += self.num_nodes * self.num_resources
            elif elm == "services_nodes":
                # add the one hot endoded services_resources
                # number of elements
                obs_size += (self.num_nodes+1) * self.num_services

        # add the one hot endoded users_stations
        # number of elements
        obs_size += (self.num_users) * self.num_stations

        higher_bound = 10 # TODO just for test
        # generate observation and action spaces
        observation_space = Box(low=0, high=higher_bound, shape=(obs_size, ))
        action_space = MultiDiscrete(np.ones(self.num_services) *
                                     self.num_nodes)

        return observation_space, action_space

    def preprocessor(self, obs):
        """
        environment preprocessor
        depeiding on the observation (state) definition
        """
        prep = Preprocessor(self.nodes_resources_cap,
                            self.services_resources_request,
                            self.num_stations)
        obs = prep.transform(obs)
        return obs

    @property
    def raw_observation(self) -> Dict[str, np.ndarray]:
        """
        returns only the raw observations requested by the user
        in the config input through obs_elements
        """
        observation = super().raw_observation
        observation.update({'users_stations': self.users_stations})
        return observation
