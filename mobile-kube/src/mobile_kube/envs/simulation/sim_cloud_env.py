"""base class for centralised cloud enviornments
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
    override,
    check_config_cloud
)

from .sim_base_env import SimBaseEnv

class SimCloudEnv(SimBaseEnv):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        check_config_cloud(config)
        self.observation_space, self.action_space =\
            self._setup_space(config['action_method'])
        _ = self.reset()

    @override(SimBaseEnv)
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
                        self.services_nodes
            }
        Actions:
            probabilistic:
                                    nodes_priorities_probabilities
                                |                                   |
                    services    |                                   |
                                |                                   |
            absolute:
                                nodes
                services    [           ]
        """
        # numuber of elements based on the obs in the observation space
        obs_size = 0
        # TODO make it dictionary
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
            elif elm == "auxiliary_resources_usage":
                # add the auxiliary resource usage to the number of elements
                obs_size += self.num_resources

        higher_bound = 3 # TODO just for test
        # generate observation and action spaces
        observation_space = Box(low=0, high=higher_bound, shape=(obs_size, ))
        if action_space_type == 'probabilistic':
            action_space = Box(low=self.action_min, high=self.action_max,
                               shape=(self.num_services * self.num_nodes,))

        elif action_space_type == 'absolute':
            action_space = MultiDiscrete(np.ones(self.num_services) *
                                         self.num_nodes)
        else:
            raise ValueError("unkown type of action space "
                             f"--> {action_space_type}")

        return observation_space, action_space
