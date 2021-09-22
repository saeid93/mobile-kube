"""
base class for non learning environments
"""
from copy import deepcopy
from typing import (
    Dict,
    Any
)
from gym.spaces import (
    Box,
    MultiDiscrete
)
import numpy as np

from mobile_kube.util import (
    override
)
from .kube_base_env import KubeBaseEnv

class KubeBinpackingEnv(KubeBaseEnv):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # reset the environment to the initial state
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
                        self.services_nodes
            }
        Actions:
            absolute:
                                nodes
                services  [           ]
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

        # generate observation and action spaces
        observation_space = Box(low=0, high=1, shape=(obs_size, ))
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

    def _next_greedy_action(self, prev_services_nodes) -> np.ndarray:
        """
        Return the next bestfit greedy move
        generate the intitial state for nodes_services with a
        bestfit greedy algorithm:
            1. pop a node from the nodes lists
            2. find the nodes with least remained resources
            3. try to allocate the services to them
            4. if not possible pop another node
            5. if not possible to allocate it to the node with the largest
               remaining resources (already the last popped node)
        """
        self.services_nodes = np.ones(self.num_services, dtype=int) * (-1)
        nodes = list(np.arange(self.num_nodes))
        popped_nodes = []
        node_id = nodes.pop()
        popped_nodes.append(node_id)
        for service_id in range(self.num_services):
            try:
                # iterate through the currently popped nodes
                # remmaining resources
                # and check whether it is possible to fit the
                # current service inside it
                nodes_sorted = [node for _, node in
                                sorted(zip(
                                    self.nodes_resources_remained_frac_avg[
                                        popped_nodes], popped_nodes))]
                for node in nodes_sorted:
                    if np.alltrue(
                        self.services_resources_usage[service_id] <
                            self.nodes_resources_remained[node]):
                        self.services_nodes[service_id] = node
                        break
                else:  # no-break
                    node_id = nodes.pop()
                    popped_nodes.append(node_id)
                    self.services_nodes[service_id] = node_id
            except IndexError:
                self.services_nodes[service_id] = node

        action = deepcopy(self.services_nodes)
        # reset the services_nodes to the previous observation
        # for consistency with other envs environments
        self.services_nodes = deepcopy(prev_services_nodes)
        num_consolidated = self._num_consolidated(action)
        return num_consolidated, action
