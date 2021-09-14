"""base class for non learning environments
"""
import numpy as np
from copy import deepcopy
from typing import (
    Dict,
    Any,
    Tuple
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

from .sim_base_env import SimBaseEnv

class SimBinpackingEnv(SimBaseEnv):
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
        # self.normalise_factor = self.edge_simulator.get_largest_station_node_path() TODO check
        check_config_edge(config)
        super().__init__(config)
        # TODO change here and remove the initialiser in the envs
        # TODO add trace here to the nuseretwork
        self.observation_space, self.action_space =\
            self._setup_space()
        _ = self.reset()

    @override(SimBaseEnv)
    def _setup_space(self):
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

        # add the one hot endoded users_stations
        # number of elements
        obs_size += (self.num_users) * self.num_stations

        higher_bound = 10 # TODO TEMP just for test - find a cleaner way
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
        # initialise with auxiliary
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
                                    self.nodes_resources_available_frac_avg[
                                        popped_nodes], popped_nodes))]
                for node in nodes_sorted:
                    if np.alltrue(
                        self.services_resources_request[service_id] <
                            self.nodes_resources_available[node]):
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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, int, bool, dict]:
        """
        General overivew:
            1. try binpacking without auxiliary
            2. if doesn't fit add auxiliary as the last server
                and allocate with considering it (auxiliary has
                unlimited space)
        Binpacking:
        generate the intitial state for nodes_services with a
        bestfit greedy algorithm.
            1. iterate over nodes one by one
            2. pop a node from the nodes lists
            3. find the nodes with least remained resources
            4. try to allocate the services to them
            5. if not possible pop another node
        We use mitigation after bin-packing because
        the action is based-on the current timestep but
        next timestep might result in illegal resource usage
        with the current placement
        """
        # save previous action for computing the
        # final (after mitigation) num_of_moves
        prev_services_nodes = deepcopy(self.services_nodes)

        # get the next greedy actoin from the env
        _, action = self._next_greedy_action(prev_services_nodes)

        # move the timestep to the next point of time
        # and then take the action
        assert self.action_space.contains(action)
        self.global_timestep += 1
        # find the reward and compute the remainder for round-robin
        self.timestep = self.global_timestep % self.workload.shape[1]
        self.services_nodes = deepcopy(action)

        num_moves = len(np.where(
            self.services_nodes != prev_services_nodes)[0])

        info = {'num_moves': num_moves,
                'num_consolidated': self.num_consolidated,
                'timestep': self.timestep}

        assert self.observation_space.contains(self.observation),\
                (f"observation:\n<{self.raw_observation}>\noutside of "
                f"observation_space:\n <{self.observation_space}>")

        return self.observation, None, self.done, info