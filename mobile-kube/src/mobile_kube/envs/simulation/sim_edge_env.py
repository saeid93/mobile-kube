"""base class of edge enviornments
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

class SimEdgeEnv(SimBaseEnv):
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
                        self.services_nodes,
                    "users_stations": --> always in the observation
                        self.users_stations
            }
        users_stations:
             user_id        user_id                 user_id
            [station_id,    station_id,    , ... ,  station_id  ]
                        range:
                            indices: [0, num_users)
                            contents: [0, num_stations)
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

    @override(SimBaseEnv)
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, int, bool, dict]:
        """
        edge servers here
        1. move the services based-on current network and nodes state
        2. do one step of user movements
        3. update the nodes
        """
        # save previous action for computing the
        # final num_of_moves
        prev_services_nodes = deepcopy(self.services_nodes)
        if self.timestep == 3452:
            a = 1

        # TODO add action of the users here too
        assert self.action_space.contains(action)
        self.global_timestep += 1
        # find the reward and compute the remainder for round-robin
        self.timestep = self.global_timestep % self.workload.shape[1]
        self.services_nodes = deepcopy(action)
        
        # make user movements --> network parts
        self.users_stations = self.edge_simulator.sample_users_stations(
            timestep=self.timestep)
        users_distances = self.edge_simulator.users_distances
        # update network with the new placements
        self.edge_simulator.update_services_nodes(self.services_nodes)


        num_moves = len(np.where(
            self.services_nodes != prev_services_nodes)[0])

        reward, rewards = self._reward(
            users_distances=users_distances,
            num_moves=num_moves
            )

        info = {'num_moves': num_moves,
                'num_consolidated': self.num_consolidated,
                'total_reward': reward,
                'rewards': rewards,
                'timestep': self.timestep}

        assert self.observation_space.contains(self.observation),\
                (f"observation:\n<{self.raw_observation}>\noutside of "
                f"observation_space:\n <{self.observation_space}>")

        return self.observation, reward, self.done, info
