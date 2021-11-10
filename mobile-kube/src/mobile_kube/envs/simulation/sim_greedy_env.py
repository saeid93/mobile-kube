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
    load_object
)
from mobile_kube.network import NetworkSimulator

from .sim_edge_env import SimEdgeEnv

class SimGreedyEnv(SimEdgeEnv):
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, int, bool, dict]:
        """
        General overivew:
            1. moves the users to the nearest available server
        """
        # find take the action
        prev_services_nodes = deepcopy(self.services_nodes)
        action = self._next_greedy_action(prev_services_nodes)
        assert self.action_space.contains(action)
        self.services_nodes = deepcopy(action)

        # move to the next timestep
        self.global_timestep += 1
        self.timestep = self.global_timestep % self.workload.shape[1]

        # make user movements --> network parts
        self.users_stations = self.edge_simulator.sample_users_stations(
            timestep=self.timestep)
        users_distances = self.edge_simulator.users_distances
        # update network with the new placements
        self.edge_simulator.update_services_nodes(self.services_nodes)

        num_moves = len(np.where(
            self.services_nodes != prev_services_nodes)[0])

        reward, rewards = self._reward(
            num_overloaded=self.num_overloaded,
            users_distances=users_distances,
            num_moves=num_moves
            )

        info = {'num_consolidated': self.num_consolidated,
                'num_moves': num_moves,
                'num_overloaded': self.num_overloaded,
                'users_distances': np.sum(users_distances),
                'total_reward': reward,
                'timestep': self.timestep,
                'rewards': rewards}

        assert self.observation_space.contains(self.observation),\
                (f"observation:\n<{self.raw_observation}>\noutside of "
                f"observation_space:\n <{self.observation_space}>")

        return self.observation, reward, self.done, info

    def _next_greedy_action(self, prev_services_nodes) -> np.ndarray:
        """
        """
        action = self.edge_simulator._next_greedy_action(
            nodes_mem_cap=self.nodes_resources_cap[:, 0],
            services_mem_request=self.services_resources_request[:, 0],
            nodes_cpu_cap=self.nodes_resources_cap[:, 1],
            services_cpu_request=self.services_resources_request[:, 1])
        return action
