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

from .sim_edge_env import SimEdgeEnv

class SimBinpackingEnv(SimEdgeEnv):
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
        # find take the action
        prev_services_nodes = deepcopy(self.services_nodes)
        _, action = self._next_greedy_action(prev_services_nodes)
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
            # greedy_mitigation_needed,
            # auxiliary_node_mitigation_needed=self,
            # prev_num_overloaded,
            users_distances=users_distances,
            num_moves=num_moves
            )


# (self, *, num_moves: int,
#                       greedy_mitigation_needed: bool,
#                       auxiliary_node_mitigation_needed: bool,
#                       prev_num_overloaded: int,
#                       users_distances: np.array = None)

        info = {'num_moves': num_moves,
                'num_consolidated': self.num_consolidated,
                'total_reward': reward,
                'rewards': rewards,
                'timestep': self.timestep}

        assert self.observation_space.contains(self.observation),\
                (f"observation:\n<{self.raw_observation}>\noutside of "
                f"observation_space:\n <{self.observation_space}>")

        return self.observation, None, self.done, info

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
