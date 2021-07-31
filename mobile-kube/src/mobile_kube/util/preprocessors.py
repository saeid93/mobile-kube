import numpy as np
from typing import Dict
from mobile_kube.util import rounding

class Preprocessor():
    def __init__(self, nodes_resources_cap: np.ndarray,
                 services_resources_cap: np.ndarray,
                 num_stations: int = 0):
        self.nodes_resources_cap = nodes_resources_cap
        self.services_resources_cap = services_resources_cap
        self.num_nodes = nodes_resources_cap.shape[0]
        self.num_stations = num_stations

    @rounding
    def transform(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        transform the input observation as the dictionary
        sends each key of the dictionary to the approperiate preprocessor
        and returns the concatenated and flattened numpy array
        """
        obs = np.array([])
        transformers = {
           'services_resources_usage': self._services_usage_normalizer,
           'nodes_resources_usage': self._nodes_usage_normalizer,
           'services_resources_usage_frac': self._none,
           'nodes_resources_usage_frac': self._none,
           'services_nodes': self._one_hot_services_nodes,
           'users_stations': self._one_hot_users_stations,
           'auxiliary_resources_usage': self._auxiliary_usage_normalizer
        }
        for key, val in observation.items():
            obs = np.concatenate((obs, transformers.get(
                key, self._invalid_operation)(val).flatten()))
        return obs

    def _services_usage_normalizer(self, obs: np.ndarray) -> np.ndarray:
        """
        divides the largest available of each resource by the
        capacity of the largest size of that resource in the cluster
        in any service
        e.g for ram:
            ram_usage_of_a_service / largest_ram_capacity_of_any_conrainer
        """
        lst = []
        for index in range(self.services_resources_cap.shape[1]):
            lst.append(max(self.services_resources_cap[:, index]))
        return obs/lst

    def _nodes_usage_normalizer(self, obs: np.ndarray) -> np.ndarray:
        """
        divides the largest available of each resource by the
        capacity of the largest size of that resource in the cluster
        in any node
        e.g for ram:
            ram_usage_of_a_node / largest_ram_capacity_of_any_node
        """
        lst = []
        for index in range(self.nodes_resources_cap.shape[1]):
            lst.append(max(self.nodes_resources_cap[:, index]))
        return obs/lst

    def _auxiliary_usage_normalizer(self, obs: np.ndarray) -> np.ndarray:
        """
        divides the largest available of each resource by the
        total capacity of that resource in the cluster
        e.g for ram:
            ram_usage_of_auxiliary / total_ram_cap_of_cluster
        so the size of auxiliary node is proportional to the cluster size
        """
        nodes_total = sum(self.nodes_resources_cap)
        return obs/nodes_total

    def _none(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def _one_hot_services_nodes(
        self, obs: np.ndarray) -> np.ndarray:
        """
        one hot encoding of the services_nodes
        e.g in a cluster of 2 nodes and 4 services:
            [0, 1, 1, 0]
        results in:
            [0, 0, 0, 1, 0, 1, 0, 0]
        """
        # TODO make a cleaner version of the one-hot encoding
        #      look at ray for inspriatino
        obs_prep = np.array([])
        for node in obs:
            # +1 for auxiliary node
            one_hot_encoded = np.zeros(self.num_nodes+1)
            one_hot_encoded[node] = 1
            obs_prep = np.concatenate((obs_prep, one_hot_encoded))
        return obs_prep

    def _one_hot_users_stations(
        self, obs: np.ndarray) -> np.ndarray:
        """
        one hot encoding of the services_nodes
        e.g in a cluster of 2 nodes and 4 services:
            [0, 1, 1, 0]
        results in:
            [0, 0, 0, 1, 0, 1, 0, 0]
        """
        # TODO make a cleaner version of the one-hot encoding
        #      look at ray for inspriatino
        obs_prep = np.array([])
        for user in obs:
            one_hot_encoded = np.zeros(self.num_stations)
            one_hot_encoded[user] = 1
            obs_prep = np.concatenate((obs_prep, one_hot_encoded))
        return obs_prep

    def _invalid_operation(self, obs: np.ndarray) -> None:
        # TODO get key instead of value
        raise ValueError(f"invalid observation: <{obs}>")