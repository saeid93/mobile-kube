import networkx as nx
from networkx.algorithms import tree
from operator import itemgetter
from networkx.algorithms.shortest_paths.generic import shortest_path 
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import (
    List,
    Tuple,
    Dict
)

from .network_base import NetworkSimulatorBase
from mobile_kube.util import override

class NetworkSimulator(NetworkSimulatorBase):
    def __init__(
        self, *, services_nodes: np.array,
        users_services: np.array, num_nodes: np.array,
        num_stations: int, width: int, length: int,
        speed_limit: int, nodes_stations_con: int,
        from_dataset: bool, users_dataset_path: str = "",
        stations_dataset_path: str = "",
        network: nx.Graph, raw_network: nx.Graph,
        trace: List[Dict[int, Tuple[float, float]]],
        seed: int, selected_nodes: list=None):
        """
            The initialiser for using the simulator in the envs
        """
        super().__init__(
            services_nodes=services_nodes,
            users_services=users_services,
            num_nodes=num_nodes,
            num_stations=num_stations, width=width,
            length=length, speed_limit=speed_limit,
            nodes_stations_con=nodes_stations_con, seed=seed)
        self.network = network
        self.raw_network = network
        self.trace = trace

    @override(NetworkSimulatorBase)
    def _users_move_from_trace(self, timestep: int=None):
        """
            move users from the loaded trace
        """
        timestep = timestep % len(self.trace)
        users_next_location = self.trace[timestep]
        self._change_users_loactions(users_next_location)

    def _change_users_loactions(self, users_next_location):
        """
            function to change the users locations from the trace
        """
        locs_updates = dict(map(lambda a: ((a[0], 'user'), a[1]),
                                users_next_location.items()))
        for user in self.users_idx:
            self.network.nodes[user]['loc'] = locs_updates[user]
