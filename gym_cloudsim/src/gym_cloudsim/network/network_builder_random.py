from copy import deepcopy
import networkx as nx
from networkx.algorithms import tree
from operator import itemgetter
import numpy as np
import random

from .network_base import NetworkSimulatorBase
from gym_cloudsim.util import (
    override
)

class NetworkBuilderRandom(NetworkSimulatorBase):
    """class to make the network and trace from the dataset
    """
    def __init__(
        self, *, services_nodes: np.array,
        users_services: np.array, num_nodes: np.array,
        num_stations: int, width: int, length: int,
        speed_limit: int, nodes_stations_con: int ,seed: int) -> None:
        """The initialiser for generating the dataset network
        """
        super().__init__(
            services_nodes=services_nodes,
            users_services=users_services,
            num_nodes=num_nodes,
            num_stations=num_stations,
            width=width, length=length,
            speed_limit=speed_limit,
            nodes_stations_con=nodes_stations_con,
            seed=seed)
        self.raw_network = self._make_raw_network()
        self.network = deepcopy(self.raw_network)
        self._make_complete_network()
        self.trace = None

    @classmethod
    def with_network(
        cls, *, services_nodes: np.array,
        users_services: np.array, num_nodes: np.array,
        num_stations: int, width: int, length: int,
        speed_limit: int, nodes_stations_con: int,
        network: nx.Graph, raw_network: nx.Graph,
        seed: int):
        """The initialiser for generating the random network
        """
        ins = cls(
            services_nodes=services_nodes,
            users_services=users_services,
            num_nodes=num_nodes,
            num_stations=num_stations, width=width,
            length=length, speed_limit=speed_limit,
            nodes_stations_con=nodes_stations_con, seed=seed)
        ins.network = network
        ins.raw_network = raw_network
        return ins

    @override(NetworkSimulatorBase)
    def _make_raw_network(self) -> nx.Graph:
        """make the raw network:
           the network only with
           nodes and stations not the users
           """
        network = nx.Graph()
        
        # add raw nodes
        network.add_nodes_from(self.nodes_idx)
        network.add_nodes_from(self.stations_idx)
        
        # add location to nodes
        occupied_locs = set()
        for node in network.nodes:
            while True:
                loc = (round(random.uniform(0, self.length), 2),
                       round(random.uniform(0, self.width), 2))
                if loc not in occupied_locs:
                    network.nodes[node]['loc'] = loc
                    occupied_locs.add(loc)
                    break

        # adding node-node edges to the network with spanning tree
        nodes_subgraph = nx.complete_graph(
            network.subgraph(self.nodes_idx).copy())
        for edge in nodes_subgraph.edges:
            weight = self._euclidean_dis(network.nodes[edge[0]]['loc'],
                                         network.nodes[edge[1]]['loc'])
            nodes_subgraph.edges[edge]['weight'] = weight
        mst = tree.minimum_spanning_edges(nodes_subgraph, algorithm='kruskal',
                                          weight='weight', data=True)
        for edge in mst:
            network.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

        # add station-nodes edges by connecting stations
        # to their closest nodes
        for station in self.stations_idx:
            dis_nodes = {}
            for node in self.nodes_idx:
                dis = self._euclidean_dis(network.nodes[station]['loc'],
                                          network.nodes[node]['loc'])
                dis_nodes[node] = dis
            res = dict(sorted(dis_nodes.items(),
                              key = itemgetter(1))[:self.nodes_stations_con])
            for key, value in res.items():
                network.add_edge(station, key, weight=value)
        return network

    @override(NetworkSimulatorBase)
    def _make_users(self, network) -> nx.Graph:
        """
           place users randomly
           at locations in the map
        """
        network.add_nodes_from(self.users_idx)
        # add location to nodes
        occupied_loc = set(nx.get_node_attributes(network,'loc').values())
        for node in self.users_idx:
            while True:
                loc = (round(random.uniform(0, self.length), 2),
                       round(random.uniform(0, self.width), 2))
                if loc not in occupied_loc:
                    network.nodes[node]['loc'] = loc
                    occupied_loc.add(loc)
                    break
        return network

    @override(NetworkSimulatorBase)
    def _users_move_random(self) -> None:
        """
            Randomly moves users with `User.SPEED_LIMIT` in 2d surface
        """
        for node_id, node_type in self.network.nodes:
            if node_type == 'user':
                user_speed = self.SPEED_LIMIT
                # user_speed = np.random.randint(SPEED_LIMIT)
                dx = (random.random() - 0.5)*user_speed/100*self.width
                dy = (random.random() - 0.5)*user_speed/100*self.length
                node_location = self.network.nodes[
                    (node_id, node_type)]['loc']
                new_x = node_location[0] + dx
                new_y = node_location[1] + dy
                if new_x > self.width:
                    new_x = self.width
                if new_x < 0:
                    new_x = 0
                if new_y > self.length:
                    new_y = self.length
                if new_y < 0:
                    new_y = 0
                self.network.nodes[(node_id, node_type)]['loc'] =\
                    (round(new_x, 2), round(new_y, 2))
