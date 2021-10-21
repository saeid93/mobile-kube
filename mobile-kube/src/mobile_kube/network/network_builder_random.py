from copy import deepcopy
import networkx as nx
from networkx.algorithms import tree
from operator import itemgetter
import numpy as np
import random

from .network_base import NetworkSimulatorBase
from mobile_kube.util import (
    override
)

class NetworkBuilderRandom(NetworkSimulatorBase):
    """class to make the network and trace from the dataset
    """
    def __init__(
        self, *, services_nodes: np.array,
        users_services: np.array, num_nodes: np.array,
        num_stations: int, width: int, length: int, colocated: bool,
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
        self.colocated = colocated
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
        users_dataset_path: str = "",
        stations_dataset_path: str = "", seed: int,
        selected_nodes: np.array):
        """The initialiser for generating the random network
        """
        ins = cls(
            services_nodes=services_nodes,
            users_services=users_services,
            num_nodes=num_nodes,
            num_stations=num_stations, width=width,
            length=length, speed_limit=speed_limit,
            colocated=True, nodes_stations_con=nodes_stations_con,
            seed=seed)
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
        if not self.colocated:
            occupied_locs = set()
            for node in network.nodes:
                while True:
                    loc = (round(random.uniform(0, 1), 2),
                        round(random.uniform(0, 1), 2))
                    if loc not in occupied_locs:
                        network.nodes[node]['loc'] = loc
                        occupied_locs.add(loc)
                        break
        else:
            assert self.num_nodes == self.num_stations, \
                "num of stations and nodes should be equal in colocated mode"
            nudge = 0.0001
            for station in self.stations_idx:
                # random station location
                loc = (round(random.uniform(0, 1), 2),
                       round(random.uniform(0, 1), 2))
                network.nodes[station]['loc'] = loc
                # colocated node in a near location
                node_loc = tuple(map(lambda x: x + nudge, loc))
                node_index = (station[0], 'node')
                network.nodes[node_index]['loc'] = node_loc

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
        xs = [v['loc'][0] for k, v in self.raw_network.nodes._nodes.items()]
        ys = [v['loc'][1] for k, v in self.raw_network.nodes._nodes.items()]
        self.x_min = min(xs) - self.length
        self.x_max = max(xs) + self.length
        self.y_min = min(ys) - self.width
        self.y_max = max(ys) + self.width
        network.add_nodes_from(self.users_idx)
        # add location to nodes
        occupied_loc = set(nx.get_node_attributes(network,'loc').values())
        for node in self.users_idx:
            while True:
                loc = (round(random.uniform(self.x_min, self.x_max), 2),
                       round(random.uniform(self.y_min, self.y_max), 2))
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
        xs = [v['loc'][0] for k, v in self.raw_network.nodes._nodes.items()]
        ys = [v['loc'][1] for k, v in self.raw_network.nodes._nodes.items()]
        self.x_min = min(xs) - self.length
        self.x_max = max(xs) + self.length
        self.y_min = min(ys) - self.width
        self.y_max = max(ys) + self.width
        for node_id, node_type in self.network.nodes:
            if node_type == 'user':
                user_speed = self.SPEED_LIMIT
                # user_speed = np.random.randint(SPEED_LIMIT)
                dx = (random.random() - 0.5)*user_speed/100* (self.x_max - self.x_min)
                dy = (random.random() - 0.5)*user_speed/100* (self.y_max - self.y_min)
                node_location = self.network.nodes[
                    (node_id, node_type)]['loc']
                new_x = node_location[0] + dx
                new_y = node_location[1] + dy
                if new_x > self.x_max:
                    new_x = self.x_max
                if new_x < self.x_min:
                    new_x = self.x_min
                if new_y > self.y_max:
                    new_y = self.y_max
                if new_y < self.y_min:
                    new_y = self.y_min
                self.network.nodes[(node_id, node_type)]['loc'] = (new_x, new_y)
