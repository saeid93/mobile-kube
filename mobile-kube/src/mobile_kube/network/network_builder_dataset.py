from copy import deepcopy
import networkx as nx
from networkx.algorithms import tree
from operator import itemgetter
import numpy as np
from typing import Tuple

from .network_base import NetworkSimulatorBase
from mobile_kube.util import (
    UserParser,
    StationParser,
    NodeParser,
    override
)

class NetworkBuilderDataset(NetworkSimulatorBase):
    """class to make the network and trace from the dataset
    """
    def __init__(
        self, *, services_nodes: np.array,
        users_services: np.array, num_nodes: np.array,
        num_stations: int, width: int, length: int,
        speed_limit: int, nodes_stations_con: int ,seed: int,
        stations_dataset_path: str, users_dataset_path: str,
        nodes_selection: str, nodes_list: list) -> None:
        """The initialiser for generating the dataset network
        """
        assert num_nodes == num_stations, \
            (f"The number of stations <{num_stations}> is not equal to the",
             f"number of nodes <{num_nodes}>")

        self.node_selection = nodes_selection
        self.node_list = nodes_list
        self.stations_parser = StationParser(stations_dataset_path)
        self.nodes_parser = NodeParser(stations_dataset_path)
        self.users_parser = UserParser(users_dataset_path)

        super().__init__(
            services_nodes=services_nodes,
            users_services=users_services,
            num_nodes=num_nodes,
            num_stations=num_stations,
            width=width, length=length,
            speed_limit=speed_limit,
            nodes_stations_con=nodes_stations_con,
            seed=seed)
        self.raw_network, self.selected_nodes_indexes =\
            self._make_raw_network()
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
        users_dataset_path: str,
        stations_dataset_path: str, seed: int,
        selected_nodes: np.array):
        """
            The initialiser for generating the trace
        """
        ins = cls(services_nodes=services_nodes,
                  users_services=users_services,
                  num_nodes=num_nodes,
                  num_stations=num_stations, width=width,
                  length=length, speed_limit=speed_limit,
                  stations_dataset_path=stations_dataset_path,
                  users_dataset_path=users_dataset_path,
                  nodes_stations_con=nodes_stations_con,
                  seed=seed,
                  nodes_selection='node_list',
                  nodes_list=list(selected_nodes))
                #   TODO check this
        ins.network = network
        ins.raw_network = raw_network
        ins.users_parser = UserParser(users_dataset_path)
        return ins

    @override(NetworkSimulatorBase)
    def _make_raw_network(self) -> Tuple[nx.Graph, np.array]:
        """make the raw network:
           the network only with
           nodes and stations not the users
           """

        # create a graph
        network = nx.Graph()

        # read nodes and stations from the dataset
        nodes = self.nodes_parser.serialize()
        stations = self.stations_parser.serialize()
        selected_nodes_indexes=np.array([])
        assert len(nodes) == len(stations),\
            "number of nodes and stations must be equal"
        if self.node_selection == 'random':
            selected_nodes_indexes = np.random.randint(
                len(nodes), size=self.num_nodes)
        elif self.node_selection == 'node_list':
            assert len(self.node_list) == self.num_nodes,\
                "length of nodes list and the number of nodes must be equal"
            selected_nodes_indexes = self.node_list
        elif self.node_selection == 'ordered':
            selected_nodes_indexes = np.arange(self.num_nodes)

        # add selected nodes to the network
        selected_nodes = np.array(nodes)[selected_nodes_indexes]            
        selected_stations = np.array(nodes)[selected_nodes_indexes]
        for node_id, node in enumerate(selected_nodes):
            network.add_node(
                (node_id, 'node'), loc=(node[0], node[1]))
        for station_id, station in enumerate(selected_stations):
            network.add_node(
                (station_id, 'station'), loc=(station[0], station[1]))
        # adding node-node edges to the network with spanning tree
        nodes_subgraph = nx.complete_graph(
            network.subgraph(self.nodes_idx).copy())
        for edge in nodes_subgraph.edges:
            weight = self._euclidean_dis(network.nodes[edge[0]]['loc'],
                                         network.nodes[edge[1]]['loc'])
            nodes_subgraph.edges[edge]['weight'] = weight

        mst = tree.minimum_spanning_edges(nodes_subgraph, algorithm='kruskal',
                                          weight='weight', data=True)

        # add mst edges to the network
        for edge in mst:
            network.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

        # add station-server edges by connecting stations to their closest servers
        for station in self.stations_idx:
            dis_nodes = {}
            for node in self.nodes_idx:
                dis = self._euclidean_dis(network.nodes[station]['loc'],
                                          network.nodes[node]['loc'])
                dis_nodes[node] = dis
            res = dict(sorted(dis_nodes.items(),
                              key=itemgetter(1))[:self.nodes_stations_con])

            # add station-server edges
            for key, value in res.items():
                network.add_edge(station, key, weight=value)

        return network, selected_nodes_indexes

    @override(NetworkSimulatorBase)
    def _make_users(self, network) -> nx.Graph:
        network.add_nodes_from(self.users_idx)
        self.users_parser.read_line()
        for user in self.users_idx:
            network.nodes[user]['loc'] = (self.users_parser.get_user(user[0]))
        return network

    @override(NetworkSimulatorBase)
    def _users_move_random(self) -> None:
        """
        users being moved from dataset not random
           fix the naming, the entire structure will be polished
        """
        self.users_parser.read_line()

        for node_id, node_type in self.network.nodes:
            if node_type == 'user':
                self.network.nodes[(node_id, node_type)]['loc'] =\
                    self.users_parser.get_user(node_id)
