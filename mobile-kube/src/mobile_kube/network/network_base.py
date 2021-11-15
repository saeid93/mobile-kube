import networkx as nx
import copy
from networkx.algorithms import tree
from operator import itemgetter
from networkx.algorithms.shortest_paths.generic import shortest_path 
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import abc
from typing import Union

class NetworkSimulatorBase:
    def __init__(self, *, services_nodes: np.array,
                 users_services: np.array, num_nodes: np.array,
                 num_stations: int, width: int, length: int,
                 speed_limit: int, nodes_stations_con: int,
                 seed: int):
        """
            The base initialiser only used in 
        """
        self.SPEED_LIMIT = speed_limit
        self.width = width
        self.length = length

        # number of connection between station and nodes
        self.nodes_stations_con = nodes_stations_con

        random.seed(seed)
        np.random.seed(seed)

        self.services_nodes = services_nodes
        self.users_services = users_services
        self.num_services = len(services_nodes)
        self.num_users = len(users_services)
        self.num_nodes = num_nodes
        self.num_stations = num_stations
        self.nodes_idx = list(map(lambda i: (i,'node'),
                                    range(self.num_nodes)))
        self.stations_idx = list(map(lambda i: (i,'station'),
                                     range(self.num_stations)))
        self.users_idx = list(map(lambda i: (i,'user'),
                                  range(self.num_users)))

        # to be instantiated in one of the classmethods
        self.users_stations = np.zeros(self.num_users, dtype=int)
        self.users_distances = np.zeros(self.num_users)

    # ------------ network making options ------------
    @abc.abstractmethod
    def _make_raw_network(self):
        """make the raw network:
           the network only with
           nodes and stations not the users
           """
        raise NotImplementedError

    def _make_complete_network(self):
        """
            do the initialization of the
            users and their distances in
            the network
        """
        self.network = self._make_users(copy.deepcopy(self.network))
        self._update_users_connections()

    def _update_users_connections(self):
        self._update_users_stations()
        self._update_users_distances() 

    @abc.abstractmethod
    def _make_users(self, network):
        """
           place users randomly or from dataset
           at locations in the map
        """
        raise NotImplementedError

    # ----------- updating and sampling funcitons -----------

    def update_users_services(self, users_services):
        """
            apply changes to the services_nodes
            to the simulator
        """
        self.users_services = users_services
        self._update_users_distances()

    def update_services_nodes(self, services_nodes):
        """
            apply changes to the services_nodes
            to the simulator
        """
        self.services_nodes = services_nodes
        self._update_users_distances()

    def _update_users_distances(self):
        """
            find the distances of the users to their connected
            service/node
        """
        for user in self.users_idx:
            service_idx = self.users_services[user[0]]
            node_idx = self.services_nodes[service_idx]
            station_idx = self.users_stations[user[0]]
            path_length = nx.shortest_path_length(
                self.network,
                source=(station_idx, 'station'),
                target=(node_idx, 'node'),
                weight='weight',
                method='dijkstra')
            self.users_distances[user[0]] = path_length

    def _update_users_stations(self):
        """
            find the nearset station to each user
        """

        for user in self.users_idx:
            dis_station = {}
            for station in self.stations_idx:
                dis = self._euclidean_dis(self.network.nodes[user]['loc'],
                                          self.network.nodes[station]['loc'])
                dis_station[station] = dis

            res = min(dis_station.items(), key=itemgetter(1))[0][0]
            self.users_stations[user[0]] = res

    def sample_users_stations(self, from_trace: bool =True,
                              timestep: int=None):
        """
            sample the next users_station array
            for the simulation
            and run the three innter functions in order
            and return the new users' stations:
            1. users_move
            2. update_users_stations
            3. update_users_distances
        """
        self._users_move_dataset(from_trace, timestep)
        self._update_users_connections()
        return self.users_stations

    def _users_move_dataset(self, from_trace: bool =True,
                            timestep: int=None):
        """
            perform one step of user's movements
        """
        if from_trace:
            self._users_move_from_trace(timestep)
        else:
            self._users_move_random()

    @abc.abstractmethod
    def _users_move_random(self):
        """
            Moves users with `User.SPEED_LIMIT` in 2d surface
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _users_move_from_trace(self, tiemstep: int=None):
        """
            Moves users with `User.SPEED_LIMIT` in 2d surface
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _change_users_loactions(self, users_next_location):
        """
            function to change the users locations from the trace
        """
        raise NotImplementedError

    def get_largest_station_node_path(self):
        max_station_node = 0
        min_station_node = 10000
        for start, dests in nx.floyd_warshall(self.network).items():
            if start[1] == 'station':
                for dest, path_length in dests.items():
                    if dest[1] == 'node' and path_length != np.inf\
                        and path_length > max_station_node:
                        max_station_node = path_length
                    # choose the second smallest as the min station_node
                    if dest[1] == 'node' and path_length != 0\
                        and path_length < min_station_node:
                        min_station_node = path_length
        return min_station_node, max_station_node

    # def get_largets_station_node(self):
    #     """
    #         get the largest station-node distance in the entire network
    #     """
    #     edges = nx.get_edge_attributes(self.network,'weight')
    #     nodes_stations_edges = dict(filter(lambda edge: edge[0][0][1] !=\
    #         edge[0][1][1], edges.items()))
    #     return max(nodes_stations_edges.values())

    def _euclidean_dis(self, loc1, loc2):
        dis = math.sqrt(sum([(a - b) ** 2 for a, b in zip(loc1, loc2)]))
        return dis

    @property
    def all_nodes_location(self):
        return nx.get_node_attributes(self.network, 'loc')

    @property
    def users_location(self):
        return dict((key[0], self.all_nodes_location[key])\
            for key in self.users_idx)

    @property
    def stations_location(self):
        return dict((key[0], self.all_nodes_location[key])\
            for key in self.stations_idx)

    @property
    def nodes_location(self):
        return dict((key[0], self.all_nodes_location[key])\
            for key in self.nodes_idx)

    def visualize_debug(self, raw=False):
        """ 
            simple network
            visualizer
            for debugging
            raw: if True network without users
        """

        f = plt.figure(figsize = [12.9, 9.6])
        if raw:
            all_pos = nx.get_node_attributes(self.raw_network,'loc')
            labels = {k: k[0] for k in self.raw_network.nodes}
            nx.draw_networkx(self.raw_network,
                            with_labels=True, labels=labels,
                            font_size = 9,
                            nodelist=self.nodes_idx,
                            node_size=100, node_shape='s',
                            pos=all_pos,
                            node_color='#eba134')
            nx.draw_networkx(self.raw_network,
                            with_labels=True, labels=labels,
                            font_size = 9,
                            nodelist=self.stations_idx,
                            node_size=100, node_shape='^',
                            pos=all_pos,
                            node_color='#e8eb34')  
            # edge_labels = {k: self.raw_network.edges[k]['weight']\
            #     for k in self.raw_network.edges}
            # nx.draw_networkx_edge_labels(
            #     self.raw_network,
            #     pos=nx.get_node_attributes(self.raw_network,'loc'),
            #     edge_labels=edge_labels, font_size=6)          
        else:
            all_pos = nx.get_node_attributes(self.network,'loc')
            labels = {k: k[0] for k in self.network.nodes}
            nx.draw_networkx(self.network,
                            with_labels=True, labels=labels,
                            font_size = 9,
                            nodelist=self.nodes_idx,
                            node_size=100, node_shape='s',
                            pos=all_pos,
                            node_color='#eba134')
            nx.draw_networkx(self.network,
                            with_labels=True, labels=labels,
                            font_size = 9,
                            nodelist=self.stations_idx,
                            node_size=100, node_shape='^',
                            pos=all_pos,
                            node_color='#e8eb34')
            nx.draw_networkx(self.network,
                            with_labels=True, labels=labels,
                            font_size = 9,
                            nodelist=self.users_idx,
                            node_size=100, node_shape='o',
                            pos=all_pos,
                            node_color='#34ebdf')
            # edge_labels = {k: self.network.edges[k]['weight']\
            #     for k in self.network.edges}
            # nx.draw_networkx_edge_labels(
            #     self.network,
            #     pos=nx.get_node_attributes(self.network,'loc'),
            #     edge_labels=edge_labels, font_size=9)
        return f

    def visualize_paper_style(self):
        """ 
            simple network
            visualizer
            like the style of the paper
            raw: if True network without users
        """
        f = plt.figure(figsize = [12.9, 9.6], frameon=False)
        ax = f.add_axes([0, 0, 1, 1])
        ax.axis('off')
        all_pos = nx.get_node_attributes(self.network,'loc')
        labels = {k: k[0] for k in self.network.nodes}
        nx.draw_networkx_nodes(self.network,
                        with_labels=True, labels=labels,
                        font_size = 9,
                        nodelist=self.nodes_idx,
                        node_size=300, node_shape='s',
                        pos=all_pos,
                        node_color='red', label='node')
        nx.draw_networkx_nodes(self.network,
                        with_labels=True, labels=labels,
                        font_size = 9,
                        nodelist=self.stations_idx,
                        node_size=300, node_shape='^',
                        pos=all_pos,
                        node_color='green', label='station')
        nx.draw_networkx_nodes(self.network,
                        with_labels=True, labels=labels,
                        font_size = 9,
                        nodelist=self.users_idx,
                        node_size=300, node_shape='o',
                        pos=all_pos,
                        node_color='blue', label='user')
        plt.legend(scatterpoints = 1, fontsize=20)
        edge_labels = {k: self.network.edges[k]['weight']\
            for k in self.network.edges}
        nx.draw_networkx_edges(self.network,
                               pos=nx.get_node_attributes(self.network,'loc'),
                               edge_labels=edge_labels, font_size=9)

        return f

    # ----------- greedy action -----------
    def _next_greedy_action(self, nodes_mem_cap, services_mem_request,
                            nodes_cpu_cap, services_cpu_request):
        temp_services_nodes = copy.deepcopy(self.services_nodes)
        def node_station_dis(node_id, station_id):
            dis = nx.shortest_path_length(self.network,
                                          source=(station_id, 'station'),
                                          target=(node_id, 'node'),
                                          weight='weight',
                                          method='dijkstra')
            return dis
        # iterate through each service one by one to find a new placement for them
        for service in range(0, self.num_services):
            service_connected_users = np.argwhere(self.users_services==service).flatten()
            service_connected_users_stations = self.users_stations[service_connected_users]
            service_average_nodes_penalties = []
            for node in range(0, self.num_nodes):
                station_nodes_penalty = np.array([node_station_dis(node, station_id)
                                                    for station_id in service_connected_users_stations])
                service_average_nodes_penalty = np.average(station_nodes_penalty)
                service_average_nodes_penalties.append(service_average_nodes_penalty)
            # move it to the first server with lowest latency and available memory
            sorted_nodes_indices = np.argsort(service_average_nodes_penalties)
            for node in sorted_nodes_indices:
                services_in_node = np.argwhere(temp_services_nodes==node).flatten()
                used_mem = sum(services_mem_request[services_in_node])
                used_cpu = sum(services_cpu_request[services_in_node])
                if used_mem + services_mem_request[service] <= nodes_mem_cap[node] and \
                    used_cpu + services_cpu_request[service] <= nodes_cpu_cap[node]:
                    temp_services_nodes[service] = node
                    break
        return temp_services_nodes
