import numpy as np
import itertools
import random


class DatasetGenerator:
    def __init__(self, nums, metrics, nodes_cap_rng,
                 services_cap_rng, start_workload,
                 cutoff, seed):
        """dataset generator
        """
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(seed)

        self.num_nodes = nums['nodes']
        self.num_services = nums['services']
        self.num_services_types = nums['services_types']
        self.num_resources = nums['resources']
        self.services_types_map = nums['services_types_map']
        assert self.num_services_types == len(self.services_types_map)
        assert self.num_services == sum(self.services_types_map)

        # label of the used metrics
        self.metrics = metrics
        assert len(self.metrics) == self.num_resources,\
            "number of metrics is not equal to the number of resources"

        # TODO the number of resources should be dynamic
        #      rather than reading their name hardcoded
        assert len(nodes_cap_rng) == self.num_resources
        self.nodes_rng_ram = nodes_cap_rng['ram']
        self.nodes_rng_cpu = nodes_cap_rng['cpu']

        # percentage of used resources in the empty servers
        self.cutoff = cutoff
        assert len(self.cutoff) == self.num_resources,\
            ("number of elements in the cutoff",
             " is not equal to the number of reousrces")
        self.nodes_rng_ram['min'] *= self.cutoff['ram']
        self.nodes_rng_ram['max'] *= self.cutoff['ram']
        self.nodes_rng_cpu['min'] *= self.cutoff['cpu']
        self.nodes_rng_cpu['max'] *= self.cutoff['ram']

        assert len(services_cap_rng) == self.num_resources
        self.services_rng_ram = services_cap_rng['ram']
        self.services_rng_cpu = services_cap_rng['cpu']

        self.start_workload = np.array(start_workload)
        assert self.num_services_types == self.start_workload.shape[0]
        assert self.num_resources == self.start_workload.shape[1]

        self.services_nodes = np.ones(self.num_services, dtype=int) * (-1)

        

        # TEMPS
        nodes_ram_range = np.arange(start=self.nodes_rng_ram['min'],
                                    stop=self.nodes_rng_ram['max']
                                    + self.nodes_rng_ram['step'],
                                    step=self.nodes_rng_ram['step'])
        nodes_ram = np.random.choice(nodes_ram_range,
                                        size=(self.num_nodes, 1))

        nodes_cpu_range = np.arange(start=self.nodes_rng_cpu['min'],
                                    stop=self.nodes_rng_cpu['max']
                                    + self.nodes_rng_cpu['step'],
                                    step=self.nodes_rng_cpu['step'])
        nodes_cpu = np.random.choice(nodes_cpu_range,
                                        size=(self.num_nodes, 1))


    def make_dataset(self):
        """combining state initilizer for consolidation
            and latency to make the real initial configuration

            nodes_resources_cap:

                        ram - cpu
                       |         |
                nodes  |         |
                       |         |

            services_resources_cap:

                             ram - cpu
                            |         |
                services    |         |
                            |         |

            services_nodes:

                contianer_id    contianer_id          contianer_id
                [node_id,       node_id,    , ... ,   node_id     ]

            services_types:

                 service_id      service_id              service_id
                [service_type,   service_type,   ....,   service_type]

            start_workload:

                                   ram_frac - cpu_frac
                                  |                   |
                services_types    |                   |
                                  |                   |
        """
        # 1. generate resource and node-services placements
        self._make_capacities()
        self._make_nodes_services()

        dataset = {
            'nodes_resources_cap': self.nodes_resources_cap,
            'services_resources_cap': self.services_resources_cap,
            'services_nodes': self.services_nodes,
            'services_types': self.services_types,
            'start_workload': self.start_workload
        }
        return dataset

    def _make_capacities(self):
        """make random initialization of nodes and contianers capacities
        """   
        # 1. generage the nodes_resources
        #    doning it in a while statement to make sure
        # sum(contianers_res) < sum(nodes_res)
        tries_limit = 100
        for _ in range(tries_limit):
            # 2. generate nodes
            nodes_ram_range = np.arange(start=self.nodes_rng_ram['min'],
                                        stop=self.nodes_rng_ram['max']
                                        + self.nodes_rng_ram['step'],
                                        step=self.nodes_rng_ram['step'])
            nodes_ram = np.random.choice(nodes_ram_range,
                                         size=(self.num_nodes, 1))

            nodes_cpu_range = np.arange(start=self.nodes_rng_cpu['min'],
                                        stop=self.nodes_rng_cpu['max']
                                        + self.nodes_rng_cpu['step'],
                                        step=self.nodes_rng_cpu['step'])
            nodes_cpu = np.random.choice(nodes_cpu_range,
                                         size=(self.num_nodes, 1))

            self.nodes_resources_cap = np.concatenate((nodes_ram, nodes_cpu),
                                                      axis=1)

            # 3. generate contaienrs
            contaienrs_ram_range = np.arange(
                start=self.services_rng_ram['min'],
                stop=self.services_rng_ram['max']
                + self.services_rng_ram['step'],
                step=self.services_rng_ram['step'])
            services_ram = np.random.choice(contaienrs_ram_range,
                                              size=(self.num_services, 1))
            services_cpu_range = np.arange(
                start=self.services_rng_cpu['min'],
                stop=self.services_rng_cpu['max']
                + self.services_rng_cpu['step'],
                step=self.services_rng_cpu['step'])
            services_cpu = np.random.choice(services_cpu_range,
                                              size=(self.num_services, 1))
            services_types = []
            for index, value in enumerate(self.services_types_map):
                services_types.extend(list(itertools.repeat(index, value)))
            self.services_types = np.array(services_types)
            np.random.shuffle(services_types)
            self.services_resources_cap = np.concatenate((services_ram,
                                                            services_cpu),
                                                           axis=1)
            if np.alltrue(sum(self.services_resources_usage) <=
                          sum(self.nodes_resources_cap)):
                break
        else:  # no-break
            raise RuntimeError((f"tried <{tries_limit}> times to make"
                                "memory allocations , memeory allocation"
                                " is to tight, try eiher smaller range for"
                                " services or larger range for nodes"))

    def _make_nodes_services(self):

        """generate the intitial state for nodes_services with a
            bestfit greedy algorithm:
                1. shuffle the nodes list
                2. pop a node from the nodes lists
                3. find the nodes with least remained resources
                4. try to allocate the services to them
                5. if not possible pop another node
                6. if not possible to allocate then go back to step 1
                and reshuffle

            Remarks:
            if the resources couldn't be allocated with this then
            we say the allocation is not possible and return an
            exception
            FIXME find a better solution with Dynamic Programming
            FIXME works only if: largest service <= smallest node
        """
        # popped_nodes: the node that are out for use according to the
        #               bestfit greeedy algorithm
        tries_limit = 100
        for try_id in range(tries_limit):
            nodes = list(np.arange(self.num_nodes))
            random.shuffle(nodes)
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
                                        self.nodes_resources_remained_frac_avg[popped_nodes],
                                               popped_nodes))]
                    for node in nodes_sorted:
                        if np.alltrue(self.services_resources_usage[service_id] <
                                      self.nodes_resources_remained[node]):
                            self.services_nodes[service_id] = node
                            break
                    else:  # no-break 
                        node_id = nodes.pop()
                        popped_nodes.append(node_id)
                        self.services_nodes[service_id] = node_id
                except IndexError:
                    if try_id < tries_limit-1:
                        break
                    else:
                        raise RuntimeError((f"tried <{tries_limit}> times but "
                                            "couldn't allocate services to"
                                            "node try eiher smaller range for"
                                            " services or larger range for"
                                            "nodes"))

    @property
    def services_resources_usage(self):
        """return the fraction of resource usage for each node
        workload at current timestep e.g. at time step 0:

            resource_contaienrs_type

                            services_types
                ram      |                    |
                cpu      |                    |
        """
        services_workloads = np.array(list(map(
            lambda service_type: self.start_workload[service_type],
            self.services_types)))
        services_resources_usage = services_workloads *\
            self.services_resources_cap
        return services_resources_usage

    @property
    def nodes_resources_usage(self):
        """return the fraction of resource usage of
        each node
        """
        nodes_resources_usage = []
        for node in range(self.num_nodes):
            services_in_node = np.where(
                self.services_nodes == node)[0]
            node_resources_usage = sum(
                self.services_resources_usage[services_in_node])
            if type(node_resources_usage) != np.ndarray:
                node_resources_usage = np.zeros(self.num_resources)
            nodes_resources_usage.append(node_resources_usage)
        return np.array(nodes_resources_usage)

    @property
    def services_resources_remained(self):
        return self.services_resources_cap - self.services_resources_usage

    @property
    def nodes_resources_remained(self):
        return self.nodes_resources_cap - self.nodes_resources_usage

    @property
    def nodes_resources_remained_frac(self):
        return self.nodes_resources_remained / self.nodes_resources_cap

    @property
    def nodes_resources_remained_frac_avg(self):
        return np.average(self.nodes_resources_remained_frac, axis=1)

    @property
    def services_resources_usage_frac(self):
        return self.services_resources_usage / self.services_resources_cap

    @property
    def nodes_resources_usage_frac(self):
        return self.nodes_resources_usage / self.nodes_resources_cap

    @property
    def services_nodes_alloc(self):
        """convert the service allocation from:

            contianer_id    contianer_id          contianer_id
            [node_id,       node_id,    , ... ,   node_id     ]
        to:

                    node_id
            [[...service_id...], [], ..., [], []]
        """
        services_nodes_alloc = [[] for _ in range(self.num_nodes)]
        for service, node in enumerate(self.services_nodes):
            if node != -1:
                services_nodes_alloc[node].append(service)
        return services_nodes_alloc
