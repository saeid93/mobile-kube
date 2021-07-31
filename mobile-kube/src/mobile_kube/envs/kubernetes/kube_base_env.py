"""base class of all kubernetes enviornments
"""
import abc
import numpy as np
from copy import deepcopy
from typing import (
    List,
    Dict,
    Any
)

import gym
from gym.utils import seeding
import types

from mobile_kube.util import (
    Preprocessor,
    override,
    rounding,
    check_config,
    load_object,
    ACTION_MIN,
    ACTION_MAX
)
from mobile_kube.envs_extensions import (
    get_render_method,
    get_action_method_kube,
    get_step_method,
    get_reward_method
)
from mobile_kube.util.kubernetes_utils import (
    Cluster,
    Node,
    Service,
    ResourceUsage
)
from mobile_kube.util.kubernetes_utils import (
    generate_random_service_name,
    construct_pod,
    mapper, construct_svc
)
from mobile_kube.util import logger


class KubeBaseEnv(gym.Env):
    """
    what differs between different enviornments
    1. States
    2. Actions and related functions
    3. Reward

    common variables:

        services_nodes:

             service_id     service_id            service_id
            [node_id,       node_id,    , ... ,   node_id     ]

            range:
                indices: [0, num_services)
                contents: [0, num_nodes+1) (+1 for one auxiliary node)

        config['workload_start']:
            the start timestep of the entire dataset that we want to use

        config['workload_stop']:
            the final timestep of the entire dataset that we want to use

    Remarks:
            the main indicator of the state (obseravation) if services_nodes
            the rest of the observation dictionary is updated with decorators
            automatically

    Variables:
            self.mitigation_needed: indicator of whether the mitigation of
                                    overloading was successful
            self.mitigation_tries: indicator of how many binpacking mitigations
                                should be tried
            self.auxiliary_node_needed: indicator of whether the
                                        _binpacking_mitigator was successful or
                                        not and if we need to use the
                                        auxiliary node
    """

    # ------------------ common functions ------------------

    def __init__(self, config: Dict[str, Any]):
        
        # action min and max
        self.action_min, self.action_max = (
            ACTION_MIN, ACTION_MAX
        )

        # initialize seed to ensure reproducible resutls
        self.seed: int = config['seed']
        # np.random.seed(self.seed)
        self.np_random = seeding.np_random(self.seed)

        check_config(config)

        # observation elements
        self.obs_elements: List[str] = config['obs_elements']

        # path of the dataset and workload
        self.dataset_path = config['dataset_path']
        self.workload_path = config['workload_path']

        # node, services resources and workload
        self.dataset = load_object(self.dataset_path)
        self.workload = load_object(self.workload_path)

        self.nodes_resources_cap: np.array = self.dataset['nodes_resources_cap']
        self.services_resources_cap: np.array = self.dataset[
            'services_resources_cap']
        self.services_types: np.array = self.dataset['services_types']

        # find the number of nodes, services, service types and timesteps
        self.num_resources: int = self.nodes_resources_cap.shape[1]
        self.num_nodes: int = self.nodes_resources_cap.shape[0]
        self.num_services: int = self.services_resources_cap.shape[0]
        self.num_services_types: int = self.workload.shape[2]
        self.total_timesteps: int = self.workload.shape[1]

        # start and stop timestep
        stop_timestep: int = int(
            config['workload_stop']*self.total_timesteps)
        self.workload = self.workload[:, 0:stop_timestep, :]

        # initial states
        self.initial_services_nodes: np.array = self.dataset['services_nodes']

        # reward penalties
        self.penalty_illegal: float = config['penalty_illegal']
        self.penalty_move: float = config['penalty_move']
        self.penalty_variance: float = config['penalty_variance']
        self.penalty_consolidated: float = config['penalty_consolidated']
        self.penalty_latency: float = config['penalty_latency']

        # binpacking mitigator variables
        # !!! not all of them are used in all envs
        self.mitigation_tries: int = config['mitigation_tries']
        self.mitigation_needed: bool = False
        self.auxiliary_node_needed: bool = False

        # render flags
        # !!! not all of them are used in all envs
        self.greedy_mitigation_needed_for_render: bool = False
        self.auxiliary_node_mitigation_needed_for_render: bool = False

        # episode length
        self.episode_length: int = config['episode_length']

        # whether to reset timestep and placement at every episode
        self.timestep_reset: bool = config['timestep_reset']
        self.placement_reset: bool = config['placement_reset']
        self.global_timestep: int = 0
        self.timestep: int = 0
        # self.services_nodes = deepcopy(self.initial_services_nodes)

        # what type of environemnt is used
        self.reward_mode = config['reward_mode']

        # whether to compute the greedy_num_consolidated at each
        # step or not TODO 
        self.compute_greedy_num_consolidated = config[
            'compute_greedy_num_consolidated']

        # set the take_action method
        _take_action, _validate_action = get_action_method_kube(config['action_method'])
        self._take_action = types.MethodType(_take_action, self)
        self._validate_action = types.MethodType(_validate_action, self)

        # set the step method
        step = get_step_method(config['step_method'])
        self.step = types.MethodType(step, self)

        # set the render method
        render = get_render_method(config['step_method'])
        self.render = types.MethodType(render, self)

        # set the reward method
        _reward = get_reward_method(config['reward_mode'])
        self._reward = types.MethodType(_reward, self)

        # TODO add it if used
        # add one auxiliary node with unlimited and autoscaled capacity
        # TODO make it consistent with Alireza
        # self.total_num_nodes: int = self.num_nodes + 1

        # get the kubernetes information
        self.kube_config_file: str = config['kube']['admin_config']
        self.service_image: str = config['kube']['service_image']
        self.utilization_image: str = config['kube']['utilization_image']
        self.namespace: str = config['kube']['namespace']
        self.clean_after_exit: bool = config['kube']['clean_after_exit']
        self.using_auxiliary_server: bool = config['kube']['using_auxiliary_server']

        # construct the kubernetes cluster
        self._cluster = Cluster(
            self.kube_config_file,
            self.namespace,
            self.workload_path,
            self.dataset_path,
            self.utilization_image,
            self.clean_after_exit,
            self.using_auxiliary_server,
        )

        # get Nodes (an array of Nodes)
        self.nodes, self.aux_server = self._cluster.monitor.get_nodes()

        # check if the number of nodes in the cluster equals to
        # the number of nodes in the dataset
        self.nodes = np.array([
            Node(id, node) for id, node in enumerate(self.nodes)
        ])

        # aux server added if exists
        if self.aux_server is not None:
            self.aux_server = Node(len(self.nodes) + 1, self.aux_server, is_auxiliary=True)

        assert len(self.nodes) == self.num_nodes, \
            (f"number of nodes in the cluster <{len(self.nodes)}>"
             " is not consistent with"
             f" the number of nodes in the dataset <{self.num_nodes}>")

        self.services_nodes_obj: np.array = np.array([
            self.nodes[id] for id in self.dataset['services_nodes']
        ])

    @override(gym.Env)
    def reset(self) -> np.ndarray:
        """Resets the state of the environment and returns
        an initial observation.
        Returns:
            (object): the initial observation.
        Remarks:
            each time resets to a different initial state
        """
        if self.timestep_reset:
            self.global_timestep = 0
            self.timestep = 0
        if self.placement_reset:
            self.services_nodes = deepcopy(self.initial_services_nodes)
        self._initialisation()
        return self.observation

    def preprocessor(self, obs):
        """
        environment preprocessor
        depeiding on the observation (state) definition
        """
        prep = Preprocessor(self.nodes_resources_cap,
                            self.services_resources_cap)        
        obs = prep.transform(obs)
        return obs

    # ------------------ common properties ------------------

    @property
    def services_nodes(self):
        return np.array(list(map(lambda node: node.id, self.services_nodes_obj)))

    @property
    @rounding
    def services_resources_usage_frac(self) -> np.ndarray:
        """fraction of usage:

                         ram - cpu
                        |         |
            services    |         |
                        |         |

            range:
                row inidices: (0, num_services]
                columns indices: (0, num_resources]
                enteries: [0, 1] type: float
        """
        return self.services_resources_usage / self.services_resources_cap

    @property
    @rounding
    def services_resources_usage(self) -> np.ndarray:
        """return the fraction of resource usage for each node
        workload at current timestep e.g. at time step 0.
                         ram - cpu
                        |         |
            services    |         |
                        |         |
            range:
                row inidices: (0, num_services]
                columns indices: (0, num_resources]
                enteries: [0, node_resource_cap] type: float
        """
        self.service_metrics = self._load_services_metrics()
        lst = []
        for service in self.services:
            used_memory = ResourceUsage(self.service_metrics.get(service.metadata_name)).memory / 1000
            used_cpu = ResourceUsage(self.service_metrics.get(service.metadata_name)).cpu / 1000000000
            lst.append([used_memory, used_cpu])
        services_resources_usage = np.array(lst)
        return services_resources_usage

    @property
    @rounding
    def nodes_resources_usage(self):
        """returns the resource usage of
        each node
                     ram - cpu
                    |         |
            nodes   |         |
                    |         |

            range:
                row inidices: (0, num_nodes]
                columns indices: (0, num_resources]
                enteries: [0, node_resource_cap] type: float
        """
        node_metrics = self._cluster.monitor.get_nodes_metrics()
        lst = []
        for node in self.nodes:
            used_memory = ResourceUsage(node_metrics.get(node.name)).memory / 1000
            used_cpu = ResourceUsage(node_metrics.get(node.name)).cpu / 1000000000
            lst.append([used_memory, used_cpu])
        nodes_resources_usage = np.array(lst)
        return np.array(nodes_resources_usage)

    @property
    def services_resources_remained(self) -> np.ndarray:
        return self.services_resources_cap - self.services_resources_usage

    @property
    def nodes_resources_remained(self) -> np.ndarray:
        return self.nodes_resources_cap - self.nodes_resources_usage

    @property
    @rounding
    def services_types_usage(self) -> np.ndarray:
        """each service type resource usage

                                 ram  cpu
                                |        |
            services_types      |        |
                                |        |
        """
        services_types_usage = np.transpose(self.workload[
            :, self.timestep, :])
        return services_types_usage

    @property
    @rounding
    def nodes_resources_usage_frac(self) -> np.ndarray:
        """nodes_resources_usage_frac (fraction of usage):
                     ram - cpu
                    |         |
            nodes   |         |
                    |         |

            range:
                row inidices: (0, num_nodes]
                columns indices: (0, num_resources]
                enteries: [0, 1] type: float
        """
        return self.nodes_resources_usage / self.nodes_resources_cap

    @property
    def num_consolidated(self) -> int:
        """returns the number of consolidated nodes
        """
        return self._num_consolidated(self.services_nodes)

    @property
    def num_overloaded(self) -> int:
        """return the number of resource exceeding nodes
        """
        overloaded_nodes = np.unique(np.where(
            self.nodes_resources_usage_frac > 1)[0])
        return len(overloaded_nodes)

    @property
    def nodes_services(self) -> np.ndarray:
        """change the representation of placements from:
             contianer_id   contianer_id          contianer_id
            [node_id,       node_id,    , ... ,   node_id     ]
        to:
                       node_id                    node_id
            [[service_id, service_id], ...,[service_id]]
        """
        nodes_services = []
        for node in range(self.num_nodes):
            services_in_node = np.where(self.services_nodes ==
                                          node)[0].tolist()
            nodes_services.append(services_in_node)
        return nodes_services

    @property
    def services_in_auxiliary(self) -> np.ndarray:
        """services in the auxiliary node
        """
        services_in_auxiliary = np.where(self.services_nodes ==
                                           self.num_nodes)[0]
        return services_in_auxiliary

    @property
    def nodes_resources_remained_frac(self) -> np.ndarray:
        return self.nodes_resources_remained / self.nodes_resources_cap

    @property
    def nodes_resources_remained_frac_avg(self):
        return np.average(self.nodes_resources_remained_frac, axis=1)

    @property
    def num_in_auxiliary(self) -> int:
        """num of services in the auxiliary node
        """
        return len(self.services_in_auxiliary)

    @property
    @rounding
    def auxiliary_resources_usage(self) -> np.ndarray:
        """resource usage in auxiliary node
        """
        services_in_auxiliary = np.where(self.services_nodes ==
                                           self.num_nodes)[0]
        auxiliary_resources_usage = sum(self.services_resources_usage[
            services_in_auxiliary])
        if type(auxiliary_resources_usage) != np.ndarray:
            auxiliary_resources_usage = np.zeros(self.num_resources)
        return auxiliary_resources_usage

    @property
    def done(self):
        """check at every step that if we have reached the
        final state of the simulation of not
        """
        done = True if self.timestep % self.episode_length == 0 else False
        return done

    @property
    def complete_raw_observation(self) -> Dict[str, np.ndarray]:
        observation = {
                "services_resources_usage": self.services_resources_usage,
                "nodes_resources_usage": self.nodes_resources_usage,
                "services_resources_usage_frac":
                self.services_resources_usage_frac,
                "nodes_resources_usage_frac": self.nodes_resources_usage_frac,
                "services_nodes": self.services_nodes,
                "auxiliary_resources_usage": self.auxiliary_resources_usage
        }
        return observation

    @property
    def raw_observation(self) -> Dict[str, np.ndarray]:
        """returns only the raw observations requested by the user
        in the config input through obs_elements
        """
        observation = {
                "services_resources_usage": self.services_resources_usage,
                "nodes_resources_usage": self.nodes_resources_usage,
                "services_resources_usage_frac":
                self.services_resources_usage_frac,
                "nodes_resources_usage_frac": self.nodes_resources_usage_frac,
                "services_nodes": self.services_nodes,
                "auxiliary_resources_usage": self.auxiliary_resources_usage
        }
        selected = dict(zip(self.obs_elements,
                            [observation[k] for k in self.obs_elements]))
        return selected

    @property
    def observation(self) -> np.ndarray:
        """preprocessed observation of each environment
        """
        obs = self.preprocessor(self.raw_observation)
        return obs

    def _num_consolidated(self, services_nodes) -> int:
        """functional version of num_services
        returns the number of consolidated nodes
        """
        a = set(services_nodes)
        b = set(np.arange(self.num_nodes))
        intersect = b - a
        return len(intersect)

    # ------------------ abstract functions and properties ------------------

    @abc.abstractmethod
    def _setup_space(self, action_space_type: str):
        """setting up the spaces
        """
        raise NotImplementedError("_setup_space not implemented")


    # --------------------- kubernetes operations ----------------------------

    def _initialisation(self):
        """Initialisation step

        services will be created from this list. if n_services
        equals to number of images, one service will be created from
        each image, otherwise, select an image randomly for creating
        a service (default: ['r0ot/stress:memory'])

        NOTE: This function will be called, when call the `reset` function
        """

        logger.info('Initialise services on the cluster')

        # create a list of services (an array of Services)
        self.services = list()
        for service_id, node in enumerate(self.services_nodes_obj):
            # The 
            name = generate_random_service_name(service_id=service_id, node_id=node.id)

            pod = construct_pod(
                name=name,
                image=self.service_image,
                node_name=node.name,
                namespace=self._cluster.namespace,
                limit_mem="{}Mi".format(self.services_resources_cap[service_id][0]),
                limit_cpu="{}".format(self.services_resources_cap[service_id][1]),
            )

            svc = construct_svc(
                name=name,
                namespace=self._cluster.namespace,
                portName='web',
                port=80,
                targetPort=80,
                portProtocol='TCP',
            )

            service = Service(id=service_id + 1, pod=pod, svc=svc)
            self.services.append(service)

        self.services: np.array = np.array(self.services)

        logger.info('Create "{}" pods on nodes'.format(
            self.num_services
        ))

        # run pods on the cluster
        self._cluster.action.create_pods(mapper(
            lambda service: service.pod, self.services
        ))

        # run svcs on the cluster
        self._cluster.action.create_services(mapper(
            lambda service: service.svc, self.services
        ))

    def _load_services_metrics(self):
        """Load Services Metrics

        :return dict
        """
        # we should ensure about existance of all service metrics
        service_metrics = dict()
        logger.info('Waiting for metrics ...')
        while len(service_metrics.keys()) != self.num_services + 1:
            service_metrics = self._cluster.monitor.get_pods_metrics()
        return service_metrics

    def _migrate(self, observations):
        # we can get the information of pods from self.observation
        services = list()
        for service, node in zip(self.services, observations):
            pod, svc = self._cluster.action.move_pod(
                previousPod=service.pod,
                previousService=service.svc,
                to_node_name=node.name,
                to_node_id=node.id
            )

            if pod is None:
                raise ValueError('pod should not be None')

            if svc is None:
                raise ValueError('svc should not be None')

            # create a service for new Pod
            service = Service(service.id, pod, svc)

            # append to the list
            services.append(service)
        self.services = np.array(services)

    # def render(self):
    #     """Show current state of cluster"""

    #     if hasattr(self, 'service_metrics'):
    #         self.service_metrics = self._load_services_metrics()
    #         print('\n----[Services Memory and CPU]----\n')
    #         for service in self.services:
    #             used_memory = ResourceUsage(self.service_metrics.get(service.metadata_name)).memory / 1000
    #             used_cpu = ResourceUsage(self.service_metrics.get(service.metadata_name)).cpu / 1000
    #             print('Memory')
    #             print("Service({})           ->           {} Mi".format(
    #                 service.id,
    #                 used_memory
    #             ))
    #             print('CPU:')
    #             print("Service({})           ->           {} mCPU".format(
    #                 service.id,
    #                 used_cpu,
    #             ))

    #     print('\n----[Servers Memory and CPU Capacity]----\n')
    #     node_metrics = self._cluster.monitor.get_nodes_metrics()
    #     for node in self.nodes:
    #         print('Memory:')
    #         used_memory = ResourceUsage(node_metrics.get(node.name)).memory / 1000
    #         total_memory = node.memory / 1000
    #         print("Node({})           ->           {} Mi / {} Mi ({:.2f}%)".format(
    #             node.id,
    #             used_memory,
    #             total_memory,
    #             (used_memory / total_memory) * 100
    #         ))
    #         print('CPU:')
    #         used_cpu = ResourceUsage(node_metrics.get(node.name)).cpu / 1000000
    #         total_cpu = node.cpu * 1000
    #         print("Node({})           ->           {} mCPU / {} mCPU ({:.2f}%)".format(
    #             node.id,
    #             used_cpu,
    #             total_cpu,
    #             (used_cpu / total_cpu) * 100
    #         ))

    #     print('\n----[Service Placements]----\n')
    #     for service, node in zip(self.services, self.services_nodes_obj):
    #         print("Service({})           ->           Node({})".format(
    #             service.id,
    #             node.id
    #         ))

    # def _is_illegal_state_done(self, observations):
    #     """Check if the current memory allocation is feasible

    #     :param observations: np.array
    #         an array of observations

    #     :return: bool
    #     """

    #     logger.info('Calculating resources by new observations ...')

    #     # get memory busy of each node
    #     node_mem_busy = np.zeros(self.num_nodes)
    #     for i, node in enumerate(self.nodes):
    #         node_mem_busy[i] = np.sum(mapper(
    #             lambda service: ResourceUsage(
    #                 self.service_metrics.get(service.metadata_name)
    #             ).memory,
    #             self.services[observations == node]
    #         ))

    #     # get memory capacity of each node
    #     node_capacities = np.array(mapper(
    #         lambda node: node.memory,
    #         self.nodes
    #     ))

    #     return np.all(node_capacities < node_mem_busy)
