"""base class of all simulation enviornments
"""
import abc
import numpy as np
from copy import deepcopy
from typing import (
    List,
    Dict,
    Any,
    Literal
)

from colorama import (
    Fore,
    Style
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
    ACTION_MAX,
    plot_resource_allocation
)
from mobile_kube.envs_extensions import (
    reward
)


class SimBaseEnv(gym.Env):
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
                contents: [0, num_nodes)

        config['workload_start']:
            the start timestep of the entire dataset that we want to use

        config['workload_stop']:
            the final timestep of the entire dataset that we want to use

    Remarks:
            the main indicator of the state (obseravation) if services_nodes
            the rest of the observation dictionary is updated with decorators
            automatically
    """

    # ------------------ common functions ------------------

    def __init__(self, config: Dict[str, Any]):
        # action min and max
        self.action_min, self.action_max = (
            ACTION_MIN, ACTION_MAX
        )

        # initialize seed to ensure reproducible resutls
        self.seed(config['seed'])

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
        self.services_resources_request: np.array = self.dataset[
            'services_resources_request']
        self.services_types: np.array = self.dataset['services_types']

        # find the number of nodes, services, service types and timesteps
        self.num_resources: int = self.nodes_resources_cap.shape[1]
        self.num_nodes: int = self.nodes_resources_cap.shape[0]
        self.num_services: int = self.services_resources_request.shape[0]
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

        # episode length
        self.episode_length: int = config['episode_length']

        # whether to reset timestep and placement at every episode
        self.timestep_reset: bool = config['timestep_reset']
        self.placement_reset: bool = config['placement_reset']
        self.global_timestep: int = 0
        self.timestep: int = 0
        self.services_nodes = deepcopy(self.initial_services_nodes)

        # set the reward method
        self._reward = types.MethodType(reward, self)

        # whether to take the overloaded action with negative reward or not
        self.no_action_on_overloaded = config['no_action_on_overloaded']

    def seed(self, seed):
        np.random.seed(seed)
        self.np_random = seeding.np_random(seed)

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
        return self.observation

    def render(self, mode: Literal['human', 'ansi'] ='human') -> None:
        """
        """
        print("--------state--------")
        if not self.num_overloaded:
            print("nodes_resources_request_frac:")
            print(self.nodes_resources_request_frac)
            print("services_nodes:")
            print(self.services_nodes)
            if mode == 'ansi':
                plot_resource_allocation(self.services_nodes,
                                        self.nodes_resources_cap,
                                        self.services_resources_request,
                                        self.services_resources_usage,
                                        plot_length=80)
        else:
            print(Fore.RED, "agent's action lead to an overloaded state!")
            print("nodes_resources_usage_frac:")
            print(self.nodes_resources_request_frac)
            print("services_nodes:")
            print(self.services_nodes)
            if mode == 'ansi':
                plot_resource_allocation(self.services_nodes,
                                        self.nodes_resources_cap,
                                        self.services_resources_request,
                                        self.services_resources_usage,
                                        plot_length=80)
            print(Style.RESET_ALL)

    # ------------------ common properties ------------------

    @property
    @rounding
    def services_resources_usage(self) -> np.ndarray:
        """return the fraction of resource usage for each node
        workload at current timestep e.g. at time step 0.
                         ram cpu
                        |       |
            services    |       |
                        |       |
            range:
                row inidices: (0, num_services]
                columns indices: (0, num_resources]
                enteries: [0, node_resource_cap] type: float
        """
        services_resources_usage = (self.services_resources_usage_frac *
                                      self.services_resources_request)
        return services_resources_usage

    @property
    def nodes_resources_usage(self):
        """return the amount of resource usage
        on each node
                     ram - cpu
                    |         |
            nodes   |         |
                    |         |

            range:
                row inidices: (0, num_nodes]
                columns indices: (0, num_resources]
                enteries: [0, node_resource_cap] type: float
        """
        nodes_resources_usage = []
        for node in range(self.num_nodes):
            services_in_node = np.where(self.services_nodes == node)[0]
            node_resources_usage = sum(self.services_resources_usage[
                services_in_node])
            if type(node_resources_usage) != np.ndarray:
                node_resources_usage = np.zeros(self.num_resources)
            nodes_resources_usage.append(node_resources_usage)
        return np.array(nodes_resources_usage)

    @property
    def nodes_resources_request(self):
        """return the amount of resource usage
        on each node
        """
        nodes_resources_request = []
        for node in range(self.num_nodes):
            services_in_node = np.where(
                self.services_nodes == node)[0]
            node_resources_usage = sum(
                self.services_resources_request[services_in_node])
            if type(node_resources_usage) != np.ndarray:
                node_resources_usage = np.zeros(self.num_resources)
            nodes_resources_request.append(node_resources_usage)
        return np.array(nodes_resources_request)

    @property
    def services_resources_remained(self) -> np.ndarray:
        return self.services_resources_request - self.services_resources_usage

    @property
    def nodes_resources_remained(self):
        # The amount of acutally used resources
        # on the nodes
        return self.nodes_resources_cap - self.nodes_resources_usage

    @property
    def nodes_resources_available(self):
        # The amount of the available
        # non-requested resources on the nodes
        return self.nodes_resources_cap - self.nodes_resources_request

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
        workload_services_types = self.workload[:, self.timestep, :]
        services_resources_usage_frac = list(map(lambda service_type:
                                                   workload_services_types
                                                   [:, service_type],
                                                   self.services_types))
        services_resources_usage_frac = np.array(
            services_resources_usage_frac)
        return services_resources_usage_frac

    @property
    @rounding
    def nodes_resources_usage_frac(self) -> np.ndarray:
        """returns the resource usage of
        each node
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
    @rounding
    def nodes_resources_request_frac(self):
        """returns the resource requested on
        each node
                     ram - cpu
                    |         |
            nodes   |         |
                    |         |

            range:
                row inidices: (0, num_nodes]
                columns indices: (0, num_resources]
                enteries: [0, 1] type: float
        """
        return self.nodes_resources_request / self.nodes_resources_cap

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
            self.nodes_resources_request_frac > 1)[0])
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
    def nodes_resources_remained_frac(self):
        return self.nodes_resources_remained / self.nodes_resources_cap

    @property
    def nodes_resources_available_frac(self):
        return self.nodes_resources_available / self.nodes_resources_cap

    @property
    def nodes_resources_remained_frac_avg(self):
        return np.average(self.nodes_resources_remained_frac, axis=1)

    @property
    def nodes_resources_available_frac_avg(self):
        return np.average(self.nodes_resources_available_frac, axis=1)

    @property
    def done(self):
        """check at every step that if we have reached the
        final state of the simulation of not
        """
        done = True if self.timestep % self.episode_length == 0 else False
        return done

    @property
    def complete_raw_observation(self) -> Dict[str, np.ndarray]:
        """complete observation with all the available elements
        """
        observation = {
                "services_resources_usage": self.services_resources_usage,
                "nodes_resources_usage": self.nodes_resources_usage,
                "services_resources_usage_frac":
                self.services_resources_usage_frac,
                "nodes_resources_usage_frac": self.nodes_resources_usage_frac,
                "services_nodes": self.services_nodes
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
                "services_nodes": self.services_nodes
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
