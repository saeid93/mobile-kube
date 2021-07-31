import numpy as np
from copy import deepcopy
from mobile_kube.util import (
    ACTION_MIN,
    ACTION_MAX
)
from typing import Union, Callable

def get_action_method_kube(action_method: str) -> Union[Callable, Callable]:
    action_methods = {
        'absolute': (_take_action_absolute, _validate_action_absolute),
        'probabilistic': (_take_action_probabilistic,
                          _validate_action_probabilistic)
    }
    return action_methods[action_method]

def _take_action_probabilistic(self, action: np.ndarray) -> None:
    """
    greedily choose the new placement from the
    new mapping
    Algorithm:
        1. go line by line through each nodes servers
        2. find the first node that have free space and the service
            can be fit into that
        3. if no server to migrate then don't migrage
    Remark:
        - The current model of allocation is
            if maximum_resource_usage_by_service_for_all_resources
                < current_timestep_used_resources:
                allocate
        - This is a heuristic, depending on the ordering of the
        contaienrs we try to allocate the result changes,
        we only try the 0-last_service by the service_id ordering
        - NOTE some form of bestfit might improve this
    """
    absolute_action = np.zeros(self.num_services)
    # reshape the actions to the columnar format
    action = action.reshape(self.num_services, self.num_nodes)
    nodes_orders = (-action).argsort()

    # move the servers based-on the arrival
    for service in range(self.num_services):
        node_order = nodes_orders[service, :]
        for node in node_order:
            if np.alltrue(self.services_resources_usage[service] <
                            self.nodes_resources_remained[node]):
                absolute_action[service] = node
                break
    self.services_nodes_obj = self.nodes[list(map(int, absolute_action))]
    self._migrate(self.services_nodes_obj)

def _validate_action_probabilistic(self, action: np.ndarray) -> None:
    """
    check the probabilistic recieved action from outside the env
    """
    assert np.all(action >= ACTION_MIN),\
        ("some actions at indexes"
            f" <{np.where(action < ACTION_MIN)[0]}>"
            " have exceeded the lower bound (0) with"
            f" values\n<{action[action < ACTION_MIN]}>"
            f" \nThe recieved action:\n<{action}>")
    assert np.all(action <= ACTION_MAX),\
        ("some actions at indexes"
            f" <{np.where(action > ACTION_MAX)[0]}>"
            " have exceeded the upper bound (1) with"
            f" values\n<{action[action > ACTION_MAX]}>"
            f" \nThe recieved action:\n<{action}>")
    assert action.shape[0] == self.num_services * self.num_nodes,\
        ("the shape of the"
            f" recieved action <{action.shape[0]}>"
            " is not consistent with the"
            f" expected shape <{self.num_services * self.num_nodes}>")

def _take_action_absolute(self, action: np.ndarray) -> None:
    """
    each action is the next placement of the servers
    """
    self.services_nodes_obj =  self.nodes[action]
    self._migrate(self.services_nodes_obj)

def _validate_action_absolute(self, action: np.ndarray) -> None:
    """
    check the absolute recieved action from outside the env
    """
    assert np.all(action >= 0),\
        ("some actions at indexes"
            f" <{np.where(action<0)[0]}>"
            " have exceeded lower bound (0) with"
            f" values\n<{action[action<0]}>"
            f" \nThe recieved action:\n<{action}>")
    assert np.all(action <= self.num_nodes-1),\
        ("some actions at indexes"
            f" {np.where(action>self.num_nodes-1)[0]}"
            " have exceeded upper bound (0) with"
            f" values\n<{action[action>self.num_nodes-1]}>"
            f" \nThe recieved action:\n<{action}>")
    assert action.shape[0] == self.num_services,\
        (f"the recieved action length <{action.shape[0]}>"
            " is not consistent with the number of"
            f" services <{self.num_nodes}>")
