import numpy as np
from copy import deepcopy
from typing import (
    Tuple,
    Dict,
    Union
)
from .mitigators import (
    auxilary_node_mitigation,
    greedy_mitigator,
)

def get_step_method(mitigation_method):
    mitigation_methods = {
        'none': _step_without_mitigators,
        'all': _step_with_all_mitigators,
        'aux': _step_with_aux_mitigators,
        'greedy': _step_greedy,
        'edge': _step_edge
    }
    return mitigation_methods[mitigation_method]

def _step_without_mitigators(self, action: np.ndarray) \
    -> Tuple[np.ndarray, int, bool, dict]:
    """simple cloud step without any mitigations
    """
    # save previous action for computing the
    # final (after mitigation) num_of_moves
    prev_services_nodes = deepcopy(self.services_nodes)

    common_step(self, action)

    num_moves = len(np.where(
        self.services_nodes != prev_services_nodes)[0])

    reward, rewards = self._reward(
        num_moves=num_moves
        )

    info = {'num_moves': num_moves,
            'num_consolidated': self.num_consolidated,
            'total_reward': reward,
            'rewards': rewards,
            'timestep': self.timestep}

    assert self.observation_space.contains(self.observation),\
            (f"observation:\n<{self.raw_observation}>\noutside of "
            f"observation_space:\n <{self.observation_space}>")

    return self.observation, reward, self.done, info


def _step_with_all_mitigators(self, action: np.ndarray) -> Tuple[
    np.ndarray, int, bool, Dict[str, Union[int, float]]]:
    """
    variables:
        get the num_of_overloaded to see if there are any
        if there are any left then fix it with a simple overloaded
            mitigator
        prev_num_overloaded: number of overloaded nodes before the
            run of the mitigation
        prev_num_consolidated: number of consolidated nodes before the run
            of the mitigation
    """
    # save previous action for computing the
    # final (after mitigation) num_of_moves
    prev_services_nodes = deepcopy(self.services_nodes)

    # move the timestep to the next point of time
    # and then take the action
    common_step(self, action)

    # flags for overloading mitigation
    greedy_mitigation_needed = False
    auxiliary_node_mitigation_needed = False
    prev_num_overloaded = 0

    # save previous observation for later use in render and
    # greedy_mitigator()
    self.before_mitigation_observation = deepcopy(
        self.complete_raw_observation)

    if self.num_overloaded:
        greedy_mitigation_needed = True
        prev_num_overloaded = self.num_overloaded

        # save the necessary before mitigation
        # variable to render
        self.greedy_mitigation_needed_for_render = \
            greedy_mitigation_needed

    if greedy_mitigation_needed:
        # mitigations
        auxiliary_node_mitigation_needed = greedy_mitigator(
            self,
            self.before_mitigation_observation) # TODO this should be fixed
        if auxiliary_node_mitigation_needed:
            auxilary_node_mitigation(self)

        # save auxiliary_node_needed for render function
        self.auxiliary_node_mitigation_needed_for_render = \
            auxiliary_node_mitigation_needed

    num_moves = len(np.where(
        self.services_nodes != prev_services_nodes)[0])

    reward, rewards = self._reward(
        num_moves=num_moves,
        greedy_mitigation_needed=greedy_mitigation_needed,
        auxiliary_node_mitigation_needed=auxiliary_node_mitigation_needed,
        prev_num_overloaded=prev_num_overloaded)

    info = {'num_moves': num_moves,
            'num_overloaded': prev_num_overloaded,
            'num_consolidated': self.num_consolidated,
            'total_reward': reward,
            'rewards': rewards,
            'timestep': self.timestep}

    assert self.observation_space.contains(self.observation),\
            (f"observation:\n<{self.raw_observation}>\noutside of "
            f"observation_space:\n <{self.observation_space}>")

    return self.observation, reward, self.done, info


def _step_with_aux_mitigators(self, action: np.ndarray) -> Tuple[
    np.ndarray, int, bool, Dict[str, Union[int, float]]]:
    """
    Step operations:
        1. Environment recieves an action map from the environment
        2. Agent try to perform the action in the take_action function
        3. If after the action is taken we end up in an illegal state
            then the mitigation_greedy is called
        4. If mitigation_greedy couldn't solve the problem then the
            problem is solved with the help of unlimited auxiliary nodes

    variables:
        get the num_of_overloaded to see if there are any
        if there are any left then fix it with a simple overloaded
        mitigator
        prev_num_overloaded: number of overloaded nodes before the
                            run of the mitigation
        prev_num_consolidated: number of consolidated nodes before the run
                            of the mitigation
    """
    # save previous action for computing the
    # final (after mitigation) num_of_moves
    prev_services_nodes = deepcopy(self.services_nodes)

    # move the timestep to the next point of time
    # and then take the action
    common_step(self, action)

    # flags for overloading mitigation
    greedy_mitigation_needed = False
    auxiliary_node_mitigation_needed = False
    prev_num_overloaded = 0

    # save previous observation for later use in render
    self.before_mitigation_observation = deepcopy(
        self.complete_raw_observation)

    if self.num_overloaded:
        auxiliary_node_mitigation_needed = True
        prev_num_overloaded = self.num_overloaded

        # save the necessary prev (before mitigation) variable to render
        self.auxiliary_node_mitigation_needed_for_render = \
            auxiliary_node_mitigation_needed
        # self.prev_observation = deepcopy(self.complete_raw_observation)

    if auxiliary_node_mitigation_needed:
        # mitigation
        auxilary_node_mitigation(self)

    num_moves = len(np.where(
        self.services_nodes != prev_services_nodes)[0])

    reward, rewards = self._reward(
        num_moves=num_moves,
        greedy_mitigation_needed=greedy_mitigation_needed,
        auxiliary_node_mitigation_needed=auxiliary_node_mitigation_needed,
        prev_num_overloaded=prev_num_overloaded)

    info = {'num_moves': num_moves,
            'num_overloaded': prev_num_overloaded,
            'num_consolidated': self.num_consolidated,
            'total_reward': reward,
            'rewards': rewards,
            'timestep': self.timestep}

    assert self.observation_space.contains(self.observation),\
            (f"observation:\n<{self.raw_observation}>\noutside of "
            f"observation_space:\n <{self.observation_space}>")

    return self.observation, reward, self.done, info


def _step_edge(self, action: np.ndarray) -> Tuple[np.ndarray, int, bool, dict]:
    """
    edge servers here
    1. move the services based-on current network and nodes state
    2. do one step of user movements
    3. update the nodes
    """
    # save previous action for computing the
    # final (after mitigation) num_of_moves
    prev_services_nodes = deepcopy(self.services_nodes)

    # TODO add action of the users here too
    common_step(self, action)
    
    # make user movements --> network parts
    self.users_stations = self.edge_simulator.sample_users_stations(
        timestep=self.timestep)
    users_distances = self.edge_simulator.users_distances
    # update network with the new placements
    self.edge_simulator.update_services_nodes(self.services_nodes)

    # if auxiliary_node_mitigation_needed:
    #     # mitigation
    #     auxilary_node_mitigation(self)

    num_moves = len(np.where(
        self.services_nodes != prev_services_nodes)[0])

    # reward, rewards = self._reward(users_distances=users_distances)
    reward, rewards = self._reward(
        users_distances=users_distances,
        num_moves=num_moves
        )

    info = {'num_moves': num_moves,
            'num_consolidated': self.num_consolidated,
            'total_reward': reward,
            'rewards': rewards,
            'timestep': self.timestep}

    assert self.observation_space.contains(self.observation),\
            (f"observation:\n<{self.raw_observation}>\noutside of "
            f"observation_space:\n <{self.observation_space}>")

    return self.observation, reward, self.done, info


def _step_greedy(self, action: np.ndarray) -> Tuple[np.ndarray, int, bool, dict]:
    """
    General overivew:
        1. try binpacking without auxiliary
        2. if doesn't fit add auxiliary as the last server
            and allocate with considering it (auxiliary has
            unlimited space)
    Binpacking:
    generate the intitial state for nodes_services with a
    bestfit greedy algorithm.
        1. iterate over nodes one by one
        2. pop a node from the nodes lists
        3. find the nodes with least remained resources
        4. try to allocate the services to them
        5. if not possible pop another node
    We use mitigation after bin-packing because
    the action is based-on the current timestep but
    next timestep might result in illegal resource usage
    with the current placement
    """
    # save previous action for computing the
    # final (after mitigation) num_of_moves
    prev_services_nodes = deepcopy(self.services_nodes)

    # get the next greedy actoin from the env
    _, action = self._next_greedy_action(prev_services_nodes)

    # move the timestep to the next point of time
    # and then take the action
    common_step(self, action)

    # flags for overloading mitigation
    greedy_mitigation_needed = False
    auxiliary_node_mitigation_needed = False
    prev_num_overloaded = 0

    # save previous observation for later use in render and
    # greedy_mitigator()
    self.before_mitigation_observation = deepcopy(
        self.complete_raw_observation)

    if self.num_overloaded:
        greedy_mitigation_needed = True
        prev_num_overloaded = self.num_overloaded

        # save the necessary before mitigation
        # variable to render
        self.greedy_mitigation_needed_for_render = \
            greedy_mitigation_needed

    if greedy_mitigation_needed:
        # mitigations
        auxiliary_node_mitigation_needed = greedy_mitigator(
            self,
            self.before_mitigation_observation)
        if auxiliary_node_mitigation_needed:
            auxilary_node_mitigation(self)

        # save auxiliary_node_needed for render function
        self.auxiliary_node_mitigation_needed_for_render = \
            auxiliary_node_mitigation_needed

    num_moves = len(np.where(
        self.services_nodes != prev_services_nodes)[0])

    info = {'num_moves': num_moves,
            'num_overloaded': prev_num_overloaded,
            'num_consolidated': self.num_consolidated,
            'timestep': self.timestep}

    assert self.observation_space.contains(self.observation),\
            (f"observation:\n<{self.raw_observation}>\noutside of "
            f"observation_space:\n <{self.observation_space}>")

    return self.observation, None, self.done, info


def common_step(self, action: np.ndarray):
    """variables:
        get the num_of_overloaded to see if there are any
        if there are any left then fix it with a simple overloaded
        mitigator
        prev_num_overloaded: number of overloaded nodes before the
                            run of the mitigation
        prev_num_consolidated: number of consolidated nodes before the run
                            of the mitigation
    """
    # move the timestep to the next point of time
    # in the workload
    assert self.action_space.contains(action)
    self.global_timestep += 1
    # find the reward and compute the remainder for round-robin
    self.timestep = self.global_timestep % self.workload.shape[1]
    self._take_action(action)

