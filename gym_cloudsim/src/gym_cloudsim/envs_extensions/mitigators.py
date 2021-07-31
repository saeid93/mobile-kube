import numpy as np
from copy import deepcopy


def greedy_mitigator(self, prev_observation) -> np.ndarray:
    """
    Although the premise is to do learning in a way that
    we don't end up with overloaded nodes but since the
    workload is some external dynamic we still might end up
    with overloaded states, this functions mitigate that
    FIXME find a better solution with Dynamic Programming
    FIXME other than Dynamic Programming this could also be improved
            in other ways too
    FIXME for example many heuristics based-on bestfit algs
    with a randomized greedy algorithm:
        1. extract overloaded nodes
        2. extract services in overloaded nodes
        3. for a number of tries
            3.1 shuffle normal_nodes
            3.2 shuffle services in overloaded node
            3.3 iterate one by one over overloaded nodes
                3.3.1 extract services from it and allocated them to one
                    of the normal_nodes
                3.3.2 if the overloading is solved then go to the next
                        overloaded node otherwise try to allocate the next
                        service
            3.4. if the overloading is solved for all the nodes then
                    return the result otherwise reset and go to step 3
    """
    auxiliary_node_needed = False
    overloaded_nodes = np.unique(np.where(self.nodes_resources_usage_frac
                                            > 1)[0]).tolist()
    normal_nodes = list(set(np.arange(self.num_nodes).tolist()) -
                        set(overloaded_nodes))
    for _ in range(self.mitigation_tries):
        # random.shuffle(normal_nodes)
        for node in overloaded_nodes:
            services_in_node = self.nodes_services[node]
            # random.shuffle(services_in_node)
            for service in services_in_node:
                for dest_node in normal_nodes:
                    if np.alltrue(self.services_resources_usage[
                        service] <
                                    self.nodes_resources_remained[
                                        dest_node]):
                        self.services_nodes[service] = dest_node
                        break
                else:  # no-break
                    continue
                break

    if self.num_overloaded:
        auxiliary_node_needed = True
        self.services_nodes = deepcopy(
            prev_observation['services_nodes'])
    return auxiliary_node_needed

def auxilary_node_mitigation(self) -> None:
    """
    move the remianed nodes to the auxiliary ones
        1. find overloaded nodes
        2. move the contaienrs out of the overloaded nodes to the
            auxilary node
    """
    overloaded_nodes = np.unique(np.where(self.nodes_resources_usage_frac
                                            > 1)[0]).tolist()
    for node in overloaded_nodes:
        services_in_node = self.nodes_services[node]
        for service in services_in_node:
            self.services_nodes[service] = self.num_nodes
            if np.alltrue(self.nodes_resources_usage_frac[node] <= 1):
                break

def _all_mitigators(self):
    # TODO merge two mitigators if possible
    pass
