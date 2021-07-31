from colorama import (
    Fore,
    Style
)

def get_render_method(mitigation_method):
    render_methods = {
       'none': _step_without_mitigators,
       'aux': _render_with_aux_mitigators,
       'greedy': _render_with_aux_mitigators,
       'all': _render_with_all_mitigators,
       'edge': _edge_render
    }
    return render_methods[mitigation_method]

def _step_without_mitigators(self) -> None:
    """
    render for the case with both
        1. aux_mitigator
    """
    print("--------state--------")
    print("services_types_usage:")
    if not self.num_overloaded:
        print("nodes_resources_usage_frac:")
        print(self.nodes_resources_usage_frac)
        print("services_nodes:")
        print(self.services_nodes)
        # plot_resource_allocation(self.services_nodes,
        #                          self.nodes_resources_cap,
        #                          self.services_resources_cap,
        #                          self.services_resources_usage,
        #                          plot_length=80)
    else:
        print(Fore.RED, "agent's action lead to an overloaded state!")
        # before using auxiliary
        print("nodes_resources_usage_frac:")
        print(self.nodes_resources_usage_frac)
        print("services_nodes:")
        print(self.services_nodes)
        # plot_resource_allocation(self.services_nodes,
        #                          self.nodes_resources_cap,
        #                          self.services_resources_cap,
        #                          self.services_resources_usage,
        #                          plot_length=80)
        # after using auxiliary
        print(Style.RESET_ALL)


def _render_with_aux_mitigators(self) -> None:
    """
    render for the case with both
        1. aux_mitigator
    """
    print("--------state--------")
    print("services_types_usage:")
    print(Fore.RED,self.services_types_usage)
    if not self.auxiliary_node_mitigation_needed_for_render:
        print("nodes_resources_usage_frac:")
        print(self.nodes_resources_usage_frac)
        print("services_nodes:")
        print(self.services_nodes)
        print("services in auxiliary nodes:")
        print(self.services_in_auxiliary)
        print("auxiliary reosource usage:")
        print(self.auxiliary_resources_usage)
        # plot_resource_allocation(self.services_nodes,
        #                          self.nodes_resources_cap,
        #                          self.services_resources_cap,
        #                          self.services_resources_usage,
        #                          plot_length=80)
    else:
        print(Fore.RED, "agent's action lead to an overloaded state!")
        # before using auxiliary
        print("nodes_resources_usage_frac:")
        print(Fore.RED,
                self.before_mitigation_observation[
                    "nodes_resources_usage_frac"])
        print("services_nodes:")
        print(Fore.RED, self.before_mitigation_observation[
            "services_nodes"])
        # plot_resource_allocation(self.services_nodes,
        #                          self.nodes_resources_cap,
        #                          self.services_resources_cap,
        #                          self.services_resources_usage,
        #                          plot_length=80)
        # after using auxiliary
        print("---resutls of using auxiliary---")
        print("nodes_resources_usage_frac:")
        print(Fore.GREEN, self.nodes_resources_usage_frac)
        print("services_nodes:")
        print(Fore.GREEN, self.services_nodes)
        print("services in auxiliary nodes:")
        print(Fore.GREEN, self.services_in_auxiliary)
        print("auxiliary reosource usage:")
        print(Fore.GREEN, self.auxiliary_resources_usage)
        print(Style.RESET_ALL)
    self.auxiliary_node_mitigation_needed_for_render = False

def _render_with_all_mitigators(self) -> None:
    """
    render for the case with both
        1. greedy_mitigator
        2. aux_mitigator
    """
    print("--------state--------")
    # print("services_types_usage:")
    # print(self.services_types_usage)
    if not self.greedy_mitigation_needed_for_render:
        print("nodes_resources_usage_frac:")
        print(self.nodes_resources_usage_frac)
        print("services_nodes:")
        print(self.services_nodes)
        # plot_resource_allocation(self.services_nodes,
        #                          self.nodes_resources_cap,
        #                          self.services_resources_cap,
        #                          self.services_resources_usage,
        #                          plot_length=80)
    else:
        print(Fore.RED, "agent's action lead to an overloaded state!")
        # before mitigation
        print("nodes_resources_usage_frac:")
        print(Fore.RED,
                self.before_mitigation_observation[
                    "nodes_resources_usage_frac"])
        print("services_nodes:")
        print(Fore.RED, self.before_mitigation_observation[
            "services_nodes"])
        # plot_resource_allocation(self.services_nodes,
        #                          self.nodes_resources_cap,
        #                          self.services_resources_cap,
        #                          self.services_resources_usage,
        #                          plot_length=80)
        # after mitigation
        print("---resutls of mitigation---")
        if not self.auxiliary_node_mitigation_needed_for_render:
            print(Fore.GREEN, "Greedy Mitigation successful.")
            print("nodes_resources_usage_frac:")
            print(Fore.GREEN, self.nodes_resources_usage_frac)
            print("services_nodes:")
            print(Fore.GREEN, self.services_nodes)
            # plot_resource_allocation(self.services_nodes,
            #                          self.nodes_resources_cap,
            #                          self.services_resources_cap,
            #                          self.services_resources_usage,
            #                          plot_length=80)
        else:
            print(Fore.RED, "Greedy Mitigation unsuccessful.")
            print(Fore.RED, "auxilary node were used.")
            print("nodes_resources_usage_frac:")
            print(Fore.RED, self.nodes_resources_usage_frac)
            print("services_nodes:")
            print(Fore.RED, self.services_nodes)
            print("services in auxiliary nodes:")
            print(Fore.RED, self.services_in_auxiliary)
            print("auxiliary reosource usage:")
            print(Fore.RED, self.auxiliary_resources_usage)
        print(Style.RESET_ALL)
    self.greedy_mitigation_needed_for_render = False
    self.auxiliary_node_mitigation_needed_for_render = False

def _edge_render(self) -> None:
    """
    just render the last observation
    without greedy and aux mitigatins
    """
    if not self.auxiliary_node_mitigation_needed_for_render:
        print("--------state--------")
        print("users_stations:")
        print(self.users_stations)
        print("nodes_resources_usage_frac:")
        print(self.nodes_resources_usage_frac)
        print("services_nodes:")
        print(self.services_nodes)
        # plot_resource_allocation(self.services_nodes,
        #                          self.nodes_resources_cap,
        #                          self.services_resources_cap,
        #                          self.services_resources_usage,
        #                          plot_length=80)
    else:
        print(Fore.RED, "--------state--------")
        print(Fore.RED, "nodes_resources_usage_frac:")
        print(Fore.RED, self.nodes_resources_usage_frac)
        print(Fore.RED, "services_nodes:")
        print(Fore.RED, self.services_nodes)
        # plot_resource_allocation(self.services_nodes,
        #                          self.nodes_resources_cap,
        #                          self.services_resources_cap,
        #                          self.services_resources_usage,
        #                          plot_length=80)
    print(Style.RESET_ALL)
    self.auxiliary_node_mitigation_needed_for_render = False
