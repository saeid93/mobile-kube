from mobile_kube.util.kubernetes_utils import (
    Cluster,
    Node,
    construct_pod
)

CONFIG_FILE = "~/.kube/config"
cluster = Cluster(CONFIG_FILE)

# Get Nodes
nodes = cluster.monitor.get_nodes()
for node in nodes:
    print(Node(node))

# Get Nodes
print(cluster.monitor.get_nodes())

# Create a Pod in a specific node
print(cluster.action.create_pod(
    construct_pod(name='nginx-1', image='nginx', node_name='NAME-OF-NODE')
))

# Get a specific Pod
print(cluster.monitor.get_pod(name='nginx-1'))

# Get Pods
print(cluster.monitor.get_pods())

# Get Pods metrics
print(cluster.monitor.get_pods_metrics())

# Get Nodes metrics
print(cluster.monitor.get_nodes_metrics())

# Delete a specific Pod
print(cluster.action.delete_pod(name='nginx-1'))
