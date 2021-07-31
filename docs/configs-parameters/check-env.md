```
{
    "env_config_base": {
        "obs_elements": ["services_resources_usage" |
                         "nodes_resources_usage" |
                         "services_resources_usage_frac" |
                         "nodes_resources_usage_frac" |
                         "services_nodes" |
                         "auxiliary_resources_usage"],
        "penalty_illegal": [int], // penalty for illegal moves
        "penalty_move": [int], // penalty for every moves (to reduce the number of moves)
        "penalty_variance": [int], // penalty for the variance reward
        "penalty_consolidatefd": [int], // penalty for the number of consolidated
        "penalty_latency": [int], // penalty for the latency
        "mitigation_tries": [int],
        "workload_stop": [int],
        "episode_length": [int],
        "timestep_reset": [bool],
        "placement_reset": [bool],
        "from_dataset": [bool],
        "reward_mode": ["cloud", "edge", "both"],
        "action_method": ["absolute", "probabilistic"],
        "step_method": ["all" | "edge" | "aux" | "binpacking"],
        "normalise_latency": [bool],
        "seed": [int] // seed for generating reproducible results
    }
}
```