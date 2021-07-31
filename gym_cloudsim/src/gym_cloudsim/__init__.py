from gym.envs.registration import register

register(
    id='SimCloudEnv-v0',
    entry_point='gym_cloudsim.envs:SimCloudEnv',
)
register(
    id='SimGreedyEnv-v0',
    entry_point='gym_cloudsim.envs:SimGreedyEnv',
)
register(
    id='SimEdgeEnv-v0',
    entry_point='gym_cloudsim.envs:SimEdgeEnv',
)
register(
    id='KubeCloudEnv-v0',
    entry_point='gym_cloudsim.envs:KubeCloudEnv',
)
register(
    id='KubeEdgeEnv-v0',
    entry_point='gym_cloudsim.envs:KubeEdgeEnv',
)
register(
    id='KubeGreedyEnv-v0',
    entry_point='gym_cloudsim.envs:KubeGreedyEnv',
)
register(
    id='KubernetesEnv-v0',
    entry_point='gym_cloudsim.envs:KubernetesConsolidationEnv'
)
