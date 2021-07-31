from gym.envs.registration import register

register(
    id='SimGreedyEnv-v0',
    entry_point='mobile_kube.envs:SimGreedyEnv',
)
register(
    id='SimEdgeEnv-v0',
    entry_point='mobile_kube.envs:SimEdgeEnv',
)
register(
    id='KubeEdgeEnv-v0',
    entry_point='mobile_kube.envs:KubeEdgeEnv',
)
register(
    id='KubeGreedyEnv-v0',
    entry_point='mobile_kube.envs:KubeGreedyEnv',
)