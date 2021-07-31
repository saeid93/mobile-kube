from gym.envs.registration import register

register(
    id='SimBinpackingEnv-v0',
    entry_point='mobile_kube.envs:SimBinpackingEnv',
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
    id='KubeBinpackingEnv-v0',
    entry_point='mobile_kube.envs:KubeBinpackingEnv',
)