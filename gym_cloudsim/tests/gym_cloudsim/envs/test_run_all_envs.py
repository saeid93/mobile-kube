import pytest
import gym

@pytest.mark.parametrize("type_env, reward_mode", [
    (0, 'cloud'),
    (1, 'cloud'),
    (2, 'cloud'),
    (3, 'cloud'),
    (4, 'none'),
    (5, 'edge'),
    (5, 'both'),
    (6, 'edge'),
    (6, 'both')
])
def test_run_envs(env_config, type_env, reward_mode):
    env_config.update({'reward_mode': reward_mode})
    env = gym.make(f'CloudSim-v{type_env}', config=env_config)
    obs = env.reset()
    for iter in range(100):
        action = env.action_space.sample()
        _ = env.step(action)
