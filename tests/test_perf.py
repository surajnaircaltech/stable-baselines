import pytest
import numpy as np
import gym

from stable_baselines import A2C, ACER, ACKTR, DeepQ, DDPG, PPO1, PPO2, TRPO
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv

MODEL_LIST_DISCRETE = [
    A2C,
    ACER,
    ACKTR,
    DeepQ,
    PPO1,
    PPO2,
    TRPO
]

MODEL_LIST_CONTINUOUS = [
    A2C,
    # ACER,
    # ACKTR,
    DDPG,
    PPO1,
    PPO2,
    TRPO
]


if not pytest.config.getoption("--perf-tests"):
    pytest.skip("--perf-tests is missing, skipping performance tests", allow_module_level=True)

@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST_DISCRETE)
def test_perf_cartpole(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn something on the simple CartPole environment

    :param model_class: (BaseRLModel) A model
    """

    # TODO: multiprocess if possible
    model = model_class(policy="MlpPolicy", env='CartPole-v1',
                        tensorboard_log="/tmp/log/perf/cartpole")
    model.learn(total_timesteps=100000, seed=0)

    env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
    # env = model.env
    n_trials = 2000
    set_global_seeds(0)
    obs = env.reset()
    episode_rewards = []
    reward_sum = 0
    for _ in range(n_trials):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            episode_rewards.append(reward_sum)
            reward_sum = 0
    assert np.mean(episode_rewards) >= 100
    # Free memory
    del model, env


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST_CONTINUOUS)
def test_perf_marslander(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn something on the LunarLanderContinuous-v2 environment

    :param model_class: (BaseRLModel) A model
    """

    model = model_class(policy="MlpPolicy", env='LunarLanderContinuous-v2',
                        tensorboard_log="/tmp/log/perf/mars/")
    model.learn(total_timesteps=200000, seed=0)

    env = DummyVecEnv([lambda: gym.make('LunarLanderContinuous-v2')])
    # env = model.env
    n_trials = 2000
    set_global_seeds(0)
    obs = env.reset()
    episode_rewards = []
    reward_sum = 0
    for _ in range(n_trials):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            episode_rewards.append(reward_sum)
            reward_sum = 0
    assert np.mean(episode_rewards) >= -200
    # Free memory
    del model, env
