import pytest
import numpy as np

from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, PPO1, PPO2, TRPO
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines.common import set_global_seeds

MODEL_LIST_DISCRETE = [
    A2C,
    ACER,
    ACKTR,
    DQN,
    PPO1,
    PPO2,
    TRPO
]

PARAM_NOISE_DDPG = AdaptiveParamNoiseSpec(initial_stddev=float(0.2), desired_action_stddev=float(0.2))
LOG_DIR_CONTINUOUS = "/tmp/log/perf/mountain/"

# Hyperparameters for learning MountainCarContinuous for each RL model
LEARN_FUNC_DICT = {
    'a2c': lambda e: A2C(policy="MlpPolicy", env=e, tensorboard_log=LOG_DIR_CONTINUOUS),
    'ddpg': lambda e: DDPG(policy="MlpPolicy", env=e,
                           param_noise=PARAM_NOISE_DDPG, tensorboard_log=LOG_DIR_CONTINUOUS),
    'ppo1': lambda e: PPO1(policy="MlpPolicy", env=e, tensorboard_log=LOG_DIR_CONTINUOUS),
    'ppo2': lambda e: PPO2(policy="MlpPolicy", env=e, tensorboard_log=LOG_DIR_CONTINUOUS),
    'trpo': lambda e: TRPO(policy="MlpPolicy", env=e, tensorboard_log=LOG_DIR_CONTINUOUS),
}



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
    model.learn(total_timesteps=int(1e5), seed=0)

    env = model.get_env()
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
@pytest.mark.parametrize("model_name", ['a2c', 'ddpg', 'ppo1', 'ppo2', 'trpo'])
def test_perf_mountain(model_name):
    """
    Test if the algorithm (with a given policy)
    can learn something on the MountainCarContinuous-v0 environment

    :param model_name: (str) Name of the RL model
    """
    # TODO: tune Hyperparameters, so each algo can pass this test
    model = LEARN_FUNC_DICT[model_name]('MountainCarContinuous-v0')
    model.learn(total_timesteps=int(2e5), seed=0)

    env = model.get_env()
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
    assert np.mean(episode_rewards) >= 0
    # Free memory
    del model, env
