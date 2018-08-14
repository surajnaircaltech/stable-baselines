#!/usr/bin/env python3
<<<<<<< HEAD:stable_baselines/ppo2/run_atari.py
from stable_baselines import logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines.ppo2 import ppo2
from stable_baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
=======
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import PPO2
from baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
>>>>>>> refactoring:baselines/ppo2/run_atari.py


def train(env_id, num_timesteps, seed, policy):
    """
    Train PPO2 model for atari environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
    """

    env = VecFrameStack(make_atari_env(env_id, 8, seed), 4)
    policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy}[policy]
    model = PPO2(policy=policy, env=env, n_steps=128, nminibatches=4, lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
                 learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
    model.learn(total_timesteps=num_timesteps)


def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy)


if __name__ == '__main__':
    main()