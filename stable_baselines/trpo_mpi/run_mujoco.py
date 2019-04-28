#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI

from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.trpo_mpi import TRPO
import stable_baselines.common.tf_util as tf_util


def train(env_id, num_timesteps, seed):
    """
    Train TRPO model for the mujoco environment, for testing purposes

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    with tf_util.single_threaded_session():
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            logger.configure()
        else:
            logger.configure(format_strs=[])
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

        tblog = "/cvgl2/u/surajn/workspace/tb_logs/reacher/"
        env = make_mujoco_env(env_id, workerseed)
        model = TRPO(MlpPolicy, env, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1, entcoeff=0.0,
                     gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, tensorboard_log)
        model.learn(total_timesteps=num_timesteps)
        env.close()


def main():
    """
    Runs the test
    """
    args = mujoco_arg_parser().parse_args()
    model, env = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

    if args.play:
        logger.log("Running trained model")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs[:] = env.step(actions)[0]
            env.render()


if __name__ == '__main__':
    main()
