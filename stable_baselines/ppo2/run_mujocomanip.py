#!/usr/bin/env python3
import numpy as np
import gym

from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines import bench, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo2 import PPO2
from stable_baselines.ppo1 import PPO1
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.ddpg import DDPG
from stable_baselines.ddpg.memory import Memory
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLnLstmPolicy
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common import tf_util
import RoboticsSuite as suite

from RoboticsSuite.wrappers import GymWrapper


def train(env_id, num_timesteps, seed, model_path = None, images = False):
    """
    Train PPO2 model for Mujoco environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    """
    def make_env():
        if images:
             env_out = GymWrapper(
            suite.make(
                "SawyerLift",
                use_object_obs=False,
                use_camera_obs=True,  # do not use pixel observations
                has_offscreen_renderer=True,  # not needed since not using pixel obs
                has_renderer=False,  # make sure we can render to the screen
                camera_depth=True,
                reward_shaping=True,  # use dense rewards
                control_freq=10,  # control should happen fast enough so that simulation looks smooth
                render_visual_mesh=False,
            ), keys=["image", "depth"], images=True,
            )
        else:
            env_out = GymWrapper(
            suite.make(
                "SawyerLift",
                use_object_obs=True,
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=False,  # make sure we can render to the screen
                camera_depth=False,
                reward_shaping=True,  # use dense rewards
                control_freq=10,  # control should happen fast enough so that simulation looks smooth
                render_visual_mesh=False,
            )#, keys=["image", "depth"], images=True,
            )
        env_out.reward_range = None
        env_out.metadata = None
        env_out.spec = None
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out

    #env = make_env()

    if images:
        env = DummyVecEnv([make_env])
        env = VecNormalize(env)

        set_global_seeds(seed)
        policy = CnnPolicy
        tblog = "/cvgl2/u/surajn/workspace/tb_logs/sawyerlift_all/"
    else:
        env = DummyVecEnv([make_env])
        env = VecNormalize(env)

        set_global_seeds(seed)
        policy = MlpPolicy
        tblog = "/cvgl2/u/surajn/workspace/tb_logs/sawyerlift_all/"
    nb_actions = env.action_space.shape[-1]
    #model = PPO2(policy=policy, env=env, n_steps=2048, nminibatches=32, lam=0.95, gamma=0.99, noptepochs=10,
    #             ent_coef=0.0, learning_rate=3e-4, cliprange=0.2, verbose=1, tensorboard_log=tblog)
    #model = TRPO(policy=policy, env, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1, entcoeff=0.0,
    #                 gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, tensorboard_log=tblog, verbose=1)
    model = DDPG(policy=ddpgMlpPolicy, env=env, memory_policy=Memory, eval_env=None, param_noise=AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2),
        action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(nb_actions), sigma=float(0.2) * np.ones(nb_actions)), memory_limit=int(1e6), verbose=2,tensorboard_log=tblog)

    model.learn(total_timesteps=num_timesteps)
    env.close()

    if model_path:
        model.save(model_path)
        #tf_util.save_state(model_path)

    return model, env

def main():
    """
    Runs the test
    """
    parser = mujoco_arg_parser()
    parser.add_argument('--model-path', default="/cvgl2/u/surajn/workspace/saved_models/sawyerlift_ppo2/model")
    parser.add_argument('--images', default=False)
    args = parser.parse_args()

    logger.configure()
    if not args.play:
        model, env = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, model_path=args.model_path, images=args.images)

    if args.play:

        def make_env():
            env_out = GymWrapper(
            suite.make(
                "SawyerLift",
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=True,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=10,  # control should happen fast enough so that simulation looks smooth
            )
            )
            env_out.reward_range = None
            env_out.metadata = None
            env_out.spec = None
            env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
            return env_out

        #env = make_env()
        env = DummyVecEnv([make_env])
        env = VecNormalize(env)



        policy = MlpPolicy
        #model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
        #         optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear', verbose=1)
        model = TRPO(MlpPolicy, env, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1, entcoeff=0.0,
                     gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
        model.load(args.model_path)
        logger.log("Running trained model")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        while True:
            env.render()
            actions = model.step(obs)[0]
            obs[:] = env.step(actions)[0]
            


if __name__ == '__main__':
    main()
