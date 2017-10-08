import argparse
import time
import os
from tempfile import mkdtemp
import sys
import subprocess
import threading
import json
from gym import utils

from baselines.common.mpi_fork import mpi_fork
from baselines import logger
from baselines.logger import Logger
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
    SimpleMonitor
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI


from custom_env import Custom0Env

from gym.envs.registration import register

register(
    id='Custom0-v0',
    entry_point='custom_env:Custom0Env',
    max_episode_steps=2000,
    reward_threshold=1000.0,
)


def run(env_id, seed, noise_type, num_cpu, layer_norm, logdir, gym_monitor, evaluation, bind_to_core, hidden_size, nb_layers, portnum, **kwargs):
    kwargs['logdir'] = logdir
    whoami = mpi_fork(num_cpu, bind_to_core=bind_to_core)
    if whoami == 'parent':
        sys.exit(0)

    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    utils.portnum = portnum + rank
    if rank != 0:
        # Write to temp directory for all non-master workers.
        actual_dir = None
        Logger.CURRENT.close()
        Logger.CURRENT = Logger(dir=mkdtemp(), output_formats=[])
        logger.set_level(logger.DISABLED)

    # Create envs.
    if rank == 0:
        env = gym.make(env_id)
        if gym_monitor and logdir:
            env = gym.wrappers.Monitor(env, os.path.join(logdir, 'gym_train'), force=True)
        env = SimpleMonitor(env)

        if evaluation:
            eval_env = gym.make(env_id)
            if gym_monitor and logdir:
                eval_env = gym.wrappers.Monitor(eval_env, os.path.join(logdir, 'gym_eval'), force=True)
            eval_env = SimpleMonitor(eval_env)
        else:
            eval_env = None
    else:
        env = gym.make(env_id)
        if evaluation:
            eval_env = gym.make(env_id)
        else:
            eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(5e5), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_size=hidden_size, nb_layers=nb_layers, layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_size=hidden_size, nb_layers=nb_layers, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    Logger.CURRENT.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-id', type=str, default='Custom0-v0')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'overwrite-memory', default=True)
    boolean_flag(parser, 'render', default=False)
    parser.add_argument('--num-cpu', type=int, default=1)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', type=int, default=59)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--nb_layers', type=int, default=2)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=5.0)
    parser.add_argument('--nb-epochs', type=int, default=10000)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-rollout-steps', type=int, default=200)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=500)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--logdir', type=str, default='Fetch_DDPG') #default=None)
    parser.add_argument('--agentName',type=str, default='Fetch_DDPG_256_2')
    parser.add_argument('--resume', type=int, default=0)
    boolean_flag(parser, 'gym-monitor', default=False)
    boolean_flag(parser, 'evaluation', default=False)
    boolean_flag(parser, 'bind-to-core', default=False)
    parser.add_argument("--portnum", required=False, type=int, default=5050)
    parser.add_argument("--server_ip", required=False, default="localhost")
  #  parser.add_argument("--server_ip", required=False, default='172.17.171.126')

 #   boolean_flag(parser, 'test', default=False)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    portnum = args['portnum']
    utils.server_ip = args['server_ip']
    del args['portnum']
    del args['server_ip']

    hidden_size = args['hidden_size']
    del args['hidden_size']

    nb_layers = args['nb_layers']
    del args['nb_layers']

    # Figure out what logdir to use.
    if args['logdir'] is None:
        args['logdir'] = os.getenv('OPENAI_LOGDIR')

    # Print and save arguments.
    logger.info('Arguments:')
    for key in sorted(args.keys()):
        logger.info('{}: {}'.format(key, args[key]))
    logger.info('')
    if args['logdir']:
        with open(os.path.join(args['logdir'], 'args.json'), 'w') as f:
            json.dump(args, f)

    # Run actual script.
    run(hidden_size=hidden_size, nb_layers=nb_layers, portnum=portnum, **args)
