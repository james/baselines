#!/usr/bin/env python
import argparse
import logging
import os
import tensorflow as tf
import gym
from gym import utils

from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_cont_parallel import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction

#from parallel_env import CustomParallelEnv

#from custom_env import Custom0Env

#from gym.envs.registration import register

#register(
#    id='Custom0-v0',
#    entry_point='custom_env:Custom0Env',
#    max_episode_steps=2000,
#    reward_threshold=1000.0,
#)

def train(env, num_timesteps, timesteps_per_batch, seed, num_cpu, resume, 
          logdir, agentName, desired_kl, gamma, lam,
          portnum, num_parallel
):
    if num_parallel > 1:
        env = CustomParallelEnv(num_parallel)
    else:
        env = gym.make(env)
        env.seed(seed) # Todo: add seed to the random env too

    if logger.get_dir():
        env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"))
    set_global_seeds(seed)
    
    gym.logger.setLevel(logging.WARN)

    with tf.Session(config=tf.ConfigProto()) as session:
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env, policy=policy, vf=vf,
            gamma=gamma, lam=0.97, timesteps_per_batch=timesteps_per_batch,
            desired_kl=desired_kl, resume=resume, logdir=logdir, agentName=agentName, 
            num_timesteps=num_timesteps, animate=True)

        env.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Custom0-v0')
    parser.add_argument('--num-cpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--agentName', type=str, default='Humanoid-ACKTR-256')
    parser.add_argument('--resume', type=int, default=1790)

    parser.add_argument('--num_timesteps', type=int, default=1e7)
    parser.add_argument('--timesteps_per_batch', type=int, default=5000)
    parser.add_argument('--desired_kl', type=float, default=0.002)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)

    parser.add_argument("--portnum", required=False, type=int, default=5000)
    parser.add_argument("--server_ip", required=False, default="localhost")

    parser.add_argument("--num_parallel", type=int, default=1)

    return vars(parser.parse_args())

if __name__ == "__main__":

    args = parse_args()
    utils.portnum = args['portnum']
    utils.server_ip = args['server_ip']
    del args['portnum']
    del args['server_ip']

    train(env='Humanoid-v1', num_timesteps=args['num_timesteps'], timesteps_per_batch=args['timesteps_per_batch'],
          seed=args['seed'], num_cpu=args['num_cpu'], resume=args['resume'], logdir='Humanoid', agentName=args['agentName'], 
          desired_kl=args['desired_kl'], gamma=args['gamma'], lam=args['lam'], 
          portnum=utils.portnum, num_parallel=args['num_parallel']
          )
