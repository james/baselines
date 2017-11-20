#!/usr/bin/env python
import argparse
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench, logger
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
from baselines.common.mpi_fork import mpi_fork

import os.path as osp
import gym, logging
from mpi4py import MPI
from gym import utils, spaces
import numpy as np
from numpy import Inf

import sys

import zmq
#from baselines.common import IsaacClient, Custom0Env, CustomParallelEnv
from baselines.common.isaac_client import IsaacClient
from baselines.common.custom_env import Custom0Env
# from baselines.common.parallel_env import CustomParallelEnv

# from parallel_env import CustomParallelEnv

USE_ISAAC_COMM = True
USE_BINARY_PROTO = False
USE_COALESCED_ARRAYS = False


class CustomParallelEnvSpec:
    def __init__(self):
        self.timestep_limit = 2000
        self.id = 1


class CustomParallelEnv:
    def __init__(self, npar):

        self.n_parallel = npar
        if USE_ISAAC_COMM:
            self.isaac = IsaacClient()
            if not self.isaac.Connect(npar):
                print('*** Failed to connect to simulation server')
                quit()
            numo = self.isaac.numObservations
            numa = self.isaac.numActions

        else:
            self.context = zmq.Context()
            #  Socket to talk to server
            print(("portnum is " + utils.portnum))
            print("Connecting to environment server")
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect("tcp://"+utils.server_ip+":" + utils.portnum)

            if USE_BINARY_PROTO:
                req = struct.pack('=Bi', 99, npar)
                self.socket.send(req)
                print("** Recv")
                message = self.socket.recv()
                numo, numa = struct.unpack('=ii', message)
            else:
                req = "99 " + str(npar)
                self.socket.send_string(req)
                print("** Recv")
                message = self.socket.recv()
                vals = message.split()
                numo = int(vals[0])
                numa = int(vals[1])

        print('Got numo = %d, numa = %d' % (numo, numa))

        minso = np.zeros(numo)
        maxso = np.zeros(numo)

        for i in range(numo):
            minso[i] = -Inf
            maxso[i] = Inf
        minsa = np.zeros(numa)
        maxsa = np.zeros(numa)
        for i in range(numa):
            minsa[i] = -1.0
            maxsa[i] = 1.0
        self.numo = numo
        self.numa = numa
        self.observation_space = spaces.Box(minso, maxso)
        self.action_space = spaces.Box(minsa, maxsa)
        self.spec = CustomParallelEnvSpec()
        print("Done")

    def reset(self):
        #print('+++ single-agent reset')

        global itrnum
        if 'itrnum' in globals():
            pass
        else:
            itrnum = 0
        print ("Itr num = " + str(itrnum))
        itrnum = itrnum + 1
        if USE_ISAAC_COMM:
            obs, _ = self.isaac.Reset()
            return obs
        else:
            if USE_BINARY_PROTO:
                req = struct.pack('B', 0)
                socket.send(req)
                #  Get the reply.
                message = socket.recv()
                if USE_COALESCED_ARRAYS:
                    buf = memoryview(message)
                    obs = np.frombuffer(buf, dtype=np.float32, count=self.numo)
                    return obs
                else:
                    ss = np.zeros([self.numo])
                    off = 0
                    for i in range(self.numo):
                        ss[i] = struct.unpack_from('=f', message, offset=off)[0]
                        off += 4
                    return ss
            else:
                req = "0"
                # print("** Send")
                socket.send_string(req)
                #  Get the reply.
                ss = np.zeros([self.numo])
                # print("** Recv")
                message = socket.recv()
                vals = message.split()
                for i in range(self.numo):
                    ss[i] = vals[i]
                return ss

    def step(self, actions):

        #print('+++ Single-agent step')
        if USE_ISAAC_COMM:
            obs, rew, die, _ = self.isaac.Step(actions)
            return obs, rew, die, {}
        else:
            if USE_BINARY_PROTO:
                if USE_COALESCED_ARRAYS:
                    req = bytearray()
                    req.extend(struct.pack('B', 1))
                    req.extend(actions)
                    socket.send(req)

                    # print("Wait for reply")
                    message = socket.recv()

                    buf = memoryview(message)
                    obs = np.frombuffer(buf, dtype=np.float32, count=self.numo)
                    # obs = obs.reshape((1, self.numo))
                    # rew = np.frombuffer(buf, dtype=np.float32, count=self.n_parallel, offset=self.n_parallel*self.numo*4)
                    # done = np.frombuffer(buf, dtype=np.uint8, count=self.n_parallel, offset=self.n_parallel*self.numo*4 + self.n_parallel*4)
                    off = self.numo * 4
                    r = struct.unpack_from('@f', message, offset=off)[0]
                    off += 4
                    done = struct.unpack_from('B', message, offset=off)[0] > 0
                    return obs, r, done, {}
                else:
                    req = bytearray()
                    req.extend(struct.pack('B', 1))
                    for i in range(self.numa):
                        req.extend(struct.pack('=f', actions[i]))
                    socket.send(req)

                    # print("Wait for reply")
                    message = socket.recv()

                    ss = np.zeros([self.numo])
                    # r = np.zeros([self.n_parallel])
                    # done = [False] * self.n_parallel
                    off = 0
                    for i in range(self.numo):
                        ss[i] = struct.unpack_from('@f', message, offset=off)[0]
                        off += 4
                    r = struct.unpack_from('@f', message, offset=off)[0]
                    off += 4
                    done = struct.unpack_from('B', message, offset=off)[0] > 0
                    # off += 1
                    return ss, r, done, {}
            else:
                req = "1"
                for i in range(self.numa):
                    req += " " + str(actions[i])

                # print("Sending request %s" % req)
                # print("** Send")
                socket.send_string(req)
                # print("** Done send")

                #  Get the reply.
                ss = np.zeros([self.numo])
                # print("Wait for reply")
                # print("** Recv")
                message = socket.recv()
                # print("** Done recv")

                # print("Get reply %s" % message)
                vals = message.split()
                for i in range(self.numo):
                    ss[i] = vals[i]
                r = float(vals[self.numo])
                # print "Val is " + vals[self.numo+1];
                done = bool(int(vals[self.numo + 1]))
                # print("Done is " + str(done));
                return ss, r, done, {}

    def reset_parallel(self):
        global itrnum
        if 'itrnum' in globals():
            pass
        else:
            itrnum = 0
        print(("Itr num = " + str(itrnum)))
        itrnum = itrnum + 1

        if USE_ISAAC_COMM:
            obs, _ = self.isaac.Reset()
            return obs
        else:
            if USE_BINARY_PROTO:
                req = struct.pack('B', 0)
                self.socket.send(req)
                #  Get the reply.
                message = self.socket.recv()
                if USE_COALESCED_ARRAYS:
                    buf = memoryview(message)
                    obs = np.frombuffer(buf, dtype=np.float32, count=self.n_parallel * self.numo)
                    obs = obs.reshape((self.n_parallel, self.numo))
                    return obs
                else:
                    ss = np.zeros([self.n_parallel, self.numo])
                    off = 0
                    for k in range(self.n_parallel):
                        for i in range(self.numo):
                            ss[k][i] = struct.unpack_from('=f', message, offset=off)[0]
                            off += 4
                        off += 4 + 1  # skip reward and die/reset flag
                    return ss
            else:
                self.socket.send_string("0")
                #  Get the reply.
                ss = np.zeros([self.n_parallel, self.numo])
                # print("** Recv")
                message = self.socket.recv()
                vals = message.split()
                for k in range(self.n_parallel):
                    for i in range(self.numo):
                        ss[k][i] = vals[k * (self.numo + 2) + i]
                return ss

    def step_parallel(self, actions):
        if USE_ISAAC_COMM:
            obs, rew, die, _ = self.isaac.Step(actions)
            return obs, rew, die, [{}] * self.n_parallel
        else:
            if USE_BINARY_PROTO:
                if USE_COALESCED_ARRAYS:
                    req = bytearray()
                    req.extend(struct.pack('B', 1))
                    req.extend(actions)
                    self.socket.send(req)

                    # print("Wait for reply")
                    message = self.socket.recv()

                    buf = memoryview(message)
                    obs = np.frombuffer(buf, dtype=np.float32, count=self.n_parallel * self.numo)
                    obs = obs.reshape((self.n_parallel, self.numo))
                    rew = np.frombuffer(buf, dtype=np.float32, count=self.n_parallel,
                                        offset=self.n_parallel * self.numo * 4)
                    done = np.frombuffer(buf, dtype=np.uint8, count=self.n_parallel,
                                         offset=self.n_parallel * self.numo * 4 + self.n_parallel * 4)
                    # print(done)
                    return obs, rew, done, [{}] * self.n_parallel
                else:
                    req = bytearray()
                    req.extend(struct.pack('B', 1))
                    for k in range(self.n_parallel):
                        for i in range(self.numa):
                            req.extend(struct.pack('=f', actions[k][i]))
                    self.socket.send(req)

                    # print("Wait for reply")
                    message = self.socket.recv()

                    ss = np.zeros([self.n_parallel, self.numo])
                    r = np.zeros([self.n_parallel])
                    done = [False] * self.n_parallel
                    off = 0
                    for k in range(self.n_parallel):
                        for i in range(self.numo):
                            ss[k][i] = struct.unpack_from('@f', message, offset=off)[0]
                            off += 4
                        r[k] = struct.unpack_from('@f', message, offset=off)[0]
                        off += 4
                        done[k] = struct.unpack_from('B', message, offset=off)[0] > 0
                        off += 1
                    return ss, r, done, [{}] * self.n_parallel
            else:
                req = "1"
                for k in range(self.n_parallel):
                    for i in range(self.numa):
                        req += " " + str(actions[k][i])
                self.socket.send_string(req)

                #  Get the reply.
                ss = np.zeros([self.n_parallel, self.numo])
                # print("Wait for reply")
                # print("** Recv")
                message = self.socket.recv()
                # print("Get reply %s" % message)
                vals = message.split()
                r = np.zeros([self.n_parallel])
                done = []
                for k in range(self.n_parallel):
                    for i in range(self.numo):
                        ss[k][i] = vals[k * (self.numo + 2) + i]
                    r[k] = float(vals[k * (self.numo + 2) + self.numo])

                    done.append(bool(int(vals[k * (self.numo + 2) + self.numo + 1])))
                # print("Done is " + str(done));
                return ss, r, done, [{}] * self.n_parallel

    def render(self):
        return


def train(env_id, num_timesteps, timesteps_per_batch, seed, num_cpu, resume,
          agentName, logdir, hid_size, num_hid_layers, noisy_nets, clip_param,
          entcoeff, optim_epochs, optim_batchsize, optim_stepsize, optim_schedule,
          desired_kl, gamma, lam, portnum, num_parallel
):
    from baselines.ppo1 import mlp_policy, pposgd_parallel
    print("num cpu = " + str(num_cpu))
    if (num_cpu > 1) and (num_parallel > 1):
        print("num_cpu > 1 and num_parallel > 0 can't be used together at the moment!")
        exit(0)

    whoami  = mpi_fork(num_cpu)
    if whoami == "parent": return
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()

    if rank != 0: logger.set_level(logger.DISABLED)
    utils.portnum = portnum + rank
    workerseed = seed + 10000 * rank

    if utils.server_list != "":
        servers = utils.server_list.split(",")
        num_thread = utils.num_thread_list.split(",")
        tmp = 0
        a = 0
        snum = -1
        num_total = 0
        for t in num_thread:
            num_total += int(t)

        for t in num_thread:
            if rank < tmp + int(t):
                snum = a
                break
            tmp += int(t)
            a += 1
        if num_total != num_cpu:
            print("Sum of num_thread_list must be equal to num_cpu")
            quit()
        print("Connect to tcp://" + servers[snum] + ":" + str(utils.portnum))
        utils.server_ip = servers[snum]

    set_global_seeds(workerseed)
    if num_parallel > 1:
        env = CustomParallelEnv(num_parallel)
    else:
        env = gym.make(env_id)
        env.seed(seed)

    if logger.get_dir():
        if num_parallel <= 1:
            env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))

    def policy_fn(name, ob_space, ac_space, noisy_nets=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hid_size, num_hid_layers=num_hid_layers, noisy_nets=noisy_nets)

    gym.logger.setLevel(logging.WARN)
    pposgd_parallel.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=timesteps_per_batch,
            clip_param=clip_param, entcoeff=entcoeff,
            optim_epochs=optim_epochs, optim_stepsize=optim_stepsize,
            optim_batchsize=optim_batchsize, schedule=optim_schedule,
            desired_kl=desired_kl, gamma=gamma, lam=lam,
            resume=resume, noisy_nets=noisy_nets,
            agentName=agentName, logdir=logdir,
            num_parallel=num_parallel, num_cpu=num_cpu
        )
    if num_parallel <= 1:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='Custom0-v0')
    parser.add_argument('--num-cpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--logdir', type=str, default='Humanoid')
    parser.add_argument('--agentName', type=str, default='Humanoid_128')
    parser.add_argument('--resume', type=int, default=0)

    parser.add_argument('--num_timesteps', type=int, default=1e7)
    parser.add_argument('--timesteps_per_batch', type=int, default=4096)
    parser.add_argument('--hid_size', type=int, default=128)
    parser.add_argument('--num_hid_layers', type=int, default=2)
    boolean_flag(parser, 'noisy_nets', default=False)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entcoeff', type=float, default=0.0)
    parser.add_argument('--optim_epochs', type=int, default=20)
    parser.add_argument('--optim_batchsize', type=int, default=64)
    parser.add_argument('--optim_stepsize', type=float, default=5e-4) # 3e-4 isefault for single agent training with constant schedule
    parser.add_argument('--optim_schedule', type=str, default='constant') # Other options: 'adaptive', 'linear'
    parser.add_argument('--desired_kl', type=float, default=0.02)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)

    parser.add_argument('--portnum', required=False, type=int, default=5050)
    parser.add_argument('--server_ip', required=False, default='localhost')
    parser.add_argument('--num_parallel', type=int, default=0)
    parser.add_argument('--server_list', required=False, type=str, default='')
    parser.add_argument('--num_thread_list', required=False, type=str, default='') #Must either be "" or summed to num_cpu

    return vars(parser.parse_args())

def main():
    args = parse_args()
    utils.portnum = args['portnum']
    utils.server_ip = args['server_ip']
    utils.server_list = args['server_list']
    utils.num_thread_list = args['num_thread_list']

    del args['server_list']
    del args['num_thread_list']
    del args['portnum']
    del args['server_ip']

    train(args['env_id'], num_timesteps=args['num_timesteps'], timesteps_per_batch=args['timesteps_per_batch'],
          seed=args['seed'], num_cpu=args['num_cpu'], resume=args['resume'], agentName=args['agentName'], 
          logdir=args['logdir'], hid_size=args['hid_size'], num_hid_layers=args['num_hid_layers'],
          noisy_nets=args['noisy_nets'], clip_param=args['clip_param'], entcoeff=args['entcoeff'],
          optim_epochs=args['optim_epochs'], optim_batchsize=args['optim_batchsize'],
          optim_stepsize=args['optim_stepsize'], optim_schedule=args['optim_schedule'],
          desired_kl=args['desired_kl'], gamma=args['gamma'], lam=args['lam'], 
          portnum=utils.portnum, num_parallel=args['num_parallel']
          )


if __name__ == '__main__':
    main()
