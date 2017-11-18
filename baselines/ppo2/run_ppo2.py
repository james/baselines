#!/usr/bin/env python
import argparse
from baselines import bench, logger
from baselines.common.isaac_client import IsaacClient
from baselines.common.custom_env import Custom0Env

USE_ISAAC_COMM = True
USE_BINARY_PROTO = False
USE_COALESCED_ARRAYS = False


class CustomVecEnvSpec:
    def __init__(self):
        self.timestep_limit = 2000
        self.id = 1


class CustomVecEnv:
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
        self.spec = CustomVecEnvSpec()
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

class CustomVecNormalize(VecEnv):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        self.venv = venv
        self._observation_space = self.venv.observation_space
        self._action_space = venv.action_space
        self.ob_rms = RunningMeanStd(shape=self._observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_parallel(vac)
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms: 
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos
    def _obfilt(self, obs):
        if self.ob_rms: 
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs
    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset_parallel()
        return self._obfilt(obs)
    @property
    def action_space(self):
        return self._action_space
    @property
    def observation_space(self):
        return self._observation_space
    def close(self):
        self.venv.close()
    @property
    def num_envs(self):
        return self.venv.n_parallel


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.zeros(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count        
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count        

def test_runningmeanstd():
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
        ]:

        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        assert np.allclose(ms1, ms2)    

def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()

