#import sys
import gym, logging
from mpi4py import MPI
from gym import utils, spaces
import numpy as np
from numpy import Inf
import zmq
from . import isaac_client
#from isaac_client import IsaacClient

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
            self.socket.connect("tcp://" + utils.server_ip+":" + utils.portnum)

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
            obs = self.isaac.Reset()
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
            obs, rew, die = self.isaac.Step(actions)
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
            obs = self.isaac.Reset()
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
            obs, rew, die = self.isaac.Step(actions)
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