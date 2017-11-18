import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import zmq
import sys
from numpy import Inf
import struct

context = zmq.Context()
#  Socket to talk to server
pn = 5050
s_ip = "localhost"
print("portnum is x " + str(pn))
print("server ip is " + s_ip)
print("Connecting to environment server")
socket = context.socket(zmq.REQ)
socket.connect("tcp://" + s_ip + ":" + str(pn))
print("done connect")

USE_BINARY_PROTO        = True
USE_COALESCED_ARRAYS    = False

class Custom0Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        print('+++ Initializing gym env')

        if USE_BINARY_PROTO:
            req = struct.pack('=Bi', 99, 1)
            socket.send(req)
            message = socket.recv()
            numo, numa = struct.unpack('=ii', message)
        else:
            req = "99 1"
            socket.send_string(req)
            message = socket.recv()
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

    def _step(self, actions):

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
                #obs = obs.reshape((1, self.numo))
                #rew = np.frombuffer(buf, dtype=np.float32, count=self.n_parallel, offset=self.n_parallel*self.numo*4)
                #done = np.frombuffer(buf, dtype=np.uint8, count=self.n_parallel, offset=self.n_parallel*self.numo*4 + self.n_parallel*4)
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

                ss = np.zeros([self.numo]);
                #r = np.zeros([self.n_parallel])
                #done = [False] * self.n_parallel
                off = 0
                for i in range(self.numo):
                    ss[i] = struct.unpack_from('@f', message, offset=off)[0]
                    off += 4
                r = struct.unpack_from('@f', message, offset=off)[0]
                off += 4
                done = struct.unpack_from('B', message, offset=off)[0] > 0
                #off += 1
                return ss, r, done, {}
        else:
            req = "1";
            for i in range(self.numa):
                req += " " + str(actions[i])

            # print("Sending request %s" % req)
            #print("** Send")
            socket.send_string(req)
            #print("** Done send")

            #  Get the reply.
            ss = np.zeros([self.numo]);
            # print("Wait for reply")
            #print("** Recv")
            message = socket.recv()
            #print("** Done recv")

            # print("Get reply %s" % message)
            vals = message.split()
            for i in range(self.numo):
                ss[i] = vals[i]
            r = float(vals[self.numo])
            # print "Val is " + vals[self.numo+1];
            done = bool(int(vals[self.numo + 1]))
            # print("Done is " + str(done));
            return ss, r, done, {}

    def _reset(self):

        global itrnum
        if 'itrnum' in globals():
            pass
        else:
            itrnum = 0
        print ("Itr num = " + str(itrnum))
        itrnum = itrnum + 1

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
                ss = np.zeros([self.numo]);
                off = 0
                for i in range(self.numo):
                    ss[i] = struct.unpack_from('=f', message, offset=off)[0]
                    off += 4
                return ss
        else:
            req = "0";
            # print("** Send")
            socket.send_string(req)
            #  Get the reply.
            ss = np.zeros([self.numo]);
            # print("** Recv")
            message = socket.recv()
            vals = message.split()
            for i in range(self.numo):
                ss[i] = vals[i]
            return ss

    def _render(self, mode='human', close=False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        # outfile.write("X");
        # outfile.write("\n");
        if (mode != 'human'):
            return outfile

