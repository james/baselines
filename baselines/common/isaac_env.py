import struct

import gym
import numpy as np
import zmq
from gym.spaces import Box


ISAAC_ENDPOINT = "10.60.10.26:5050"

class IsaacClient:

    COMMAND = dict(
		FASTMODE=96,
		REALMODE=97,
        TERMINATE=98,
        START=99,
        RESET=0,
        ACTION=1,
        )

    def __init__(self, endpoint):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect("tcp://{}".format(endpoint))

    def send_start(self):
        req = struct.pack('=Bi', IsaacClient.COMMAND["START"], 1)
        self._socket.send(req)
        message = self._socket.recv()
        dim_obs, dim_actions = struct.unpack('=ii', message)
        self.numa = dim_actions
        self.numo = dim_obs
        return dim_obs, dim_actions

    def send_actions(self, actions):
        req = bytearray()
        req.extend(struct.pack('B', IsaacClient.COMMAND["ACTION"]))
        for i in range(self.numa):
            req.extend(struct.pack('=f', actions[i]))
        self._socket.send(req)

        # print("Wait for reply")
        message = self._socket.recv()

        state = np.zeros([self.numo])

        off = 0
        for i in range(self.numo):
            state[i] = struct.unpack_from('@f', message, offset=off)[0]
            off += 4
        reward = struct.unpack_from('@f', message, offset=off)[0]
        off += 4
        done = struct.unpack_from('B', message, offset=off)[0] > 0

        return state, reward, done

    def send_reset(self):
        req = struct.pack('B', IsaacClient.COMMAND["RESET"])
        self._socket.send(req)
        #  Get the reply.
        message = self._socket.recv()

        states = np.zeros([self.numo])
        off = 0
        for i in range(self.numo):
            states[i] = struct.unpack_from('=f', message, offset=off)[0]
            off += 4
        return states

    def send_fastmode(self):
        req = struct.pack('B', IsaacClient.COMMAND["FASTMODE"])
        self._socket.send(req)
        message = self._socket.recv()
        success = struct.unpack('B', message)
        return success
		
    def send_realmode(self):
        req = struct.pack('B', IsaacClient.COMMAND["REALMODE"])
        self._socket.send(req)
        message = self._socket.recv()
        success = struct.unpack('B', message)
        return success


class IsaacEnv(gym.Env):

    def __init__(self, endpoint="localhost:5000"):
        print("IsaacEnv: Building IsaacClient")
        self._client = IsaacClient(endpoint)

        print("IsaacEnv: Sending Start")
        dim_obs, dim_actions = self._client.send_start()

        self.observation_space = Box(
            -np.ones(dim_obs), np.ones(dim_obs))
        self.action_space = Box(
            -np.ones(dim_actions), -np.ones(dim_actions))

    def _step(self, actions):
        state, reward, done = self._client.send_actions(actions)
        return state, reward, done, {}

    def _reset(self):
        self.T = 0
        return self._client.send_reset()

    def _render(self, mode='human', close=False):
        pass


if __name__ == "__main__":
    env = IsaacEnv(endpoint=ISAAC_ENDPOINT)

    print(env.reset())
