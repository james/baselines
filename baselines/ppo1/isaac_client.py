from __future__ import print_function
import ctypes
import sys
import os
import numpy as np

c_float_p   = ctypes.POINTER(ctypes.c_float)
c_byte_p    = ctypes.POINTER(ctypes.c_uint8)

class IsaacClient:
    def __init__(self, pluginDir = '.'):

        self.numActors = 0
        self.numObservations = 0
        self.numActions = 0

        if sys.platform == "win32":
            pluginName = 'IsaacClient.dll'
        else:
            pluginName = 'IsaacClient.so'
        pluginPath = os.path.join(pluginDir, pluginName)
        if not os.path.isfile(pluginPath):
            raise Exception('*** Plugin %s not found in directory %s' % (pluginName, os.path.abspath(pluginDir)))
        print('Working directory:', os.getcwd())
        print('Loading plugin %s from directory %s' % (pluginName, os.path.abspath(pluginDir)))
        #self.plugin = ctypes.cdll.LoadLibrary(pluginPath)
        self.plugin = np.ctypeslib.load_library(pluginName, os.getcwd())
        #self.plugin.IsaacGetObservations.restype = ctypes.POINTER(ctypes.c_float)

    def Connect(self, numActors):
        if self.plugin.IsaacConnect(int(numActors)):

            self.numActors = self.plugin.IsaacNumActors()
            self.numObservations = self.plugin.IsaacNumObservations()
            self.numActions = self.plugin.IsaacNumActions()

            #obs_buf_t = ctypes.POINTER(ctypes.c_float * self.numActors * self.numObservations)

            print('1')
            #self.plugin.IsaacGetObservations.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_float, shape=(self.numActors, self.numObservations))
            #self.plugin.IsaacGetObservations.restype = np.ctypeslib.ndpointer(dtype=np.float32, shape=(5, 2), flags='CONTIGUOUS')
            #print('fuck', self.plugin.IsaacGetObservations())
            #self.plugin.IsaacGetObservations.restype = ctypes.c_float * (self.numActors * self.numObservations)

            self.plugin.IsaacGetObservations.restype = c_float_p
            self.plugin.IsaacGetRewards.restype = c_float_p
            self.plugin.IsaacGetDeaths.restype = c_byte_p

            print('restype is', self.plugin.IsaacGetObservations.restype)

            print('2')
            self.obsBuffer = self.plugin.IsaacGetObservations()
            self.rewBuffer = self.plugin.IsaacGetRewards()
            self.dieBuffer = self.plugin.IsaacGetDeaths()
            print(self.obsBuffer)
            #print(np.ctypeslib.as_array(self.obsBuffer, shape=(self.numActors, self.numObservations)).reshape(self.numActors, self.numObservations))
            #print(np.frombuffer(ctypes.c_float * 3).from_address(self.plugin.IsaacGetObservations()), np.float32)
            print('3')

            return True
        else:
            return False

    def Disconnect(self):
        self.plugin.IsaacDisconnect()

    def Reset(self):
        if self.plugin.IsaacReset():
            return np.ctypeslib.as_array(self.obsBuffer, shape=(self.numActors, self.numObservations))
        else:
            return None

    def Step(self, actions):
        if self.plugin.IsaacStep(actions.ctypes.data_as(c_float_p)):
            obs = np.ctypeslib.as_array(self.obsBuffer, shape=(self.numActors, self.numObservations))
            rew = np.ctypeslib.as_array(self.rewBuffer, shape=(self.numActors,))
            die = np.ctypeslib.as_array(self.dieBuffer, shape=(self.numActors,))
            return obs, rew, die
        else:
            return None
