from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype


class Classifier(object):

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, num_input, num_output, hid_size, num_hid_layers, reward_scale, learning_rate):
        sequence_length = None

        self.input = U.get_placeholder(name="input", dtype=tf.float32, shape=[None,num_input])
        self.label = U.get_placeholder(name="label", dtype=tf.float32, shape=[None,num_output])
        last_out = self.input

        for i in range(num_hid_layers - 1):
            last_out = tf.nn.elu(U.dense(last_out, hid_size, "cls%i" % (i + 1), weight_init=U.normc_initializer(1.0)))
        last_out = tf.nn.elu(U.dense(last_out, hid_size, "cls%i" % (num_hid_layers), weight_init=U.normc_initializer(1.0)))
        self.logits = U.dense(last_out, num_output, "cls_final", weight_init=U.normc_initializer(1.0))
        self.pred = tf.nn.softmax(self.logits)
        self.prob_fake = tf.slice(self.pred,[0,num_output-1],[-1,1]) # Extract the last column
        self.scope = tf.get_variable_scope().name
        all_var_list = self.get_trainable_variables()
        var_list = [v for v in all_var_list if v.name.split("/")[2].startswith("w")]
        #name_list = [v.name for v in all_var_list]
        #print(str(name_list))
        #name_list = [v.name for v in var_list]
        #print(str(name_list))
        #print("There are "+str(len(var_list))+" vars out of "+str(len(all_var_list)))
        #exit(0)
        regularizer_weight = 0.0001
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label))
        for v in var_list:
            self.loss += tf.nn.l2_loss(v)*regularizer_weight
        print("Learning rate is "+str(learning_rate)+" reward scale is "+str(reward_scale))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.optimizer_first = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        self.extra_reward = -tf.log(self.prob_fake)*reward_scale
        #self._extra_reward = U.function([self.input], [self.extra_reward, self.pred, self.prob_fake])
        self._extra_reward = U.function([self.input], [self.extra_reward])

        correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def compute_extra_reward(self, input):
        #rew, pred, pf = self._extra_reward(input)
        #print("Pred fake = " + str(pred))
        #print("Prob fake = " + str(pf))
        #return rew
        return self._extra_reward(input)


    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

