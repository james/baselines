from sklearn.neighbors import NearestNeighbors
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import os
import json
import pickle
import random



class MyReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._full = False

    def __len__(self):
        return len(self._storage)

    def add(self, exts):
        if not self._full:
            num = len(exts)
            if len(self._storage) + num >= self._maxsize:
                num = self._maxsize - len(self._storage)
                self._storage.extend(exts[0:num])
                exts = exts[num:len(exts)]
                self._full = True
            else:
                self._storage.extend(exts)
        num = len(exts)
        if self._full:

            if self._next_idx+num <= self._maxsize:
                self._storage[self._next_idx:self._next_idx+num] = exts
            else:
                self._storage[self._next_idx:self._maxsize] = exts[0:self._maxsize-self._next_idx]
                self._storage[0:num-(self._maxsize-self._next_idx)] = exts[self._maxsize-self._next_idx:num]
        self._next_idx = (self._next_idx + num) % self._maxsize

    def sample(self, batch_size):
        choices = np.random.choice(len(self._storage), batch_size, replace=True)
        return np.array([self._storage[i] for i in choices])


def traj_segment_generator(pi, env, disc, horizon, stochastic, num_parallel, num_cpu, rank, ob_size, ac_size, com, num_extra, iters_so_far, use_distance, nn):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype

    new = np.ones(num_parallel, 'int32')  # marks if we're on first timestep of an episode

    ob,ex = env.reset_parallel()
    cur_ep_ret = [0 for _ in range(num_parallel)]
    cur_ep_len = [0 for _ in range(num_parallel)]
    ep_rets = [[] for _ in range(num_parallel)] # returns of completed episodes in this segment
    ep_lens = [[] for _ in range(num_parallel)] # lengths of ...

    # Initialize history arrays
    #obs = np.array([[ob for _ in range(horizon)] for __ in range(num_parallel)])
    obs = np.zeros((num_parallel, horizon, ob.shape[1]),'float32')
    extra = np.zeros((num_parallel, horizon, num_extra),'float32')
    rews = np.zeros((num_parallel, horizon), 'float32')
    ext_rews = np.zeros((num_parallel, horizon), 'float32')
    vpreds = np.zeros((num_parallel, horizon), 'float32')
    news = np.zeros((num_parallel, horizon), 'int32')
    #acs = np.array([[ac for _ in range(horizon)] for __ in range(num_parallel)])
    acs = np.zeros((num_parallel, horizon, ac.shape[0]),'float32')
    prevacs = acs.copy()
    ac = np.zeros((num_parallel, ac.shape[0]),'float32')
    num_yield = 0
    while True:
        prevac = ac
        ac, vpred = pi.act_parallel(stochastic, ob)
        #print("t = " + str(t) + " ac = " + str(ac) + " vpred = " + str(vpred))
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:

            obs_all = np.reshape(obs, (horizon * num_parallel,-1))
            rews_all = np.reshape(rews, horizon * num_parallel)
            extra_all = np.reshape(extra, (horizon * num_parallel, -1))
            mr = 0
            if (num_yield > 0) or (use_distance):
                ext_rews_all = np.reshape(ext_rews, horizon * num_parallel)
                mr = np.mean(ext_rews_all)
                print("Average extra reward over the whole iteration = " + str(mr))
            #for j in range(num_parallel):
            #    vpreds[j][horizon] = vpred[j] * (1 - new[j])
            #vpreds_all = np.reshape(vpreds, horizon * num_parallel)
            vpreds_all = vpreds
            news_all = np.reshape(news, horizon * num_parallel)
            acs_all = np.reshape(acs, (horizon * num_parallel, -1))
            prevacs_all = np.reshape(prevacs, (horizon * num_parallel,-1))
            ep_rets_all = [item for sublist in ep_rets for item in sublist]
            ep_lens_all = [item for sublist in ep_lens for item in sublist]
            yield {"ob" : obs_all, "rew" : rews_all, "vpred" : vpreds_all, "new" : news_all,
                    "ac" : acs_all, "prevac" : prevacs_all, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets_all, "ep_lens" : ep_lens_all, "extra":extra_all, "mean_ext_rew":mr}
            num_yield+=1
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            for j in range(num_parallel):
                ep_rets[j] = []
                ep_lens[j] = []
        i = t % horizon
        for j in range(num_parallel):
            extra[j][i] = ex[j]
            obs[j][i] = ob[j]
            vpreds[j][i] = vpred[j]
            news[j][i] = new[j]
            acs[j][i] = ac[j]
            prevacs[j][i] = prevac[j]
        ob, rew, new, ex, _ = env.step_parallel(ac)

        if (num_yield > 0) or (use_distance):
            if use_distance:
                distances, _ = nn.kneighbors(ex)
                #print("Shape = " + str(distances.shape) + " Distances = " + str(distances))
                distances = np.reshape(distances,rew.shape)
                ext_rew = (13-distances)/4

                #print(str(ext_rew.shape))
            else:
                ext_rew = np.reshape(disc.compute_extra_reward(ex),rew.shape)

            #print("Average ext reward = " + np.mean(ext_rew))
            #print("rew shape is " + str(rew.shape) + " ext_rew shape is " + str(ext_rew.shape))
            rew += ext_rew # Extra reward from discriminator

        #print("t = " + str(t) + " ob = " + str(ob) + " rew = " + str(rew) + " new = " + str(new))
        for j in range(num_parallel):
            rews[j][i] = rew[j]
            if num_yield > 0:
                ext_rews[j][i] = ext_rew[j]
            cur_ep_ret[j] += rew[j]
            cur_ep_len[j] += 1
            if new[j]:
                ep_rets[j].append(cur_ep_ret[j])
                ep_lens[j].append(cur_ep_len[j])
                cur_ep_ret[j] = 0
                cur_ep_len[j] = 0
                #ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam, horizon, num_parallel, num_cpu):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    if (num_parallel == 0) or (num_cpu == num_parallel):
        new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-new[t+1]
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    else:
        #vpred = np.reshape(np.concatenate((seg["vpred"], np.reshape(seg["nextvpred"], (num_parallel,1)) ), axis=1), (horizon + 1) * num_parallel)
        new = seg["new"]
        vpred = seg["vpred"] = np.reshape(seg["vpred"], horizon * num_parallel)
        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            if t % horizon == horizon-1: # last time step of an agent
                nonterminal = 1
                e = t // horizon
                delta = rew[t] + gamma * seg["nextvpred"][e] * nonterminal - vpred[t]
            else:
                nonterminal = 1 - new[t + 1]
                delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    #print("convpred = " + str(vpred))
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_func, disc, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        logdir=".",
        agentName="PPO-Agent",
        resume = 0,
        num_parallel=0,
        num_cpu=1,
        num_extra=0,
        gan_batch_size=128,
        gan_num_epochs=5,
        gan_display_step=40,
        resume_disc=0,
        resume_non_disc=0,
        mocap_path="",
        gan_replay_buffer_size=1000000,
        gan_prob_to_put_in_replay = 0.01,
        gan_reward_to_retrain_discriminator = 5,
        use_distance = 0,
        use_blend = 0
        ):
    # Deal with GAN
    if not use_distance:
        replay_buf = MyReplayBuffer(gan_replay_buffer_size)
    data = np.loadtxt(mocap_path+".dat") #"D:/p4sw/devrel/libdev/flex/dev/rbd/data/bvh/motion_simple.dat");
    label = np.concatenate((np.ones((data.shape[0],1)), np.zeros((data.shape[0],1))), axis=1)

    print("Real data label = " + str(label))

    mocap_set = Dataset(dict(data=data, label=label), shuffle=True)

    # Setup losses and stuff
    # ----------------------------------------
    rank = MPI.COMM_WORLD.Get_rank()
    ob_space = env.observation_space
    ac_space = env.action_space

    ob_size = ob_space.shape[0]
    ac_size = ac_space.shape[0]

    #print("rank = " + str(rank) + " ob_space = "+str(ob_space.shape) + " ac_space = "+str(ac_space.shape))
    #exit(0)
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vfloss1 = tf.square(pi.vpred - ret)
    vpredclipped = oldpi.vpred + tf.clip_by_value(pi.vpred - oldpi.vpred, -clip_param, clip_param)
    vfloss2 = tf.square(vpredclipped - ret)
    vf_loss = .5 * U.mean(tf.maximum(vfloss1, vfloss2)) # we do the same clipping-based trust region for the value function
    #vf_loss = U.mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    sess = tf.get_default_session()

    avars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    non_disc_vars = [a for a in avars if not a.name.split("/")[0].startswith("discriminator")]
    disc_vars = [a for a in avars if a.name.split("/")[0].startswith("discriminator")]
    #print(str(non_disc_names))
    #print(str(disc_names))
    #exit(0)
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    disc_saver = tf.train.Saver(disc_vars,max_to_keep=None)
    non_disc_saver = tf.train.Saver(non_disc_vars, max_to_keep=None)
    saver = tf.train.Saver(max_to_keep=None)
    if resume > 0:
        saver.restore(tf.get_default_session(), os.path.join(os.path.abspath(logdir), "{}-{}".format(agentName, resume)))
        if not use_distance:
            if os.path.exists(logdir + "\\" +'replay_buf_'+str(int(resume / 100)*100)+'.pkl'):
                print("Load replay buf")
                with open(logdir + "\\" +'replay_buf_'+str(int(resume / 100)*100)+'.pkl', 'rb') as f:
                    replay_buf = pickle.load(f)
            else:
                print("Can't load replay buf "+logdir + "\\" +'replay_buf_'+str(int(resume / 100)*100)+'.pkl')
    iters_so_far = resume

    if resume_non_disc > 0:
        non_disc_saver.restore(tf.get_default_session(), os.path.join(os.path.abspath(logdir), "{}-{}".format(agentName + "_non_disc", resume_non_disc)))
        iters_so_far = resume_non_disc


    if use_distance:
        print("Use distance")
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data)
    else:
        nn = None
    seg_gen = traj_segment_generator(pi, env, disc, timesteps_per_batch, stochastic=True, num_parallel=num_parallel, num_cpu=num_cpu, rank=rank, ob_size=ob_size, ac_size=ac_size,com=MPI.COMM_WORLD, num_extra=num_extra, iters_so_far=iters_so_far, use_distance=use_distance,nn=nn)

    if resume_disc > 0:
        disc_saver.restore(tf.get_default_session(), os.path.join(os.path.abspath(logdir), "{}-{}".format(agentName + "_disc", resume_disc)))

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"
    logF = open(logdir + "\\" + 'log.txt', 'a')
    logR = open(logdir + "\\" + 'log_rew.txt', 'a')
    logStats = open(logdir + "\\" + 'log_stats.txt', 'a')
    if os.path.exists(logdir + "\\" +'ob_list_'+str(rank)+'.pkl'):
        with open(logdir + "\\" +'ob_list_'+str(rank)+'.pkl', 'rb') as f:
            ob_list = pickle.load(f)
    else:
        ob_list = []

    dump_training = 0
    learn_from_training = 0
    if dump_training:
        # , "mean": pi.ob_rms.mean, "std": pi.ob_rms.std
        saverRMS = tf.train.Saver({"_sum": pi.ob_rms._sum, "_sumsq": pi.ob_rms._sumsq, "_count": pi.ob_rms._count})
        saverRMS.save(tf.get_default_session(), os.path.join(os.path.abspath(logdir), "rms.tf"))

        ob_np_a = np.asarray(ob_list)
        ob_np = np.reshape(ob_np_a, (-1,ob_size))
        [vpred, pdparam] = pi._vpred_pdparam(ob_np)

        print("vpred = " + str(vpred))
        print("pd_param = " + str(pdparam))
        with open('training.pkl', 'wb') as f:
            pickle.dump(ob_np, f)
            pickle.dump(vpred, f)
            pickle.dump(pdparam, f)
        exit(0)
    if learn_from_training:
        # , "mean": pi.ob_rms.mean, "std": pi.ob_rms.std


        with open('training.pkl', 'rb') as f:
            ob_np = pickle.load(f)
            vpred = pickle.load(f)
            pdparam = pickle.load(f)
        num = ob_np.shape[0]
        for i in range(num):
            xp = ob_np[i][1]
            ob_np[i][1] = 0.0
            ob_np[i][18] -= xp
            ob_np[i][22] -= xp
            ob_np[i][24] -= xp
            ob_np[i][26] -= xp
            ob_np[i][28] -= xp
            ob_np[i][30] -= xp
            ob_np[i][32] -= xp
            ob_np[i][34] -= xp
        print("ob_np = " + str(ob_np))
        print("vpred = " + str(vpred))
        print("pdparam = " + str(pdparam))
        batch_size = 128

        y_vpred = tf.placeholder(tf.float32, [batch_size, ])
        y_pdparam = tf.placeholder(tf.float32, [batch_size, pdparam.shape[1]])

        vpred_loss = U.mean(tf.square(pi.vpred - y_vpred))
        vpdparam_loss = U.mean(tf.square(pi.pdparam - y_pdparam))

        total_train_loss = vpred_loss + vpdparam_loss
        #total_train_loss = vpdparam_loss
        #total_train_loss = vpred_loss
        #coef = 0.01
        #dense_all = U.dense_all
        #for a in dense_all:
        #   total_train_loss += coef * tf.nn.l2_loss(a)
        #total_train_loss = vpdparam_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_train_loss)
        d = Dataset(dict(ob=ob_np, vpred=vpred, pdparam=pdparam), shuffle=not pi.recurrent)
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        saverRMS = tf.train.Saver({"_sum": pi.ob_rms._sum, "_sumsq": pi.ob_rms._sumsq, "_count": pi.ob_rms._count})
        saverRMS.restore(tf.get_default_session(), os.path.join(os.path.abspath(logdir), "rms.tf"))
        if resume > 0:
            saver.restore(tf.get_default_session(),
                          os.path.join(os.path.abspath(logdir), "{}-{}".format(agentName, resume)))

        for q in range(100):
            sumLoss = 0
            for batch in d.iterate_once(batch_size):
                tl, _ = sess.run([total_train_loss, optimizer], feed_dict={pi.ob: batch["ob"], y_vpred: batch["vpred"], y_pdparam:batch["pdparam"]})
                sumLoss += tl
            print("Iteration " + str(q)+ " Loss = " + str(sumLoss))
        assign_old_eq_new()  # set old parameter values to new parameter values

        # Save as frame 1
        try:
            saver.save(tf.get_default_session(), os.path.join(logdir, agentName), global_step=1)
        except:
            pass
        #exit(0)
    if resume > 0:
        firstTime = False
    else:
        firstTime = True

    # Check accuracy
    #amocap = sess.run([disc.accuracy],
    #                feed_dict={disc.input: data,
    #                           disc.label: label})
    #print("Mocap accuracy = " + str(amocap))
    #print("Mocap label is " + str(label))

    #adata = np.array(replay_buf._storage)
    #print("adata shape = " + str(adata.shape))
    #alabel = np.concatenate((np.zeros((adata.shape[0], 1)), np.ones((adata.shape[0], 1))), axis=1)

    #areplay = sess.run([disc.accuracy],
    #                feed_dict={disc.input: adata,
    #                           disc.label: alabel})
    #print("Replay accuracy = " + str(areplay))
    #print("Replay label is " + str(alabel))
    #exit(0)
    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()

        add_vtarg_and_adv(seg, gamma, lam, timesteps_per_batch, num_parallel, num_cpu)
        #print(" ob= " + str(seg["ob"])+ " rew= " + str(seg["rew"])+ " vpred= " + str(seg["vpred"])+ " new= " + str(seg["new"])+ " ac= " + str(seg["ac"])+ " prevac= " + str(seg["prevac"])+ " nextvpred= " + str(seg["nextvpred"])+ " ep_rets= " + str(seg["ep_rets"])+ " ep_lens= " + str(seg["ep_lens"]))

        #exit(0)
        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret, extra = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["extra"]

        #ob_list.append(ob.tolist())
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            #print(str(losses))
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        rewmean = np.mean(rewbuffer)
        logger.record_tabular("EpRewMean", rewmean)
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        # Train discriminator
        if not use_distance:
            print("Put in replay buf " +str((int)(gan_prob_to_put_in_replay*extra.shape[0] + 1)) )
            replay_buf.add(extra[np.random.choice(extra.shape[0], (int)(gan_prob_to_put_in_replay*extra.shape[0] + 1), replace=True)])
            #if iters_so_far == 1:
            if not use_blend:
                if firstTime:
                    firstTime = False
                    # Train with everything we got
                    lb = np.concatenate((np.zeros((extra.shape[0],1)),np.ones((extra.shape[0],1))),axis=1)
                    extra_set = Dataset(dict(data=extra,label=lb), shuffle=True)
                    for e in range(10):
                        i = 0
                        for mbatch in mocap_set.iterate_once(gan_batch_size):
                            batch = extra_set.next_batch(gan_batch_size)
                            _, l = sess.run([disc.optimizer_first, disc.loss],
                                            feed_dict={disc.input: np.concatenate((mbatch['data'], batch['data'])),
                                                       disc.label: np.concatenate((mbatch['label'], batch['label']))})
                            i = i + 1
                            # Display logs per step
                            if i % gan_display_step == 0 or i == 1:
                                print('discriminator epoch %i Step %i: Minibatch Loss: %f' % (e, i, l))
                        print('discriminator epoch %i Step %i: Minibatch Loss: %f' % (e, i, l))
                if seg['mean_ext_rew'] > gan_reward_to_retrain_discriminator:
                    for e in range(gan_num_epochs):
                        i = 0
                        for mbatch in mocap_set.iterate_once(gan_batch_size):
                            data = replay_buf.sample(mbatch['data'].shape[0])
                            lb = np.concatenate((np.zeros((data.shape[0], 1)), np.ones((data.shape[0], 1))), axis=1)
                            _, l = sess.run([disc.optimizer, disc.loss],
                                        feed_dict={disc.input: np.concatenate((mbatch['data'], data)),
                                                   disc.label: np.concatenate((mbatch['label'], lb))})
                            i = i + 1
                            # Display logs per step
                            if i % gan_display_step == 0 or i == 1:
                                print('discriminator epoch %i Step %i: Minibatch Loss: %f' % (e, i, l))
                        print('discriminator epoch %i Step %i: Minibatch Loss: %f' % (e, i, l))
            else:
                if firstTime:
                    firstTime = False
                    # Train with everything we got
                    extra_set = Dataset(dict(data=extra), shuffle=True)
                    for e in range(10):
                        i = 0
                        for mbatch in mocap_set.iterate_once(gan_batch_size):
                            batch = extra_set.next_batch(gan_batch_size)
                            bf = np.random.uniform(0,1,(gan_batch_size,1))
                            onembf = 1-bf
                            my_label = np.concatenate((bf,onembf),axis=1)
                            my_data = np.multiply(mbatch['data'], bf) + np.multiply(batch['data'], onembf)
                            _, l = sess.run([disc.optimizer_first, disc.loss],
                                            feed_dict={disc.input: my_data,
                                                       disc.label: my_label})
                            i = i + 1
                            # Display logs per step
                            if i % gan_display_step == 0 or i == 1:
                                print('discriminator epoch %i Step %i: Minibatch Loss: %f' % (e, i, l))
                        print('discriminator epoch %i Step %i: Minibatch Loss: %f' % (e, i, l))
                if seg['mean_ext_rew'] > gan_reward_to_retrain_discriminator:
                    for e in range(gan_num_epochs):
                        i = 0
                        for mbatch in mocap_set.iterate_once(gan_batch_size):
                            data = replay_buf.sample(mbatch['data'].shape[0])

                            bf = np.random.uniform(0,1,(gan_batch_size,1))
                            onembf = 1-bf
                            my_label = np.concatenate((bf,onembf),axis=1)
                            my_data = np.multiply(mbatch['data'], bf) + np.multiply(data, onembf)

                            _, l = sess.run([disc.optimizer_first, disc.loss],
                                            feed_dict={disc.input: my_data,
                                                       disc.label: my_label})
                            i = i + 1
                            # Display logs per step
                            if i % gan_display_step == 0 or i == 1:
                                print('discriminator epoch %i Step %i: Minibatch Loss: %f' % (e, i, l))
                        print('discriminator epoch %i Step %i: Minibatch Loss: %f' % (e, i, l))

        # if True:
        #     lb = np.concatenate((np.zeros((extra.shape[0],1)),np.ones((extra.shape[0],1))),axis=1)
        #     extra_set = Dataset(dict(data=extra,label=lb), shuffle=True)
        #     num_r = 1
        #     if iters_so_far == 1:
        #         num_r = gan_num_epochs
        #     for e in range(num_r):
        #         i = 0
        #         for batch in extra_set.iterate_once(gan_batch_size):
        #             mbatch = mocap_set.next_batch(gan_batch_size)
        #             _, l = sess.run([disc.optimizer, disc.loss], feed_dict={disc.input: np.concatenate((mbatch['data'],batch['data'])), disc.label: np.concatenate((mbatch['label'],batch['label']))})
        #             i = i + 1
        #             # Display logs per step
        #             if i % gan_display_step == 0 or i == 1:
        #                 print('discriminator epoch %i Step %i: Minibatch Loss: %f' % (e, i, l))
        #         print('discriminator epoch %i Step %i: Minibatch Loss: %f' % (e, i, l))

        if not use_distance:
            if iters_so_far % 100 == 0:
                with open(logdir + "\\" + 'replay_buf_' + str(iters_so_far) + '.pkl', 'wb') as f:
                    pickle.dump(replay_buf, f)

        with open(logdir + "\\" + 'ob_list_' + str(rank) + '.pkl', 'wb') as f:
            pickle.dump(ob_list, f)
        if MPI.COMM_WORLD.Get_rank()==0:
            logF.write(str(rewmean) + "\n")
            logR.write(str(seg['mean_ext_rew']) + "\n")
            logStats.write(logger.get_str() + "\n")
            logF.flush()
            logStats.flush()
            logR.flush()

            logger.dump_tabular()

            try:
                os.remove(logdir + "/checkpoint")
            except OSError:
                pass
            try:
                saver.save(tf.get_default_session(), os.path.join(logdir, agentName), global_step=iters_so_far)
            except:
                pass
            try:
                non_disc_saver.save(tf.get_default_session(), os.path.join(logdir, agentName+ "_non_disc"), global_step=iters_so_far)
            except:
                pass
            try:
                disc_saver.save(tf.get_default_session(), os.path.join(logdir, agentName+ "_disc"), global_step=iters_so_far)
            except:
                pass

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
