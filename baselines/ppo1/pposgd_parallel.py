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

def traj_segment_generator(pi, env, horizon, stochastic, num_parallel, num_cpu, rank, ob_size, ac_size, com):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    if num_parallel == 0:
        new = True  # marks if we're on first timestep of an episode
        ob = env.reset()

        cur_ep_ret = 0 # return in current episode
        cur_ep_len = 0 # len of current episode
        ep_rets = [] # returns of completed episodes in this segment
        ep_lens = [] # lengths of ...

        # Initialize history arrays
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()

        while True:
            prevac = ac
            ac, vpred = pi.act(stochastic, ob)
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                        "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                        "ep_rets" : ep_rets, "ep_lens" : ep_lens}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            ob, rew, new, _ = env.step(ac)
            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if new:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()
            t += 1
    elif num_cpu == num_parallel:
        new = True  # marks if we're on first timestep of an episode

        if rank == 0:
            ob_whole = env.reset()
            #print("t = " + str(t) + " ob_whole = " + str(ob_whole))
            ob_flat = np.reshape(ob_whole, num_parallel * ob_size)
            #print("Rank = 0 obs_whole = " + str(ob_whole))
        else:
            ob_flat = None
        ob = np.zeros(ob_size, 'float32')
        com.Scatter(ob_flat, ob, root=0)

        #print("Rank = " + str(rank) + " ob = " + str(ob))
        #exit(0)

        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # lengths of ...

        # Initialize history arrays
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')

        acs = np.zeros((horizon, ac.shape[0]), 'float32')
        prevacs = acs.copy()
        ac = np.zeros(ac.shape[0], 'float32')

        vpred_a = np.zeros(1,'float32')
        rew_a = np.zeros(1,'float32')
        new_a = np.zeros(1,'uint8')
        while True:
            prevac = ac
            #ac, vpred = pi.act(stochastic, ob)


            if rank == 0:
                ac_whole, vpred_whole = pi.act_parallel(stochastic, ob_whole)
                #print("t = " + str(t) + " ac_whole = " + str(ac_whole) + " vpred_whole = " + str(vpred_whole))
                ac_flat = np.reshape(ac_whole, num_parallel * ac_size)
                vpred_flat = np.reshape(vpred_whole, num_parallel)
            else:
                ac_flat = None
                vpred_flat = None

            com.Scatter(ac_flat, ac, root=0)
            com.Scatter(vpred_flat, vpred_a, root=0)
            vpred = vpred_a[0]
            #print("t = " + str(t) + " rank = " + str(rank) + " ac = " + str(ac) + "vpred = " + str(vpred))

            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                       "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                       "ep_rets": ep_rets, "ep_lens": ep_lens}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac
            #print("rank = " + str(rank)+ " i = " + str(i) + " ob = " + str(ob))
            if rank == 0:
                ob_whole, rew_whole, new_whole, _ = env.step_parallel(ac_whole)
                #print("t = " + str(t) + " ob_whole = " + str(ob_whole) + " rew_whole = " + str(rew_whole) + " new_whole = " + str(new_whole))

                ob_flat = np.reshape(ob_whole, num_parallel * ob_size)
                rew_flat = np.reshape(rew_whole, num_parallel)
                new_flat = np.reshape(new_whole, num_parallel)
                #print("t = " + str(t) + " ob_flat= " + str(ob_flat) + " rew_flat = " + str(rew_flat) + " new_flat = " + str(new_flat))
            else:
                ob_flat = None
                rew_flat = None
                new_flat = None

            com.Scatter(ob_flat, ob, root=0)
            com.Scatter(rew_flat, rew_a, root=0)
            rew = rew_a[0]
            com.Scatter(new_flat, new_a, root=0)
            new = new_a[0]

            #print("t = " + str(t) + " rank = " + str(rank) + " ob = " + str(ob) + " rew = " + str(rew)+ " new = " + str(new))

            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if new:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                #ob = env.reset()
            t += 1
    else:

        new = np.ones(num_parallel, 'int32')  # marks if we're on first timestep of an episode

        ob = env.reset_parallel()
        cur_ep_ret = [0 for _ in range(num_parallel)]
        cur_ep_len = [0 for _ in range(num_parallel)]
        ep_rets = [[] for _ in range(num_parallel)] # returns of completed episodes in this segment
        ep_lens = [[] for _ in range(num_parallel)] # lengths of ...

        # Initialize history arrays
        #obs = np.array([[ob for _ in range(horizon)] for __ in range(num_parallel)])
        obs = np.zeros((num_parallel, horizon, ob.shape[1]),'float32')
        rews = np.zeros((num_parallel, horizon), 'float32')
        vpreds = np.zeros((num_parallel, horizon), 'float32')
        news = np.zeros((num_parallel, horizon), 'int32')
        #acs = np.array([[ac for _ in range(horizon)] for __ in range(num_parallel)])
        acs = np.zeros((num_parallel, horizon, ac.shape[0]),'float32')
        prevacs = acs.copy()
        ac = np.zeros((num_parallel, ac.shape[0]),'float32')

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
                        "ep_rets" : ep_rets_all, "ep_lens" : ep_lens_all}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                for j in range(num_parallel):
                    ep_rets[j] = []
                    ep_lens[j] = []
            i = t % horizon
            for j in range(num_parallel):
                obs[j][i] = ob[j]
                vpreds[j][i] = vpred[j]
                news[j][i] = new[j]
                acs[j][i] = ac[j]
                prevacs[j][i] = prevac[j]
            ob, rew, new, _ = env.step_parallel(ac)
            #print("t = " + str(t) + " ob = " + str(ob) + " rew = " + str(rew) + " new = " + str(new))
            for j in range(num_parallel):
                rews[j][i] = rew[j]
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

def learn(env, policy_func, *,
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
        num_cpu=1
        ):
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
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True, num_parallel=num_parallel, num_cpu=num_cpu, rank=rank, ob_size=ob_size, ac_size=ac_size,com=MPI.COMM_WORLD)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    saver = tf.train.Saver()
    if resume > 0:
        saver.restore(tf.get_default_session(), os.path.join(os.path.abspath(logdir), "{}-{}".format(agentName, resume)))
    iters_so_far = resume
    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"
    logF = open(logdir + "\\" + 'log.txt', 'a')
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
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]

        ob_list.append(ob.tolist())
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

        with open(logdir + "\\" + 'ob_list_' + str(rank) + '.pkl', 'wb') as f:
            pickle.dump(ob_list, f)
        if MPI.COMM_WORLD.Get_rank()==0:
            logF.write(str(rewmean) + "\n")
            logStats.write(logger.get_str() + "\n")
            logF.flush()
            logStats.flush()

            logger.dump_tabular()

            try:
                os.remove(logdir + "/checkpoint")
            except OSError:
                pass
            try:
                saver.save(tf.get_default_session(), os.path.join(logdir, agentName), global_step=iters_so_far)
            except:
                pass

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
