import nengo
import numpy as np

D = 3
N = D*50

model = nengo.Network(label="My Network")
with model:

    def us_stim(t):
        t = t % 3
        if 0.9 < t< 1: return [1, 0, 0]
        if 1.9 < t< 2: return [0, 1, 0]
        if 2.9 < t< 3: return [0, 0, 1]
        return [0, 0, 0]
    us_stim = nengo.Node(us_stim)

    def cs_stim(t):
        t = t % 3
        if 0.7 < t< 1: return [0.7, 0, 0.5]
        if 1.7 < t< 2: return [0.6, 0.7, 0.8]
        if 2.7 < t< 3: return [0, 1, 0]
        return [0, 0, 0]
    cs_stim = nengo.Node(cs_stim)

    us = nengo.Ensemble(N, D)
    cs = nengo.Ensemble(N*2, D*2)

    nengo.Connection(us_stim, us[:D])
    nengo.Connection(cs_stim, cs[:D])
    nengo.Connection(cs[:D], cs[D:], synapse=0.2)


    ur = nengo.Ensemble(N, D)

    nengo.Connection(us, ur)

    cr = nengo.Ensemble(N, D)

    learn_conn = nengo.Connection(cs, cr, function=lambda x: [0]*D)
    learn_conn.learning_rule_type = nengo.PES(learning_rate=3e-4)
    
    error = nengo.Ensemble(N, D)
    nengo.Connection(error, learn_conn.learning_rule)

    nengo.Connection(ur, error, transform=-1)
    nengo.Connection(cr, error, transform=1, synapse=0.1)
    
    stop_learn = nengo.Node([0])
    nengo.Connection(stop_learn, error.neurons, transform=-10*np.ones((N, 1)))

    response = nengo.Ensemble(N, D)
    nengo.Connection(ur, response)
    nengo.Connection(cr, response)