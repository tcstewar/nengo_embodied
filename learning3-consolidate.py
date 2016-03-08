import nengo
import numpy as np

model = nengo.Network()
with model:
    frontal = nengo.Ensemble(50, 2, radius=1.4)
    
    sma = nengo.Ensemble(50, 1)
    sma_direct = nengo.Ensemble(50, 1)
    #nengo.Connection(sma, sma_final)
    
    visual = nengo.Ensemble(50,1)
    
    stim = nengo.Node(lambda t: np.sin(t))
    wm = nengo.Node(1)
    
    nengo.Connection(stim, visual)
    
    nengo.Connection(visual, frontal[0], synapse=0.1)
    nengo.Connection(wm, frontal[1])
    
    nengo.Connection(frontal, sma, synapse=0.1, 
            function=lambda x: x[0]*x[1])
            
    
    conn = nengo.Connection(visual, sma_direct, synapse=0.01,
            function=lambda x: 0)
            
    error = nengo.Ensemble(50, 1)
    
    nengo.Connection(sma, error, transform=-1)
    nengo.Connection(sma_direct, error, transform=1)
    conn.learning_rule_type = nengo.PES()
    
    error_conn = nengo.Connection(error, conn.learning_rule)
    
    sma_total = nengo.Ensemble(50, 1)
    nengo.Connection(sma, sma_total)
    nengo.Connection(sma_direct, sma_total)

