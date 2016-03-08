import nengo
import numpy as np

model = nengo.Network()
with model:
    def stim_pulse(t):
        return np.sin(t)
        index = int(t / 1.0)
        values = [1, 0, -1, 0]
        return values[index % len(values)]
    pre_value = nengo.Node(stim_pulse)

    tau_slow = 0.3
    
    pre = nengo.Ensemble(100, 1)
    post = nengo.Ensemble(100, 1)
    target = nengo.Ensemble(100, 1)
    nengo.Connection(pre_value, pre)

    conn = nengo.Connection(pre, post, function=lambda x: np.random.random(),
                learning_rule_type=nengo.PES())
    
    #error_1 = nengo.Connection(post, post, transform=-1, modulatory=True,
    #                            synapse=tau_slow*2)
    #error_2 = nengo.Connection(target, post, modulatory=True)
    #    
  

    wm = nengo.Ensemble(300, 2, radius=1.4)
    context = nengo.Node(1)
    nengo.Connection(context, wm[1])
    nengo.Connection(pre, wm[0], synapse=tau_slow)
    
    nengo.Connection(wm, target, synapse=tau_slow, 
                     function=lambda x: x[0]*x[1])
                     
    error = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(post, error, synapse=tau_slow*2, transform=1)
    nengo.Connection(target, error, transform=-1)
    
    nengo.Connection(error, conn.learning_rule)

    



    stop_learn = nengo.Node([1])
    nengo.Connection(stop_learn, error.neurons, transform=-10*np.ones((100,1)))
    
    
    
    
    
    
    
    
    
    
    