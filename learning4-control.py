import nengo
import ctn_benchmark.control as ctrl
import numpy as np

class Param:
    pass
p = Param()
p.D = 1
p.dt=0.001
p.seed=1
p.noise=0.0
p.Kp=2
p.Kd=1
p.Ki=0
p.tau_d=0.001
p.period=4
p.n_neurons=500
p.learning_rate=1
p.max_freq=1.0
p.synapse=0.01
p.scale_add=1
p.delay=0.01
p.filter=0.01
p.radius=1



model = nengo.Network()
with model:

    system = ctrl.System(p.D, p.D, dt=p.dt, seed=p.seed,
            motor_noise=p.noise, sense_noise=p.noise,
            scale_add=p.scale_add,
            motor_scale=10,
            motor_delay=p.delay, sensor_delay=p.delay,
            motor_filter=p.filter, sensor_filter=p.filter)

    def minsim_system(t, x):
        return system.step(x)

    minsim = nengo.Node(minsim_system, size_in=p.D, size_out=p.D,
                        label='minsim')

    state_node = nengo.Node(lambda t: system.state, label='state')

    pid = ctrl.PID(p.Kp, p.Kd, p.Ki, tau_d=p.tau_d)
    control = nengo.Node(lambda t, x: pid.step(x[:p.D], x[p.D:]),
                         size_in=p.D*2, label='control')
    nengo.Connection(minsim, control[:p.D], synapse=0)

    adapt = nengo.Ensemble(p.n_neurons, dimensions=p.D,
                           radius=p.radius, label='adapt')
    nengo.Connection(minsim, adapt, synapse=None)
    
    motor = nengo.Ensemble(p.n_neurons, p.D, radius=p.radius)
    nengo.Connection(motor, minsim, synapse=None)
    nengo.Connection(control, motor, synapse=None)
    
    #adapt_signal = nengo.Node(None, size_in=p.D, label='adapt_signal')
    #nengo.Connection(adapt_signal, minsim, synapse=None)
    conn = nengo.Connection(adapt, motor, synapse=p.synapse,
            function=lambda x: [0]*p.D,
            #solver=ZeroDecoder(),
            learning_rule_type=nengo.PES(1e-4 * p.learning_rate))

    error = nengo.Ensemble(p.n_neurons, p.D)
    nengo.Connection(control, error, synapse=None,
                        transform=-1)
    nengo.Connection(error, conn.learning_rule)
    

    signal = ctrl.Signal(p.D, p.period, dt=p.dt, max_freq=p.max_freq, seed=p.seed)
    desired = nengo.Node(signal.value, label='desired')
    nengo.Connection(desired, control[p.D:], synapse=None)
    
    
    stop_learn = nengo.Node([1])
    nengo.Connection(stop_learn, error.neurons, transform=np.ones((p.n_neurons,1))*-10)