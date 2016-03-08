import nengo
import numpy as np    

import ccm.lib.grid
import ccm.lib.continuous
import ccm.ui.nengo

mymap="""
#########
#       #
#       #
#   ##  #
#   ##  #
#       #
#########

"""

class Cell(ccm.lib.grid.Cell):
    def color(self):
        return 'black' if self.wall else None
    def load(self, char):
        if char == '#':
            self.wall = True

world = ccm.lib.grid.World(Cell, map=mymap, directions=4)

body = ccm.lib.continuous.Body()
world.add(body, x=1, y=3, dir=2)

model = nengo.Network()
with model:
    def move(t, x):
        speed, rotation = x
        dt = 0.001
        max_speed = 20.0
        max_rotate = 10.0
        body.turn(rotation * dt * max_rotate)
        success = body.go_forward(speed * dt * max_speed)
        if not success:
            return -1
        else:
            return speed


    movement = nengo.Ensemble(n_neurons=100, dimensions=2, radius=1.4)
    
    movement_node = nengo.Node(move, size_in=2)
    nengo.Connection(movement, movement_node)

    env = ccm.ui.nengo.GridNode(world, dt=0.005)

    def detect(t):
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir ) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
    stim_radar = nengo.Node(detect)

    radar = nengo.Ensemble(n_neurons=50, dimensions=3, radius=4)
    nengo.Connection(stim_radar, radar)
    
    
    bg = nengo.networks.actionselection.BasalGanglia(3)
    thal = nengo.networks.actionselection.Thalamus(3)
    nengo.Connection(bg.output, thal.input)
    
    def u_fwd(x):
        if x[1] > 2:
            return 1
        else:
            return 0
    def u_left(x):
        if x[1] < 2:
            if x[2] > x[0]:
                return 1
        return 0
    def u_right(x):
        if x[1] < 2:
            if x[2] < x[0]:
                return 1
        return 0
    
    conn_fwd = nengo.Connection(radar, bg.input[0], function=u_fwd, learning_rule_type=nengo.PES())
    conn_left = nengo.Connection(radar, bg.input[1], function=u_left, learning_rule_type=nengo.PES())
    conn_right = nengo.Connection(radar, bg.input[2], function=u_right, learning_rule_type=nengo.PES())
        
    nengo.Connection(thal.output[0], movement, transform=[[1],[0]])
    nengo.Connection(thal.output[1], movement, transform=[[0],[1]])
    nengo.Connection(thal.output[2], movement, transform=[[0],[-1]])
    
    errors = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=3)
    nengo.Connection(movement_node, errors.input, transform=-np.ones((3,1)))
    nengo.Connection(bg.output[0], errors.ensembles[0].neurons, transform=np.ones((50,1))*-10)    
    nengo.Connection(bg.output[1], errors.ensembles[1].neurons, transform=np.ones((50,1))*-10)    
    nengo.Connection(bg.output[2], errors.ensembles[2].neurons, transform=np.ones((50,1))*-10)    
    nengo.Connection(bg.input, errors.input, transform=1)
    
    nengo.Connection(errors.ensembles[0], conn_fwd.learning_rule)
    nengo.Connection(errors.ensembles[1], conn_left.learning_rule)
    nengo.Connection(errors.ensembles[2], conn_right.learning_rule)
    
    
    
    
    