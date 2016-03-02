import nengo
import numpy as np    

import ccm.lib.grid
import ccm.lib.continuous
import ccm.ui.nengo

mymap="""
################
#              #
#              #
#  ####  ####  #
#  ####  ####  #
#  ####  ####  #
#  ####  ####  #
#  ####  ####  #
#  ####  ####  #
#  ####  ####  #
#  ###    ###  #
#      ##      #
#      ##      #
################
"""

class Cell(ccm.lib.grid.Cell):
    def color(self):
        return 'black' if self.wall else None
    def load(self, char):
        if char == '#':
            self.wall = True

world = ccm.lib.grid.World(Cell, map=mymap, directions=4)

body = ccm.lib.continuous.Body()
world.add(body, x=2, y=11, dir=1)




model = nengo.Network()
with model:
    def move(t, x):
        speed, rotation = x
        dt = 0.001
        max_speed = 30.0
        max_rotate = 20.0
        body.turn(rotation * dt * max_rotate)
        body.go_forward(speed * dt * max_speed)

    movement = nengo.Node(move, size_in=2)

    env = ccm.ui.nengo.GridNode(world, dt=0.005)

    def detect(t):
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir ) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
    stim_radar = nengo.Node(detect)

    radar = nengo.Ensemble(n_neurons=50, dimensions=3, radius=4)
    nengo.Connection(stim_radar, radar)
    
    
    actions = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=4,
                                           encoders=nengo.dists.Choice([[1]]),
                                           intercepts=nengo.dists.Uniform(0.1,0.9))
    
    nengo.Connection(actions.ensembles[0], movement, function=lambda x: [1,0])
    nengo.Connection(actions.ensembles[1], movement, function=lambda x: [-1,0])
    nengo.Connection(actions.ensembles[2], movement, function=lambda x: [0,1])
    nengo.Connection(actions.ensembles[3], movement, function=lambda x: [0,-1])
    
    bg = nengo.networks.actionselection.BasalGanglia(4)

    stim_utility = nengo.Node([0]*4)
    nengo.Connection(stim_utility, bg.input)
    nengo.Connection(bg.output, actions.input)
    with actions:
        nengo.Connection(nengo.Node([1]*4), actions.input)
        
    def u_forward(x):
        return 1 if x[1] > 1 else 0
    nengo.Connection(radar, bg.input[0], function=u_forward)
    
    def u_backward(x):
        return 1 if x[1] < 0.1 else 0
    nengo.Connection(radar, bg.input[1], function=u_backward)
    
    def u_left(x):
        return min(1, 0.5 * (x[1] - x[0]))
    nengo.Connection(radar, bg.input[2], function=u_left)
        
    def u_right(x):
        return min(1, 0.5 * (x[0] - x[1]))
    nengo.Connection(radar, bg.input[3], function=u_right)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
