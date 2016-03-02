import nengo
import numpy as np    

import ccm.lib.grid
import ccm.lib.continuous
import ccm.ui.nengo
reload(ccm.ui.nengo)

mymap="""
################
#   #    #     #
#              #
#              #
#              #
#  #####    ####
#     #        #
#              #
#              #
#              #
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
world.add(body, x=3, y=3)

def move(t, x):
    speed, rotation = x
    dt = 0.001
    max_speed = 30.0
    max_rotate = 10.0
    body.turn(rotation * dt * max_rotate)
    body.go_forward(speed * dt * max_speed)

def detect(t):
    angles = (np.linspace(-0.5, 0.5, 3) + body.dir ) % world.directions
    return [body.detect(d, max_distance=4)[0] for d in angles]


model = nengo.Network()
with model:
    movement = nengo.Node(move, size_in=2)

    env = ccm.ui.nengo.GridNode(world, dt=0.005)

    stim_radar = nengo.Node(detect)

    radar = nengo.Ensemble(n_neurons=50, dimensions=3, radius=4)
    nengo.Connection(stim_radar, radar)

    def braiten(x):
        turn = x[2] - x[0]
        spd = x[1] - 1
        return spd, turn
    nengo.Connection(radar, movement, function=braiten)  