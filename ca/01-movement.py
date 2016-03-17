import nengo
import numpy as np    
import cellular
reload(cellular)

mymap="""
################
#   #    #     #
#           T  #
#  T    S      #
#              #
#  #####    ####
#     #        #
#              #
#  T        T  #
#              #
################
"""

model = nengo.Network()
with model:
    body = cellular.make_body(map=mymap)
    
    control_forward = nengo.Node([0])
    forward = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(control_forward, forward)
    nengo.Connection(forward, body.movement[0])
    
    control_turn = nengo.Node([0])
    turn = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(control_turn, turn)
    nengo.Connection(turn, body.movement[1])
    
