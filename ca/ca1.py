import nengo
import numpy as np    
import cellular


mymap="""
################
#   #    #     #
# S          T #
# T            #
#              #
#  #####    ####
#     #        #
#           S  #
#  T        T  #
#              #
################
"""
model = nengo.Network()
with model:
    body = cellular.make_body(map=mymap)

    whisker = nengo.Ensemble(n_neurons=50, dimensions=3, radius=1.7)
    nengo.Connection(body.whisker, whisker)

    def braiten(x):
        turn = x[2] - x[0]
        spd = x[1] - 0.25
        return spd*100, turn*40
    nengo.Connection(whisker, body.movement, function=braiten)  
    
    target = nengo.Ensemble(n_neurons=200, dimensions=2)
    nengo.Connection(body.target, target)
    
    def turn_to_target(x):
        return 0, x[0]*10
    nengo.Connection(target, body.movement, function=turn_to_target)
