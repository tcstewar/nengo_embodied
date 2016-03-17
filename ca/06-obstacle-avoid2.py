import nengo
import numpy as np    
import cellular

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
    
    whiskers = nengo.Ensemble(n_neurons=100, dimensions=3)
    nengo.Connection(body.whisker, whiskers)
    
    target = nengo.Ensemble(n_neurons=100, dimensions=2)
    nengo.Connection(body.target, target)
    
    def turn_toward(target):
        if target[0] < 0:
            return -1
        elif target[0] > 0:
            return 1
        else:
            return 0
    nengo.Connection(target, turn, function=turn_toward)
            
    def go_toward(target):
        if -0.2 < target[0] < 0.2:
            return 1
        else:
            return 0
    nengo.Connection(target, forward, function=go_toward)
            
        
    def avoid(whiskers):
        left, mid, right = whiskers
        return (right-left)*(2-mid)
    nengo.Connection(whiskers, turn, function=avoid)
        
    
        
        
        
        
        
        
        
        
        
        
        
        
    
    
