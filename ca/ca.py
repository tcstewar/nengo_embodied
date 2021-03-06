import nengo
import numpy as np    

mymap="""
################
#   #    #     #
# S          T #
#            T #
#              #
#  #####    ####
#     #        #
#           S  #
#              #
#              #
################
"""


import ccm.lib.grid
import ccm.lib.continuous
import ccm.ui.nengo


class Cell(ccm.lib.grid.Cell):
    has_start = False
    possible_target = False
    target = False
    
    def color(self):
        if self.target:
            return 'yellow'
        return 'black' if self.wall else None
    def load(self, char):
        if char == 'S':
            self.has_start = True
        if char == 'T':
            self.possible_target = True
        if char == '#':
            self.wall = True

def make_body(map, whisker_count=3, whisker_range=0.5, whisker_max=4,
              movement_dt=0.001, start_dir=0):
    net = nengo.Network()
    with net:
        world = ccm.lib.grid.World(Cell, map=map, directions=4)
        net.environment = ccm.ui.nengo.GridNode(world, dt=0.005)
    
        starts = []
        for row in world.grid:
            for cell in row:
                if cell.has_start:
                    starts.append(cell)
        if len(starts) == 0:
            for row in world.grid:
                for cell in row:
                    if not cell.wall:
                        starts.append(cell)
        start = np.random.choice(starts)
        
        targets = []
        for row in world.grid:
            for cell in row:
                if cell.possible_target:
                    targets.append(cell)
        if len(targets) > 0:
            np.random.choice(targets).target = True
    
        net.body = ccm.lib.continuous.Body()
        world.add(net.body, x=start.x, y=start.y, dir=start_dir)
        
        net.last_time = 0
        net.score = 0
    
        def move(t, x):
            speed, rotation = x
            net.body.turn(rotation * movement_dt)
            net.body.go_forward(speed * movement_dt)
            if t < net.last_time:
                start = np.random.choice(starts)
                net.body.x = start.x
                net.body.y = start.y
                net.body.dir = start_dir
                net.score = 0
            net.last_time = t
            if net.body.cell.target:
                net.score += 1
                net.body.cell.target = False
                next_targets = [c for c in targets if c is not net.body.cell]
                if len(next_targets) > 0:
                    np.random.choice(next_targets).target=True
                    
        net.movement = nengo.Node(move, size_in=2)
        
        def detect(t):
            angles = (np.linspace(-whisker_range, whisker_range, whisker_count) + net.body.dir ) % world.directions
            value = [net.body.detect(d, max_distance=whisker_max)[0]/whisker_max for d in angles]
            
            return value
        net.whisker = nengo.Node(detect)
        
        def score_func(t):
            if t <= 0:
                score_rate = 0
            else:
                score_rate = net.score / t
            
            score_func._nengo_html_ = '''
            <br/>
            <ul>
             <li>total score: %d
             <li>score per second: %1.3f
            </ul> 
            ''' % (net.score, score_rate)
        score = nengo.Node(score_func, label='score')
    
    
    return net

model = nengo.Network()
with model:
    body = make_body(map=mymap)

    whisker = nengo.Ensemble(n_neurons=200, dimensions=3, radius=1.7)
    nengo.Connection(body.whisker, whisker)

    def braiten(x):
        turn = x[2] - x[0]
        spd = x[1] - 0.25
        return spd*100, turn*40
    nengo.Connection(whisker, body.movement, function=braiten)  