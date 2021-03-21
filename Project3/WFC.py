import random
import io
import base64
from collections import namedtuple
import numpy as np
from PIL import Image

LENGTH = 10
WIDTH = 10

class WaveFunctionCollaps:
     
    def __init__(self, images, constraints):
        self.tiles = self.getImage(images)
        self.constraints = self.buildConstraints(constraints)
        self.matrix = np.full((LENGTH, WIDTH, len(self.tiles)), True)
        self.process = []
        #  self.showImage()
    
    def getImage(self, images):
        tiles = []
        for i in range(len(images[0])):
            tmp = {}
            name, angle = images[0][i].split(',')
            tmp['name'] = images[0][i]
            tmp['bitmap'] = Image.open(name).rotate(int(angle))
            tmp['prob'] = images[1][i]
            tiles.append(tmp)
        # print(tiles)
        return tiles

    def setProbs(self, probs):
        if len(self.tiles) != len(probs):
            SystemExit(0)
        for i in range(len(self.tiles)):
            self.tiles[i]['prob'] = probs[i]

    def buildConstraints(self, constraints):
        res = {}
        for constraint in constraints:
            key = constraint['target']
            res[key] = {}
            res[key]['top'] = constraint['top']
            res[key]['bot'] = constraint['bot']
            res[key]['left'] = constraint['left']
            res[key]['right'] = constraint['right']
        return res

    def find_true(self, array):
        """
        Like np.nonzero, except it makes sense.
        """
        transform = int if len(np.asarray(array).shape) == 1 else tuple
        return list(map(transform, np.transpose(np.nonzero(array))))

    def location_with_fewest_choices(self, potential):
        num_choices = np.sum(potential, axis=2, dtype='float32')
        num_choices[num_choices == 1] = np.inf
        candidate_locations = self.find_true(num_choices == num_choices.min())
        # random.seed(0)
        location = random.choice(candidate_locations)
        if num_choices[location] == np.inf:
            return None

        return location

    def reverseDirection(self, direction):
        if direction == 'top':
            return 'bot'
        if direction == 'bot':
            return 'top'
        if direction == 'left':
            return 'right'
        if direction == 'right':
            return 'left'
        print('wrong with direction!!')
        SystemExit(0)
        
    def add_constraint(self, potential, location, incoming_direction, possible_tiles):
        neighbor_constraint = set()
        direction = self.reverseDirection(incoming_direction)
        for t in possible_tiles:
            neighbor_constraint |= set(self.constraints[t['name']][direction])
        changed = False
        for i_p, p in enumerate(potential[location]):
            if not p:
                continue
            if self.tiles[i_p]['name'] not in neighbor_constraint:
                potential[location][i_p] = False
                changed = True
        if not np.any(potential[location]):
            raise Exception(f"No patterns left at {location}")
        return changed

    def neighbors(self, location, height, width):
        res = []
        x, y = location
        if x != 0:
            res.append(('bot', x-1, y))
        if y != 0:
            res.append(('right', x, y-1))
        if x < height - 1:
            res.append(('top', x+1, y))
        if y < width - 1:
            res.append(('left', x, y+1))
        return res

    def propagate(self, potential, start_location):
        height, width = potential.shape[:2]
        needs_update = np.full((height, width), False)
        needs_update[start_location] = True
        while np.any(needs_update):
            needs_update_next = np.full((height, width), False)
            locations = self.find_true(needs_update)
            for location in locations:
                possible_tiles = [self.tiles[n] for n in self.find_true(potential[location])]
                for neighbor in self.neighbors(location, height, width):
                    neighbor_direction, neighbor_x, neighbor_y = neighbor
                    neighbor_location = (neighbor_x, neighbor_y)
                    was_updated = self.add_constraint(potential, neighbor_location,
                                                neighbor_direction, possible_tiles)
                    needs_update_next[neighbor_location] |= was_updated
            needs_update = needs_update_next

    def run_iteration(self, old_potential):
        potential = old_potential.copy()
        to_collapse = self.location_with_fewest_choices(potential) #3
        if to_collapse is None:                               #1
            raise StopIteration()
        elif not np.any(potential[to_collapse]):              #2
            raise Exception(f"No choices left at {to_collapse}")
        else:                                                 #4 ↓
            nonzero = self.find_true(potential[to_collapse])
            # np.random.seed(0)
            # print(nonzero)
            probs = []
            sumOfProbs = 0
            for i in nonzero:
                sumOfProbs += self.tiles[i]['prob']
            for i in nonzero:
                probs.append(self.tiles[i]['prob']/sumOfProbs)
            # print(probs)
            selected_tile = np.random.choice(nonzero, p=probs)
            potential[to_collapse] = False
            potential[to_collapse][selected_tile] = True
            self.propagate(potential, to_collapse)                 #5
        return potential

    def run(self):
        self.p = self.matrix
        # self.process = [self.showResult()]
        while True:
            try:
                self.p = self.run_iteration(self.p)
                # self.process.append(self.showResult())  # Move me for speed
            except StopIteration as e:
                # print('test')
                break
            except Exception as e:
                print(e)
                break
        # print(timer)
        return self.showResult()
        

    def showImage(self):
        for tile in self.tiles:
            image = tile['bitmap']
            image.show()
    
    def blend_many(self, ims):
        """
        Blends a sequence of images.
        """
        current, *ims = ims
        for i, im in enumerate(ims):
            current = Image.blend(current, im, 1/(i+2))
        return current
        
    def blend_tiles(self, choices, tiles):
        """
        Given a list of states (True if ruled out, False if not) for each tile,
        and a list of tiles, return a blend of all the tiles that haven't been
        ruled out.
        """
        to_blend = [self.tiles[i]['bitmap'] for i in range(len(choices)) if choices[i]]
        return self.blend_many(to_blend)

    def showResult(self):
        rows = []
        for row in self.p:
            rows.append([np.asarray(self.blend_tiles(t, self.tiles)) for t in row])

        rows = np.array(rows)
        n_rows, n_cols, tile_height, tile_width, _ = rows.shape
        images = np.swapaxes(rows, 1, 2)
        res = Image.fromarray(images.reshape(n_rows*tile_height, n_cols*tile_width, 3))
        return res
        # res.show()



# from enum import Enum, auto
# class Direction(Enum):
#     RIGHT = 0; UP = 1; LEFT = 2; DOWN = 3
    
#     def reverse(self):
#         return {Direction.RIGHT: Direction.LEFT,
#                 Direction.LEFT: Direction.RIGHT,
#                 Direction.UP: Direction.DOWN,
#                 Direction.DOWN: Direction.UP}[self]

if __name__ == "__main__":
    # images = ['./Images/empty.png,0', './Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270']
    # constraints = [{'target': './Images/empty.png,0', 'top': ['./Images/empty.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/corner.png,90', './Images/t.png,180'], 'bot': ['./Images/empty.png,0', './Images/t.png,0', './Images/line.png,0', './Images/corner.png,180', './Images/corner.png,270'], 'left': ['./Images/empty.png,0', './Images/corner.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270'], 'right': ['./Images/empty.png,0', './Images/corner.png,0', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,270']}, {'target': './Images/cross.png,0', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, {'target': './Images/t.png,0', 'top': ['./Images/empty.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/corner.png,90', './Images/t.png,180'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, {'target': './Images/corner.png,0', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/empty.png,0', './Images/t.png,0', './Images/line.png,0', './Images/corner.png,180', './Images/corner.png,270'], 'left': ['./Images/empty.png,0', './Images/corner.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, {'target': './Images/line.png,0', 'top': ['./Images/empty.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/corner.png,90', './Images/t.png,180'], 'bot': ['./Images/empty.png,0', './Images/t.png,0', './Images/line.png,0', './Images/corner.png,180', './Images/corner.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, {'target': './Images/cross.png,90', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, {'target': './Images/t.png,90', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/empty.png,0', './Images/corner.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, {'target': './Images/corner.png,90', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/empty.png,0', './Images/t.png,0', './Images/line.png,0', './Images/corner.png,180', './Images/corner.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/empty.png,0', './Images/corner.png,0', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,270']}, {'target': './Images/line.png,90', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/empty.png,0', './Images/corner.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270'], 'right': ['./Images/empty.png,0', './Images/corner.png,0', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,270']}, {'target': './Images/t.png,180', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/empty.png,0', './Images/t.png,0', './Images/line.png,0', './Images/corner.png,180', './Images/corner.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, {'target': './Images/corner.png,180', 'top': ['./Images/empty.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/corner.png,90', './Images/t.png,180'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/empty.png,0', './Images/corner.png,0', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,270']}, {'target': './Images/t.png,270', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/empty.png,0', './Images/corner.png,0', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,270']}, {'target': './Images/corner.png,270', 'top': ['./Images/empty.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/corner.png,90', './Images/t.png,180'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/empty.png,0', './Images/corner.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}]
    images = [['./Images/empty.png,0', './Images/cross.png,0', './Images/line.png,0', './Images/line.png,90'], [5,1,1,1]]
    constraints = [{'target': './Images/empty.png,0', 'top': ['./Images/empty.png,0', './Images/line.png,0'], 'bot': ['./Images/empty.png,0', './Images/line.png,0'], 'left': ['./Images/empty.png,0', './Images/line.png,90'], 'right': ['./Images/empty.png,0', './Images/line.png,90']}, {'target': './Images/cross.png,0', 'top': ['./Images/cross.png,0','./Images/line.png,90'], 'bot': ['./Images/cross.png,0', './Images/line.png,90'], 'left': ['./Images/cross.png,0', './Images/line.png,0'], 'right': ['./Images/cross.png,0', './Images/line.png,0']}, {'target': './Images/line.png,0', 'top': ['./Images/empty.png,0', './Images/line.png,0'], 'bot': ['./Images/empty.png,0', './Images/line.png,0'], 'left': ['./Images/cross.png,0', './Images/line.png,0'], 'right': ['./Images/cross.png,0', './Images/line.png,0']}, {'target': './Images/line.png,90', 'top': ['./Images/cross.png,0', './Images/line.png,90'], 'bot': ['./Images/cross.png,0', './Images/line.png,90'], 'left': ['./Images/empty.png,0', './Images/line.png,90'], 'right': ['./Images/empty.png,0', './Images/line.png,90']}]
    WFC = WaveFunctionCollaps(images, constraints)
    for i in range(5):
        res = WFC.run()
        res.show()
    # WFC.showImage()
    a = [1,5,1,1]
    WFC.setProbs(a)
    num = 0
    for i in range(5):
        res = WFC.run()
        res.show()
    # print(res)
    # WFC.showResult()
    # print(len(res))
    # res[0].save('out.gif', format='gif', save_all=True, append_images=WFC.process[1:],
    #         duration=50, loop=0)

    # [{'target': './Images/empty.png,0', 
    # 'top': ['./Images/empty.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/corner.png,90', './Images/t.png,180'], 
    # 'bot': ['./Images/empty.png,0', './Images/t.png,0', './Images/line.png,0', './Images/corner.png,180', './Images/corner.png,270'], 
    # 'left': ['./Images/empty.png,0', './Images/corner.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270'], 
    # 'right': ['./Images/empty.png,0', './Images/corner.png,0', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,270']}, 
    # {'target': './Images/cross.png,0', 
    # 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 
    # 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 
    # 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 
    # 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, 
    # {'target': './Images/t.png,0', 
    # 'top': ['./Images/empty.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/corner.png,90', './Images/t.png,180'], 
    # 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 
    # 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 
    # 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, 
    # {'target': './Images/corner.png,0', 
    # 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 
    # 'bot': ['./Images/empty.png,0', './Images/t.png,0', './Images/line.png,0', './Images/corner.png,180', './Images/corner.png,270'], 
    # 'left': ['./Images/empty.png,0', './Images/corner.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270'], 
    # 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, {'target': './Images/line.png,0', 'top': ['./Images/empty.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/corner.png,90', './Images/t.png,180'], 'bot': ['./Images/empty.png,0', './Images/t.png,0', './Images/line.png,0', './Images/corner.png,180', './Images/corner.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, {'target': './Images/cross.png,90', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, {'target': './Images/t.png,90', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/empty.png,0', './Images/corner.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, {'target': './Images/corner.png,90', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/empty.png,0', './Images/t.png,0', './Images/line.png,0', './Images/corner.png,180', './Images/corner.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/empty.png,0', './Images/corner.png,0', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,270']}, {'target': './Images/line.png,90', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/empty.png,0', './Images/corner.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270'], 'right': ['./Images/empty.png,0', './Images/corner.png,0', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,270']}, {'target': './Images/t.png,180', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/empty.png,0', './Images/t.png,0', './Images/line.png,0', './Images/corner.png,180', './Images/corner.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}, {'target': './Images/corner.png,180', 'top': ['./Images/empty.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/corner.png,90', './Images/t.png,180'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/empty.png,0', './Images/corner.png,0', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,270']}, {'target': './Images/t.png,270', 'top': ['./Images/cross.png,0', './Images/t.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270', './Images/corner.png,270'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/cross.png,0', './Images/t.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/t.png,180', './Images/corner.png,270'], 'right': ['./Images/empty.png,0', './Images/corner.png,0', './Images/t.png,90', './Images/line.png,90', './Images/corner.png,270']}, {'target': './Images/corner.png,270', 'top': ['./Images/empty.png,0', './Images/corner.png,0', './Images/line.png,0', './Images/corner.png,90', './Images/t.png,180'], 'bot': ['./Images/cross.png,0', './Images/corner.png,0', './Images/cross.png,90', './Images/t.png,90', './Images/corner.png,90', './Images/line.png,90', './Images/t.png,180', './Images/t.png,270'], 'left': ['./Images/empty.png,0', './Images/corner.png,90', './Images/line.png,90', './Images/corner.png,180', './Images/t.png,270'], 'right': ['./Images/cross.png,0', './Images/t.png,0', './Images/line.png,0', './Images/cross.png,90', './Images/corner.png,90', './Images/t.png,180', './Images/corner.png,180', './Images/t.png,270']}]

    # a = set([1,2,3,4,5])
    # b = set([1,4,7,8])
    # a |= b
    # print(a)
    # test = {'top':1}
    # a = 'top'
    # print(test[a])
    # test = Direction.DOWN
    # print(test.reverse())
    # images = './Images/empty.png,0'
    # name, angle = images.split(',')
    # print(name)
    # print(angle)
    # constraints = [1,2,3]
    # test = WaveFunctionCollaps(images, constraints)
    # image = Image.open('./Images/corner.png').rotate(90)
    # image.show()
    # test = np.full((LENGTH, WIDTH, 12), False)
    # test[1][1][1] = True
    # test[1][2][1] = True
    # test1 = test[(13)]
    # print(test1.shape)
    # print(len(test1.shape))
    # num_choices = np.sum(test, axis=2, dtype='float32')
    # print(num_choices[num_choices == 1])
    # num_choices[num_choices == 1] = np.inf
    # print(num_choices)
    # print(np.inf)
    # transform = int if len(np.asarray(test).shape) == 1 else tuple
    # print(transform)
    # print(np.transpose(np.nonzero(test)))
    # print(list(map(transform, np.transpose(np.nonzero(test)))))
    # test = list(map(tuple, [[1,2,3],[4,5,6]]))
    # print(test)