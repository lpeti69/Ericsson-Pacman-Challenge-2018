import sys
import queue
import operator
import numpy as np
import random

class Agent():

    def __init__(self):
        self.featureExtractor = FeatureExtractor()
        self.weights = Counter()
        self.weights['eats-food'] = 215.1656253667629
        self.weights['closest-food'] = -1.1901203570404904
        self.weights['bias'] = 113.4627147751201
        self.weights['capsules'] = 12.8180044889
        self.weights['#-of-ghosts-1-step-away'] = -250.5676392640621

    def Qsa(self, state, action):
        Qsa = 0.0
        features = self.featureExtractor.getFeatures(state, action)
        for key, val in features.items():
            sys.stderr.write("%s: %f" % (key, val))
        for key in features.keys():
            Qsa += self.weights[key] * features[key]
        return Qsa

    def getPolicy(self, state):
        actions = state.getLegalActions()
        values = [self.Qsa(state, a) for a in actions]
        if len(values) == 0:
            return (0,0)
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        if len(bestActions) == 0:
            return (0,0)
        return random.choice(bestActions)

class FeatureExtractor:

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.M.getFoods()
        walls = state.M.getWalls()
        ghostPositions = state.getGhostPositions()
        capsulesLeft = len(state.getCapsPositions())
        scaredGhost = []
        activeGhost = []
        features = Counter()
        for ghost in state.G:
            if not ghost.eatable:
                activeGhost.append(ghost)
            else:
                scaredGhost.append(ghost)

        pos = state.getOwn().getPos()
        def getManhattanDistances(ghosts):
            return map(lambda g: abs(pos[0]-g.getPos()[0]) + abs(pos[1]-g.getPos()[1]), ghosts)

        distanceToClosestActiveGhost = distanceToClosestScaredGhost = 0
        features["bias"] = 1.0

        y,x = pos
        dy, dx = action
        next_y, next_x = int(y + dy), int(x + dx)

        features["#-of-ghosts-1-step-away"] = sum(
            (next_y, next_x) in state.getLegalNeighbors(g) for g in ghostPositions)

        if not features["#-of-ghosts-1-step-away"] and food[next_y][next_x]:
            features["eats-food"] = 1.0

        closestFood = state.getClosests((next_y, next_x))[0] ## food
        if closestFood != []:
            features["closest-food"] = float(closestFood[0][1]) / \
                (state.M.width * state.M.height)
        if scaredGhost:
            distanceToClosestScaredGhost = min(
                getManhattanDistances(scaredGhost))
            if activeGhost:
                distanceToClosestActiveGhost = min(
                    getManhattanDistances(activeGhost))
            else:
                distanceToClosestActiveGhost = 10
            features["capsules"] = capsulesLeft
            if distanceToClosestScaredGhost <= 8 and distanceToClosestActiveGhost >= 2:
                features["#-of-ghosts-1-step-away"] = 0
                features["eats-food"] = 0.0

        sys.stderr.write("active: %d, scared: %d, closestF: %d\n" % (distanceToClosestActiveGhost, distanceToClosestScaredGhost, closestFood[0][1]))
        for ghost in activeGhost:
            sys.stderr.write("active: (%d, %d)" % (ghost.getPos()[0], ghost.getPos()[1]))
        for ghost in scaredGhost:
            sys.stderr.write("active: (%d, %d)" % (ghost.getPos()[0], ghost.getPos()[1]))
        features.divideAll(10.0)
        return features

class Counter(dict):
    
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        for key in keys:
            self[key] += count

    def argMax(self):
        if len(self.keys()) == 0: return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        sign = lambda x: 1 if x >=0 else -1
        sortedItems = self.items()
        compare = lambda x, y:  sign(y[1] - x[1])
        sortedItems.sort(cmp=compare)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        return sum(self.values())

    def normalize(self):
        total = float(self.totalCount())
        if total == 0: return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        return Counter(dict.copy(self))

    def __mul__(self, y ):
        sum = 0
        x = self
        if len(x) > len(y):
            x,y = y,x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        for key, value in y.items():
            self[key] += value

    def __add__( self, y ):
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__( self, y ):
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend
    

class Map:
    def __init__(self, matrix=None):
        if matrix == None:
            self.width = 0
            self.height = 0
            self._map = []
        else:
            self.width = len(matrix[0])
            self.height = len(matrix)
            self._map = matrix
        self._walls = None
        self._foods = None
            
    def resize(self, h, w):
        self.height = h
        self.width = w
    
    def read(self):
        self._map = [list(sys.stdin.readline().strip('\n')) for i in range(self.height)]

    def translate(self):
        for y in range(self.height):
            for x in range(self.width):
                self._map[y][x] = self._translateChar(self._map[y][x])
        self._walls = Map([['%#'.find(c)+1 for c in r] for r in self._map])
        self._foods  = Map([['.'.find(c)+1 for c in r] for r in self._map])

    def emplaceP(self, P, owns):
        for p in P:
            if p.id in owns: self._map[p.y][p.x] = 'P'
            else: self._map[p.y][p.x] = 'E'
    
    def emplaceG(self, G):
        for g in G: self._map[g.y][g.x] = 'G'
    
    def getWalls(self):
        return self._walls
    
    def getFoods(self):
        return self._foods
    
    def _translateChar(self, chr):
        tr_from = 'F 1+G'
        tr_to   = '% .o#'
        return tr_to[tr_from.index(chr)]

    def __getitem__(self, i):
        return self._map[i]
    
    def __str__(self):
        return '\n'.join([''.join(list(map(str,r))) for r in self._map])

class Pacman:
    def __init__(self):
        self.id = 0
        self.team = ''
        self.y = 0
        self.x = 0
        self.boost = 0
        self.points = 0
        self.ppoints = ''
        self.msg = ''
    
    def read(self):
        line = sys.stdin.readline().strip().split(" ")
        self.id = int(line[0])
        self.team = line[1]
        self.y, self.x, self.boost, self.points = list(map(int, line[2:6]))
        self.ppoints = line[6]
    
    def getPos(self):
        return (self.y, self.x)
    
    def getBoosterRemain(self):
        return self.boost

    def __str__(self):
        return ' '.join([str(self.id), str(self.team), str(self.y), str(self.x), str(self.boost), str(self.points), self.ppoints, self.msg])


class Ghost:
    def __init__(self):
        self.id = ''
        self.y = 0
        self.x = 0
        self.eatable = 0
        self.frozen = 0
    
    def read(self):
        line = sys.stdin.readline().strip().split(" ")
        self.id = line[0]
        self.y, self.x, self.eatable, self.frozen = list(map(int, line[1:5]))
    
    def getPos(self):
        return (self.y, self.x)
    
    def getEatableRemain(self):
        return self.eatable
    
    def getFrozenRemain(self):
        return self.frozen
    
    def __str__(self):
        return ' '.join([self.id, str(self.y), str(self.x), str(self.eatable), str(self.frozen)])


class Game:
    def __init__(self):
        # meta
        self.id = 0
        self.tick = 0
        self.pacmanid = 0
        self.msg = ''
        # objects
        self.M = Map()
        self.P = []
        self.G = []
        self.weights = Counter()
        #
        self._ownIndex = 0
        self.agent = Agent()

    def read(self):

        # meta
        self.id, self.tick, self.pacmanid = list(map(int, self._readline()))

        # objects' meta
        line = self._readline()
        self.M.resize(int(line[0]), int(line[1]))
        if len(self.P) != int(line[2]): self.P = [Pacman() for i in range(int(line[2]))]
        if len(self.G) != int(line[3]): self.G = [Ghost() for i in range(int(line[3]))]
        self.msg = ' '.join(line[4:])

        # objects
        self.M.read()
        for p in self.P: p.read()
        for g in self.G: g.read()
        
        # translate
        self.M.translate()

        # emplace
        self.M.emplaceP(self.P, [self.pacmanid])
        self.M.emplaceG(self.G)
        
        # helpers
        #_ownIndex
        _ownIndex = 0
        while _ownIndex < len(self.P) and self.P[_ownIndex].id != self.pacmanid: _ownIndex += 1
        if _ownIndex < len(self.P): self._ownIndex = _ownIndex
        else: self._ownIndex = None
        #_capsPos
        self._capsPos = [(x,y) for x in range(self.M.height) for y in range(self.M.width) if G.M[x][y] == 'o']
        
        # booster fix
        if G.pacmanid != -1 and self.getOwn().getBoosterRemain():
            for g in self.G: g.eatable = 0

        # successful reading
        return G.pacmanid != -1
    
    def getOwn(self):
        return self.P[self._ownIndex]

    def clip(self, M, y, x):
        if y < 0: y += M.height
        elif y >= M.height: y -= M.height
        if x < 0: x += M.width
        elif x >= M.width: x -= M.width
        return (y, x)
        
    
    def getLegalActions(self):
        actions = []
        pos = self.getOwn().getPos()
        for (dy,dx) in [(0,-1), (0,1), (-1,0), (1,0)]:
            y, x = self.clip(self.M, pos[0]+dy, pos[1]+dx)
            if not self.M.getWalls()[y][x]:
                actions.append((dy,dx))
        return actions

    def getLegalNeighbors(self, pos):
        neighbors = []
        for (dy,dx) in [(0,-1), (0,1), (-1,0), (1,0)]:
            y, x = self.clip(self.M, pos[0]+dy, pos[1]+dx)
            if not self.M.getWalls()[y][x]:
                neighbors.append((y,x))
        return neighbors
  
    def getGhostPositions(self):
        return [g.getPos() for g in self.G]
    
    def getPacmanPositions(self):
        return [p.getPos() for p in self.P]
    
    def getEnemyPositions(self):
        return [p.getPos() for p in self.P if p.id != self.pacmanid]
    
    def getCapsPositions(self):
        return self._capsPos
    
    def getPacmanFromPosition(self, y, x):
        for p in self.P:
            if p.x == x and p.y == y:
                return p
    
    def getGhostFromPosition(self, y, x):
        for g in self.G:
            if g.x == x and g.y == y:
                return g
    
    # call: getClosests((y,x))
    # [food, cap, pacman, biggerPacmap, smallerPacman, ghost, scaradGhost, activeGhost]
    # format: [ ((y1,x1),dist1), ((y2,x2),dist2), ... ]
    def getClosests(self, pos):
        return G._BFS(M=self.M,
                      starts=[pos],
                      isTarget=[
                        lambda m,y,x:m[y][x]=='.',
                        lambda m,y,x:m[y][x]=='o',
                        lambda m,y,x:m[y][x]=='E',
                        lambda m,y,x:m[y][x]=='E' and G.getPacmanFromPosition(y,x).points>G.getOwn().points,
                        lambda m,y,x:m[y][x]=='E' and G.getPacmanFromPosition(y,x).points<G.getOwn().points,
                        lambda m,y,x:m[y][x]=='G',
                        lambda m,y,x:m[y][x]=='G' and G.getGhostFromPosition(y,x).eatable>0,
                        lambda m,y,x:m[y][x]=='G' and G.getGhostFromPosition(y,x).eatable==0 and G.getGhostFromPosition(y,x).frozen==0])
    
    # call: getGhostsDistance((y,x))
    # format: [ ((y1,x1),dist1), ((y2,x2),dist2), ... ]
    def getGhostsDistance(self, pos):
        return G._BFS(M=self.M,
                      starts=[pos],
                      isTarget=[
                        lambda m,y,x:m[y][x]=='G'
                      ],
                      firstOnly=False)[0]
    
    def update(self, d):
        pos  = ( y,  x) = self.getOwn().getPos()
        npos = (ny, nx) = (y+d[0], x+d[1])
        nfld = self.M[ny][nx]
        # map
        self.M[y][x] = ' '
        self.M[ny][nx] = 'P'
        # pacman
        P = self.getOwn()
        P.y, P.x = npos
        if nfld == '.': P.points += 10
        elif nfld == 'o': P.points += 50
    
    def out(self, a1, a2=''):
        a1 = self.getDir(a1)
        a2 = self.getDir(a2) if a2!='' else ''
        sys.stdout.write("%s %s %s %s\n" % (G.id, G.tick, G.getOwn().id, a1+a2))
        sys.stderr.write("%s %s %s %s\n" % (G.id, G.tick, G.getOwn().id, a1+a2))
        sys.stderr.write("Pos: (%d, %d)\n" %(G.getOwn().getPos()[0], G.getOwn().getPos()[1]))

    def getDir(self, d):
        if d == (0,1):
            return '>'
        if d == (1,0):
            return 'v'
        if d == (0,-1):
            return '<'
        if d == (-1,0):
            return '^'
        return "ERROR DIR"
        
    
    def _readline(self):
        return sys.stdin.readline().strip().split(" ")
    
    def _BFS(self,
             M=Map(),
             starts=[],
             isWall=lambda m,y,x:m[y][x]=='%',
             isTarget=[],
             firstOnly=True,
             maxDistance=sys.maxsize):
        Q = queue.Queue()
        visited  = np.full((M.height,M.width), False)
        distance = np.full((M.height,M.width), 0)
        targets  = [[]] * len(isTarget)
        # init
        for start in starts:
            visited[start] = True
            Q.put(start)
        # search
        while not Q.empty():
            pos  = (y, x) = Q.get()
            dist = distance[pos]
            for (dy,dx) in [(0,-1), (0,1), (-1,0), (1,0)]:
                npos = (ny, nx) = self.clip(self.M, y+dy, x+dx)
                if isWall(M,ny,nx) or visited[npos]: continue
                # add
                visited[npos] = True
                distance[npos] = dist+1
                Q.put(npos)
                # targets
                nt = [[(npos,dist+1)] if t(M,ny,nx) else [] for t in isTarget]
                targets = list(map(operator.add, targets, nt))
                if firstOnly and all(targets):
                    Q = queue.Queue()
                    break
        if firstOnly:
            targets = [t[:1] for t in targets]
        return targets       
            

G = Game()
while G.read():
    #sys.stderr.write("%s\n" % G.M)
    #sys.stderr.write("%d, %d\n" % (G.M.width, G.M.height))
    sys.stderr.write("%s\n" % G.getClosests(G.getOwn().getPos()))
    
    a1 = G.agent.getPolicy(G)
    #sys.stdout.write("%s" % G.getDir(action))
    #sys.stderr.write("%s" % G.M)
    if G.getOwn().getBoosterRemain() > 0:
        G.update(a1) ## TODO
        a2 = G.agent.getPolicy(G)
        G.out(a1, a2)
    else:
        G.out(a1)
    #  c o d e   g o e s   h e r e
    