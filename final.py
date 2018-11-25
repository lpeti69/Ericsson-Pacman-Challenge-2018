import sys
import queue
import operator
import numpy as np

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
        self._foods  = Map([['.o'.find(c)+1 for c in r] for r in self._map])

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
        self.ppoints = 0
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
        #
        self._ownIndex = 0

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

        # successful reading
        return G.pacmanid != -1
    
    def getOwn(self):
        return self.P[self._ownIndex]
    
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
                        lambda m,y,x:m[y][x]=='P',
                        lambda m,y,x:m[y][x]=='P' and G.getPacmanFromPosition(y,x).points>G.getOwn().points,
                        lambda m,y,x:m[y][x]=='P' and G.getPacmanFromPosition(y,x).points<G.getOwn().points,
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
                npos = (ny, nx) = (y+dy, x+dx)
                if ny<0 or M.height<=ny or nx<0 or M.width<=nx: continue
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
    print(G.M)
    print(G.getClosests((17,13)))
    
    #  c o d e   g o e s   h e r e
    