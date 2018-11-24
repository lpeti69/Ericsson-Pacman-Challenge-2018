import sys

class Map:
    def __init__(self):
        self.width = 0
        self.height = 0
        self._map = []
    
    def resize(self, h, w):
        self.height = h
        self.width = w
    
    def read(self):
        self._map = [list(sys.stdin.readline().strip('\n')) for i in range(self.height)]

    def translate(self):
        for y in range(self.height):
            for x in range(self.width):
                self._map[y][x] = self._translateChar(self._map[y][x])

    def emplaceP(self, P, owns):
        for p in P:
            if p.id in owns: self._map[p.y][p.x] = 'P'
            else: self._map[p.y][p.x] = 'E'
    
    def emplaceG(self, G):
        for g in G: self._map[g.y][g.x] = 'G'
    
    def _translateChar(self, chr):
        tr_from = 'F 1+G'
        tr_to   = '% .o#'
        return tr_to[tr_from.index(chr)]

    def __getitem__(self, i):
        return self._map[i]
    
    def __str__(self):
        return '\n'.join([''.join(r) for r in self._map])

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
    
    def pos(self):
        return (self.y, self.x)
    
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
    
    def pos(self):
        return (self.y, self.x)
    
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
    
    def _readline(self):
        return sys.stdin.readline().strip().split(" ")

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

        # successful reading
        return G.pacmanid != -1

G = Game()
while G.read():
    print(G.tick)

    #  c o d e   g o e s   h e r e
    