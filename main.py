import sys
import queue
import numpy as np

class Game:
    def __init__(self):
        self.GHOST_DEATH_TIME                   = 5
        self.FOOD_SCORE                         = 10
        self.SPEED_BOOSTER_SCORE                = 50
        self.SPEED_BOOSTER_DURATION             = 21
        self.GHOST_SCORE                        = 100
        self.BOOSTER_GHOST_SCORE_MULTIPLICATOR  = 2
        self.MAX_TICK                           = 480
        self.pacmanId                           = 0

    def __str__(self):
        return ("GAME_ID:{id}, PACMAN_ID:{pacmanId}, TICK:{tick}, MESSAGE:{message}\n"
            .format(id=self.gameId, pacmanId=self.pacmanId, tick=self.tick, message=self.currentMessage))

    def setValues(self, gameId, tick, pacmanId, mapHeight, mapWidth, message = "NONE"):
        self.gameId    = gameId
        self.tick      = tick
        self.pacmanId  = pacmanId
        self.map       = dict(
            width   = mapWidth,
            height  = mapHeight,
            state   = []
        )
        self.pacmanInfo     = []
        self.ghostInfo      = []
        self.currentMessage = message
    
    def updateState(self):
        getLine = lambda: sys.stdin.readline().strip().split(" ")

        gameId, tick, pacmanId = list(map(int, getLine()))
        mapHeight, mapWidth, pacmanCount, ghostCount, *message = getLine()

        self.setValues(gameId, tick, pacmanId, int(mapHeight), int(mapWidth), message)

        ## TODO: Change code not to reinitialize
        for _ in range(int(mapHeight)):
            self.map["state"].append(list(sys.stdin.readline())[:int(mapWidth)])

        for _ in range(int(pacmanCount)):
            id, team, y, x, remBoostTime, score, bonusPoints, *effects = getLine()
            self.pacmanInfo.append(dict(
                id              = int(id),
                team            = team,
                pos             = (int(y), int(x)),
                remBoostTime    = int(remBoostTime),
                score           = int(score),
                bonusPoints     = bonusPoints,
                effects         = effects
            ))
        
        for _ in range(int(ghostCount)):
            id, y, x, remEatableTime, remStandbyTime = getLine()
            self.ghostInfo.append(dict(
                id                  = id,
                pos                 = (int(y), int(x)),
                dir                 = None,
                remEatableTime      = int(remEatableTime),
                remStandbyTime      = int(remStandbyTime)
            ))
        
        return 0
        
class Pacman:
    def __init__(self, id = 0, pos = (0,0)):
        self.dirs   = "v<>^"
        self.id     = id
        self.pos    = pos
        self.dir    = None
        self.score  = 0
        self.remBoostTime = 0
        self.GHOST_SPREAD_DEATH_RADIUS = 10
        self.GHOST_SPREAD_DEATH_DECAY  = 5

    def __str__(self):
        return ("ID:{id}, POS:({y},{x}), DIR:{dir}\n"
            .format(id=self.id, y=self.pos[0], x=self.pos[1], dir=self.dir))

    def updateStatus(self, pos, score, remBoostTime):
        self.pos            = pos
        self.score          = score
        self.remBoostTime   = remBoostTime

    
    def getFieldScore(self, pos):
        y,x = pos[0],pos[1]
        state = game.map["state"]
        if state[y][x] == '1':
            return game.FOOD_SCORE
        if state[y][x] == '+':
            multiplier = sum(np.exp(1 / np.arange(1, game.SPEED_BOOSTER_DURATION - self.remBoostTime)) - 1)
            return multiplier * game.SPEED_BOOSTER_SCORE
        for ghost in game.ghostInfo:
            if ghost["pos"] == (y,x) and pacman.remBoostTime > 0:
                ## TODO: Add ghost kill streak && better heuristicts
                return game.GHOST_SCORE / 2
            elif ghost["pos"] == (y,x) and pacman.remBoostTime == 0:
                #return -2*game.GHOST_SCORE
                return 0
        return 0

    def coverPath(self, start, end, parents):
        while end != start:
            if (game.tick == 5 or game.tick == 6) or game.tick == 7:
                sys.stderr.write("(%d,%d)\n" % (end[0],end[1]))
            prev = parents[end[0]][end[1]]
            dir = (end[0]-prev[0], end[1]-prev[1])
            end = prev
        if (game.tick == 5 or game.tick == 6) or game.tick == 7:
            sys.stderr.write("(%d,%d)\n" % (end[0],end[1]))
        return dir

    def breadthFirstSearch(self, starts, evalFun, maxDist = sys.maxsize):
        height  = game.map["height"]
        width   = game.map["width"]
        dist    = np.zeros(shape  = (height, width))
        state = game.map["state"]
        targets = queue.Queue()
        for pos in starts:
            sys.stderr.write("(%d,%d)\n" % (pos[0], pos[1]))
            targets.put(pos)
            visited = [[False for _ in range(width)] for _ in range(height)]

            while not targets.empty():
                field = targets.get()
                for dir in [(0,1), (1,0), (-1,0), (0,-1)]:
                    y,x = field[0]+dir[0], field[1]+dir[1]
                    if ((0 <= y and y < height) and (0 <= x and x < width)) and (state[y][x] != 'F' and state[y][x] != 'G'):
                        if not visited[y][x]:
                            dist[y][x] = dist[field[0]][field[1]] + 1
                            if dist[y][x] <= maxDist:
                                evalFun(y,x, field, dist[y][x])
                                visited[y][x]   = True
                                targets.put((y,x))

        return dist

    def getHeuristics(self):
        height  = game.map["height"]
        width   = game.map["width"]
        pos = self.pos
        scores  = np.zeros(shape = (height, width))
        dangers = np.zeros(shape = (height, width))
        parents = [[None for _ in range(width)] for _ in range(height)]

        ghostsPos = []
        for ghost in game.ghostInfo:
            if ghost["remStandbyTime"] <= 1:
                ghostY, ghostX = ghost["pos"]
                dangers[ghostY][ghostX] = game.GHOST_SCORE/2
                ghostsPos.append((ghostY, ghostX))

        def distAndScoreEval(y,x,field, dist):
            gain = scores[field[0]][field[1]] + self.getFieldScore((y,x))
            expense = dangers[y][x]
            scores[y][x]    = gain - expense
            parents[y][x]   = field

        def ghostDangerEval(y,x,field, dist):
            diff = np.exp(1/dist) * self.GHOST_SPREAD_DEATH_DECAY
            dangers[y][x]   += max(0, dangers[field[0]][field[1]] - diff)

        ## TODO: push ghost pos by their dir
        ## TODO: change boost handling
        if self.remBoostTime <= game.SPEED_BOOSTER_DURATION / 4:
            self.breadthFirstSearch(ghostsPos, ghostDangerEval, 
                                    self.GHOST_SPREAD_DEATH_RADIUS)
        dist = self.breadthFirstSearch([self.pos], distAndScoreEval)

        ## TODO: Change it
        distM = 2
        heatMap = scores - distM * dist

        if game.tick % 10 == 0:
            sys.stderr.write("%s\n" % ghostsPos)
            for container in [scores, dist, dangers]:
                for i in range(height):
                    for j in range(width):
                        sys.stderr.write("%d " % int(container[i][j]))
                    sys.stderr.write("\n")
                sys.stderr.write("\n\n")

        index = np.argmax(heatMap)
        newY, newX = int(index / width), int(index % width)
        sys.stderr.write("target: (%d,%d)\n" % (newY ,newX))

        return self.coverPath(pos, (newY, newX), parents)


    def move(self):
        dir = self.getHeuristics()
        if dir == (0,1):
            self.dir = '>'
        if dir == (1,0):
            self.dir = 'v'
        if dir == (0,-1):
            self.dir = '<'
        if dir == (-1,0):
            self.dir = '^'

game = Game()
pacman = Pacman()

while True:
    game.updateState()
    if game.pacmanId == -1:
        break

    pInfo = game.pacmanInfo[0]
    pacman.updateStatus(pInfo["pos"], pInfo["score"], pInfo["remBoostTime"])
    pacman.move()

    if game.currentMessage != []:
        sys.stderr.write("gameState:\n%spacman:\n%s\n" % (game, pacman))
    sys.stdout.write("%s %s %s %c\n" % (game.gameId, game.tick, game.pacmanId, pacman.dir))