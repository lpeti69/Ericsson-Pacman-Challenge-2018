import sys
import queue
import numpy as np

class CONSTANTS(object):
    FIELD_WALL                          = 'F'
    FIELD_EMPTY                         = ' '
    FIELD_FOOD                          = '1'
    FIELD_BOOSTER                       = '+'
    FIELD_GHOST_WALL                    = 'G'
    GHOST_DEATH_TIME                    = 5
    FOOD_SCORE                          = 10
    SPEED_BOOSTER_SCORE                 = 50
    SPEED_BOOSTER_DURATION              = 21
    GHOST_SCORE                         = 100
    BOOSTER_GHOST_SCORE_MULTIPLICATOR   = 2
    MAX_TICK                            = 480

class Game:
    def __init__(self):
        self.pacmanId                           = 0
        self.GHOST_SPREAD_DEATH_RADIUS          = 10
        self.GHOST_SPREAD_DEATH_DECAY           = 4
        self.GHOST_PUSH_DIST                    = 1
        self.GHOST_DANGER_EATABLE_TIME          = 2
        self.pacmans                            = dict()
        self.ghosts                             = dict()
        self.map                                = dict()

    def __str__(self):
        return ("GAME_ID:{id}, PACMAN_ID:{pacmanId}, TICK:{tick}, MESSAGE:{message}\n"
            .format(id=self.gameId, pacmanId=self.pacmanId, tick=self.tick, message=self.currentMessage))

    def setValues(self, gameId, tick, pacmanId, message = "NONE"):
        self.gameId    = gameId
        self.tick      = tick
        self.pacmanId  = pacmanId
        self.currentMessage = message
    
    def updateState(self):
        getLine = lambda: sys.stdin.readline().strip().split(" ")

        gameId, tick, pacmanId = list(map(int, getLine()))
        mapHeight, mapWidth, pacmanCount, ghostCount, *message = getLine()

        self.map["width"]   = int(mapWidth)
        self.map["height"]  = int(mapHeight)
        self.gameId    = gameId
        self.tick      = tick
        self.pacmanId  = pacmanId
        self.currentMessage = message

        ## TODO: Low Prio, optimize input reading
        self.map["state"] = []
        for _ in range(int(mapHeight)):
            self.map["state"].append(list(sys.stdin.readline())[:int(mapWidth)])

        for _ in range(int(pacmanCount)):
            id, team, y, x, remBoostTime, score, bonusPoints, *effects = getLine()
            id = int(id)
            if game.tick == 0:
                self.pacmans[id] = Pacman(id = id, team = team)
            self.pacmans[id].updateStatus(
                pos             = (int(y), int(x)),
                remBoostTime    = int(remBoostTime),
                score           = int(score),
                bonusPoints     = bonusPoints,
                effects         = effects
            )
        
        for _ in range(int(ghostCount)):
            id, y, x, remEatableTime, remStandbyTime = getLine()
            if game.tick == 0:
                self.ghosts[id] = Ghost(id = id)
            self.ghosts[id].updateStatus(
                pos                 = (int(y), int(x)),
                remEatableTime      = int(remEatableTime),
                remStandbyTime      = int(remStandbyTime)
            )

        return 0


class Ghost:
    def __init__(self, id):
        self.id         = id
        self.dir        = (0,0)
        self.prevPos    = (0,0)

    def updateStatus(self, pos, remEatableTime, remStandbyTime):
        self.pos = pos
        self.remEatableTime = remEatableTime
        self.remStandbyTime = remStandbyTime
        self.move()

    def learnStrategy(self):
        ## TODO: Add ghost statistics collection
        ## TODO: Add accumulated Reinforcement learning for the strategy of the ghosts
        pass

    def move(self):
        self.learnStrategy()
        if self.prevPos != (0,0):
            self.dir = (self.pos[0]-self.prevPos[0], self.pos[1]-self.prevPos[1])
        self.prevPos = self.pos
        
class Pacman:
    def __init__(self, id, team):
        self.dirs       = "v<>^"
        self.dir        = None
        self.id         = id
        self.team       = team

    def __str__(self):
        return ("ID:{id}, POS:({y},{x}), DIR:{dir}\n"
            .format(id=self.id, y=self.pos[0], x=self.pos[1], dir=self.dir))

    def updateStatus(self, pos, score, remBoostTime, bonusPoints, effects):
        self.pos            = pos
        self.score          = score
        self.effects        = effects
        self.remBoostTime   = remBoostTime
        self.bonusPoints    = bonusPoints

    
    def getFieldScore(self, pos):
        y,x = pos
        state = game.map["state"]
        if state[y][x] == CONSTANTS.FIELD_FOOD:
            return CONSTANTS.FOOD_SCORE
        if state[y][x] == CONSTANTS.FIELD_BOOSTER:
            ## Expected utility
            multiplier = sum(np.exp(1 / np.arange(1, CONSTANTS.SPEED_BOOSTER_DURATION - self.remBoostTime)) - 1)
            return multiplier * CONSTANTS.SPEED_BOOSTER_SCORE
        for _, ghost in game.ghosts.items():
            if ghost.pos == (y,x) and pacman.remBoostTime > 0:
                ## TODO: Add ghost kill streak && better heuristicts
                return CONSTANTS.GHOST_SCORE / 2
            elif ghost.pos == (y,x) and pacman.remBoostTime == 0:
                #return -2*game.GHOST_SCORE
                return 0
        return 0

    def coverPath(self, start, end, parents):
        sys.stderr.write("%s %s" % (start, end))
        while end != start:
            prev = parents[end[0]][end[1]]
            dir = (end[0]-prev[0], end[1]-prev[1])
            end = prev
        return dir

    def breadthFirstSearch(self, starts, evalFun, maxDist = sys.maxsize):
        height  = game.map["height"]
        width   = game.map["width"]
        state   = game.map["state"]
        dist    = np.zeros(shape = (height, width))
        targets = queue.Queue()
        for pos in starts:
            targets.put(pos)
            visited = [[False for _ in range(width)] for _ in range(height)]
            while not targets.empty():
                field = targets.get()
                for dir in [(0,1), (1,0), (-1,0), (0,-1)]:
                    y,x = field[0]+dir[0], field[1]+dir[1]
                    if ((0 <= y and y < height) and (0 <= x and x < width)) \
                    and (state[y][x] != CONSTANTS.FIELD_WALL and state[y][x] != CONSTANTS.FIELD_GHOST_WALL):
                        if not visited[y][x]:
                            dist[y][x] = dist[field[0]][field[1]] + 1
                            if dist[y][x] <= maxDist:
                                evalFun(y,x, field, dist[y][x])
                                visited[y][x] = True
                                targets.put((y,x))
        return dist

    def getStepCount(self):
        if self.remBoostTime > 0:
            return 2
        return 1

    def getDir(self, dir):
        if dir == (0,1):
            return '>'
        if dir == (1,0):
            return 'v'
        if dir == (0,-1):
            return '<'
        if dir == (-1,0):
            return '^'
        return "ERROR DIR"

    def getHeuristics(self):
        height  = game.map["height"]
        width   = game.map["width"]
        state   = game.map["state"]
        pos     = self.pos
        scores  = np.zeros(shape = (height, width))
        dangers = np.zeros(shape = (height, width))
        parents = [[None for _ in range(width)] for _ in range(height)]

        ghostsPos = []
        for _, ghost in game.ghosts.items():
            ## TODO: change boost handling
            if ghost.remStandbyTime <= 1 and (ghost.remEatableTime <= game.GHOST_DANGER_EATABLE_TIME \
            or self.remBoostTime <= CONSTANTS.SPEED_BOOSTER_DURATION / 4):
                y,x = ghost.pos
                pushDist = 0
                ## Push ghost by its dir
                while state[y][x] != CONSTANTS.FIELD_WALL \
                and state[y][x] != game.pacmans[game.pacmanId] \
                and pushDist <= game.GHOST_PUSH_DIST \
                and ghost.dir != (0,0):
                    ny,nx = y + ghost.dir[0], x + ghost.dir[1]
                    if not ((0 <= ny and ny < height) and (0 <= nx and nx < width)):
                        break
                    y,x = ny,nx
                    pushDist += 1
                dangers[y][x] = CONSTANTS.GHOST_SCORE / 2
                ghostsPos.append((y, x))

        def distAndScoreEval(y,x,field, dist):
            gain = scores[field[0]][field[1]] + self.getFieldScore((y,x))
            expense = dangers[y][x]
            scores[y][x]    = gain - expense
            parents[y][x]   = field

        def ghostDangerEval(y,x,field, dist):
            diff = dist * game.GHOST_SPREAD_DEATH_DECAY
            #sys.stderr.write("%.1f " % diff)
            dangers[y][x]   += max(0, diff)

        self.breadthFirstSearch(ghostsPos, ghostDangerEval, 
                                game.GHOST_SPREAD_DEATH_RADIUS)
        dist = self.breadthFirstSearch([self.pos], distAndScoreEval)

        ## TODO: Adjust distance multiplier
        distM = 2
        heatMap = scores - distM * dist 
        ## Making walls heat very low
        for i in range(height):
            for j in range(width):
                if dist[i][j] == 0:
                    heatMap[i,j] = -sys.maxsize
        index = np.argmax(heatMap)

        if game.tick % 10 == 0:
            for container in [heatMap]:
                for i in range(height):
                    for j in range(width):
                        if container[i][j] == -sys.maxsize:
                            sys.stderr.write("  ")
                        else:
                            sys.stderr.write("%d " % int(container[i][j]))
                    sys.stderr.write("\n")
                sys.stderr.write("\n\n")

        target = (int(index / width), int(index % width))
        sys.stderr.write("target: (%d,%d)\n" % (target[0], target[1]))

        ## TODO: Fix double step
        dir = self.coverPath(pos, target, parents)
        dirs = [dir]
        if self.getStepCount() > 1:
            dirs.append(self.coverPath((pos[0]+dir[0], pos[1]+dir[1]), target, parents))
        return dirs

    def move(self):
        dir1, *dir2 = self.getHeuristics()
        self.dir = self.getDir(dir1)
        if dir2 != []:
            self.dir += self.getDir(dir2[0])

game = Game()
while True:
    game.updateState()
    if game.pacmanId == -1:
        break

    pacman = game.pacmans[game.pacmanId]
    pacman.move()

    if game.currentMessage != []:
        sys.stderr.write("gameState:\n%spacman:\n%s\n" % (game, pacman))
    sys.stdout.write("%s %s %s %s\n" % (game.gameId, game.tick, game.pacmanId, pacman.dir))