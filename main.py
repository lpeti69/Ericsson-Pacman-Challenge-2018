import sys
import queue
import numpy as np

## Add more heat for utility points nearby
## Distributed positive heat generation
## Search for safety places => bonus heat for them
## Dont search paths which goes though a ghost when a ghost is 7(or x) distance away
## Bonus heat for territories/field when a certain criteria meets (example: 25 food in 6 radius neighbourhod)

class CNST(object):
    FIELD_WALL                          = 'F'
    FIELD_EMPTY                         = ' '
    FIELD_FOOD                          = '1'
    FIELD_BOOSTER                       = '+'
    FIELD_GHOST_WALL                    = 'G'
    SCORE_FOOD                          = 10
    SCORE_GHOST_EAT                     = 100
    SCORE_GHOST_DEATH                   = -200
    SCORE_BOOSTER                       = 50
    GHOST_DEATH_TIME                    = 5
    BOOSTER_DURATION                    = 21
    BOOSTER_GHOST_SCORE_MULTIPLICATOR   = 2
    MAX_TICK                            = 480


class Logger:
    def logContainer(self, container, evalFun):
        for i in range(len(container)):
            for j in range(len(container[i])):
                if evalFun(container[i][j]):
                    sys.stderr.write("%d " % container[i][j])
                else:
                    sys.stderr.write("A ")
            sys.stderr.write("\n")
        sys.stderr.write("\n")

class Game:
    def __init__(self):
        self.gameId                             = 0
        self.tick                               = 0
        self.pacmanId                           = 0
        self.currentMessage                     = ''
        
        self.pacmanId                           = 0
        self.GHOST_SPREAD_DEATH_RADIUS          = 10
        self.GHOST_SPREAD_DEATH_DECAY           = 4
        self.GHOST_PUSH_DIST                    = 1
        self.GHOST_DANGER_EATABLE_TIME          = 2
        self.pacmans                            = dict({})
        self.ghosts                             = dict({})
        self.map                                = dict({'width':0,'height':0})
        self.logger         = Logger()
        self.gameId         = 0
        self.tick           = 0
        self.pacmanId       = 0
        self.currentMessage = 0

    def setValues(self, gameId, tick, pacmanId, message = "NONE"):
        self.gameId    = gameId
        self.tick      = tick
        self.pacmanId  = pacmanId
        self.currentMessage = message
    
    def updateState(self):
        getLine = lambda: sys.stdin.readline().strip().split(" ")

        # Game meta data
        gameId, tick, pacmanId = list(map(int, getLine()))
        mapHeight, mapWidth, pacmanCount, ghostCount, *message = getLine()
        
        self.map['width']   = int(mapWidth)
        self.map['height']  = int(mapHeight)
        self.gameId         = gameId
        self.tick           = tick
        self.pacmanId       = pacmanId
        self.currentMessage = message
        
        # Map
        for i in range(int(mapHeight)):
            self.map[i] = sys.stdin.readline()[:int(mapWidth)]
        
        # Pacmans
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
        
        # Ghosts
        for _ in range(int(ghostCount)):
            id, y, x, remEatableTime, remStandbyTime = getLine()
            if game.tick == 0:
                self.ghosts[id] = Ghost(id = id)
            self.ghosts[id].updateStatus(
                pos                 = (int(y), int(x)),
                remEatableTime      = int(remEatableTime),
                remStandbyTime      = int(remStandbyTime)
            )


class Ghost:
    def __init__(self, id):
        self.id             = id
        self.direction      = (0,0)
        self.prevPos        = (0,0)
        
        self.pos            = (0,0)
        self.remEatableTime = 0
        self.remStandbyTime = 0

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
        self.dir            = (0,0)
        self.id             = id
        self.team           = team
        
        self.pos            = (0,0)
        self.score          = 0
        self.effects        = None
        self.remBoostTime   = 0
        self.bonusPoints    = 0

    def updateStatus(self, pos, score, remBoostTime, bonusPoints, effects):
        self.pos            = pos
        self.score          = score
        self.effects        = effects
        self.remBoostTime   = remBoostTime
        self.bonusPoints    = bonusPoints
    
    def getFieldScore(self, pos):
        y, x = pos
        field = game.map[y][x]
        # FIELD_FOOD
        if field == CNST.FIELD_FOOD:
            return CNST.SCORE_FOOD
        # FIELD_BOOSTER
        elif field == CNST.FIELD_BOOSTER:
            multiplier = sum(np.exp(1 / np.arange(1, CNST.BOOSTER_DURATION - self.remBoostTime)) - 1)
            return multiplier * CNST.SCORE_BOOSTER
        # Ghosts
        else:
            for _, ghost in game.ghosts.items():
                if ghost.pos == pos and pacman.remBoostTime > 0:
                    ## TODO: Add ghost kill streak && better heuristicts && distance
                    return CNST.SCORE_GHOST_EAT / 2
                elif ghost.pos == pos and pacman.remBoostTime == 0:
                    # return SCORE_GHOST_DEATH
                    return 0
        # otherwise
        return 0

    def coverPath(self, start, end, parents):
        sys.stderr.write("%s %s" % (start, end))
        while end != start:
            prev = parents[end[0]][end[1]]
            dir = (end[0]-prev[0], end[1]-prev[1])
            end = prev
        return dir

    def breadthFirstSearch(self, starts, evalFun, maxDist = sys.maxsize):
        width   = game.map['width']
        height  = game.map['height']
        dist    = np.zeros(shape=(height,width))
        walls = [CNST.FIELD_WALL, CNST.FIELD_GHOST_WALL]
        targets = queue.Queue()
        for pos in starts:
            targets.put(pos)
            visited = [[False for _ in range(width)] for _ in range(height)]
            visited[pos[0]][pos[1]] = True
            while not targets.empty():
                field = targets.get()
                for direction in [(0,1), (1,0), (-1,0), (0,-1)]:
                    # Map clip
                    y, x = field[0]+direction[0], field[1]+direction[1]
                    if y < 0 or y >= height:
                        y = height - y
                    if x < 0 or x >= width:
                        x = width - x
                    # Check
                    if (0 <= y and y < height) and (0 <= x and x < width) \
                    and game.map[y][x] not in walls and not visited[y][x]:
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
        width   = game.map['width']
        height  = game.map['height']
        pos     = self.pos
        scores  = np.zeros(shape=(height,width))
        dangers = np.zeros(shape=(height,width))
        parents = [[None for _ in range(width)] for _ in range(height)]

        ghostsPos = []
        for _, ghost in game.ghosts.items():
            ## TODO: change boost handling
            if ghost.remStandbyTime <= 1 \
            and (ghost.remEatableTime <= game.GHOST_DANGER_EATABLE_TIME \
            or self.remBoostTime <= CNST.BOOSTER_DURATION / 4):
                y, x = ghost.pos
                pushDist = 0
                ## Push ghost by its dir
                while game.map[y][x] != CNST.FIELD_WALL \
                and game.map[y][x] != game.pacmans[game.pacmanId] \
                and pushDist <= game.GHOST_PUSH_DIST \
                and ghost.dir != (0,0):
                    ny,nx = y + ghost.dir[0], x + ghost.dir[1]
                    if not ((0 <= ny and ny < height) and (0 <= nx and nx < width)):
                        break
                    y,x = ny,nx
                    pushDist += 1
                dangers[y][x] = CNST.SCORE_GHOST_EAT / 2
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
                game.logger.logContainer(container, lambda heatScore: heatScore != -sys.maxsize)

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