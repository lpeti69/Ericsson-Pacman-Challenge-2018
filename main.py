
import sys
import queue
import numpy as np

<<<<<<< HEAD
class CONSTANTS(object):
=======
class CNST(object):
>>>>>>> master
    FIELD_WALL                          = 'F'
    FIELD_EMPTY                         = ' '
    FIELD_FOOD                          = '1'
    FIELD_BOOSTER                       = '+'
    FIELD_GHOST_WALL                    = 'G'
<<<<<<< HEAD
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
=======
    SCORE_FOOD                          = 10
    SCORE_GHOST_EATH                    = 100
    SCORE_GHOST_DEATH                   = -200
    SCORE_BOOSTER                       = 50
    GHOST_DEATH_TIME                    = 5
    BOOSTER_DURATION                    = 21
    BOOSTER_GHOST_SCORE_MULTIPLICATOR   = 2
    MAX_TICK                            = 480


class Game:
    def __init__(self):
        self.gameId                             = 0
        self.tick                               = 0
        self.pacmanId                           = 0
        self.currentMessage                     = ''
        
        self.pacmanId                           = 0
>>>>>>> master
        self.GHOST_SPREAD_DEATH_RADIUS          = 10
        self.GHOST_SPREAD_DEATH_DECAY           = 4
        self.GHOST_PUSH_DIST                    = 1
        self.GHOST_DANGER_EATABLE_TIME          = 2
<<<<<<< HEAD
        self.pacmans                            = dict()
        self.ghosts                             = dict()
        self.map                                = dict()

    def __str__(self):
        return ("GAME_ID:{id}, PACMAN_ID:{pacmanId}, TICK:{tick}, MESSAGE:{message}\n"
            .format(id=self.gameId, pacmanId=self.pacmanId, tick=self.tick, message=self.currentMessage))

=======
        self.pacmans                            = dict({})
        self.ghosts                             = dict({})
        self.map                                = dict({'width':0,'height':0})
        
        self.gameId         = 0
        self.tick           = 0
        self.pacmanId       = 0
        self.currentMessage = 0

>>>>>>> master
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
<<<<<<< HEAD

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

=======
        
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
>>>>>>> master
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

<<<<<<< HEAD
        return 0


class Ghost:
    def __init__(self, id):
        self.id         = id
        self.dir        = (0,0)
        self.prevPos    = (0,0)
=======

class Ghost:
    def __init__(self, id):
        self.id             = id
        self.direction      = (0,0)
        self.prevPos        = (0,0)
        
        self.pos            = (0,0)
        self.remEatableTime = 0
        self.remStandbyTime = 0
>>>>>>> master

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
<<<<<<< HEAD
        self.dirs       = "v<>^"
        self.dir        = None
        self.id         = id
        self.team       = team

    def __str__(self):
        return ("ID:{id}, POS:({y},{x}), DIR:{dir}\n"
            .format(id=self.id, y=self.pos[0], x=self.pos[1], dir=self.dir))
=======
        self.dir            = (0,0)
        self.id             = id
        self.team           = team
        
        self.pos            = (0,0)
        self.score          = 0
        self.effects        = None
        self.remBoostTime   = 0
        self.bonusPoints    = 0
>>>>>>> master

    def updateStatus(self, pos, score, remBoostTime, bonusPoints, effects):
        self.pos            = pos
        self.score          = score
        self.effects        = effects
        self.remBoostTime   = remBoostTime
        self.bonusPoints    = bonusPoints
<<<<<<< HEAD

    
    def getFieldScore(self, pos):
        y,x = pos[0],pos[1]
        state = game.map["state"]
        if state[y][x] == CONSTANTS.FIELD_FOOD:
            return CONSTANTS.FOOD_SCORE
        if state[y][x] == CONSTANTS.FIELD_BOOSTER:
            multiplier = sum(np.exp(1 / np.arange(1, CONSTANTS.SPEED_BOOSTER_DURATION - self.remBoostTime)) - 1)
            return multiplier * CONSTANTS.SPEED_BOOSTER_SCORE
        for _, ghost in game.ghosts.items():
            if ghost.pos == (y,x) and pacman.remBoostTime > 0:
                ## TODO: Add ghost kill streak && better heuristicts
                return CONSTANTS.GHOST_SCORE / 2
            elif ghost.pos == (y,x) and pacman.remBoostTime == 0:
                #return -2*game.GHOST_SCORE
                return 0
=======
    
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
                    return CNST.SCORE_GHOST_EATH / 2
                elif ghost.pos == pos and pacman.remBoostTime == 0:
                    # return SCORE_GHOST_DEATH
                    return 0
        # otherwise
>>>>>>> master
        return 0

    def coverPath(self, start, end, parents):
        while end != start:
            prev = parents[end[0]][end[1]]
            dir = (end[0]-prev[0], end[1]-prev[1])
            end = prev
        return dir

    def breadthFirstSearch(self, starts, evalFun, maxDist = sys.maxsize):
<<<<<<< HEAD
        height  = game.map["height"]
        width   = game.map["width"]
        state   = game.map["state"]
        dist    = np.zeros(shape  = (height, width))
=======
        width   = game.map['width']
        height  = game.map['height']
        dist    = np.zeros(shape=(height,width))
>>>>>>> master
        targets = queue.Queue()
        for pos in starts:
            targets.put(pos)
            visited = [[False for _ in range(width)] for _ in range(height)]
            while not targets.empty():
                field = targets.get()
<<<<<<< HEAD
                for dir in [(0,1), (1,0), (-1,0), (0,-1)]:
                    y,x = field[0]+dir[0], field[1]+dir[1]
                    if ((0 <= y and y < height) and (0 <= x and x < width)) \
                    and (state[y][x] != CONSTANTS.FIELD_WALL and state[y][x] != CONSTANTS.FIELD_GHOST_WALL):
=======
                for direction in [(0,1), (1,0), (-1,0), (0,-1)]:
                    y,x = field[0]+direction[0], field[1]+direction[1]
                    if ((0 <= y and y < height) \
                    and (0 <= x and x < width)) \
                    and (game.map[y][x] != CNST.FIELD_WALL and game.map[y][x] != CNST.FIELD_GHOST_WALL):
>>>>>>> master
                        if not visited[y][x]:
                            dist[y][x] = dist[field[0]][field[1]] + 1
                            if dist[y][x] <= maxDist:
                                evalFun(y,x, field, dist[y][x])
                                visited[y][x] = True
                                targets.put((y,x))
        return dist

    def getHeuristics(self):
<<<<<<< HEAD
        height  = game.map["height"]
        width   = game.map["width"]
        state   = game.map["state"]
        pos     = self.pos
        scores  = np.zeros(shape = (height, width))
        dangers = np.zeros(shape = (height, width))
=======
        width   = game.map['width']
        height  = game.map['height']
        pos     = self.pos
        scores  = np.zeros(shape=(height,width))
        dangers = np.zeros(shape=(height,width))
>>>>>>> master
        parents = [[None for _ in range(width)] for _ in range(height)]

        ghostsPos = []
        for _, ghost in game.ghosts.items():
            ## TODO: change boost handling
<<<<<<< HEAD
            if ghost.remStandbyTime <= 1 and (ghost.remEatableTime <= game.GHOST_DANGER_EATABLE_TIME \
            or self.remBoostTime <= CONSTANTS.SPEED_BOOSTER_DURATION / 4):
                y,x = ghost.pos
                pushDist = 0
                ## Push ghost by its dir
                while state[y][x] != CONSTANTS.FIELD_WALL \
                and state[y][x] != game.pacmans[game.pacmanId] \
=======
            if ghost.remStandbyTime <= 1 \
            and (ghost.remEatableTime <= game.GHOST_DANGER_EATABLE_TIME \
            or self.remBoostTime <= CNST.BOOSTER_DURATION / 4):
                y, x = ghost.pos
                pushDist = 0
                ## Push ghost by its dir
                while game.map[y][x] != CNST.FIELD_WALL \
                and game.map[y][x] != game.pacmans[game.pacmanId] \
>>>>>>> master
                and pushDist <= game.GHOST_PUSH_DIST \
                and ghost.dir != (0,0):
                    ny,nx = y + ghost.dir[0], x + ghost.dir[1]
                    if not ((0 <= ny and ny < height) and (0 <= nx and nx < width)):
                        break
                    y,x = ny,nx
                    pushDist += 1
<<<<<<< HEAD
                dangers[y][x] = CONSTANTS.GHOST_SCORE / 2
=======
                dangers[y][x] = CNST.SCORE_GHOST_EATH / 2
>>>>>>> master
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

<<<<<<< HEAD
        if game.tick % 10 == 0:
            for container in [scores, dangers]:
                for i in range(height):
                    for j in range(width):
                        sys.stderr.write("%d " % int(container[i][j]))
                    sys.stderr.write("\n")
                sys.stderr.write("\n\n")

        target = (int(index / width), int(index % width))
        sys.stderr.write("target: (%d,%d)\n" % (target[0], target[1]))
=======
        #if game.tick % 10 == 0:
        #    for container in [scores, dangers]:
        #        for i in range(height):
        #            for j in range(width):
        #                sys.stderr.write("%d " % int(container[i][j]))
        #            sys.stderr.write("\n")
        #        sys.stderr.write("\n\n")

        target = (int(index / width), int(index % width))
        #sys.stderr.write("target: (%d,%d)\n" % (target[0], target[1]))
>>>>>>> master

        ## TODO: Fix double step
        dirs = [self.coverPath(pos, target, parents)]
        if self.remBoostTime > 0:
            dirs += self.coverPath((pos[0]+dirs[0][0], pos[1]+dirs[0][1]), target, parents)
        return dirs


    def move(self):
        dirs = self.getHeuristics()
        self.dir = ''
        for dir in dirs:
            if dir == (0,1):
                self.dir += '>'
            if dir == (1,0):
                self.dir += 'v'
            if dir == (0,-1):
                self.dir += '<'
            if dir == (-1,0):
                self.dir += '^'

game = Game()
while True:
    game.updateState()
    if game.pacmanId == -1:
        break

    pacman = game.pacmans[game.pacmanId]
    pacman.move()

    if game.currentMessage != []:
        sys.stderr.write("gameState:\n%spacman:\n%s\n" % (game, pacman))
    sys.stdout.write("%s %s %s %c\n" % (game.gameId, game.tick, game.pacmanId, pacman.dir))