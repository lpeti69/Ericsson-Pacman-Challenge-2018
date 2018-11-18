# pacman.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
Pacman.py holds the logic for the classic pacman game along with the main
code to run a game.  This file is divided into three sections:

  (i)  Your interface to the pacman world:
          Pacman is a complex environment.  You probably don't want to
          read through all of the code we wrote to make the game runs
          correctly.  This section contains the parts of the code
          that you will need to understand in order to complete the
          project.  There is also some code in game.py that you should
          understand.

  (ii)  The hidden secrets of pacman:
          This section contains all of the logic code that the pacman
          environment uses to decide who can move where, who dies when
          things collide, etc.  You shouldn't need to read this section
          of code, but you can if you want.

  (iii) Framework to start a game:
          The final section contains the code for reading the command
          you use to set up the game, then starting up a new game, along with
          linking in all the external parts (agent functions, graphics).
          Check this section out to see all the options available to you.

To play your first game, type 'python pacman.py' from the command line.
The keys are 'a', 's', 'd', and 'w' to move (or arrow keys).  Have fun!
"""

class CNST(object):
    FIELD_WALL                          = 'F'
    FIELD_EMPTY                         = ' '
    FIELD_FOOD                          = '1'
    FIELD_BOOSTER                       = '+'
    FIELD_GHOST_WALL                    = 'G'
    SCORE_FOOD                          = 10
    SCORE_GHOST_EAT                     = 50
    SCORE_GHOST_DEATH                   = -200
    SCORE_BOOSTER                       = 50
    GHOST_DEATH_TIME                    = 5
    BOOSTER_DURATION                    = 21
    BOOSTER_GHOST_SCORE_MULTIPLICATOR   = 2
    MAX_TICK                            = 480

## TODO:
## Change misc funs in GameState
##

from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from game import Configuration
from util import nearestPoint
from util import manhattanDistance
import util, layout
import sys, types, time, random, os

###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################

class GameState:
    """
    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes.

    GameStates are used by the Game object to capture the actual state of the game and
    can be used by agents to reason about the game.

    Much of the information in a GameState is stored in a GameStateData object.  We
    strongly suggest that you access that data via the accessor methods below rather
    than referring to the GameStateData object directly.

    Note that in classic Pacman, Pacman is always agent 0.
    """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    # static variable keeps track of which states have had getLegalActions called
    explored = set()
    def getAndResetExplored():
        tmp = GameState.explored.copy()
        GameState.explored = set()
        return tmp
    getAndResetExplored = staticmethod(getAndResetExplored)

    def getLegalActions( self, agentIndex=0 ):
        """
        Returns the legal actions for the agent specified.
        """
#        GameState.explored.add(self)
        if self.isGameOver(): return []

        if self.isPacman(agentIndex):  # Pacman is moving
            return PacmanRules.getLegalActions( self, agentIndex )
        else:
            return GhostRules.getLegalActions( self, agentIndex )

    def generateSuccessor( self, agentIndex, action):

        ## TODO: ADD timing
        """
        Returns the successor state after the specified agent takes the action.
        """
        # Check that successors exist
        if self.isGameOver(): raise Exception('Can\'t generate a successor of a terminal state.')

        # Copy current state
        state = GameState(self)

        if state.data.tick == CNST.MAX_TICK:
            state.data._isGameOver = True
        else:
            # Let agent's logic deal with its action's effects on the board
            if self.isPacman(agentIndex):  # Pacman is moving
                state.data._eaten = [False for i in range(state.getNumAgents())]
                PacmanRules.applyAction( state, action, agentIndex )
            else:                # A ghost is moving
                GhostRules.applyAction( state, action, agentIndex )

            # Time passes
            if self.isPacman(agentIndex):
                state.data.reward[agentIndex] += -TIME_PENALTY # Penalty for waiting around
            else:
                GhostRules.decrementTimer( state.data.agentStates[agentIndex] )

            # Resolve multi-agent effects
            GhostRules.checkDeath( state, agentIndex )
            
            # Resolve pacman-pacman collisions
            if self.isPacman(agentIndex):
                PacmanRules.checkCollision(state, agentIndex)

            # Book keeping
            state.data._agentMoved = agentIndex
            state.data.score[agentIndex] += state.data.reward[agentIndex]
            GameState.explored.add(self)
            GameState.explored.add(state)
        return state
    
    def isPacman(self, agentIndex):
        if agentIndex < 0 or agentIndex >= self.getNumAgents():
            raise Exception("%d: Index out of range" % agentIndex)
        return self.data.agentStates[agentIndex].isPacman

    def getLegalPacmanActions( self, agentIndex ):
        if not self.isPacman(agentIndex):
            raise Exception("Ghost's index passed to getLegalPacmanActions")
        return self.getLegalActions( agentIndex )

    def generatePacmanSuccessor( self, agentIndex, action ):
        """
        Generates the successor state after the specified pacman move
        """
        if not self.isPacman(agentIndex):
            raise Exception("Ghost's index passed to generatePacmanSuccessor")
        return self.generateSuccessor( agentIndex, action )

    def getMyPacmanState(self):
        return self.getPacmanState(0)

    def getPacmanState( self, agentIndex ):
        """
        Returns an AgentState object for pacman (in game.py)

        state.pos gives the current position
        state.direction gives the travel vector
        """
        if not self.isPacman(agentIndex):
            raise Exception("Ghost's index passed to getPacmanState")
        return self.data.agentStates[agentIndex].copy()

    def getMyPacmanPosition(self):
        return self.getPacmanPosition(0)

    def getPacmanPosition( self, agentIndex ):
        if not self.isPacman(agentIndex):
            raise Exception("Ghost's index passed to getPacmanPosition")
        return self.data.agentStates[agentIndex].getPosition()

    def getEnemyPacmanStates(self):
        return [pacmanState for agent in self.data.agentStates if agent.isPacman and agent.index > 0]

    def getEnemyPacmanPositions(self):
        return [enemy.getPosition() for enemy in self.getEnemyPacmanStates()]

    def getGhostStates( self ):
        return [ghostState for agent in self.data.agentStates if not agent.isPacman]

    def getGhostState( self, agentIndex ): ## todo atirni
        if self.isPacman(agentIndex) or agentIndex >= self.getNumAgents():
            raise Exception("Invalid index passed to getGhostState")
        return self.data.agentStates[agentIndex]

    def getGhostPosition( self, agentIndex ): ## todo
        if self.isPacman(agentIndex):
            raise Exception("Pacman's index passed to getGhostPosition")
        return self.data.agentStates[agentIndex].getPosition()

    def getGhostPositions(self): ## todo
        return [s.getPosition() for s in self.getGhostStates()]

    def getNumGhost(self):
        return self.data.numGhosts

    def getNumAgents( self ): ## todo
        return len( self.data.agentStates )

    def getScore( self, agentIndex ):
        if agentIndex < 0 or agentIndex >= self.getNumAgents():
            raise Exception("Wrong index added to getScore")
        return float(self.data.score[agentIndex])

    def getCapsules(self):
        """
        Returns a list of positions (x,y) of the remaining capsules.
        """
        return self.data.capsules

    def getNumFood( self ):
        return self.data.food.count()

    def getFood(self):
        """
        Returns a Grid of boolean food indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        currentFood = state.getFood()
        if currentFood[x][y] == True: ...
        """
        return self.data.food

    def getWalls(self):
        """
        Returns a Grid of boolean wall indicator variables.

        Grids can be accessed via list notation, so to check
        if there is a wall at (x,y), just call

        walls = state.getWalls()
        if walls[x][y] == True: ...
        """
        return self.data.layout.walls

    def hasFood(self, x, y):
        return self.data.food[x][y]

    def hasWall(self, x, y):
        return self.data.layout.walls[x][y] == 1

    def hasGhostWall(self, x, y):
        return self.data.layout.walls[x][y] == 2

    def isGameOver( self ):
        return self.data._isGameOver

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__( self, prevState = None ):
        """
        Generates a new state by copying information from its predecessor.
        """
        if prevState != None: # Initial state
            self.data = GameStateData(prevState.data)
        else:
            self.data = GameStateData()

    def deepCopy( self ):
        state = GameState( self )
        state.data = self.data.deepCopy()
        return state

    def __eq__( self, other ):
        """
        Allows two states to be compared.
        """
        return hasattr(other, 'data') and self.data == other.data

    def __hash__( self ):
        """
        Allows states to be keys of dictionaries.
        """
        return hash( self.data )

    def __str__( self ):

        return str(self.data)

    def initialize( self, layout, numGhostAgents=1000, numEnemyPacmanAgents = 3 ):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.data.initialize(layout, numGhostAgents, numEnemyPacmanAgents)

############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

SCARED_TIME = CNST.BOOSTER_DURATION    # Moves ghosts are scared
COLLISION_TOLERANCE = 0.7 # How close ghosts must be to Pacman to kill TODO
TIME_PENALTY = 1 # Number of points lost each round TODO

class EriccsonPacmanRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """
    def __init__(self, timeout=30):
        self.timeout = timeout

    def newGame( self, layout, pacmanAgents, ghostAgents, display, quiet = False, catchExceptions=False):
        ## TODO: Pacman index 0, enemy pacmans [1,k], ghosts [k+1,n]
        ## TODO: Check if merge agent arrays or not
        agents = pacmanAgents[:1] + ghostAgents[:layout.getNumGhosts()] + pacmanAgents[1:layout.getNumEnemyPacmans() + 1]
        initState = GameState()
        ## Sets underlying GameState's gamestatedata from game.py
        initState.initialize( layout, len(ghostAgents), len(pacmanAgents) - 1 )
        game = Game(agents, display, self, catchExceptions=catchExceptions)
        game.state = initState
        self.initialState = initState.deepCopy()
        self.quiet = quiet
        return game

    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if not self.quiet and state.isGameOver(): 
            print "Game ended, Scores:\n(Agent,Score)"
            i = 0
            for score in state.data.score:
                print "(%d,%d)" % (i, score)
                i += 1
            game.gameOver = True

    def getProgress(self, game):
        return float(game.state.getNumFood()) / self.initialState.getNumFood()

    def agentCrash(self, game, agentIndex):
        if self.game.state.isPacman(agentIndex):
            if agentIndex == 0:
                print "Our pacman crashed"
            else:
                print "Other pacman crashed"
        else:
            print "A ghost crashed"

    def getMaxTotalTime(self, agentIndex):
        return self.timeout

    def getMaxStartupTime(self, agentIndex):
        return self.timeout

    def getMoveWarningTime(self, agentIndex):
        return self.timeout

    def getMoveTimeout(self, agentIndex):
        return self.timeout

    def getMaxTimeWarnings(self, agentIndex):
        return 0

class PacmanRules:
    """
    These functions govern how pacman interacts with his environment under
    the classic game rules.
    """
    PACMAN_SPEED=1

    def getLegalActions( state, agentIndex ):
        """
        Returns a list of possible actions.
        """
        return Actions.getPossibleActions( state.getPacmanState(agentIndex).configuration, state.data.layout.walls, [1,2] )
    getLegalActions = staticmethod( getLegalActions )

    def applyAction( state, action, agentIndex ):
        """
        Edits the state to reflect the results of the action.
        """
        legal = PacmanRules.getLegalActions( state, agentIndex )
        if action not in legal:
            raise Exception("Illegal action " + str(action))

        pacmanState = state.data.agentStates[agentIndex]

        # Update Configuration
        speed = PacmanRules.PACMAN_SPEED
        if pacmanState.boosterTimer > 0:
            pacmanState.boosterTimer -= 1
            speed *= 2
        vector = Actions.directionToVector( action, speed ) ## TODO?
        pacmanState.configuration = pacmanState.configuration.generateSuccessor( vector )

        # Eat
        next = pacmanState.configuration.getPosition()
        nearest = nearestPoint( next )
        if manhattanDistance( nearest, next ) <= 0.5 :
            # Remove food
            PacmanRules.consume( nearest, state, agentIndex )
    applyAction = staticmethod( applyAction )

    def consume( position, state, agentIndex ):
        x,y = position
        # Eat food
        if state.data.food[x][y]:
            state.data.reward[agentIndex] += CNST.SCORE_FOOD  ## TODO: Change for quest
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._foodEaten = position
            # TODO: cache numFood?
            numFood = state.getNumFood()
            if numFood == 0 and not state.data._isGameOver:
                state.data.reward += 5 ## TODO
                state.data._isGameOver = True
        # Eat capsule
        if( position in state.getCapsules() ): ## TODO: Set scared time tracking for every pacman
            state.data.capsules.remove( position )
            state.data._capsuleEaten = position ## TODO: Add score
            state.data.reward[agentIndex] = CNST.SCORE_BOOSTER
            # Reset all ghosts' scared timers
            for index in range( 1, state.data.numGhosts + 1 ): ## TODO: only ghosts
                state.data.agentStates[index].scaredTimer[agentIndex] = SCARED_TIME ## TODO: change?
    consume = staticmethod( consume )

    def checkCollision( state, agentIndex ):
        pass ## TODO
    checkCollision = staticmethod( checkCollision )

class GhostRules: ## TODO: Calculate for general pacmans
    """
    These functions dictate how ghosts interact with their environment.
    """
    GHOST_SPEED=1.0
    def getLegalActions( state, ghostIndex ): ## TODO
        """
        Ghosts cannot stop, and cannot turn around unless they
        reach a dead end, but can turn 90 degrees at intersections.
        """
        conf = state.getGhostState( ghostIndex ).configuration
        possibleActions = Actions.getPossibleActions( conf, state.data.layout.walls, [1] )
        reverse = Actions.reverseDirection( conf.direction )
        if Directions.STOP in possibleActions:
            possibleActions.remove( Directions.STOP )
        if reverse in possibleActions and len( possibleActions ) > 1:
            possibleActions.remove( reverse )
        return possibleActions
    getLegalActions = staticmethod( getLegalActions )

    def applyAction( state, action, ghostIndex):

        legal = GhostRules.getLegalActions( state, ghostIndex )
        if action not in legal:
            raise Exception("Illegal ghost action " + str(action) + " " + str(state.data.agentStates[ghostIndex]) + " " + str(ghostIndex))

        ghostState = state.getGhostState(ghostIndex)
        speed = GhostRules.GHOST_SPEED
        vector = Actions.directionToVector( action, speed )
        ghostState.configuration = ghostState.configuration.generateSuccessor( vector )
    applyAction = staticmethod( applyAction )

    def decrementTimer( ghostState):
        timer = ghostState.scaredTimer[0]
        if timer == 1:
            ghostState.configuration.pos = nearestPoint( ghostState.configuration.pos )
        ghostState.scaredTimer[0] = max( 0, timer - 1 )
    decrementTimer = staticmethod( decrementTimer )

    def checkDeath( state, agentIndex): ## TODO check for all pacmans
        pacmanPosition = state.getPacmanPosition(0)
        if agentIndex == 0: # Pacman just moved; Anyone can kill him
            for index in range( 1, state.data.numGhosts + 1 ):
                ghostState = state.data.agentStates[index]
                ghostPosition = ghostState.configuration.getPosition()
                if GhostRules.canKill( pacmanPosition, ghostPosition ):
                    GhostRules.collide( state, ghostState, index )
        else:
            ghostState = state.data.agentStates[agentIndex]
            ghostPosition = ghostState.configuration.getPosition()
            if GhostRules.canKill( pacmanPosition, ghostPosition ):
                GhostRules.collide( state, ghostState, agentIndex )
    checkDeath = staticmethod( checkDeath )

    def collide( state, ghostState, agentIndex): ## TODO apply for all pacmans
        if ghostState.scaredTimer[0] > 0:
            state.data.reward[agentIndex] += CNST.SCORE_GHOST_EAT ## TODO
            ##state.data.reward += CNST.SCORE_GHOST_EAT
            GhostRules.placeGhost(state, 
                                  ghostState,
                                  GhostRules.searchNearestGhostWall(state, agentIndex),
                                  ghostState.start.direction) ## todo
            ghostState.scaredTimer[0] = 0
            # Added for first-person
            state.data._eaten[agentIndex] = True ## TODO
        else:
            if not state.data._isGameOver:
                state.data.reward[agentIndex] -= 500 ## TODO
                state.data._isGameOver = True
    collide = staticmethod( collide )

    def searchNearestGhostWall(state, agentIndex):
        pos = state.getGhostPosition(agentIndex)
        layout = state.data.layout
        minDist, ghostWallPos = layout.width*layout.height, (0,0)
        for x in range(layout.width):
            for y in range(layout.height):
                if layout.walls[x][y] == 2:
                    dist = (pos[0]-x)**2 + (pos[1]-y)**2 ## TODO: Manhattan or euk?
                    if dist < minDist:
                        minDist = dist
                        ghostWallPos = (x,y)

        return ghostWallPos
    searchNearestGhostWall = staticmethod( searchNearestGhostWall )

    def canKill( pacmanPosition, ghostPosition ):
        return manhattanDistance( ghostPosition, pacmanPosition ) <= COLLISION_TOLERANCE
    canKill = staticmethod( canKill )

    def placeGhost(state, ghostState, pos, direction):
        ghostState.configuration = Configuration(pos, direction)
    placeGhost = staticmethod( placeGhost )

#############################
# FRAMEWORK TO START A GAME #
#############################

def default(str):
    return str + ' [Default: %default]'

def parseAgentArgs(str):
    if str == None: return {}
    pieces = str.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=')
        else:
            key,val = p, 1
        opts[key] = val
    return opts

def readCommand( argv ):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python pacman.py <options>
    EXAMPLES:   (1) python pacman.py
                    - starts an interactive game
                (2) python pacman.py --layout smallClassic --zoom 2
                OR  python pacman.py -l smallClassic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    """
    parser = OptionParser(usageStr)

    parser.add_option('-n', '--numGames', dest='numGames', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=1)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('the LAYOUT_FILE from which to load the map layout'),
                      metavar='LAYOUT_FILE', default='mediumClassic')
    parser.add_option('-p', '--pacman', dest='pacman',
                      help=default('the agent TYPE in the pacmanAgents module to use'),
                      metavar='TYPE', default='KeyboardAgent')
    parser.add_option('-t', '--textGraphics', action='store_true', dest='textGraphics',
                      help='Display output as text only', default=False)
    parser.add_option('-q', '--quietTextGraphics', action='store_true', dest='quietGraphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_option('-g', '--ghosts', dest='ghost',
                      help=default('the ghost agent TYPE in the ghostAgents module to use'),
                      metavar = 'TYPE', default='RandomGhost')
    parser.add_option('-e', '--enemy_pacmans', dest='enemyPacmans', ## TODO: Added
                      help=default('the enemy pacman agent TYPE in the pacmanAgents module to use'),
                      metavar = 'TYPE', default='GreedyAgent')
    parser.add_option('-u', '--num_enemy_pacmans', dest='numEnemyPacmans', ## TODO: Added
                      help=default('The maximum number of enemy pacmans while generation'), 
                      type='int', default=3)
    parser.add_option('-k', '--numghosts', type='int', dest='numGhosts',
                      help=default('The maximum number of ghosts to use'), default=4)
    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help=default('Zoom the size of the graphics window'), default=1.0)
    parser.add_option('-f', '--fixRandomSeed', action='store_true', dest='fixRandomSeed',
                      help='Fixes the random seed to always play the same game', default=False)
    parser.add_option('-r', '--recordActions', action='store_true', dest='record',
                      help='Writes game histories to a file (named by the time they were played)', default=False)
    parser.add_option('--replay', dest='gameToReplay',
                      help='A recorded game file (pickle) to replay', default=None)
    parser.add_option('-a','--agentArgs',dest='agentArgs',
                      help='Comma separated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"')
    parser.add_option('-x', '--numTraining', dest='numTraining', type='int',
                      help=default('How many episodes are training (suppresses output)'), default=0)
    parser.add_option('--frameTime', dest='frameTime', type='float',
                      help=default('Time to delay between frames; <0 means keyboard'), default=0.1)
    parser.add_option('-c', '--catchExceptions', action='store_true', dest='catchExceptions',
                      help='Turns on exception handling and timeouts during games', default=False)
    parser.add_option('--timeout', dest='timeout', type='int',
                      help=default('Maximum length of time an agent can spend computing in a single game'), default=30)

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = dict()

    # Fix the random seed
    if options.fixRandomSeed: random.seed('cs188')

    # Choose a layout
    args['layout'] = layout.getLayout( options.layout )
    if args['layout'] == None: raise Exception("The layout " + options.layout + " cannot be found")

    # Choose a Pacman agent
    noKeyboard = options.gameToReplay == None and (options.textGraphics or options.quietGraphics)
    pacmanType = loadAgent(options.pacman, noKeyboard)
    agentOpts = parseAgentArgs(options.agentArgs)
    if options.numTraining > 0:
        args['numTraining'] = options.numTraining
        if 'numTraining' not in agentOpts: agentOpts['numTraining'] = options.numTraining
    pacman = pacmanType(0, **agentOpts) # Instantiate Pacman with agentArgs ## TODO: Pacman always with 0 index
    args['pacmans'] = [pacman]

    # Don't display training games
    if 'numTrain' in agentOpts:
        options.numQuiet = int(agentOpts['numTrain'])
        options.numIgnore = int(agentOpts['numTrain'])
    
    # Choose a ghost agent
    ghostType = loadAgent(options.ghost, noKeyboard)
    args['ghosts'] = [ghostType(i + 1) for i in range( options.numGhosts )]

    ## Choose an enemy pacman agent
    enemyPacmanType = loadAgent(options.enemyPacmans, noKeyboard)

    args['pacmans'] += [enemyPacmanType(i + options.numGhosts + 1) for i in range(options.numEnemyPacmans)]





    # Choose a display format
    if options.quietGraphics:
        import textDisplay
        args['display'] = textDisplay.NullGraphics()
    elif options.textGraphics:
        import textDisplay
        textDisplay.SLEEP_TIME = options.frameTime
        args['display'] = textDisplay.PacmanGraphics()
    else:
        import graphicsDisplay
        args['display'] = graphicsDisplay.PacmanGraphics(options.zoom, frameTime = options.frameTime)
    args['numGames'] = options.numGames
    args['record'] = options.record
    args['catchExceptions'] = options.catchExceptions
    args['timeout'] = options.timeout

    # Special case: recorded games don't use the runGames method or args structure
    if options.gameToReplay != None:
        print 'Replaying recorded game %s.' % options.gameToReplay
        import cPickle
        f = open(options.gameToReplay)
        try: recorded = cPickle.load(f)
        finally: f.close()
        recorded['display'] = args['display']
        replayGame(**recorded)
        sys.exit(0)

    return args

def loadAgent(pacman, nographics):
    # Looks through all pythonPath Directories for the right module,
    pythonPathStr = os.path.expandvars("$PYTHONPATH")
    if pythonPathStr.find(';') == -1:
        pythonPathDirs = pythonPathStr.split(':')
    else:
        pythonPathDirs = pythonPathStr.split(';')
    pythonPathDirs.append('.')

    for moduleDir in pythonPathDirs:
        if not os.path.isdir(moduleDir): continue
        moduleNames = [f for f in os.listdir(moduleDir) if f.endswith('gents.py')]
        for modulename in moduleNames:
            try:
                module = __import__(modulename[:-3])
            except ImportError:
                continue
            if pacman in dir(module):
                if nographics and modulename == 'keyboardAgents.py':
                    raise Exception('Using the keyboard requires graphics (not text display)')
                return getattr(module, pacman)
    raise Exception('The agent ' + pacman + ' is not specified in any *Agents.py.')

def replayGame( layout, actions, display ):
    import pacmanAgents, ghostAgents
    rules = EriccsonPacmanRules()
    agents = [pacmanAgents.GreedyAgent()] + [ghostAgents.RandomGhost(i+1) for i in range(layout.getNumGhosts())]
    game = rules.newGame( layout, agents[0], agents[1:], display )
    state = game.state
    display.initialize(state.data)

    for action in actions:
            # Execute the action
        state = state.generateSuccessor( *action )
        # Change the display
        display.update( state.data )
        # Allow for game specific conditions (winning, losing, etc.)
        rules.process(state, game)

    display.finish()

def runGames( layout, pacmans, ghosts, display, numGames, record, numTraining = 0, catchExceptions=False, timeout=30 ):
    import __main__
    __main__.__dict__['_display'] = display

    rules = EriccsonPacmanRules(timeout)
    games = []

    for i in range( numGames ):
        beQuiet = i < numTraining
        if beQuiet:
                # Suppress output and graphics
            import textDisplay
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display
            rules.quiet = False
        game = rules.newGame( layout, pacmans, ghosts, gameDisplay, beQuiet, catchExceptions) ## TODO
        print game.state
        game.run()
        if not beQuiet: games.append(game)

        if record:
            import time, cPickle
            fname = ('recorded-game-%d' % (i + 1)) +  '-'.join([str(t) for t in time.localtime()[1:6]])
            f = file(fname, 'w')
            components = {'layout': layout, 'actions': game.moveHistory}
            cPickle.dump(components, f)
            f.close()

    if (numGames-numTraining) > 0:
        scores = [game.state.getScore(0) for game in games] ## TODO: print other scores
        print 'Average Score:', sum(scores) / float(len(scores))
        print 'Scores:       ', ', '.join([str(score) for score in scores])
        print 'Record:        %d' % max(scores)

    return games

if __name__ == '__main__':
    """
    The main function called when pacman.py is run
    from the command line:

    > python pacman.py

    See the usage string for more details.

    > python pacman.py --help
    """
    args = readCommand( sys.argv[1:] ) # Get game components based on input
    runGames( **args )

    # import cProfile
    # cProfile.run("runGames( **args )")
    pass
