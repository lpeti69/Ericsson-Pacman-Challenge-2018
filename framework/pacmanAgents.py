# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
from util import nearestPoint
from util import BFS
import copy, sys
import distanceCalculator
import numpy as np
import random, time, game, util

class LeftTurnAgent(Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions(self.index)
        current = state.getPacmanState(self.index).configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

class GreedyAgent(Agent):
    def __init__(self, index, evalFn="scoreEvaluation"):
        self.index = index
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions(self.index)
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(self.index, action), action) for action in legal]
        scored = [(self.evaluationFunction(state, self.index), action) for state, action in successors]
        if len(scored) == 0:
            return Directions.STOP
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)


# ide csak absztrakt dolgok jonnek
# foleg static meg egyeb helper..
class ReinforcementLearningAgent(Agent):
    def __init__( self, index, timeForComputing = .25 ):
        self.index = index
        self.observationHistory = []
        self.timeForComputing = timeForComputing
        self.computationTimes = []
        self.approximators = []
        self.isTraining = True
        self.display = None
        self.numthGame = 1

    def __str__(self):
        return "AgentState: {}\Weights: {}\nComputationTimes: {}".format(
            self.observationHistory[-1].data.agentStates[self.index],
            self.weights,
            self.computationTimes
        )

    def setisTraining(self, isTraining):
        self.isTraining = isTraining

    def observationFunction(self, gameState):
        return gameState.deepCopy()

    def registerInitialState(self, gameState):
        self.observationHistory = []
        self.numthGame += 1
        import __main__
        if '_display' in dir(__main__):
            self.display = __main__._display

    def final(self, gameState, fileName = 'weights.txt'):
        print "Avg time for evaulate: {}".format(sum(self.computationTimes) / float(len(self.computationTimes)))
        print self.weights
        with open(fileName, 'w') as file:
            for weight in self.weights:
                file.write(str(weight) + '\n')

    def updateWeights(self, state, action, alpha = .1, gamma = .9):
        ## TODO: Need to handle r(it is not provided, since the state reward not updated yet)
        Qsa = self.evaluate(state, action)
        newState = self.getSuccessor(state, action)
        r = newState.data.reward[self.index]
        ## -1 provided for epsilon => it will select according to the current strategy
        newAction = self.getPolicyAction(newState)
        maxQsa = self.evaluate(newState, newAction)
        ## TODO: Optimalize: vector function?
        #print r, self.weights, maxQsa, Qsa
        for i in range(len(self.features)):
            self.weights[i] = self.weights[i] + alpha*(r + gamma*maxQsa - Qsa)*self.features[i](newState)#(float(self.features[i](newState)) - float(self.features[i](state)))
            self.weights[i] /= np.linalg.norm(self.weights)
            #print self.features[i](newState)
        #print self.weights

    ## Approximates Q(s,a)
    def evaluate(self, gameState, action):
        if action == Directions.STOP:
            return 0
        successor = self.getSuccessor(gameState, action)
        #pfs = [feature(gameState) for feature in self.features]
        #cfs = [feature(successor) for feature in self.features]
        Qsa = 0
        for i, weight in enumerate(self.weights):
            Qsa += weight * self.features[i](successor)#(float(cfs[i]) - float(pfs[i]))
        return Qsa

    def getSuccessor(self, gameState, action):
        successor = gameState.deepCopy().generateSuccessor(self.index, action)
        pos = successor.getMyPacmanPosition()
        ## DEBUG
        lastPos = gameState.getMyPacmanPosition()
        currPos = successor.getMyPacmanPosition()
        #print "reward: {}, ({},{}) -> ({},{})".format(successor.data.reward[self.index], lastPos[0], lastPos[1], currPos[0], currPos[1])
        ## END DEBUG
        if pos != nearestPoint(pos):
            return successor.deepCopy().generateSuccessor(self.index, action)
        else:
            return successor

    def getAction(self, gameState, epsilonDecay = .99, alphaDecay = .75):
        self.observationHistory.append(gameState)
        myPos = gameState.getMyPacmanPosition()
        if myPos != nearestPoint(myPos):
            return gameState.getLegalActions(self.index)[0]
        
        epsilon = np.power(epsilonDecay, gameState.data.tick + 3*self.numthGame)
        #alpha   = np.power(alphaDecay, self.numthGame)
        alpha = 0.1
        action  = self.chooseAction(gameState, epsilon)
        if self.isTraining:
            self.updateWeights(gameState, action, alpha)
        return action

    def chooseAction(self, gameState, epsilon):
        util.raiseNotDefined()

class MyAgent(ReinforcementLearningAgent):
    def __init__(self, index = 0, weights = []):
        ReinforcementLearningAgent.__init__(self, index)
        radius = 10
        self.features = [
            lambda state: 1,
            lambda state: self.closest(state, lambda s, x, y: s.hasFood(x,y))[1],
            lambda state: 1/(self.closest(state, lambda s, x, y: s.hasFood(x,y))[1])**2,
            lambda state: self.closest(state, lambda s, x, y: (x,y) in s.getCapsules())[1],
            self.isEnemyGhostXStepsAway(1, active=True),
            self.isEnemyGhostXStepsAway(2, active=True),
            self.isEnemyGhostXStepsAway(1, active=False),
            self.isEnemyGhostXStepsAway(2, active=False),
            lambda state: self.closestGhost(state, active=True),
            lambda state: self.closestGhost(state, active=False),
            lambda state: self.closestEnemyPacman(state, higherScore=True),
            lambda state: self.closestEnemyPacman(state, higherScore=False),
            self.isEnemyPacmanXStepsAway(3, higherScore = False),
            self.numScaredGhostsXStepsAway(radius),
            lambda state: BFS(state, [state.getMyPacmanPosition()], lambda s,x,y: s.hasFood(x,y), radius)[0],
            lambda state: BFS(state, [state.getMyPacmanPosition()], lambda s,x,y: (x,y) in s.getCapsules(), radius)[0]
        ]
        if len(weights) == 0:
            self.weights = []
            for i in range(len(self.features)):
                self.weights.append(1.0 * float(i == 0))
        else:
            self.weights = weights

    def numScaredGhostsXStepsAway(self, radius):
        return lambda state: BFS(state, [state.getMyPacmanPosition()], lambda s,x,y: self.ghostEval(s,x,y,False), radius)[0]

    def isEnemyGhostXStepsAway(self, numStepsAway, active = True):
        return lambda state: self.closest(state, lambda s,x,y: self.ghostEval(s,x,y,active))[1] <= numStepsAway

    def closestGhost(self, state, active = True):
        closest = self.closest(state, lambda s,x,y: self.ghostEval(s,x,y,active))[1]
        if closest == sys.maxsize + 1:
            return 0
        return closest

    def closestEnemyPacman(self, state, higherScore = True):
        closest = self.closest(state, lambda s,x,y: self.pacmanEval(s,x,y,higherScore))[1]
        if closest == sys.maxsize + 1:
            return 0
        return closest

    def isEnemyPacmanXStepsAway(self, numStepsAway, higherScore = True):
        return lambda state: self.closest(state, lambda s,x,y: self.pacmanEval(s,x,y,higherScore))[1] <= numStepsAway

    def ghostEval(self, state, x, y, active = True):
        for ghost in state.getGhostStates():
            if (y,x) == state.getGhostPosition(ghost.index) \
            and (ghost.scaredTimer[self.index] == 0) == active:
                return True
        return False

    def pacmanEval(self, state, x, y, higherScore = True):
        for pacman in state.getPacmanStates():
            if pacman.index != 0 \
            and (state.data.score[pacman.index] > state.data.score[self.index]) == higherScore:
                return True
        return False

    # usage e.g.: (count, closest, furthest, dists, visits) = closest(state, lambda)
    def closest(self, state, evalFun):
        return BFS(state, [state.getMyPacmanPosition()], evalFun)

    def registerInitialState(self, gameState):
        self.start = gameState.getMyPacmanPosition()
        ReinforcementLearningAgent.registerInitialState(self, gameState)

    def getPolicyAction(self, gameState):
        return self.chooseAction(gameState, 0)

    def chooseAction(self, gameState, epsilon):
        actions = gameState.getLegalActions(self.index)
        if len(actions) == 0:
            return Directions.STOP
        bestActions = actions
        if np.random.uniform(0,1) >= epsilon or not self.isTraining:
            start = time.time()
            values = [self.evaluate(gameState, a) for a in actions]
            self.computationTimes.append(time.time() - start)
            if len(values) == 0:
                return Directions.STOP
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            if len(bestActions) == 0:
                return Directions.STOP
        return random.choice(bestActions)

def scoreEvaluation(state, agentIndex):
    return state.getScore(agentIndex)