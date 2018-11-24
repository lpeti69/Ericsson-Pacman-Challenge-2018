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
import copy
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
        print self.isTraining

        import __main__
        if '_display' in dir(__main__):
            self.display = __main__._display

    def final(self, gameState, fileName = 'weights.txt'):
        print "Avg time for evaulate: {}".format(sum(self.computationTimes) / float(len(self.computationTimes)))
        print self.weights
        with open(fileName, 'w') as file:
            for weight in self.weights:
                file.write(str(weight) + '\n')

    def updateWeights(self, state, action, alpha = .1, gamma = .5):
        ## TODO: Need to handle r(it is not provided, since the state reward not updated yet)
        Qsa = self.evaluate(state, action)
        newState = self.getSuccessor(state, action)
        r = newState.data.reward[self.index]
        ## -1 provided for epsilon => it will select according to the current strategy
        newAction = self.getPolicyAction(newState)
        maxQsa = self.evaluate(newState, newAction)
        ## TODO: Optimalize: vector function?
        for i in range(len(self.features)):
            self.weights[i] += alpha*(r + gamma*maxQsa - Qsa)*self.features[i](newState)
            #print r, maxQsa, Qsa, self.features[i](newState)

    ## Approximates Q(s,a)
    def evaluate(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        Qsa = 0
        for feature, weight in zip(self.features, self.weights):
            Qsa += weight * feature(successor)
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

    def getAction(self, gameState, decay = .99):
        self.observationHistory.append(gameState)
        myPos = gameState.getMyPacmanPosition()
        if myPos != nearestPoint(myPos):
            return gameState.getLegalActions(self.index)[0]
        
        epsilon = np.power(decay, gameState.data.tick)
        action = self.chooseAction(gameState, epsilon)
        self.updateWeights(gameState, action)
        return action

    def chooseAction(self, gameState, epsilon):
        util.raiseNotDefined()

class MyAgent(ReinforcementLearningAgent):
    def __init__(self, index = 0, weights = []):
        ReinforcementLearningAgent.__init__(self, index)
        self.features = [
            lambda state: self.closest(state, lambda s, x, y: s.hasFood(x,y))[1],
            self.isEnemyGhostXStepsAway(2),
            self.isEnemyGhostXStepsAway(3),
        ]
        if len(weights) == 0:
            self.weights = []
            for i in range(len(self.features)):
                self.weights.append(1.0)
        else:
            self.weights = weights

    def isEnemyGhostXStepsAway(self, numStepsAway):
        return lambda state: self.closest(state, lambda s, x, y: (x,y) in s.getGhostPositions())[1] <= numStepsAway

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
        return random.choice(bestActions)

def scoreEvaluation(state, agentIndex):
    return state.getScore(agentIndex)