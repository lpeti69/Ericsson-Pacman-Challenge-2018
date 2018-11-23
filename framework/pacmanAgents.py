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
        self.approximators = {}
        self.display = None

    def __str__(self):
        return "AgentState: {}\nApproximators: {}\nComputationTimes: {}".format(
            self.observationHistory[-1].data.agentStates[self.index],
            self.approximators,
            self.computationTimes
        )

    def observationFunction(self, gameState):
        return gameState.deepCopy()

    def registerInitialState(self, gameState):
        self.observationHistory = []

        import __main__
        if '_display' in dir(__main__):
            self.display = __main__._display

    def final(self, gameState):
        ## TODO: Add result handling
        ## ide kell irni valamit...
        print self
        print "Avg time for evaulate: {}".format(sum(self.computationTimes) / float(len(self.computationTimes)))
        print gameState.data.score[0]

    def updateWeights(self, state, action, alpha = .25, gamma = .75):
        ## TODO: Need to handle r(it is not provided, since the state reward not updated yet)
        Qsa = self.evaluate(state, action)
        newState = self.getSuccessor(state, action)
        r = newState.data.reward[self.index]
        ## -1 provided for epsilon => it will select according to the current strategy
        newAction = self.getPolicyAction(newState)
        maxQsa = self.evaluate(newState, newAction)
        ## TODO: Optimalize: vector function?
        for featureName, (feature, weight) in self.approximators.items():
            weight += alpha*(r + gamma*maxQsa - Qsa)*feature(newState)

    ## Convert {key: lambda} -> {key: (lambda, initWeight)} ex.: setApproximators(self.positionSelector, 0)
    def setApproximators(self, approximators, initialWeightValues = 1):
        for key, fun in approximators.items():
            self.approximators[key] = (fun, initialWeightValues)
        print self.approximators

    def convCurrentFeaturesToApproximators(self):
        approximators = {}
        for selector, evalFun in self.positionSelector.items():
            approximators[selector] = lambda state: self.closest(state, selector)[1]
        self.setApproximators(approximators)

    ## Approximates Q(s,a)
    def evaluate(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        Qsa = 0
        for featureName, (feature, weight) in self.approximators.items():
            temp = weight * feature(gameState)
            Qsa += temp
            print "Feature: {}, value: {}".format(featureName, temp)
        return Qsa

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getMyPacmanPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getAction(self, gameState, decay = .995):
        self.observationHistory.append(gameState)
        myPos = gameState.getMyPacmanPosition()
        if myPos != nearestPoint(myPos):
            return gameState.getLegalActions(self.index)[0]
        
        epsilon = np.power(gameState.data.tick, decay)
        action = self.chooseAction(gameState, epsilon)
        self.updateWeights(gameState, action)
        return action

    def chooseAction(self, gameState, epsilon):
        util.raiseNotDefined()

class MyAgent(ReinforcementLearningAgent):
    def __init__(self, index = 0):
        ReinforcementLearningAgent.__init__(self, index)
        self.weights = {}
        self.positionSelector = {
            'food':     lambda s, x, y: s.hasFood(x,y),
            'caps':     lambda s, x, y: (x,y) in s.getCapsules(),
            'ghosts':   lambda s, x, y: (x,y) in s.getGhostPositions(),
            'enemy':    lambda s, x, y: (x,y) in s.getEnemyPacmanPositions()
        }
        self.convCurrentFeaturesToApproximators()


    # usage e.g.: (count, closest, furthest) = closest(state, 'food')
    def closest(self, state, selector):
        return BFS(state, [state.getMyPacmanPosition()], self.positionSelector[selector])

    def registerInitialState(self, gameState):
        self.start = gameState.getMyPacmanPosition()
        ReinforcementLearningAgent.registerInitialState(self, gameState)

    def getPolicyAction(self, gameState):
        return self.chooseAction(gameState, 0)

    def chooseAction(self, gameState, epsilon):
        actions = gameState.getLegalActions(self.index)

        bestActions = actions
        if np.random.uniform(0,1) >= epsilon:
            start = time.time()
            values = [self.evaluate(gameState, a) for a in actions]
            self.computationTimes.append(time.time() - start)

            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

def scoreEvaluation(state, agentIndex):
    return state.getScore(agentIndex)