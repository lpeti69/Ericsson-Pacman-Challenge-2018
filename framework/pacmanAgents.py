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
from util import distanceCalculator
import random
import game
import util

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


class ReinforcementLearningAgent(Agent):
    def __init__( self, index, timeForComputing = .25 ):
        self.index = index
        self.distancer = None
        self.observationHistory = []
        self.timeForComputing = timeForComputing
        self.display = None

    def observationFunction(self, gameState):
        # return gameState.makeObservation(self.index)
        pass

    def registerInitialState(self, gameState):
        self.red = gameState.isOnRedTeam(self.index)
        self.distancer = distanceCalculator.Distancer(gameState.data.layout)

        # comment this out to forgo maze distance computation and use manhattan distances
        self.distancer.getMazeDistances()

        import __main__
        if '_display' in dir(__main__):
            self.display = __main__._display

    def final(self, gameState):
        self.observationHistory = []

    def getAction(self, gameState):
        self.observationHistory.append(gameState)

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        #if myPos != nearestPoint(myPos):
        # We're halfway from one position to the next
            #return gameState.getLegalActions(self.index)[0]
        #else:
        return self.chooseAction(gameState)

    
    def chooseAction(self, gameState):
        util.raiseNotDefined()

class MyAgent(ReinforcementLearningAgent): ## TODO

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        ReinforcementLearningAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
        bestDist = 9999
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(self.start,pos2)
            if dist < bestDist:
            bestAction = action
            bestDist = dist
        return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
        # Only half a grid position was covered
        return successor.generateSuccessor(self.index, action)
        else:
        return successor

    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        ## TODO
        for feature in [lambda state: 0]:
            features[feature] = feature(successor)
        return features

    def getWeights(self, gameState, action):
        return {'asd': 1.0}

    def getAction(self, state):
        return Directions.STOP

def scoreEvaluation(state, agentIndex):
    return state.getScore(agentIndex)