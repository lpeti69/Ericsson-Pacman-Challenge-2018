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
from featureExtractors import FeatureExtractor
import numpy as np
import random, time, game, util

def scoreEvaluation(state, agentIndex):
    return state.getScore(agentIndex)

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


class ReinforcementLearningAgent(Agent):
    def __init__(self, timeForComputing = .25, alpha = 0.1, epsilon = 0.05, gamma = 0.8, numTraining = 1):
        self.observationHistory = []
        self.timeForComputing   = timeForComputing
        self.computationTimes   = []
        self.numTraining        = int(numTraining)
        self.alpha              = float(alpha)
        self.epsilon            = float(epsilon)
        self.gamma              = float(gamma)
        self.display            = None
        self.episodes           = 0
        self.accumTrainRewards  = 0.0
        self.accumTestRewards   = 0.0

    def __str__(self):
        return "AgentState: {}\Weights: {}\nComputationTimes: {}".format(
            self.observationHistory[-1].data.agentStates[self.index],
            self.weights,
            self.computationTimes
        )

    def isInTraining(self):
        return self.episodes < self.numTraining

    def observeTransition(self, state, action, nextState, reward):
        self.episodeRewards += reward
        self.update(state, action, nextState, reward)

    def observationFunction(self, state):
        if not self.lastState is None:
            reward = state.getScore(self.index) - self.lastState.getScore(self.index)
            self.observeTransition(self.lastState, self.lasAction, state, reward)
        return state

    def registerInitialState(self, gameState):
        self.start = gameState.getMyPacmanPosition()
        self.startEpisode()
        if self.episodes == 0:
            print 'Beginning %d episodes of Training' % (self.numTraining)

    def getAction(self, state):
        uitl.raiseNotDefined()

    def startEpisode(self):
        self.lasAction = None
        self.lastState = None
        self.episodeRewards = 0.0
        self.observationHistory = []

    def endEpisode(self):
        if self.episodes < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        self.episodes += 1
        if self.episodes >= self.numTraining:
            self.epsilon = 0.0
            self.alpha   = 0.0

    def doAction(self, state, action):
        self.lastState = state
        self.lasAction = action

    def final(self, state, fileName = 'weights.txt'):
        reward = state.getScore(self.index) - self.lastState.getScore(self.index)
        self.observeTransition(self.lastState, self.lasAction, state, reward)
        self.endEpisode()

        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore(self.index)

        NUM_EPS_UPDATE = 10
        if self.episodes % NUM_EPS_UPDATE == 0:
            print 'Reinforcement Learning Status:'
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodes <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodes)
                print '\tCompleted %d out of %d training episodes' % (
                    self.episodes,self.numTraining)
                print '\tAverage Rewards over all training: %.2f' % (
                    trainAvg)
                print '\tAvg time for evaulate: {}'.format(sum(self.computationTimes) / float(len(self.computationTimes)))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodes - self.numTraining)
                print '\tCompleted %d test episodes' % (self.episodes - self.numTraining)
                print '\tAverage Rewards over testing: %.2f' % testAvg
            print '\tAverage Rewards for last %d episodes: %.2f'  % (
               NUM_EPS_UPDATE,windowAvg)
            print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodes == self.numTraining:
            print 'Training Done (turning off epsilon and alpha)\n'
            
class QLearningAgent(ReinforcementLearningAgent):
    def __init__(self, **args):
        ReinforcementLearningAgent.__init__(self, **args)
        print "ALPHA", self.alpha
        print "GAMMA", self.gamma
        print "EPS",   self.epsilon
    
    def Qsa(self, state, action):
        util.raiseNotDefined()

    def maxQsa(self, state):
        util.raiseNotDefined()

    def getPolicy(self, state):
        util.raiseNotDefined()

    def getAction(self, state):
        self.observationHistory.append(state)
        legalActions = state.getLegalActions(self.index)
        action = Directions.STOP
        if len(legalActions) > 0:
            if np.random.uniform(0,1) > self.epsilon and self.isInTraining():
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
        return action
    

class ApproximateQLearningAgent(QLearningAgent):
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=1, weights = util.Counter(), **args):
        self.index = 0
        self.featureExtractor = FeatureExtractor()
        self.weights = weights
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        start = time.time()
        action = QLearningAgent.getAction(self, state)
        self.computationTimes.append(time.time() - start)
        self.doAction(state, action)
        #epsilon = np.power(epsilonDecay, gameState.data.tick + 3*self.numthGame)
        #alpha   = np.power(alphaDecay, self.numthGame)
        return action

    def Qsa(self, state, action):
        Qsa = 0.0
        features = self.featureExtractor.getFeatures(state, action)
        for key in features.keys():
            Qsa += self.weights[key] * features[key]
        return Qsa

    def maxQsa(self, state):
        action = self.getPolicy(state)
        return self.Qsa(state, action)

    def getPolicy(self, state):
        actions = state.getLegalActions(self.index)
        values = [self.Qsa(state, a) for a in actions]
        if len(values) == 0:
            return Directions.STOP
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        if len(bestActions) == 0:
            return Directions.STOP
        return random.choice(bestActions)
    
    def update(self, state, action, nextState, reward):
        features = self.featureExtractor.getFeatures(state, action)
        Qsa     = self.Qsa(state, action)
        maxQsa  = self.maxQsa(nextState)
        alpha   = self.alpha
        gamma   = self.gamma
        for key in features.keys():
            self.weights[key] += alpha*(reward + gamma*maxQsa - Qsa)*features[key]

class MyPacmanAgent(ApproximateQLearningAgent):
    def __init__(self, **args):
        ApproximateQLearningAgent.__init__(self, **args)

    def final(self, state):
        ApproximateQLearningAgent.final(self, state)
        print self.weights
        if self.episodes == self.numTraining:
            with open('weights.txt', 'w') as file:
                for feature, weight in self.weights.items():
                    file.write(feature + ':' + str(weight) + '\n')