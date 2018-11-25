# featureExtractors.py
# --------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"Feature extractors for Pacman game states"

from game import Directions, Actions
import util


def closestFood(pos, food, walls):
	"""
	closestFood -- this is similar to the function that we have
	worked on in the search project; here its all in one place
	"""
	fringe = [(pos[0], pos[1], 0)]
	expanded = set()
	while fringe:
		pos_x, pos_y, dist = fringe.pop(0)
		if (pos_x, pos_y) in expanded:
			continue
		expanded.add((pos_x, pos_y))
		# if we find a food at this location then exit
		if food[pos_x][pos_y]:
			return dist
		# otherwise spread out from the location to its neighbours
		nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
		for nbr_x, nbr_y in nbrs:
			fringe.append((nbr_x, nbr_y, dist+1))
	# no food found
	return None


class FeatureExtractor:

	def getFeatures(self, state, action):
		# extract the grid of food and wall locations and get the ghost locations
		food = state.getFood()
		walls = state.getWalls()
		ghosts = state.getGhostPositions()
		capsulesLeft = len(state.getCapsules())
		scaredGhost = []
		activeGhost = []
		features = util.Counter()
		for ghost in state.getGhostStates():
			if not ghost.scaredTimer[0]:
				activeGhost.append(ghost)
			else:
				scaredGhost.append(ghost)

		pos = state.getMyPacmanPosition()

		def getManhattanDistances(ghosts):
			return map(lambda g: util.manhattanDistance(pos, g.getPosition()), ghosts)

		distanceToClosestActiveGhost = distanceToClosestScaredGhost = 0
		'''
	if activeGhost:
		distanceToClosestActiveGhost = min(getManhattanDistances(activeGhost))
	else: 
		distanceToClosestActiveGhost = float("inf")
	distanceToClosestActiveGhost = max(distanceToClosestActiveGhost, 5)
	'''

		'''else:
		distanceToClosestScaredGhost = 0 # I don't want it to count if there aren't any scared ghosts
		features["dist-to-closest-scared-ghost"] = -2*distanceToClosestScaredGhost
	'''

		features["bias"] = 1.0

		# compute the location of pacman after he takes the action
		x, y = state.getMyPacmanPosition()
		dx, dy = Actions.directionToVector(action)
		next_x, next_y = int(x + dx), int(y + dy)

		# count the number of ghosts 1-step away
		features["#-of-ghosts-1-step-away"] = sum(
			(next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
		features["#-of-ghosts-2-step-away"] = sum(
			[(next_x, next_y) in Actions.getLegalNeighbors(neighbor, walls) for neighbor in [Actions.getLegalNeighbors(g, walls) for g in ghosts]]
		)

		# if there is no danger of ghosts then add the food feature
		if (not features["#-of-ghosts-1-step-away"] and not features['#-of-ghosts-2-step-away']) and food[next_x][next_y]:
			features["eats-food"] = 1.0

		dist = closestFood((next_x, next_y), food, walls)
		if dist is not None:
			# make the distance a number less than one otherwise the update
			# will diverge wildly
			features["closest-food"] = float(dist) / \
				(walls.width * walls.height)
		if scaredGhost:  # and not activeGhost:
			distanceToClosestScaredGhost = min(
				getManhattanDistances(scaredGhost))
			if activeGhost:
				distanceToClosestActiveGhost = min(
					getManhattanDistances(activeGhost))
			else:
				distanceToClosestActiveGhost = 10
			features["capsules"] = capsulesLeft
			#features["dist-to-closest-active-ghost"] = 2*(1./distanceToClosestActiveGhost)
			# features["#-of-ghosts-1-step-away"] >= 1:
			if distanceToClosestScaredGhost <= 8 and distanceToClosestActiveGhost >= 3:
				features["#-of-ghosts-1-step-away"] = 0
				features["#-of-ghosts-2-step-away"] = 0
				features["eats-food"] = 0.0
				#features["closest-food"] = 0

		# print(features)
		features.divideAll(10.0)
		return features
