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
import numpy as np


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

		## general query functions
		minDistArray = util.getClosests(state, (next_x, next_y))
		countArray 	 = util.BFS(M=state, 
								starts=[(next_x, next_y)],
								isWall=lambda m,X,Y: m.walls[X][Y]=='%' or (X,Y) == (x,y),  
								maxDistance=10)
		minDists = [feature[0][1] for feature in minDistArray]
		counts   = [feature[0][1] for feature in countArray]

		## normalizing them
		minDists 	/= np.linalg.norm(minDists)
		counts 		/= np.linalg.norm(counts)

		## food
		features["closest-food"] = minDists[0]
		features["num-food"] = counts[0] ## TODO: Add dead-end?

		## capsules
		features["closest-capsule"] = minDists[1]
		features["num-capsules"] = counts[1]

		## enemy pacman with higher score
		features["closest-enemy-stronger"] = minDists[3]

		## enemy pacmans with lower score
		features["closest-enemy-weaker"] = minDists[4]

		## scared ghosts
		features["closest-scared-ghost"] = minDists[6]
		features["num-scared-ghosts"] = counts[6]

		## active ghosts
		features["closest-active-ghost"] = min
		features["num-active-ghosts"] = counts[7]

		## special query functions
		#cg1 = util.BFS(state, (x,y), [
		#	lambda s,x,y: 'isScared',
		#	lambda s,x,y: 'isActive'
		#	], radius=1
		#)

		#cg2 = util.BFS(state, (x,y), [
		#	lambda s,x,y: 'isScared',
		#	lambda s,x,y: 'isActive'
		#	], radius=2
		#)

		#cg5 = util.BFS(state, (x,y), [
		#	lambda s,x,y: 'isScared',
		#	lambda s,x,y: 'isActive'
		#	], radius=1
		#)

		## special features
		#features["#-of-ghosts-0-step-away"] = cg1[6] + cg1[7] ## TODO
		## etc

		if features['num-food'] > 2*features['closest-active-ghost'] and food[next_x][next_y]:
			features['eats-food'] = 1.0

		# call: getClosests((y,x))
		# [food, cap, pacman, biggerPacmap, smallerPacman, ghost, scaradGhost, activeGhost]
		# format: [ ((y1,x1),dist1), ((y2,x2),dist2), ... ]

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