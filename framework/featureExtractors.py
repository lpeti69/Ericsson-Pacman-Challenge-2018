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
		layout = state.data.layout
		labels = ['food', 'capsule', 'enemy-stronger', 'enemy-weaker', 'scared-ghost', 'active-ghost']
		features["bias"] = 1.0

		# compute the location of pacman after he takes the action
		x, y = state.getMyPacmanPosition()
		dx, dy = Actions.directionToVector(action)
		next_x, next_y = util.clip(layout, int(x + dx), int(y + dy))

		# count the number of ghosts 1-step away
		features["#-of-ghosts-1-step-away"] = sum(
			(next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

		## general query functions
		#minDists = util.getClosests(state, (next_x, next_y))
		counts 	 = util.getCount(state, (x,y), (next_x, next_y), 10)
		#						starts=[(next_x, next_y)],
		#						isWall=lambda m,X,Y: m.walls[X][Y]=='%' or (X,Y) == (x,y),  
		#						maxDistance=10)
		#countArray 	= util.getClosests(state, (next_x, next_y))
		#minDists = [feature[0][1] for feature in minDistArray if feature != []]
		#counts   = [feature[0][1] for feature in countArray]

		## normalizing them
		#for i, dist in enumerate(minDists):
		#	if dist != []:
		#		features['closest-' + labels[i]] = dist[0]
		#	else:
		#		features['closest-' + labels[i]] = 0.0
		#for i, dist in enumerate(counts):
			#features['num-' + labels[i]] = dist

		ghosts1 = util.getCount(state, (x,y), (next_x, next_y), 1,  [
			lambda s,x,y: s.isGhostPos((x,y)) and  util.getGhostFromPosition(s,x,y).scaredTimer[0]>0,
            lambda s,x,y: s.isGhostPos((x,y)) and  util.getGhostFromPosition(s,x,y).scaredTimer[0]==0])
		ghosts2 = util.getCount(state, (x,y), (next_x, next_y), 2,  [
			lambda s,x,y: s.isGhostPos((x,y)) and  util.getGhostFromPosition(s,x,y).scaredTimer[0]>0,
            lambda s,x,y: s.isGhostPos((x,y)) and  util.getGhostFromPosition(s,x,y).scaredTimer[0]==0])
		ghosts3 = util.getCount(state, (x,y), (next_x, next_y), 3,  [
			lambda s,x,y: s.isGhostPos((x,y)) and  util.getGhostFromPosition(s,x,y).scaredTimer[0]>0,
            lambda s,x,y: s.isGhostPos((x,y)) and  util.getGhostFromPosition(s,x,y).scaredTimer[0]==0])
		
		features['#-of-ghosts-1-step-away'] = ghosts1[1]
		features['#-of-ghosts-2-step-away'] = ghosts2[1]
		features['#-of-ghosts-3-step-away'] = ghosts3[1]
		features['#-of-scared-ghosts-1-step-away'] = ghosts1[0]
		features['#-of-scared-ghosts-2-step-away'] = ghosts2[0]

		pacmans1 = util.getCount(state, (x,y), (next_x, next_y), 1,  [
			lambda s,x,y: s.isEnemyPacmanPos((x,y)) and s.getScore(util.getPacmanFromPosition(s,x,y).index) > s.data.score[0],
			lambda s,x,y: s.isEnemyPacmanPos((x,y)) and s.getScore(util.getPacmanFromPosition(s,x,y).index) < s.data.score[0]])
		pacmans2 = util.getCount(state, (x,y), (next_x, next_y), 2,  [
			lambda s,x,y: s.isEnemyPacmanPos((x,y)) and s.getScore(util.getPacmanFromPosition(s,x,y).index) > s.data.score[0],
			lambda s,x,y: s.isEnemyPacmanPos((x,y)) and s.getScore(util.getPacmanFromPosition(s,x,y).index) < s.data.score[0]])
		pacmans3 = util.getCount(state, (x,y), (next_x, next_y), 3,  [
			lambda s,x,y: s.isEnemyPacmanPos((x,y)) and s.getScore(util.getPacmanFromPosition(s,x,y).index) > s.data.score[0],
			lambda s,x,y: s.isEnemyPacmanPos((x,y)) and s.getScore(util.getPacmanFromPosition(s,x,y).index) < s.data.score[0]])

		features['#-of-stronger-pacman-1-step-away'] = pacmans1[0]
		features['#-of-stronger-pacman-2-step-away'] = pacmans2[0]
		features['#-of-stronger-pacman-3-step-away'] = pacmans3[0]
		features['#-of-weaker-pacman-1-step-away'] = pacmans1[1]
		features['#-of-weaker-pacman-2-step-away'] = pacmans2[1]
		features['#-of-weaker-pacman-3-step-away'] = pacmans3[1]

		dangerous = lambda: (features['#-of-ghosts-1-step-away'] + features['#-of-ghosts-2-step-away'] + features['#-of-ghosts-3-step-away'] > 0) \
							or pacmans1[0] + pacmans2[0] + pacmans3[0] > 0
		ghostEating = lambda: ghosts1[0] + ghosts2[0]# + ghosts3[0]
		pacmanEeating = lambda: pacmans1[1] + pacmans2[1] + pacmans3[1]
		#minDists 	/= np.linalg.norm(minDists)
		#counts 		/= np.linalg.norm(counts)

		## food
		#features["closest-food"] = minDists[0]
		#features["num-food"] = counts[0]

		## capsules
		#features["closest-capsule"] = minDists[1]
		#features["num-capsules"] = counts[1]

		## enemy pacman with higher score
		#features["closest-enemy-stronger"] = minDists[3]

		## enemy pacmans with lower score
		#features["closest-enemy-weaker"] = minDists[4]

		## scared ghosts
		#features["closest-scared-ghost"] = minDists[6]
		#features["num-scared-ghosts"] = counts[6]

		## active ghosts
		#features["closest-active-ghost"] = min
		#features["num-active-ghosts"] = counts[7]

		# call: getClosests((y,x))
		# [food, cap, pacman, biggerPacmap, smallerPacman, ghost, scaradGhost, activeGhost]
		# format: [ ((y1,x1),dist1), ((y2,x2),dist2), ... ]

		# if there is no danger of ghosts then add the food feature
		if not dangerous() and food[next_x][next_y]:
			features["eats-food"] = 1.0
		if not dangerous() and state.isCapsulePos((next_x, next_y)):
			features["eats-capsule"] = 1.0
		if not dangerous() and ghostEating() > 0:
			features["eats-ghosts"] = ghostEating()
		if not dangerous() and pacmanEeating() > 0:
			features["eats-pacman"] = pacmanEeating()
		## dead end
		if features['num-food'] <= 2*features['closest-active-ghost'] and food[next_x][next_y]:
			features['eats-food'] = 0.0

		dist = closestFood((next_x, next_y), food, walls)
		if dist is not None:
			# make the distance a number less than one otherwise the update
			# will diverge wildly
			features["closest-food"] = float(dist)
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

		#print(features)
		return util.normalize(features)