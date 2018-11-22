# Ericcson-Programozo-Bajnoksag
Team Szukcesszív Approximátorok's solution

### How to run
`python pacman.py --pacman=MyAgent --layout=layouts/er1`

TODO
-
primary
-
* Generate random 31x30maps with 4 ghosts and 2 enemy pacmans
* Write the ericcson specific implementation of this program with the calculated weights
* Define metrics for evaluating the results of one or several runs(results) for a given agent's logic
* Add features (to be discussed)

secondary
-
* Remove items only at the end of the turn
* Currently every pacman get 10 for food, 50 for booster, 0 for ghosts if eaten already
* Set event sorting
  * Pacman deaths
  * Ghost deaths
  * Pacman collisions
  * Booster/food eat
* Measure: 3 ghost eaten in 10 tick
