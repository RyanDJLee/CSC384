#Look for #IMPLEMENT tags in this file. These tags indicate what has
#to be implemented to complete the Sokoban warehouse domain.

#   You may add only standard python imports---i.e., ones that are automatically
#   available on TEACH.CS
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

# import os for time functions
import os
from search import * #for search engines
from sokoban import SokobanState, Direction, PROBLEMS, sokoban_goal_state  #for Sokoban specific classes and problems

#SOKOBAN HEURISTICS
def heur_displaced(state):
  '''trivial admissible sokoban heuristic'''
  '''INPUT: a sokoban state'''
  '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
  count = 0
  # for each box in the state, count how many are not in storage
  for box in state.boxes:
    if box not in state.storage:
      count += 1
  # return number of boxes not in storage
  return count

def heur_manhattan_distance(state):
#IMPLEMENT
    '''admissible sokoban heuristic: manhattan distance'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    #We want an admissible heuristic, which is an optimistic heuristic.
    #It must always underestimate the cost to get from the current state to the goal.
    #The sum Manhattan distance of the boxes to their closest storage spaces is such a heuristic.
    #When calculating distances, assume there are no obstacles on the grid and that several boxes can fit in one storage bin.
    #You should implement this heuristic function exactly, even if it is tempting to improve it.
    #Your function should return a numeric value; this is the estimate of the distance to the goal.

    # initialize variable to keep track of sum of manhattan distances
    total_distance = 0
    # for each box in state, find minimum manhattan distance
    for box in state.boxes:
        # initialize variable to keep track of shortest manhattan distance
        # TODO: initialize to infinity?
        shortest = float("inf")
        # if box is already in a valid storage space, shortest distance is 0
        if box in state.storage and (state.restrictions is None or box in state.restrictions[state.boxes[box]]):
            pass
        # else, find valid storage space with shortest distance
        else:
            # state has restrictions, so check valid storages
            if state.restrictions is not None:
                # iterate through valid storages
                for storage in state.restrictions[state.boxes[box]]:
                    # find manhattan distance
                    manhattan_distance = abs(storage[0] - box[0]) + abs(storage[1] - box[1])
                    # update shortest accordingly
                    if manhattan_distance < shortest:
                        shortest = manhattan_distance
                # update total distance
                total_distance += shortest
            # no restrictions, check all storages
            else:
                for storage in state.storage:
                    # find manhattan distance
                    manhattan_distance = abs(storage[0] - box[0]) + abs(storage[1] - box[1])
                    # update shortest accordingly
                    if manhattan_distance < shortest:
                        shortest = manhattan_distance
                # update total distance
                total_distance += shortest
    return total_distance

# TODO: private helper method for finding shortest manhattan distance(s) per box? shortest_manhattan(storage_list)
def _shortest_manhattan(box, storage_list):
    for storage in storage_list:
        # find manhattan distance
        manhattan_distance = abs(storage[0] - box[0]) + abs(storage[1] - box[1])
        # update shortest accordingly
        if manhattan_distance < shortest:
            shortest = manhattan_distance
    # return shortest manhattan distance
    return shortest


def heur_alternate(state):
#IMPLEMENT
    '''a better sokoban heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    #heur_manhattan_distance has flaws.
    #Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    #Your function should return a numeric value for the estimate of the distance to the goal.
    return 0

def fval_function(sN, weight):
#IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """

    #Many searches will explore nodes (or states) that are ordered by their f-value.
    #For UCS, the fvalue is the same as the gval of the state. For best-first search, the fvalue is the hval of the state.
    #You can use this function to create an alternate f-value for states; this must be a function of the state and the weight.
    #The function must return a numeric f-value.
    #The value will determine your state's position on the Frontier list during a 'custom' search.
    #You must initialize your search engine object as a 'custom' search engine if you supply a custom fval function.

    # f(node) = g(node) +w * h(node)
    return sN.gval + (weight * sN.hval)


def anytime_gbfs(initial_state, heur_fn, timebound = 10):
#IMPLEMENT
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    # GBestFS is an implementation of BestFS with f(n) = h(n)
    se = SearchEngine('best_first', 'full')
    se.init_search(initial_state, goal_fn=sokoban_goal_state, heur_fn=heur_fn)
    # record time anytime_gbfs was called
    start_time = os.times()[0]
    # run first iteration of search, set initial costbound for pruning
    goal_state = se.search(timebound)
    # check if goal_state returned
    if goal_state:
        costbound = goal_state.gval
        time_left = timebound - (os.times()[0] - start_time)
        # keep track of best goal state
        best_state = goal_state
        # run iterations of anytime_gbfs until time runs out
        while time_left > 0:
            initial_time = os.times()[0]
            new_goal_state = se.search(time_left, costbound=(costbound,float("inf"),float("inf")))
            # update time left
            time_left -= os.times()[0] - initial_time
            # update costbound and best state
            if new_goal_state:
                costbound = new_goal_state.gval
                best_state = new_goal_state
        # return best goal state
        return best_state
    # no goal_state was returned from search, return False
    else:
        return False

def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound = 10):
#IMPLEMENT
    '''Provides an implementation of anytime weighted a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    se = SearchEngine('astar', 'full')
    # use lambda expression to create unary weighted fval function
    wrapped_fval_function = (lambda sN: fval_function(sN, 1))
    se.init_search(initial_state, goal_fn=sokoban_goal_state, heur_fn=heur_fn, fval_function=wrapped_fval_function)
    # record time anytime_gbfs was called
    start_time = os.times()[0]
    # run first iteration of search, set initial costbound for pruning
    goal_state = se.search(timebound)
    # check if goal_state returned
    if goal_state:
        costbound = goal_state.gval + heur_fn(goal_state)
        time_left = timebound - (os.times()[0] - start_time)
        # keep track of best goal state
        best_state = goal_state
        # run iterations of anytime weighted until time runs out
        while time_left > 0:
            initial_time = os.times()[0]
            new_goal_state = se.search(time_left, costbound=(float("inf"),float("inf"),costbound))
            # update time left
            time_left -= os.times()[0] - initial_time
            # update costbound and best state
            if new_goal_state:
                costbound = new_goal_state.gval + heur_fn(new_goal_state)
                best_state = new_goal_state
        # return best goal state
        return best_state
    # no goal_state was returned from search, return False
    else:
        return False

if __name__ == "__main__":
  #TEST CODE
  solved = 0; unsolved = []; counter = 0; percent = 0; timebound = 2; #2 second time limit for each problem
  print("*************************************")
  print("Running A-star")

  for i in range(0, 10): #note that there are 40 problems in the set that has been provided.  We just run through 10 here for illustration.

    print("*************************************")
    print("PROBLEM {}".format(i))

    s0 = PROBLEMS[i] #Problems will get harder as i gets bigger

    se = SearchEngine('astar', 'full')
    se.init_search(s0, goal_fn=sokoban_goal_state, heur_fn=heur_displaced)
    final = se.search(timebound)

    if final:
      final.print_path()
      solved += 1
    else:
      unsolved.append(i)
    counter += 1

  if counter > 0:
    percent = (solved/counter)*100

  print("*************************************")
  print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))
  print("*************************************")

  solved = 0; unsolved = []; counter = 0; percent = 0; timebound = 8; #8 second time limit
  print("Running Anytime Weighted A-star")

  for i in range(0, 10):
    print("*************************************")
    print("PROBLEM {}".format(i))

    s0 = PROBLEMS[i] #Problems get harder as i gets bigger
    weight = 10
    final = anytime_weighted_astar(s0, heur_fn=heur_displaced, weight=weight, timebound=timebound)

    if final:
      final.print_path()
      solved += 1
    else:
      unsolved.append(i)
    counter += 1

  if counter > 0:
    percent = (solved/counter)*100

  print("*************************************")
  print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))
  print("*************************************")



