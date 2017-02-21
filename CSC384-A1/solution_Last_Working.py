#Look for #IMPLEMENT tags in this file. These tags indicate what has
#to be implemented to complete the Sokoban warehouse domain.

#   You may add only standard python imports---i.e., ones that are automatically
#   available on TEACH.CS
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

# import os for time functions
import os
import sys
import copy

from search import * #for search engines
from sokoban import *  #for Sokoban specific classes and problems

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
        shortest = float("inf")
        # if box is already in a valid storage space, shortest distance is 0
        if box in state.storage and (state.restrictions is None or box in state.restrictions[state.boxes[box]]):
            pass
        # else, find valid storage space with shortest distance
        else:
            # state has restrictions, so check valid storages and update total
            if state.restrictions is not None:
                total_distance += _shortest_manhattan(box, state.restrictions[state.boxes[box]])
            # no restrictions, check all storages and update total
            else:
                total_distance += _shortest_manhattan(box, state.storage)
    return total_distance

# Helper to get shortest manhattan distance
def _shortest_manhattan(box, storages):
    """
    Returns shortest manhattan distance between box and storage
    @param 2-tuple box: coordinates of box
    @param list of 2-tuples storages: coordinates of storages
    @return number shortest: shortest manhattan distance to valid storage from box
    """
    shortest = float("inf")
    for storage in storages:
        # find manhattan distance
        manhattan_distance = abs(storage[0] - box[0]) + abs(storage[1] - box[1])
        # update shortest accordingly
        if manhattan_distance < shortest:
            shortest = manhattan_distance
    # return shortest manhattan distance
    return shortest

# TODO: global variables for alternate heurisitc

# hash columns to storage coordinates
matrix_to_storage = {}
matrix_to_box = {}
# Pattern DB? Store parent's heuristics and use this to quickly calculate successors???
pattern_db = {}
#TODO: build up and tear down of all globals before and after search!
cost_db = {}

def heur_alternate(state):
#IMPLEMENT
    '''a better sokoban heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    #heur_manhattan_distance has flaws.
    #Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    #Your function should return a numeric value for the estimate of the distance to the goal.

    # if the parent state heuristic is in the Pattern DB, use state.action to derive heuristic
    if state.parent is None or state.parent.hashable_state not in pattern_db:
        # populate matrix to box dictionary
        matrix_to_box = {}
        row_index = 0
        for box in state.boxes:
            matrix_to_box[row_index] = box
            row_index += 1
        # populate matrix to storage dictionary
        column_index = 0
        for box in state.storage:
            matrix_to_storage[column_index] = box
            column_index += 1
        matrix = [[0 for x in range(0, column_index)] for x in range(0, row_index)]
        # populate matrix
        for row in range(0,row_index):
            for column in range (0, column_index):
                # if valid storage, find cost
                if state.restrictions is None or matrix_to_storage[column] in state.restrictions[state.boxes[matrix_to_box[row]]]:
                    matrix[row][column] = abs(matrix_to_box[row][0] - matrix_to_storage[column][0]) + abs(matrix_to_box[row][1] - matrix_to_storage[column][1])
                # else set cost as infinity
                else:
                    matrix[row][column] = float("inf")

        # use hungarian algorithm and return minimum cost as heuristic value
        m = Munkres()
        indexes = m.compute(matrix)
        total = 0
        for row, column in indexes:
            value = matrix[row][column]
            total += value
        if state.hashable_state not in cost_db:
            cost_db[state.hashable_state] = total
    # else, derive heuristic directly
    else:
        # depending on the action, we can determine whether the robot moved a box
        # if robot was next to box and moved in the moves direction, we know which box it moved and where.
        # if robot in current state is occupying the coordinates of a box in parent's state, we can deduce which direction robot pushed the box
        # and thus determine where the box is now.
        # thus we simply have to update the manhattan distances for a single row then call hungarian algorithm again
        # if the robot in current state is occupying a box coordinate in the previous state, it has moved that box
        matrix = [row[:] for row in pattern_db[state.parent.hashable_state]]
        moved_box_coord = state.robot
        # robot moved a box!
        if state.robot in state.parent.boxes:
            # find which direction the robot has pushed the box to deduce box's new coordinates
            # up
            if state.parent.robot[1] > state.robot[1]:
                moved_box_coord = (state.robot[0], state.robot[1] - 1)
            # down
            elif state.parent.robot[1] < state.robot[1]:
                moved_box_coord = (state.robot[0], state.robot[1] + 1)
            # left
            elif state.parent.robot[0] > state.robot[0]:
                moved_box_coord = (state.robot[0] - 1, state.robot[1])
            # right
            elif state.parent.robot[0] < state.robot[0]:
                moved_box_coord = (state.robot[0] + 1, state.robot[1])
            # find row corresponding to moved box
            moved_box_row = [ key for key,val in matrix_to_box.items() if val==state.robot ][0]
            # TODO: can remove the if statements for hashing to make faster?
            # update coordinate of box
            # TODO: make a hashmap for each box and it's index so that we don't have to know it by it's coordinates... like since for every game
            # TODO: we only have one set of boxes..lets find a way to identify them. maybe make a box type or a NAMED TUPLE?
            ma[moved_box_row] = moved_box_coord
            if state.hashable_state not in pattern_db_boxes:
                pattern_db_boxes[state.hashable_state] = copy.deepcopy(boxes)
            # update row in matrix
            for x in range(0, len(matrix_to_storage)):
                matrix[moved_box_row][x] = abs(moved_box_coord[0] - matrix_to_storage[x][0]) + abs(moved_box_coord[1] - matrix_to_storage[x][1])
            # hash the matrix
            if state.hashable_state not in pattern_db:
                pattern_db[state.hashable_state] = [row[:] for row in matrix]
            # use hungarian algorithm to find cost
            m = Munkres()
            indexes = m.compute(matrix)
            total = 0
            for row, column in indexes:
                value = matrix[row][column]
                total += value
        # no moved box, return same cost
        else:
            total = cost_db[state.parent.hashable_state]

    return total

#######################################################################################################################################################################################################
# USING CODE FROM SOURCE SPECIFIED BELOW:
"""
References
==========
1. http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html
2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
   *Naval Research Logistics Quarterly*, 2:83-97, 1955.
3. Harold W. Kuhn. Variants of the Hungarian method for assignment
   problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.
4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
   *Journal of the Society of Industrial and Applied Mathematics*,
   5(1):32-38, March, 1957.
5. http://en.wikipedia.org/wiki/Hungarian_algorithm
Copyright and License
=====================
Copyright 2008-2016 Brian M. Clapper
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class Munkres:
    """
    Calculate the Munkres solution to the classical assignment problem.
    See the module documentation for usage.
    """

    def     __init__(self):
        """Create a new instance"""
        self.C = None
        self.row_covered = []
        self.col_covered = []
        self.n = 0
        self.Z0_r = 0
        self.Z0_c = 0
        self.marked = None
        self.path = None

    def make_cost_matrix(profit_matrix, inversion_function):
        """
        **DEPRECATED**
        Please use the module function ``make_cost_matrix()``.
        """
        import munkres
        return munkres.make_cost_matrix(profit_matrix, inversion_function)

    make_cost_matrix = staticmethod(make_cost_matrix)

    def pad_matrix(self, matrix, pad_value=0):
        """
        Pad a possibly non-square matrix to make it square.
        :Parameters:
            matrix : list of lists
                matrix to pad
            pad_value : int
                value to use to pad the matrix
        :rtype: list of lists
        :return: a new, possibly padded, matrix
        """
        max_columns = 0
        total_rows = len(matrix)

        for row in matrix:
            max_columns = max(max_columns, len(row))

        total_rows = max(max_columns, total_rows)

        new_matrix = []
        for row in matrix:
            row_len = len(row)
            new_row = row[:]
            if total_rows > row_len:
                # Row too short. Pad it.
                new_row += [pad_value] * (total_rows - row_len)
            new_matrix += [new_row]

        while len(new_matrix) < total_rows:
            new_matrix += [[pad_value] * total_rows]

        return new_matrix

    def compute(self, cost_matrix):
        """
        Compute the indexes for the lowest-cost pairings between rows and
        columns in the database. Returns a list of (row, column) tuples
        that can be used to traverse the matrix.
        :Parameters:
            cost_matrix : list of lists
                The cost matrix. If this cost matrix is not square, it
                will be padded with zeros, via a call to ``pad_matrix()``.
                (This method does *not* modify the caller's matrix. It
                operates on a copy of the matrix.)
                **WARNING**: This code handles square and rectangular
                matrices. It does *not* handle irregular matrices.
        :rtype: list
        :return: A list of ``(row, column)`` tuples that describe the lowest
                 cost path through the matrix
        """
        self.C = self.pad_matrix(cost_matrix)
        self.n = len(self.C)
        self.original_length = len(cost_matrix)
        self.original_width = len(cost_matrix[0])
        self.row_covered = [False for i in range(self.n)]
        self.col_covered = [False for i in range(self.n)]
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = self.__make_matrix(self.n * 2, 0)
        self.marked = self.__make_matrix(self.n, 0)

        done = False
        step = 1

        steps = { 1 : self.__step1,
                  2 : self.__step2,
                  3 : self.__step3,
                  4 : self.__step4,
                  5 : self.__step5,
                  6 : self.__step6 }

        while not done:
            try:
                func = steps[step]
                step = func()
            except KeyError:
                done = True

        # Look for the starred columns
        results = []
        for i in range(self.original_length):
            for j in range(self.original_width):
                if self.marked[i][j] == 1:
                    results += [(i, j)]

        return results

    def __copy_matrix(self, matrix):
        """Return an exact copy of the supplied matrix"""
        return copy.deepcopy(matrix)

    def __make_matrix(self, n, val):
        """Create an *n*x*n* matrix, populating it with the specific value."""
        matrix = []
        for i in range(n):
            matrix += [[val for j in range(n)]]
        return matrix

    def __step1(self):
        """
        For each row of the matrix, find the smallest element and
        subtract it from every element in its row. Go to Step 2.
        """
        C = self.C
        n = self.n
        for i in range(n):
            minval = min(self.C[i])
            # Find the minimum value for this row and subtract that minimum
            # from every element in the row.
            for j in range(n):
                self.C[i][j] -= minval

        return 2

    def __step2(self):
        """
        Find a zero (Z) in the resulting matrix. If there is no starred
        zero in its row or column, star Z. Repeat for each element in the
        matrix. Go to Step 3.
        """
        n = self.n
        for i in range(n):
            for j in range(n):
                if (self.C[i][j] == 0) and \
                        (not self.col_covered[j]) and \
                        (not self.row_covered[i]):
                    self.marked[i][j] = 1
                    self.col_covered[j] = True
                    self.row_covered[i] = True
                    break

        self.__clear_covers()
        return 3

    def __step3(self):
        """
        Cover each column containing a starred zero. If K columns are
        covered, the starred zeros describe a complete set of unique
        assignments. In this case, Go to DONE, otherwise, Go to Step 4.
        """
        n = self.n
        count = 0
        for i in range(n):
            for j in range(n):
                if self.marked[i][j] == 1 and not self.col_covered[j]:
                    self.col_covered[j] = True
                    count += 1

        if count >= n:
            step = 7 # done
        else:
            step = 4

        return step

    def __step4(self):
        """
        Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the starred
        zero. Continue in this manner until there are no uncovered zeros
        left. Save the smallest uncovered value and Go to Step 6.
        """
        step = 0
        done = False
        row = -1
        col = -1
        star_col = -1
        while not done:
            (row, col) = self.__find_a_zero()
            if row < 0:
                done = True
                step = 6
            else:
                self.marked[row][col] = 2
                star_col = self.__find_star_in_row(row)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    done = True
                    self.Z0_r = row
                    self.Z0_c = col
                    step = 5

        return step

    def __step5(self):
        """
        Construct a series of alternating primed and starred zeros as
        follows. Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always
        be one). Continue until the series terminates at a primed zero
        that has no starred zero in its column. Unstar each starred zero
        of the series, star each primed zero of the series, erase all
        primes and uncover every line in the matrix. Return to Step 3
        """
        count = 0
        path = self.path
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        done = False
        while not done:
            row = self.__find_star_in_col(path[count][1])
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count-1][1]
            else:
                done = True

            if not done:
                col = self.__find_prime_in_row(path[count][0])
                count += 1
                path[count][0] = path[count-1][0]
                path[count][1] = col

        self.__convert_path(path, count)
        self.__clear_covers()
        self.__erase_primes()
        return 3

    def __step6(self):
        """
        Add the value found in Step 4 to every element of each covered
        row, and subtract it from every element of each uncovered column.
        Return to Step 4 without altering any stars, primes, or covered
        lines.
        """
        minval = self.__find_smallest()
        for i in range(self.n):
            for j in range(self.n):
                if self.row_covered[i]:
                    self.C[i][j] += minval
                if not self.col_covered[j]:
                    self.C[i][j] -= minval
        return 4

    def __find_smallest(self):
        """Find the smallest uncovered value in the matrix."""
        minval = sys.maxsize
        for i in range(self.n):
            for j in range(self.n):
                if (not self.row_covered[i]) and (not self.col_covered[j]):
                    if minval > self.C[i][j]:
                        minval = self.C[i][j]
        return minval

    def __find_a_zero(self):
        """Find the first uncovered element with value 0"""
        row = -1
        col = -1
        i = 0
        n = self.n
        done = False

        while not done:
            j = 0
            while True:
                if (self.C[i][j] == 0) and \
                        (not self.row_covered[i]) and \
                        (not self.col_covered[j]):
                    row = i
                    col = j
                    done = True
                j += 1
                if j >= n:
                    break
            i += 1
            if i >= n:
                done = True

        return (row, col)

    def __find_star_in_row(self, row):
        """
        Find the first starred element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 1:
                col = j
                break

        return col

    def __find_star_in_col(self, col):
        """
        Find the first starred element in the specified row. Returns
        the row index, or -1 if no starred element was found.
        """
        row = -1
        for i in range(self.n):
            if self.marked[i][col] == 1:
                row = i
                break

        return row

    def __find_prime_in_row(self, row):
        """
        Find the first prime element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 2:
                col = j
                break

        return col

    def __convert_path(self, path, count):
        for i in range(count+1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1

    def __clear_covers(self):
        """Clear all covered matrix cells"""
        for i in range(self.n):
            self.row_covered[i] = False
            self.col_covered[i] = False

    def __erase_primes(self):
        """Erase all prime markings"""
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def make_cost_matrix(profit_matrix, inversion_function):
    """
    Create a cost matrix from a profit matrix by calling
    'inversion_function' to invert each value. The inversion
    function must take one numeric argument (of any type) and return
    another numeric argument which is presumed to be the cost inverse
    of the original profit.
    This is a static method. Call it like this:
    .. python::
        cost_matrix = Munkres.make_cost_matrix(matrix, inversion_func)
    For example:
    .. python::
        cost_matrix = Munkres.make_cost_matrix(matrix, lambda x : sys.maxsize - x)
    :Parameters:
        profit_matrix : list of lists
            The matrix to convert from a profit to a cost matrix
        inversion_function : function
            The function to use to invert each entry in the profit matrix
    :rtype: list of lists
    :return: The converted matrix
    """
    cost_matrix = []
    for row in profit_matrix:
        cost_matrix.append([inversion_function(value) for value in row])
    return cost_matrix

def print_matrix(matrix, msg=None):
    """
    Convenience function: Displays the contents of a matrix of integers.
    :Parameters:
        matrix : list of lists
            Matrix to print
        msg : str
            Optional message to print before displaying the matrix
    """
    import math

    if msg is not None:
        print(msg)

    # Calculate the appropriate format width.
    width = 0
    for row in matrix:
        for val in row:
            width = max(width, len(str(val)))

    # Make the format string
    format = '%%%dd' % width

    # Print the matrix
    for row in matrix:
        sep = '['
        for val in row:
            sys.stdout.write(sep + format % val)
            sep = ', '
        sys.stdout.write(']\n')

#################################################################################################################################################################################################


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
    se = SearchEngine('custom', 'full')
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



