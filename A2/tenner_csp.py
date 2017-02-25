#Look for #IMPLEMENT tags in this file. These tags indicate what has
#to be implemented to complete the warehouse domain.

'''
Construct and return Tenner Grid CSP models.
'''

from cspbase import *
import itertools

def tenner_csp_model_1(initial_tenner_board):
    '''Return a CSP object representing a Tenner Grid CSP problem along
       with an array of variables for the problem. That is return

       tenner_csp, variable_array

       where tenner_csp is a csp representing tenner grid using model_1
       and variable_array is a list of lists

       [ [  ]
         [  ]
         .
         .
         .
         [  ] ]

       such that variable_array[i][j] is the Variable (object) that
       you built to represent the value to be placed in cell i,j of
       the Tenner Grid (only including the first n rows, indexed from
       (0,0) to (n,9)) where n can be 3 to 8.


       The input board is specified as a pair (n_grid, last_row).
       The first element in the pair is a list of n length-10 lists.
       Each of the n lists represents a row of the grid.
       If a -1 is in the list it represents an empty cell.
       Otherwise if a number between 0--9 is in the list then this represents a
       pre-set board position. E.g., the board

       ---------------------
       |6| |1|5|7| | | |3| |
       | |9|7| | |2|1| | | |
       | | | | | |0| | | |1|
       | |9| |0|7| |3|5|4| |
       |6| | |5| |0| | | | |
       ---------------------
       would be represented by the list of lists

       [[6, -1, 1, 5, 7, -1, -1, -1, 3, -1],
        [-1, 9, 7, -1, -1, 2, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, 0, -1, -1, -1, 1],
        [-1, 9, -1, 0, 7, -1, 3, 5, 4, -1],
        [6, -1, -1, 5, -1, 0, -1, -1, -1,-1]]


       This routine returns model_1 which consists of a variable for
       each cell of the board, with domain equal to {0-9} if the board
       has a 0 at that position, and domain equal {i} if the board has
       a fixed number i at that cell.

       model_1 contains BINARY CONSTRAINTS OF NOT-EQUAL between
       all relevant variables (e.g., all pairs of variables in the
       same row, etc.).
       model_1 also constains n-nary constraints of sum constraints for each
       column.
    '''

#IMPLEMENT
    # taking in list of lists of Integers, representating game state
    # first we must instantiate the variables and constraints to be added to CSP
    # iterate through numbers in input matrix, and instantiate variables appropriately
        # each variable must have a domain
        # if -1, domain is {0,9}
        # else, it's domain is the number itself
    # we must create a constraint for every variable and add to a VARIABLE LIST
    # constraints must be initialized explicitly.. can't be done with function atm.
    # as we are instantiating variables, let's call _CONSTRAINT_MAKER(vari,j)
    # since we are building left to right, top down.. we can instantiate new constraints
        # as we iterate through matrix, creating binary constraints for every EXISTING
        # adjacent cell to the most recently instantiated variable's L, U, LU&RU Diagonals.
        #DO THIS CHECK IN O(1) TIME BY CHECKIN INDEX, DONT ACTUAL CHECK FOR EXISTENCE OF CELL
    # first make the ALL-DIF constrain for EACH ROW using ONLY BINARY constraints
        # there should be exactly n*(sigma 0-10 n) constraints total for this
    # then make all the 10*(n-1) adjacent row binary constraints
    # then make the diagonal constraints
    # for every variable, create appropriate constraints
        # we only need 1 constraint PER 2 adjacent cells
        # however to implement the ALL DIFF of each row using ONLY BINARY constraints,
            # this should take n-1 + n-2 + n-3 + ... constraints for n ALL-Diff
        # also need to implement sum of column constraint, but using n-ary constraint
    # after instantiating all variables and constraints
    # instantiate csp model with variables in our VARIABLE LIST
    # then add each constraint to csp model!
    # return csp model and variable list!
##############################

def tenner_csp_model_2(initial_tenner_board):
    '''Return a CSP object representing a Tenner Grid CSP problem along
       with an array of variables for the problem. That is return

       tenner_csp, variable_array

       where tenner_csp is a csp representing tenner using model_1
       and variable_array is a list of lists

       [ [  ]
         [  ]
         .
         .
         .
         [  ] ]

       such that variable_array[i][j] is the Variable (object) that
       you built to represent the value to be placed in cell i,j of
       the Tenner Grid (only including the first n rows, indexed from
       (0,0) to (n,9)) where n can be 3 to 8.

       The input board takes the same input format (a list of n length-10 lists
       specifying the board as tenner_csp_model_1.

       The variables of model_2 are the same as for model_1: a variable
       for each cell of the board, with domain equal to {0-9} if the
       board has a -1 at that position, and domain equal {i} if the board
       has a fixed number i at that cell.

       However, model_2 has different constraints. In particular,
       model_2 has a combination of n-nary
       all-different constraints and binary not-equal constraints: all-different
       constraints for the variables in each row, binary constraints for
       contiguous cells (including diagonally contiguous cells), and n-nary sum
       constraints for each column.
       Each n-ary all-different constraint has more than two variables (some of
       these variables will have a single value in their domain).
       model_2 should create these all-different constraints between the relevant
       variables.
    '''

#IMPLEMENT
