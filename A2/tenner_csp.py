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
       model_1 also constraints n-nary constraints of sum constraints for each
       column.
    '''
    # CONSTRAINTS
    cons_lst = []
    num_rows = len(initial_tenner_board[0])
    var_matrix = [[0 for x in range(10)] for y in range(num_rows)]
    for i in range(0, num_rows):
        # we have 10 columns
        for j in range(0, 10):
            entry = initial_tenner_board[0][i][j]
            if entry == -1: # if -1, domain is {0,9}
                curr_var = Variable("V[{}][{}]".format(i,j), list(range(0,10)))
                var_matrix[i][j] = curr_var
            else: #else, domain is restricted to pre-assigned value
                curr_var = Variable("V[{}][{}]".format(i,j), [entry])
                var_matrix[i][j] = curr_var
    for row in range(num_rows):
        for col in range(10):
            _make_Cons(cons_lst, num_rows, var_matrix, row, col)
    # final set of constraints for sum of each column; n-ary constraint per col
    for col in range(0, 10):
        # create list of domains of each variable in that column
        var_col = []
        for row in range (0, num_rows):
            var_col.append(var_matrix[row][col]) # Var[i][j] = var_lst[i*10+j]
        # call helper to derive constraint for this column
        _make_Col_Cons(var_col, initial_tenner_board[1][col], cons_lst, col)
    # initialize list of variables to return with CSP
    var_lst = []
    for row in var_matrix:
        for var in row:
            var_lst.append(var)
    # now instantiate the CSP model
    var_matrix = []
    tenner_csp_model_1 = CSP("tenner_csp_model_1", var_lst)
    for cons in cons_lst:
        tenner_csp_model_1.add_constraint(cons)
    return tenner_csp_model_1, var_matrix

def _make_Cons(cons_lst, num_rows, var_matrix, _i, _j):
    '''
    Instantiates and appends all required constraints with var in scope.

    :param var:
    :type var: Variable
    :param cons_lst:
    :type cons_lst: [Constraint]
    :return:
    :rtype: None
    '''
    i, j = _i, _j
    # for L, LU, U, UR create binary NOT-EQUALS constraint
    process = [(0,-1),(-1,-1),(-1,0),(-1,1)]
    k = j
    # for each EXISTING cell more than one to the left, add corresp. constraint
    while k > 1:
        process.append((0,-k))
        k -= 1
    for (m,n) in process:
        row = i+m
        col = j+n
        # EXISTING ADJACENT CELL
        if (row >= 0 and row <= num_rows) and (col >= 0 and col <= 9):
            # create binary NOT-EQUALS constraint
            # it's row and column indices
            curr_v1 = var_matrix[row][col]
            curr_v2 = var_matrix[i][j]
            curr_cons = Constraint("C(A[{}][{}])(A[{}][{}]".format(row,col,i,j), [curr_v1, curr_v2])
            # create satisfying tuples for constraint
            sat_tuples = []
            # derive satisfying tuples for the two variables
            _notEq_Sat_tuples(curr_v1, curr_v2, sat_tuples)
            # add all the tuples to the constraint
            curr_cons.add_satisfying_tuples(sat_tuples)
            cons_lst.append(curr_cons)

def _notEq_Sat_tuples(var1, var2, sat_tuples):
    '''
    Derive satisfying tuples for given adjacent variables.

    :param cons:
    :type cons:
    :return:
    :rtype: None
    '''
    # check the values in each domain and return all possible ordered tuples
    # such that the first and second elements in the tuple are different
    # check sat for every possible ordering of variables' domain values
    for tuple in itertools.product(var1.domain(), var2.domain()):
        if tuple[0] != tuple[1]:
            # satisfiable tuple, so append!
            sat_tuples.append(tuple)

def _make_Col_Cons(var_lst, total, cons_lst, col):
    col_Cons = Constraint("Column{}".format(col), var_lst)
    sat_tuples = []
    # get list of domains for iterating
    var_dom_lst = [var.domain() for var in var_lst]
    # user itertools to get all satisfying possible tuples
    for tuple in itertools.product(*var_dom_lst):
        if sum(tuple) == total:
            sat_tuples.append(tuple)
    col_Cons.add_satisfying_tuples(sat_tuples)
    cons_lst.append(col_Cons)

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
    # CONSTRAINTS
    cons_lst = []
    num_rows = len(initial_tenner_board[0])
    var_matrix = [[0 for x in range(10)] for y in range(num_rows)]
    for i in range(0, num_rows):
        # we have 10 columns
        for j in range(0, 10):
            entry = initial_tenner_board[0][i][j]
            if entry == -1: # if -1, domain is {0,9}
                curr_var = Variable("V[{}][{}]".format(i,j), list(range(0,10)))
                var_matrix[i][j] = curr_var
            else: #else, domain is restricted to pre-assigned value
                curr_var = Variable("V[{}][{}]".format(i,j), [entry])
                var_matrix[i][j] = curr_var
    for row in range(num_rows):
        for col in range(10):
            _make_Cons_2(cons_lst, num_rows, var_matrix, row, col)
    # n-ary NOT EQUAL row constraints
    for row in range(num_rows):
        row_lst = []
        for col in range(10):
            row_lst.append(var_matrix[row][col])
        _make_Row_Cons(row_lst, cons_lst, row)
    # constraints for sum of each column; n-ary constraint per col
    for col in range(0, 10):
        # create list of domains of each variable in that column
        var_col = []
        for row in range (0, num_rows):
            var_col.append(var_matrix[row][col]) # Var[i][j] = var_lst[i*10+j]
        # call helper to derive constraint for this column
        _make_Col_Cons(var_col, initial_tenner_board[1][col], cons_lst, col)
    # initialize list of variables to return with CSP
    var_lst = []
    for row in var_matrix:
        for var in row:
            var_lst.append(var)
    # now instantiate the CSP model
    var_matrix = []
    tenner_csp_model_1 = CSP("tenner_csp_model_1", var_lst)
    # now instantiate the CSP model
    tenner_csp_model_1 = CSP("tenner_csp_model_2", var_lst)
    for cons in cons_lst:
        tenner_csp_model_1.add_constraint(cons)
    return tenner_csp_model_1, var_matrix

def _make_Cons_2(cons_lst, num_rows, var_matrix, _i, _j):
    '''
    Instantiates and appends all required constraints with var in scope.

    :param var:
    :type var: Variable
    :param cons_lst:
    :type cons_lst: [Constraint]
    :return:
    :rtype: None
    '''
    i, j = _i, _j
    # for LU, U, UR create binary NOT-EQUALS constraint
    for (m,n) in [(-1,-1),(-1,0),(-1,1)]:
        row = i+m
        col = j+n
        # EXISTING ADJACENT CELL
        if (row >= 0 and row <= num_rows) and (col >= 0 and col <= 9):
            # create binary NOT-EQUALS constraint
            # it's row and column indices
            curr_v1 = var_matrix[row][col]
            curr_v2 = var_matrix[i][j]
            curr_cons = Constraint("C(A[{}][{}])(A[{}][{}]".format(row,col,i,j), [curr_v1, curr_v2])
            # create satisfying tuples for constraint
            sat_tuples = []
            # derive satisfying tuples for the two variables
            _notEq_Sat_tuples(curr_v1, curr_v2, sat_tuples)
            # add all the tuples to the constraint
            curr_cons.add_satisfying_tuples(sat_tuples)
            cons_lst.append(curr_cons)

def _make_Row_Cons(row_lst, cons_lst, row):
    row_Cons = Constraint("Row{}".format(row), row_lst)
    sat_tuples = []
    # get list of domains for iterating
    var_dom_lst = [var.domain() for var in row_lst]
    # user itertools to get all satisfying possible tuples
    for tuple in itertools.product(*var_dom_lst):
        # in order for the tuple to be satisfiable, it must consist of unique
        # ordering of 10 numbers {0..9} => SUM MUST BE 45
        if sum(tuple) == 45:
            # create dictionary of counts for constant time access, but break
            # once count higher than 1 is found!
            counts = dict()
            for i in tuple:
                if counts.get(i, 0) > 0:
                    break
                counts[i] = counts.get(i, 0) + 1
            else:
                sat_tuples.append(tuple)
    row_Cons.add_satisfying_tuples(sat_tuples)
    cons_lst.append(row_Cons)
