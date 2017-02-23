#Look for #IMPLEMENT tags in this file. These tags indicate what has
#to be implemented to complete problem solution.

'''This file will contain different constraint propagators to be used within
   bt_search.

   propagator == a function with the following template
      propagator(csp, newVar=None)
           ==> returns (True/False, [(Variable, Value), (Variable, Value) ...]

      csp is a CSP object---the propagator can use this to get access
      to the variables and constraints of the problem. The assigned variables
      can be accessed via methods, the values assigned can also be accessed.

      newVar (newly instantiated variable) is an optional argument.
      if newVar is not None:
          then newVar is the most
           recently assigned variable of the search.
      else:
          propagator is called before any assignments are made
          in which case it must decide what processing to do
           prior to any variables being assigned. SEE BELOW

       The propagator returns True/False and a list of (Variable, Value) pairs.
       Return is False if a dead-end has been detected by the propagator.
       in this case bt_search will backtrack
       return is true if we can continue.

      The list of variable values pairs are all of the values
      the propagator pruned (using the variable's prune_value method).
      bt_search NEEDS to know this in order to correctly restore these
      values when it undoes a variable assignment.

      NOTE propagator SHOULD NOT prune a value that has already been
      pruned! Nor should it prune a value twice

      PROPAGATOR called with newVar = None
      PROCESSING REQUIRED:
        for plain backtracking (where we only check fully instantiated
        constraints)
        we do nothing...return true, []

        for forward checking (where we only check constraints with one
        remaining variable)
        we look for unary constraints of the csp (constraints whose scope
        contains only one variable) and we forward_check these constraints.

        for gac we establish initial GAC by initializing the GAC queue
        with all constraints of the csp


      PROPAGATOR called with newVar = a variable V
      PROCESSING REQUIRED:
         for plain backtracking we check all constraints with V (see csp method
         get_cons_with_var) that are fully assigned.

         for forward checking we forward check all constraints with V
         that have one unassigned variable left

         for gac we initialize the GAC queue with all constraints containing V.
   '''

def prop_BT(csp, newVar=None):
    '''Do plain backtracking propagation. That is, do no
    propagation at all. Just check fully instantiated constraints'''

    # no newly assigned variable
    if not newVar:
        # return True, no pruned values
        return True, []
    # iterate over all constraints containing most recently assigned variable
    for c in csp.get_cons_with_var(newVar):
        # if all variables of constraint are assigned
        if c.get_n_unasgn() == 0:
            # get scope of the constraint
            vals = []
            vars = c.get_scope()
            # record values of all assigned variables in constraint
            for var in vars:
                vals.append(var.get_assigned_value())
            # if constraint is not satisfied, return False, no pruned values
            if not c.check(vals):
                return False, []
    # every constraint with fully assigned variables is satisfied.
    return True, []

def prop_FC(csp, newVar=None):
    '''Do forward checking. That is check constraints with
       only one uninstantiated variable. Remember to keep
       track of all pruned variable,value pairs and return '''
    # Boolean to indicate whether ANY constraint is unsatisfiable
    satisfiable = True
    # maintain list of pruned var, value pairs
    pruned = []
    # if no new variable assigned, FC all unary constraints in csp
    if not newVar:
        # FC all unary constraints
        for constraint in csp.get_all_cons():
            if len(constraint.get_scope()) == 1:
                satisfiable = _FC(constraint, constraint.get_scope[0], pruned)
                # DWO for var in constraint, return False and pruned for restore
                if not satisfiable:
                    return False, pruned
    # we have most recently assigned variable
    else:
        # FC check all constraints with newVar in scope and ONE unassigned variable
        for constraint in csp.get_cons_with_var(newVar):
            if len(constraint.get_unasgn_vars()) == 1:
               satisfiable = _FC(constraint, constraint.get_unasgn_vars()[0], pruned)
               # DWO for var in constraint, return False and pruned for restore
               if not satisfiable:
                   return False, pruned
    # all constraints were satisfiable using FC, return True and pruned list
    if satisfiable:
        return True, pruned
    #TODO: maybe just initialize list to be FC'd and return _FC(list) or is this inefficient?

# HELPER FUNCTION FOR FORWARD CHECKING
def _FC(cons, var, pruned):
    """
    Return True iff constraint is satisfiable with respect to var,
    update list of pruned

    :param cons: Constraint to forward check
    :type cons: Constraint
    :param var: Variable being checked for satisfying constraint
    :type var: Variable
    :param pruned: List of pruned var, value pairs
    :type pruned: List of Variable, Int tuples
    :return: Boolean indicating whether var did NOT DWO
    :rtype: Boolean
    """
    # iterate over CURRENT domain of values for var
    for value in var.cur_domain():
        # value does not satisfy constraint, prune!
        if not cons.has_support(var, value):
            pruned.append((var, value))
            var.prune_value(value)
            # check for DWO, return False since unsatisfiable
            if var.cur_domain_size() == 0:
                return False
    # no DWO, return True
    return True


def prop_GAC(csp, newVar=None):
    '''Do GAC propagation. If newVar is None we do initial GAC enforce
       processing all constraints. Otherwise we do GAC enforce with
       constraints containing newVar on GAC Queue'''
    # initialize queue to keep track of possibly unsatisfied constraints
    cons_queue = []
    # list of pruned
    pruned = []
    satisfiable = True
    # if no assigned variable, queue all constraints for pre-processing
    if not newVar:
        for cons in csp.get_all_cons():
            cons_queue.append(cons)
    # given most recently assigned variable, queue constraints with newVar
    else:
        for cons in csp.get_cons_with_var(newVar):
            cons_queue.append(cons)
    # execute _GAC_Enforce on constraints
    satisfiable = _GAC_Enforce(csp, cons_queue, pruned)
    # if any constraint is not satisfiable, return False with pruned
    if not satisfiable:
        return False, pruned
    else:
        return True, pruned

def _GAC_Enforce(csp, cons_queue, pruned):
    """
    Return True iff ALL of the queued constraints are satisfiable

    :param cons_queue: potentially unsatisfied constraints for processing
    :type cons_queue: list of Constraints
    :param pruned: list of pruned
    :type pruned: [(Variable, Int)]
    :return: Return True iff ALL of the queued constraints are satisfiable
    :rtype: Boolean
    """
    # process every constraint so ALL var are Arc Consistent
    while len(cons_queue) > 0:
        # check if each variable has consistent values
        cons = cons_queue.pop()
        # if constraint has unassigned variables
        if cons.get_n_unasgn() > 0:
            # establish consistency for all values in all variables
            for var in cons.get_unasgn_vars():
                for val in var.cur_domain():
                    # value is not consistent, prune
                    if not cons.has_support(var, val):
                        pruned.append((var,val))
                        var.prune_value(val)
                        # check for DWO
                        if var.cur_domain_size == 0:
                            return False
                        # add affected constraints to queue
                        for affected_cons in csp.get_cons_with_var(var):
                            cons_queue.append(affected_cons)
    # every constraint is Arc Consistent
    return  True
