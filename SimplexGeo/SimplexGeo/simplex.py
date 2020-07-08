import numpy as np
import itertools
from scipy.linalg import solve
from typing import List, Tuple

"""Provides an implementation of the revised simplex method.

Classes:
    UnboundedLinearProgram: Exception indicating the unboundedness of an LP.
    InvalidBasis: Exception indicating a list of indices does not form a basis.
    InfeasibleBasicSolution: Exception indicating infeasible bfs.
    LP: Maintains the coefficents and size of a linear program (LP).

Functions:
    invertible: Return true if the matrix A is invertible.
    simplex_iter: Execute a single iteration of the revised simplex method.
    simplex: Run the revised simplex method.
"""


class UnboundedLinearProgram(Exception):
    """Raised when an LP is found to be unbounded during an execution of the
    revised simplex method"""
    pass


class InvalidBasis(Exception):
    """Raised when a list of indices does not form a valid basis and prevents
    further correct execution of the function."""
    pass


class InfeasibleBasicSolution(Exception):
    """Raised when a list of indices forms a valid basis but the corresponding
    basic solution is infeasible."""
    pass


class LP:
    """Maintains the coefficents and size of a linear program (LP).

    The LP class maintains the coefficents of a linear program in both standard
    inequality and equality form. A is an m*n matrix describing the linear
    combination of variables making up the LHS of each constraint. b is a
    nonnegative vector of length m making up the RHS of each constraint.
    Lastly, c is a vector of length n describing the objective function to be
    maximized. Both the n decision variables and m slack variables must be
    nonnegative. Under these assumptions, the LP must be feasible.

    inequality        equality
    max c^Tx          max c^Tx
    s.t Ax <= b       s.t Ax + Is == b
         x >= 0               x,s >= 0

    Attributes:
        n (int): number of decision variables (excluding slack variables).
        m (int): number of constraints (excluding nonnegativity constraints).
        A (np.ndarray): An m*n matrix of coefficients.
        A_I (np.ndarray): An m*(n+m) matrix of coefficients: [A I].
        b (np.ndarray): A nonnegative vector of coefficients of length m.
        c (np.ndarray): A vector of coefficients of length n.
        c_0 (np.ndarray): A vector of coefficients of length n+m: [c^T 0^T]^T.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray):
        """Initializes an LP.

        Creates an instance of LP using the given coefficents. Note: the given
        coefficents must correspond to an LP in standard INEQUALITY form.

        max c^Tx
        s.t Ax <= b
             x >= 0

        Args:
            A (np.ndarray): An m*n matrix of coefficients.
            b (np.ndarray): A nonnegative vector of coefficients of length m.
            c (np.ndarray): A vector of coefficients of length n.

        Raises:
            ValueError: A has shape (m,n). b should have shape (m,1) but was ().
            ValueError: b is not nonnegative. Was [].
            ValueError: A has shape (m,n). c should have shape (n,1) but was ().
        """
        self.m = len(A)
        self.n = len(A[0])
        if not b.shape == (self.m, 1):
            raise ValueError('A has shape ' + str(A.shape)
                             + '. b should have shape ('+str(self.m) + ',1) '
                             + 'but was '+str(b.shape))
        if not all(b >= np.zeros((self.m, 1))):
            raise ValueError('b is not nonnegative. Was \n'+str(b))
        if not c.shape == (self.n, 1):
            raise ValueError('A has shape '+str(A.shape)
                             + '. c should have shape (' + str(self.n)+',1) '
                             + 'but was '+str(c.shape))
        self.A = A
        self.A_I = np.hstack((self.A, np.identity(self.m)))
        self.b = b
        self.c = c
        self.c_0 = np.vstack((self.c, np.zeros((self.m, 1))))

    def get_inequality_form(self):
        """Returns n,m,A,b,c describing this LP in standard inequality form."""
        return self.n, self.m, self.A, self.b, self.c

    def get_equality_form(self):
        """Returns n,m,A_I,b,c_0 describing this LP in standard equality form"""
        return self.n, self.m, self.A_I, self.b, self.c_0

    def get_basic_feasible_sol(self, B: List[int]) -> np.ndarray:
        """Return the basic feasible solution corresponding to this basis.

        By definition, B is a basis iff A_B is invertible (where A is the
        matrix of coefficents in standard equality form). The corresponding
        basic solution x satisfies A_Bx = b. By definition, x is a basic
        feasible solution iff x satisfies both Ax = b and x > 0.

        Args:
            B (List[int]): A list of indices in {1..n+m} forming a basis.

        Returns:
            np.ndarray: Basic feasible solution corresponding to the basis B.

        Raises:
            InvalidBasis: B
            InfeasibleBasicSolution: x_B
        """
        n,m,A,b,c = self.get_equality_form()
        B.sort()
        if B[-1] < n+m and invertible(A[:,B]):
            x_B = np.zeros((n+m, 1))
            x_B[B,:] = np.round(solve(A[:,B], b), 7)
            if all(x_B >= np.zeros((n+m, 1))):
                return x_B
            else:
                raise InfeasibleBasicSolution(x_B)
        else:
            raise InvalidBasis(B)

    def get_basic_feasible_solns(self) -> Tuple[List[np.ndarray],
                                                List[List[int]],
                                                List[float]]:
        """Return all basic feasible solutions, their basis, and objective value.

        Returns:
            List[np.ndarray]: The list of basic feasible solutions for this LP.
            List[List[int]]: The corresponding list of bases.
            List[float]: The corresponding list of objective values.
        """
        n, m, A, b, c = self.get_equality_form()
        bfs, bases, values = [], [], []
        for B in itertools.combinations(range(n+m), m):
            try:
                x_B = self.get_basic_feasible_sol(list(B))
                bfs.append(x_B)
                bases.append(B)
                values.append(float(np.round(np.dot(c.transpose(), x_B), 7)))
            except (InvalidBasis, InfeasibleBasicSolution):
                pass
        return (bfs, bases, values)

    def get_tableau(self, B: List[int]) -> np.ndarray:
        """Return the tableau corresponding to the basis B for this LP.

        The returned tableau has the following form:

        z - (c_N^T - y^TA_N)x_N = y^Tb  where   y^T = c_B^TA_B^(-1)
        x_B + A_B^(-1)A_Nx_N = x_B^*    where   x_B^* = A_B^(-1)b

        Args:
            B (List[int]): A valid basis for this LP

        Returns:
            np.ndarray: A numpy array representing the tableau

        Raises:
            InvalidBasis: Invalid basis. A_B is not invertible.
        """
        n,m,A,b,c = self.get_equality_form()
        if not invertible(A[:,B]):
            raise InvalidBasis('Invalid basis. A_B is not invertible.')

        N = list(set(range(n+m)) - set(B))
        B.sort()
        N.sort()
        A_B_inv = np.linalg.inv(A[:,B])
        yT = np.dot(c[B,:].transpose(), A_B_inv)

        T = np.zeros((m+1, n+m+2))
        T[0,0] = 1
        T[0,1:n+m+1][N] = c[N,:].transpose() - np.dot(yT, A[:,N])
        T[0,n+m+1] = np.dot(yT,b)
        T[1:,1:n+m+1][:,N] = np.dot(A_B_inv, A[:,N])
        T[1:,1:n+m+1][:,B] = np.identity(len(B))
        T[1:,n+m+1] = np.dot(A_B_inv, b)[:,0]
        return np.round(T,7)


def invertible(A:np.ndarray) -> bool:
    """Return true if the matrix A is invertible.

    By definition, a matrix A is invertible iff n = m and A has rank n

    Args:
        A (np.ndarray): An m*n matrix

    Returns:
        bool: True if the matrix A is invertible. False otherwise.
    """
    return len(A) == len(A[0]) and np.linalg.matrix_rank(A) == len(A)


def simplex_iteration(lp: LP,
                      x: np.ndarray,
                      B: List[int],
                      pivot_rule: str = 'bland'
                      ) -> Tuple[np.ndarray, List[int], float, bool]:
    """Execute a single iteration of the revised simplex method.

    Let x be the initial basic feasible solution with corresponding basis B. Do
    one iteration of the revised simplex method using the given pivot rule.
    Implemented pivot rules include:

    Entering variable:
        'bland' or 'min_index': minimum index
        'dantzig' or 'max_reduced_cost': most positive reduced cost
        'greatest_ascent': most positive (minimum ratio) x (reduced cost)
        'manual_select': user selects among possible entering indices
    Leaving variable:
        (All): minimum (positive) ratio (minimum index to tie break)

    Args:
        lp (LP): LP on which the simplex iteration is being done.
        x (np.ndarray): Initial basic feasible solution.
        B (List(int)): Basis corresponding to basic feasible solution x.
        pivot_rule (str): Pivot rule to be used. 'bland' by default.

    Returns:
        np.ndarray: New basic feasible solution.
        List[int]: Basis corresponding to the new basic feasible solution.
        float: Objective value of the new basic feasible solution.
        bool: An idication of optimality. True if optimal. False otherwise.

    Raises:
        ValueError: Invalid pivot rule. Select from (list).
        ValueError: x should have shape (n+m,1) but was ().

    """
    pivot_rules = ['bland','min_index','dantzig','max_reduced_cost',
                   'greatest_ascent','manual_select']
    if pivot_rule not in pivot_rules:
        raise ValueError('Invalid pivot rule. Select from ' + pivot_rules)
    n,m,A,b,c = lp.get_equality_form()
    if not x.shape == (n+m, 1):
        raise ValueError('x should have shape (' + str(n+m) + ',1) '
                         + 'but was ' + str(x.shape))
    if not np.allclose(x, lp.get_basic_feasible_sol(B), atol=1e-07):
        raise ValueError('The basis ' + str(B) + ' corresponds to a different '
                         + 'basic feasible solution.')

    N = list(set(range(n+m)) - set(B))
    y = solve(A[:,B].transpose(), c[B,:])
    red_costs = c - np.dot(y.transpose(),A).transpose()
    entering = {k: red_costs[k] for k in N if red_costs[k] > 0}
    if len(entering) == 0:
        current_value = float(np.round(np.dot(c.transpose(), x), 7))
        return x,B,current_value,True
    else:
        if pivot_rule == 'greatest_ascent':
            eligible = {}
            for k in entering:
                d = np.zeros((1,n+m))
                d[:,B] = solve(A[:,B], A[:,k])
                ratios = {i: x[i]/d[0][i] for i in B if d[0][i] > 0}
                if len(ratios) == 0:
                    raise UnboundedLinearProgram('This LP is unbounded')
                t = min(ratios.values())
                r_pos = [r for r in ratios if ratios[r] == t]
                r = min(r_pos)
                t = ratios[r]
                eligible[(t*red_costs[k])[0]] = [k,r,t,d]
            k,r,t,d = eligible[max(eligible.keys())]
        else:
            user_input = None
            if pivot_rule == 'manual_select':
                user_input = int(input('Pick one of ' + str(entering.keys())))
            k = {'bland': min(entering.keys()),
                 'min_index': min(entering.keys()),
                 'dantzig': max(entering, key=entering.get),
                 'max_reduced_cost': max(entering, key=entering.get),
                 'manual_select': user_input}[pivot_rule]
            d = np.zeros((1,n+m))
            d[:,B] = solve(A[:,B], A[:,k])
            ratios = {i: x[i]/d[0][i] for i in B if d[0][i] > 0}
            if len(ratios) == 0:
                raise UnboundedLinearProgram('This LP is unbounded')
            t = min(ratios.values())
            r_pos = [r for r in ratios if ratios[r] == t]
            r = min(r_pos)
            t = ratios[r]
        # update
        x[k] = t
        x[B,:] = x[B,:] - t*(d[:,B].transpose())
        B.append(k)
        B.remove(r)
        N.append(r)
        N.remove(k)
        current_value = float(np.round(np.dot(c.transpose(), x), 7))
        return np.round(x,7),B,current_value,False


def simplex(lp: LP,
            pivot_rule: str = 'bland',
            initial_solution: np.ndarray = None,
            iteration_limit: int = None
            ) -> Tuple[List[np.ndarray], List[List[int]], float, bool]:
    """Execute the revised simplex method on the given LP.

    Execute the revised simplex method on the given LP using the specified
    pivot rule. If a valid initial basic feasible solution is given, use it as
    the initial bfs. Otherwise, ignore it. If an iteration limit is given,
    terminate if the specified limit is reached. Output the current solution
    and indicate the solution may not be optimal.

    PIVOT RULES
    -----------
    Entering variable:
        'bland' or 'min_index': minimum index
        'dantzig' or 'max_reduced_cost': most positive reduced cost
        'greatest_ascent': most positive (minimum ratio) x (reduced cost)
        'manual_select': user selects among possible entering indices
    Leaving variable:
        (All): minimum (positive) ratio (minimum index to tie break)

    Args:
        lp (LP): LP on which to run simplex
        pivot_rule (str): Pivot rule to be used. 'bland' by default.
        initial_solution (np.ndarray): Initial bfs. None by default.
        iteration_limit (int): Simplex iteration limit. None by default.

    Return:
        List[np.ndarray]: Basic feasible solutions at each simplex iteration.
        List[List[int]]: Corresponding bases at each simplex iteration.
        float: The current objective value.
        bool: True if the current objective value is known to be optimal.

    Raises:
        ValueError: Invalid pivot rule. Select from (list).
        ValueError: Iteration limit must be strictly positive.
        ValueError: initial_solution should have shape (n,1) but was ().
    """
    pivot_rules = ['bland','min_index','dantzig','max_reduced_cost',
                   'greatest_ascent','manual_select']
    if pivot_rule not in pivot_rules:
        raise ValueError('Invalid pivot rule. Select from ' + pivot_rules)
    if iteration_limit is not None and iteration_limit <= 0:
        raise ValueError('Iteration limit must be strictly positive.')

    n,m,A,b,c = lp.get_equality_form()

    B = list(range(n,n+m))
    x = np.zeros((n+m,1))
    x[B,:] = b

    if initial_solution is not None:
        if not initial_solution.shape == (n, 1):
            raise ValueError('initial_solution should have shape (' + str(n)
                             + ',1) but was ' + str(initial_solution.shape))
        x_B = np.zeros((n+m,1))
        x_B[:n] = initial_solution
        x_B[n:] = b-np.dot(lp.A,initial_solution)
        x_B = np.round(x_B,7)
        if (all(np.round(np.dot(A,x_B)) == b) and
                all(x_B >= np.zeros((n+m,1))) and
                len(np.nonzero(x_B)[0]) <= m):
            x = x_B
            B = list(np.nonzero(x_B)[0])
            N = list(set(range(n+m)) - set(B))
            while len(B) < m:  # if initial solution is degenerate
                B.append(N.pop())
        else:
            print('Initial solution ignored.')

    path = [np.copy(x)]
    bases = [np.copy(B)]
    current_value = float(np.round(np.dot(c.transpose(), x), 7))
    optimal = False

    if iteration_limit is not None:
        lim = iteration_limit
    while(not optimal):
        x,B,value,opt = simplex_iteration(lp,x,B,pivot_rule)
        current_value = value
        if opt:
            optimal = True
        else:
            path.append(np.copy(x))
            bases.append(np.copy(B))
        if iteration_limit is not None:
            lim = lim - 1
        if iteration_limit is not None and lim == 0:
            break

    return path, bases, current_value, optimal
