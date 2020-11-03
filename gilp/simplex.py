"""Linear and integer program solver.

This module defines an LP class. LPs can be solved through an implementation
of phase I and the revised simplex method. Furthermore, integer solutions can
be obtained through an implementation of the branch and bound algorithm.
"""

__author__ = 'Henry Robbins'
__all__ = ['LP', 'simplex', 'branch_and_bound']

from collections import namedtuple
import itertools
import math
import numpy as np
from scipy.linalg import solve
from typing import List, Tuple
import warnings


class UnboundedLinearProgram(Exception):
    """Raised when an LP is found to be unbounded during an execution of the
    revised simplex method."""
    pass


class InvalidBasis(Exception):
    """Raised when a list of indices does not form a valid basis and prevents
    further correct execution of the function."""
    pass


class Infeasible(Exception):
    """Raised when an LP is found to have no feasible solution."""
    pass


class InfeasibleBasicSolution(Exception):
    """Raised when a list of indices forms a valid basis but the corresponding
    basic solution is infeasible."""
    pass


class LP:
    """Maintains the coefficents and size of a linear program (LP).

    The LP class maintains the coefficents of a linear program in either
    standard equality or inequality form. A is an m*n matrix describing the
    linear combination of variables making up the LHS of each constraint. b is
    a vector of length m making up the RHS of each constraint. Lastly, c is a
    vector of length n describing the objective function to be maximized. The
    n decision variables are all nonnegative.

    ::

        inequality        equality
        max c^Tx          max c^Tx
        s.t Ax <= b       s.t Ax == b
             x >= 0            x >= 0

    Attributes:
        n (int): Number of decision variables.
        m (int): Number of constraints (excluding nonnegativity constraints).
        A (np.ndarray): An m*n matrix of coefficients.
        b (np.ndarray): A vector of coefficients of length m.
        c (np.ndarray): A vector of coefficients of length n.
        equality (bool): True iff the LP is in standard equality form.
    """

    def __init__(self,
                 A: np.ndarray,
                 b: np.ndarray,
                 c: np.ndarray,
                 equality: bool = False):
        """Initialize an LP.

        Creates an instance of LP using the given coefficents interpreted as
        either inequality or equality form.

        ::

            inequality        equality
            max c^Tx          max c^Tx
            s.t Ax <= b       s.t Ax == b
                x >= 0            x >= 0

        Args:
            A (np.ndarray): An m*n matrix of coefficients.
            b (np.ndarray): A vector of coefficients of length m.
            c (np.ndarray): A vector of coefficients of length n.
            equality (bool): True iff the LP is in standard equality form.

        Raises:
            ValueError: b should have shape (m,1) or (m) but was ().
            ValueError: c should have shape (n,1) or (n) but was ().
        """
        self.equality = equality
        self.m = len(A)
        self.n = len(A[0])
        self.A = np.copy(A)

        if len(b.shape) == 1 and b.shape[0] == self.m:
            self.b = np.array([b]).transpose()
        elif len(b.shape) == 2 and b.shape == (self.m, 1):
            self.b = np.copy(b)
        else:
            raise ValueError('b should have shape (%d,1) or (%d) but was %s.'
                             % (self.m, self.m, str(b.shape)))

        if len(c.shape) == 1 and c.shape[0] == self.n:
            self.c = np.array([c]).transpose()
        elif len(c.shape) == 2 and c.shape == (self.n, 1):
            self.c = np.copy(c)
        else:
            raise ValueError('c should have shape (%d,1) or (%d) but was %s.'
                             % (self.n, self.n, str(b.shape)))

    def get_coefficients(self):
        """Returns n,m,A,b,c describing this LP."""
        Coefficents = namedtuple('coefficents', ['n', 'm', 'A', 'b', 'c'])
        return Coefficents(n=self.n,
                           m=self.m,
                           A=np.copy(self.A),
                           b=np.copy(self.b),
                           c=np.copy(self.c))

    def get_basic_feasible_sol(self,
                               B: List[int],
                               feas_tol: float = 1e-7) -> np.ndarray:
        """Return the basic feasible solution corresponding to this basis.

        By definition, B is a basis iff A_B is invertible (where A is the
        matrix of coefficents in standard equality form). The corresponding
        basic solution x satisfies A_Bx = b. By definition, x is a basic
        feasible solution iff x satisfies both A_Bx = b and x > 0. These
        constraints must be satisfied to a tolerance of feas_tol (which is set
        to 1e-7 by default).

        Args:
            B (List[int]): A list of indices in {0..(n+m-1)} forming a basis.
            feas_tol (float): Primal feasibility tolerance (1e-7 by default).

        Returns:
            np.ndarray: Basic feasible solution corresponding to the basis B.

        Raises:
            InvalidBasis: B
            InfeasibleBasicSolution: x_B
        """
        n,m,A,b,c = equality_form(self).get_coefficients()
        B.sort()
        if B[-1] < n and invertible(A[:,B]):
            x_B = np.zeros((n, 1))
            x_B[B,:] = solve(A[:,B], b)
            if all(x_B >= np.zeros((n, 1)) - feas_tol):
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
            Tuple:

            - bfs (List[np.ndarray]): Basic feasible solutions for this LP.
            - bases (List[List[int]]): The corresponding list of bases.
            - values (List[float]): The corresponding list of objective values.
        """
        n,m,A,b,c = equality_form(self).get_coefficients()
        bfs, bases, values = [], [], []
        for B in itertools.combinations(range(n), m):
            try:
                x_B = self.get_basic_feasible_sol(list(B))
                bfs.append(x_B)
                bases.append(list(B))
                values.append(float(np.dot(c.transpose(), x_B)))
            except (InvalidBasis, InfeasibleBasicSolution):
                pass
        BFSList = namedtuple('bfs_list', ['bfs', 'bases', 'values'])
        return BFSList(bfs=bfs, bases=bases, values=values)

    def get_tableau(self, B: List[int]) -> np.ndarray:
        """Return the tableau corresponding to the basis B for this LP.

        The returned tableau has the following form::

            z - (c_N^T - y^TA_N)x_N = y^Tb  where   y^T = c_B^TA_B^(-1)
            x_B + A_B^(-1)A_Nx_N = x_B^*    where   x_B^* = A_B^(-1)b

        Args:
            B (List[int]): A valid basis for this LP

        Returns:
            np.ndarray: A numpy array representing the tableau

        Raises:
            InvalidBasis: Invalid basis. A_B is not invertible.
        """
        n,m,A,b,c = equality_form(self).get_coefficients()
        if not invertible(A[:,B]):
            raise InvalidBasis('Invalid basis. A_B is not invertible.')

        N = list(set(range(n)) - set(B))
        B.sort()
        N.sort()
        A_B_inv = np.linalg.inv(A[:,B])
        yT = np.dot(c[B,:].transpose(), A_B_inv)

        T = np.zeros((m+1, n+2))
        T[0,0] = 1
        T[0,1:n+1][N] = -(c[N,:].transpose() - np.dot(yT, A[:,N]))
        T[0,n+1] = np.dot(yT,b)
        T[1:,1:n+1][:,N] = np.dot(A_B_inv, A[:,N])
        T[1:,1:n+1][:,B] = np.identity(len(B))
        T[1:,n+1] = np.dot(A_B_inv, b)[:,0]
        return T


def invertible(A:np.ndarray) -> bool:
    """Return true if the matrix A is invertible.

    By definition, a matrix A is invertible iff n = m and A has rank n.

    Args:
        A (np.ndarray): An m*n matrix.

    Returns:
        bool: True if the matrix A is invertible. False otherwise.
    """
    return len(A) == len(A[0]) and np.linalg.matrix_rank(A) == len(A)


def equality_form(lp: LP) -> LP:
    """Return the LP in standard equality form.

    Transform the LP (if needed) into an equivalent LP in standard equality
    form. Furthermore, ensure that every element of b is nonnegative. The
    transformation can be summariazed as follows.

    ::

        inequality        equality
        max c^Tx          max c^Tx
        s.t Ax <= b       s.t Ax + Is == b
             x >= 0               x,s >= 0

    Args:
        lp (LP): An LP in either standard inequality or equality form.

    Returns:
        LP: The corresponding standard equality form LP
    """
    n,m,A,b,c = lp.get_coefficients()
    if not lp.equality:
        # add slack variables
        A = np.hstack((A, np.identity(m)))
        c = np.vstack((c, np.zeros((m, 1))))
    # ensure every element of b is nonnegative
    neg = (b < 0)[:,0]
    b[neg] = -b[neg]
    A[neg,:] = -A[neg,:]
    return LP(A,b,c,equality=True)


def phase_one(lp: LP, feas_tol: float = 1e-7) -> Tuple[np.ndarray, List[int]]:
    """Execute Phase I of the simplex method.

    Execute Phase I of the simplex method to find an inital basic feasible
    solution to the given LP. Return a basic feasible solution if one exists.
    Otherwise, raise the Infeasible exception.

    Args:
        lp (LP): LP on which phase I of the simplex method will be done.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).

    Returns:
        Tuple:

        - x (np.ndarray): An initial basic feasible solution.
        - B (List[int]): Corresponding basis to the initial BFS.

    Raises:
        Infeasible: The LP is found to not have a feasible solution.
    """

    def delete_variables(A,c,x,B,rem):
        """Delete variables with indices in rem from the given coefficent
        matrices, basic feasible solution, and basis."""
        in_basis = np.array([int(i in B) for i in range(len(A[0]))])
        B = list(np.nonzero(np.delete(in_basis,rem))[0])
        A = np.delete(A, rem, 1)
        c = np.delete(c, rem, 0)
        x = np.delete(x, rem, 0)
        return A,c,x,B

    n,m,A,b,c = equality_form(lp).get_coefficients()

    # Introduce artificial variables
    A = np.hstack((A,np.identity(m)))
    c = np.zeros((n+m,1))
    c[n:,0] = -1

    # Use artificial variables as initial basis
    B = list(range(n,n+m))
    x = np.zeros((n+m,1))
    x[B,:] = b

    # Solve the auxiliary LP
    aux_lp = LP(A,b,c,equality=True)
    optimal = False
    current_value = float(np.dot(c.transpose(),x))
    while(not optimal):
        x, B, current_value, optimal = simplex_iteration(lp=aux_lp,
                                                         x=x, B=B,
                                                         feas_tol=feas_tol)
        # Delete appearances of nonbasic artificial variables
        rem = [i for i in list(range(n,aux_lp.n)) if i not in B]
        A,c,x,B = delete_variables(A,c,x,B,rem)
        aux_lp = LP(A,b,c,equality=True)

    # Interpret solution to the auxiliary LP
    if current_value < -feas_tol:
        raise Infeasible('The LP has no feasible solutions.')
    else:
        # Remove constraints and pivot to remove any basic artificial variables
        while(B[-1] >= n):
            j = B[-1]  # Basic artificial variable in column j
            a = aux_lp.get_tableau(B)[1:,1:-1]
            i = int(np.nonzero(a[:,j])[0])  # Corresponding constraint in row i
            nonzero_a_ij = np.nonzero(a[i,:n])[0]
            if len(nonzero_a_ij) > 0:
                # Nonzero a_ij enters and nonbasic artificial variable leaves
                B.append(nonzero_a_ij[0])
                B.remove(j)
                B.sort()
            else:
                # Redundant constraint; delete
                A = np.delete(A, i, 0)
                b = np.delete(b, i, 0)
            A,c,x,B = delete_variables(A,c,x,B,[j])
            aux_lp = LP(A,b,c,equality=True)
        InitSol = namedtuple('init_sol', ['x', 'B'])
        return InitSol(x=x, B=B)


def simplex_iteration(lp: LP,
                      x: np.ndarray,
                      B: List[int],
                      pivot_rule: str = 'bland',
                      feas_tol: float = 1e-7
                      ) -> Tuple[np.ndarray, List[int], float, bool]:
    """Execute a single iteration of the revised simplex method.

    Let x be the initial basic feasible solution with corresponding basis B.
    Use a primal feasibility tolerance of feas_tol (with default vlaue of
    1e-7). Do one iteration of the revised simplex method using the given
    pivot rule. Implemented pivot rules include:

    Entering variable:

        - 'bland' or 'min_index': minimum index
        - 'dantzig' or 'max_reduced_cost': most positive reduced cost
        - 'greatest_ascent': most positive (minimum ratio) x (reduced cost)
        - 'manual' or 'manual_select': user selects possible entering index

    Leaving variable:

        - (All): minimum (positive) ratio (minimum index to tie break)

    Args:
        lp (LP): LP on which the simplex iteration is being done.
        x (np.ndarray): Initial basic feasible solution.
        B (List(int)): Basis corresponding to basic feasible solution x.
        pivot_rule (str): Pivot rule to be used. 'bland' by default.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).

    Returns:
        Tuple:

        - x (np.ndarray): New basic feasible solution.
        - B (List[int]): Basis for the new basic feasible solution.
        - obj_val (float): Objective value of the new basic feasible solution.
        - optimal (bool_: True if x is optimal. False otherwise.

    Raises:
        ValueError: Invalid pivot rule. Select from (list).
        ValueError: x should have shape (n+m,1) but was ().

    """
    pivot_rules = ['bland','min_index','dantzig','max_reduced_cost',
                   'greatest_ascent','manual', 'manual_select']
    if pivot_rule not in pivot_rules:
        raise ValueError('Invalid pivot rule. Select from ' + str(pivot_rules))
    n,m,A,b,c = equality_form(lp).get_coefficients()
    if not x.shape == (n, 1):
        raise ValueError('x should have shape (%d,1) but was %s'
                         % (n, str(x.shape)))
    if not np.allclose(x, lp.get_basic_feasible_sol(B), atol=feas_tol):
        raise ValueError("The basis %s corresponds to a different basic "
                         "feasible solution." % (str(B)))

    # Named tuple for return value
    SimplexIter = namedtuple('simplex_iter', ['x', 'B', 'obj_val', 'optimal'])

    N = list(set(range(n)) - set(B))
    y = solve(A[:,B].transpose(), c[B,:])
    red_costs = c - np.dot(y.transpose(),A).transpose()
    entering = {k: red_costs[k] for k in N if red_costs[k] > feas_tol}
    if len(entering) == 0:
        current_value = float(np.dot(c.transpose(), x))
        return SimplexIter(x=x, B=B, obj_val=current_value, optimal=True)
    else:

        def ratio_test(k):
            """Do the ratio test assuming entering index k. Return the leaving
            index r, minimum ratio t, and d from solving A_b*d = A_k."""
            d = np.zeros((1,n))
            d[:,B] = solve(A[:,B], A[:,k])
            ratios = {i: x[i]/d[0][i] for i in B if d[0][i] > feas_tol}
            if len(ratios) == 0:
                raise UnboundedLinearProgram('This LP is unbounded')
            t = min(ratios.values())
            r_pos = [r for r in ratios if ratios[r] == t]
            r = min(r_pos)
            t = ratios[r]
            return r,t,d

        if pivot_rule == 'greatest_ascent':
            eligible = {}
            for k in entering:
                r,t,d = ratio_test(k)
                eligible[(t*red_costs[k])[0]] = [k,r,t,d]
            k,r,t,d = eligible[max(eligible.keys())]
        else:
            user_input = None
            if pivot_rule in ['manual', 'manual_select']:
                user_options = [i + 1 for i in entering.keys()]
                user_input = int(input('Pick one of ' + str(user_options))) - 1
            k = {'bland': min(entering.keys()),
                 'min_index': min(entering.keys()),
                 'dantzig': max(entering, key=entering.get),
                 'max_reduced_cost': max(entering, key=entering.get),
                 'manual_select': user_input,
                 'manual': user_input}[pivot_rule]
            r,t,d = ratio_test(k)
        # Update
        x[k] = t
        x[B,:] = x[B,:] - t*(d[:,B].transpose())
        B.append(k)
        B.remove(r)
        N.append(r)
        N.remove(k)
        current_value = float(np.dot(c.transpose(), x))
        return SimplexIter(x=x, B=B, obj_val=current_value, optimal=False)


def simplex(lp: LP,
            pivot_rule: str = 'bland',
            initial_solution: np.ndarray = None,
            iteration_limit: int = None,
            feas_tol: float = 1e-7
            ) -> Tuple[np.ndarray, List[int], float, bool,
                       Tuple[np.ndarray, List[int], float]]:
    """Execute the revised simplex method on the given LP.

    Execute the revised simplex method on the given LP using the specified
    pivot rule. If a valid initial basic feasible solution is given, use it as
    the initial bfs. Otherwise, ignore it. If an iteration limit is given,
    terminate if the specified limit is reached. Output the current solution
    and indicate the solution may not be optimal. Use a primal feasibility
    tolerance of feas_tol (with default vlaue of 1e-7).

    PIVOT RULES

    Entering variable:

        - 'bland' or 'min_index': minimum index
        - 'dantzig' or 'max_reduced_cost': most positive reduced cost
        - 'greatest_ascent': most positive (minimum ratio) x (reduced cost)
        - 'manual' or 'manual_select': user selects possible entering index

    Leaving variable:

        - (All): minimum (positive) ratio (minimum index to tie break)

    Args:
        lp (LP): LP on which to run simplex
        pivot_rule (str): Pivot rule to be used. 'bland' by default.
        initial_solution (np.ndarray): Initial bfs. None by default.
        iteration_limit (int): Simplex iteration limit. None by default.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).

    Return:
        Tuple:

        - x (np.ndarray): Current basic feasible solution.
        - B (List[int]): Corresponding bases of the current best BFS.
        - obj_val (float): The current objective value.
        - optimal (bool): True if x is optimal. False otherwise.
        - path (Tuple[np.ndarray, List[int], float]): Path of simplex.

    Raises:
        ValueError: Iteration limit must be strictly positive.
        ValueError: initial_solution should have shape (n,1) but was ().
    """
    if iteration_limit is not None and iteration_limit <= 0:
        raise ValueError('Iteration limit must be strictly positive.')

    n,m,A,b,c = equality_form(lp).get_coefficients()
    init_sol = phase_one(lp)
    x,B = init_sol.x, init_sol.B

    if initial_solution is not None:
        initial_solution = initial_solution.astype(float)
        # If the LP is in standard inequality form, the initial solution can be
        # set by only providing decision variables values; slacks computed.
        if not lp.equality and initial_solution.shape == (lp.n, 1):
            # compute slacks
            slacks = b - np.matmul(lp.A, initial_solution)
            initial_solution = np.vstack((initial_solution, slacks))

        if not initial_solution.shape == (n, 1):
            shape = str(initial_solution.shape)
            if lp.equality:
                raise ValueError("Initial solution should have shape (%d,1) "
                                 "but was %s" % (n, shape))
            else:
                raise ValueError("Initial solution should have shape (%d,1) "
                                 "or (%d,1) but was %s""" % (lp.n, n, shape))

        x_B = initial_solution
        if (np.allclose(np.dot(A,x_B), b, atol=feas_tol) and
                all(x_B >= np.zeros((n,1)) - feas_tol) and
                len(np.nonzero(x_B)[0]) <= m):
            x = x_B
            B = list(np.nonzero(x_B)[0])
            N = list(set(range(lp.n+lp.m)) - set(B))
            while len(B) < m:  # if initial solution is degenerate
                B.append(N.pop())
        else:
            warnings.warn("Provided initial solution was not a basic feasible "
                          "solution; ignored.", UserWarning)

    current_value = float(np.dot(c.transpose(), x))
    optimal = False

    path = []
    BFS = namedtuple('bfs', ['x', 'B', 'obj_val'])

    # Print instructions if manual mode is chosen.
    if pivot_rule in ['manual', 'manual_select']:
        print('''
        INSTRUCTIONS

        At each iteration of simplex, choose one of the variables with a
        positive coefficent in the objective function. The list of indices
        for possible variables (also called entering variables) is given.
        ''')

    i = 0  # number of iterations
    while(not optimal):
        path.append(BFS(x=x.copy(), B=B.copy(), obj_val=current_value))
        simplex_iter = simplex_iteration(lp=lp, x=x, B=B,
                                         pivot_rule=pivot_rule,
                                         feas_tol=feas_tol)
        x = simplex_iter.x
        B = simplex_iter.B
        current_value = simplex_iter.obj_val
        optimal = simplex_iter.optimal
        i = i + 1
        if iteration_limit is not None and i >= iteration_limit:
            break
    Simplex = namedtuple('simplex', ['x', 'B', 'obj_val', 'optimal', 'path'])
    return Simplex(x=x, B=B, obj_val=current_value, optimal=optimal, path=path)


def branch_and_bound_iteration(lp: LP,
                               incumbent: np.ndarray,
                               best_bound: float,
                               manual: bool = False,
                               feas_tol: float = 1e-7,
                               int_feas_tol: float = 1e-7
                               ) -> Tuple[bool, np.ndarray, float, LP, LP]:
    """Exectue one iteration of branch and bound on the given node.

    Execute one iteration of branch and bound on the given node (LP). Update
    the current incumbent and best bound if needed. Use the given primal
    feasibility and integer feasibility tolerance (defaults to 1e-7).

    Args:
        lp (LP): Branch and bound node.
        incumbent (np.ndarray): Current incumbent solution.
        best_bound (float): Current best bound.
        manual (bool): True if the user can choose the variable to branch on.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).
        int_feas_tol (float): Integer feasibility tolerance (1e-7 default).

    Returns:
        Tuple:

        - fathomed (bool): True if node was fathomed. False otherwise.
        - incumbent (np.ndarray): Current incumbent solution (after iteration).
        - best_bound (float): Current best bound (after iteration).
        - right_LP (LP): Left branch node (LP).
        - left_LP (LP): Right branch node (LP).
    """

    # Named tuple for return values
    BnbIter = namedtuple('bnb_iter', ['fathomed','incumbent','best_bound',
                                      'left_LP', 'right_LP'])

    try:
        sol = simplex(lp=lp, feas_tol=feas_tol)
        x = sol.x
        value = sol.obj_val
    except Infeasible:
        return BnbIter(fathomed=True, incumbent=incumbent,
                       best_bound=best_bound, left_LP=None, right_LP=None)
    if best_bound is not None and best_bound > value:
        return BnbIter(fathomed=True, incumbent=incumbent,
                       best_bound=best_bound, left_LP=None, right_LP=None)
    else:
        frac_comp = ~np.isclose(x, np.round(x), atol=int_feas_tol)[:lp.n]
        if np.sum(frac_comp) > 0:
            pos_i = np.nonzero(frac_comp)[0]  # list of indices to branch on
            if manual:
                pos_i = [i + 1 for i in pos_i]
                i = int(input('Pick one of ' + str(pos_i))) - 1
            else:
                i = pos_i[0]  # branch on first fractional component x_i
            frac_val = x[i,0]
            lb, ub = math.floor(frac_val), math.ceil(frac_val)

            def create_branch(lp, i, bound, branch):
                """Create branch off LP on fractional variable x_i."""
                s = {'left': 1, 'right': -1}[branch]
                n,m,A,b,c = lp.get_coefficients()
                v = np.zeros(n)
                v[i] = s
                A = np.vstack((A,v))
                b = np.vstack((b,np.array([[s*bound]])))
                if lp.equality:
                    A = np.hstack((A,np.zeros((len(A),1))))
                    A[-1,-1] = 1
                    c = np.vstack((c,np.array([0])))
                return LP(A,b,c)

            left_LP = create_branch(lp,i,lb,'left')
            right_LP = create_branch(lp,i,ub,'right')
        else:
            # better all integer solution
            incumbent = x
            best_bound = value
            return BnbIter(fathomed=True, incumbent=incumbent,
                           best_bound=best_bound, left_LP=None, right_LP=None)
    return BnbIter(fathomed=False, incumbent=incumbent,best_bound=best_bound,
                   left_LP=left_LP, right_LP=right_LP)


def branch_and_bound(lp: LP,
                     manual: bool = False,
                     feas_tol: float = 1e-7,
                     int_feas_tol: float = 1e-7
                     ) -> Tuple[np.ndarray, float]:
    """Execute branch and bound on the given LP.

    Execute branch and bound on the given LP assuming that all decision
    variables must be integer. Use a primal feasibility tolerance of feas_tol
    (with default vlaue of 1e-7) and an integer feasibility tolerance of
    int_feas_tol (with default vlaue of 1e-7).

    Args:
        lp (LP): LP on which to run the branch and bound algorithm.
        manual (bool): True if the user can choose the variable to branch on.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).
        int_feas_tol (float): Integer feasibility tolerance (1e-7 default).

    Return:
        Tuple:

        - x (np.ndarray): An optimal all integer solution.
        - obj_val(float): The optimal value subject to integrality constraints.
    """
    incumbent = None
    best_bound = None
    unexplored = [lp]

    while len(unexplored) > 0:
        sub = unexplored.pop()
        iteration = branch_and_bound_iteration(lp=sub,
                                               incumbent=incumbent,
                                               best_bound=best_bound,
                                               manual=manual,
                                               feas_tol=feas_tol,
                                               int_feas_tol=int_feas_tol)
        fathom = iteration.fathomed
        incumbent = iteration.incumbent
        best_bound = iteration.best_bound
        left_LP = iteration.left_LP
        right_LP = iteration.right_LP
        if not fathom:
            unexplored.append(right_LP)
            unexplored.append(left_LP)
    Bnb = namedtuple('bnb', ['x', 'obj_val'])
    return Bnb(x=incumbent[:lp.n], obj_val=best_bound)
