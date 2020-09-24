import numpy as np
import math
import itertools
from scipy.linalg import solve
from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog
#from pyhull.halfspace import Halfspace, HalfspaceIntersection
import pyhull.halfspace as hs
from typing import List, Tuple

"""Provides an implementation of the revised simplex method.

Classes:
    UnboundedLinearProgram: Exception indicating the unboundedness of an LP.
    InvalidBasis: Exception indicating a list of indices does not form a basis.
    InfeasibleBasicSolution: Exception indicating infeasible bfs.
    LP: Maintains the coefficents and size of a linear program (LP).

Functions:
    invertible: Return true if the matrix A is invertible.
    equality_form: Return the LP in standard equality form.
    phase_one: Execute Phase 1 of the simplex method.
    simplex_iter: Execute a single iteration of the revised simplex method.
    simplex: Run the revised simplex method.
"""


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
        n (int): number of decision variables.
        m (int): number of constraints (excluding nonnegativity constraints).
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
        """Initializes an LP.

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
            m = str(self.m)
            raise ValueError('b should have shape (' + m + ',1) '
                             + 'or (' + m + ') but was ' + str(b.shape) + '.')

        if len(c.shape) == 1 and c.shape[0] == self.n:
            self.c = np.array([c]).transpose()
        elif len(c.shape) == 2 and c.shape == (self.n, 1):
            self.c = np.copy(c)
        else:
            n = str(self.n)
            raise ValueError('c should have shape (' + n + ',1) '
                             + 'or (' + n + ') but was ' + str(c.shape) + '.')

    def get_coefficients(self):
        """Returns n,m,A,b,c describing this LP."""
        return (self.n, self.m,
                np.copy(self.A), np.copy(self.b), np.copy(self.c))

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
            B (List[int]): A list of indices in {0..n+m-1} forming a basis.
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

            - List[np.ndarray]: List of basic feasible solutions for this LP.
            - List[List[int]]: The corresponding list of bases.
            - List[float]: The corresponding list of objective values.
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
        return (bfs, bases, values)

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

    By definition, a matrix A is invertible iff n = m and A has rank n

    Args:
        A (np.ndarray): An m*n matrix

    Returns:
        bool: True if the matrix A is invertible. False otherwise.
    """
    return len(A) == len(A[0]) and np.linalg.matrix_rank(A) == len(A)


def intersection(A: np.ndarray,
                 b: float,
                 D: np.ndarray,
                 e: np.ndarray,
                 interior_pt: np.ndarray = None) -> List[np.ndarray]:
    """Return the intersection of the plane and convex ployhedron.

    Returns the intersection of the plane Ax = b and the convex ployhedron
    described by Dx <= e. An interior point of the intersection can be
    specified for quicker computation time.
    """
    if len(A) != 3 or len(D[0]) != 3:
        raise ValueError('Only supports the intesection of 3d objects.')

    if interior_pt is None:
        pts = []
        A_b = np.hstack((A,b))
        D = np.vstack((D,-np.identity(3)))
        e = np.vstack((e,np.zeros((3,1))))
        D_e = np.hstack((D,e))
        for indices in itertools.combinations(range(len(D)),2):
            M_c = np.vstack((A,D[list(indices)]))
            M_d = np.vstack((A_b,D_e[list(indices)]))
            if np.linalg.matrix_rank(M_c) == 3 and np.linalg.matrix_rank(M_d) == 3:
                det = np.linalg.det(M_c)
                if det != 0:
                    x_1 = np.linalg.det(M_d[:,[3,1,2]])/det
                    x_2 = np.linalg.det(M_d[:,[0,3,2]])/det
                    x_3 = np.linalg.det(M_d[:,[0,1,3]])/det
                    x = np.array([[x_1],[x_2],[x_3]])
                    if all(np.matmul(D,x) <= e + 1e-7):
                        pts.append(np.round(x,7))
    else:
        A_b_ub = np.hstack((A,-b-1e-12))
        A_b_lb = np.hstack((-A,b-1e-12))
        A_b = np.vstack((A_b_ub,A_b_lb))
        D = np.vstack((D,-np.identity(3)))
        e = np.vstack((e,np.zeros((3,1))))
        D_e = np.hstack((D,-e))
        H = np.vstack((D_e,A_b))
        pts = HalfspaceIntersection(H, interior_pt[:,0]).intersections
        pts = np.unique(np.round(pts,10),axis=0)
        pts = [np.array([pts[i,:]]).transpose() for i in range(len(pts))]
    return pts


def halfspace_intersection(A: np.ndarray,
                           b: np.ndarray,
                           interior_pt: np.ndarray = None
                           ) -> hs.HalfspaceIntersection:
    """Returns the halfspace intersection for the given halfspaces.

    If an interior point of the halfspace intersection is not given, one is
    computed using linear programming. It is assumed that a feasible
    interior point of the halfspace intersection exists. This interior point
    is then used to compute the full halfspace intersection.
    """
    n = len(A[0])
    A = np.vstack((A,-np.identity(n)))
    b = -np.vstack((b,np.zeros((n,1))))

    halfspaces = []
    for i in range(len(A)):
        halfspaces.append(hs.Halfspace(A[i],float(b[i])))

    def interior_point(A,b):
        """Get an interior point of the halfspace intersection."""
        M = np.hstack((A,b))
        norm = np.reshape(np.linalg.norm(M[:, :-1], axis=1),(M.shape[0], 1))
        obj_func = np.zeros((M.shape[1],))
        obj_func[-1] = -1
        res = linprog(obj_func,
                      A_ub=np.hstack((M[:, :-1], norm)),
                      b_ub=-M[:, -1:],
                      bounds=(None, None))
        return res.x[:-1]

    if interior_pt is None:
        interior_pt = interior_point(A,b)

    return hs.HalfspaceIntersection(halfspaces, interior_pt)


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

        - np.ndarray: An initial basic feasible solution.
        - List[int]: Corresponding basis to the initial BFS.

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

    # introduce artificial variables
    A = np.hstack((A,np.identity(m)))
    c = np.zeros((n+m,1))
    c[n:,0] = -1

    # use artificial variables as initial basis
    B = list(range(n,n+m))
    x = np.zeros((n+m,1))
    x[B,:] = b

    # solve the auxiliary LP
    aux_lp = LP(A,b,c,equality=True)
    optimal = False
    current_value = float(np.dot(c.transpose(),x))
    while(not optimal):
        x, B, current_value, optimal = simplex_iteration(lp=aux_lp,
                                                         x=x, B=B,
                                                         feas_tol=feas_tol)
        # delete appearances of nonbasic artificial variables
        rem = [i for i in list(range(n,aux_lp.n)) if i not in B]
        A,c,x,B = delete_variables(A,c,x,B,rem)
        aux_lp = LP(A,b,c,equality=True)

    if current_value < -feas_tol:
        raise Infeasible('The LP has no feasible solutions.')
    else:
        # remove constraints and pivot to remove any basic artificial variables
        while(B[-1] >= n):
            j = B[-1]  # basic artificial variable in column j
            a = aux_lp.get_tableau(B)[1:,1:-1]
            i = int(np.nonzero(a[:,j])[0])  # corresponding constraint in row i
            nonzero_a_ij = np.nonzero(a[i,:n])[0]
            if len(nonzero_a_ij) > 0:
                # nonzero a_ij enters and nonbasic artificial variable leaves
                B.append(nonzero_a_ij[0])
                B.remove(j)
                B.sort()
            else:
                # redundant constraint; delete
                A = np.delete(A, i, 0)
                b = np.delete(b, i, 0)
            A,c,x,B = delete_variables(A,c,x,B,[j])
            aux_lp = LP(A,b,c,equality=True)
        return x,B


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
        - 'manual_select': user selects among possible entering indices

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

        - np.ndarray: New basic feasible solution.
        - List[int]: Basis corresponding to the new basic feasible solution.
        - float: Objective value of the new basic feasible solution.
        - bool: An idication of optimality. True if optimal. False otherwise.

    Raises:
        ValueError: Invalid pivot rule. Select from (list).
        ValueError: x should have shape (n+m,1) but was ().

    """
    pivot_rules = ['bland','min_index','dantzig','max_reduced_cost',
                   'greatest_ascent','manual_select']
    if pivot_rule not in pivot_rules:
        raise ValueError('Invalid pivot rule. Select from ' + str(pivot_rules))
    n,m,A,b,c = equality_form(lp).get_coefficients()
    if not x.shape == (n, 1):
        raise ValueError('x should have shape (' + str(n) + ',1) '
                         + 'but was ' + str(x.shape))
    if not np.allclose(x, lp.get_basic_feasible_sol(B), atol=feas_tol):
        raise ValueError('The basis ' + str(B) + ' corresponds to a different '
                         + 'basic feasible solution.')

    N = list(set(range(n)) - set(B))
    y = solve(A[:,B].transpose(), c[B,:])
    red_costs = c - np.dot(y.transpose(),A).transpose()
    entering = {k: red_costs[k] for k in N if red_costs[k] > feas_tol}
    if len(entering) == 0:
        current_value = float(np.dot(c.transpose(), x))
        return x,B,current_value,True
    else:

        def ratio_test(k):
            """Do the ratio test assuming entering index k. Return the leaving
            index r, minimum ratio t, and d from solving A_b*d = A_k."""
            d = np.zeros((1,n))
            d[:,B] = solve(A[:,B], A[:,k])
            ratios = {i: x[i]/d[0][i] for i in B if d[0][i] > 0}
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
            if pivot_rule == 'manual_select':
                user_options = [i + 1 for i in entering.keys()]
                user_input = int(input('Pick one of ' + str(user_options))) - 1
            k = {'bland': min(entering.keys()),
                 'min_index': min(entering.keys()),
                 'dantzig': max(entering, key=entering.get),
                 'max_reduced_cost': max(entering, key=entering.get),
                 'manual_select': user_input}[pivot_rule]
            r,t,d = ratio_test(k)
        # update
        x[k] = t
        x[B,:] = x[B,:] - t*(d[:,B].transpose())
        B.append(k)
        B.remove(r)
        N.append(r)
        N.remove(k)
        current_value = float(np.dot(c.transpose(), x))
        return x,B,current_value,False


def simplex(lp: LP,
            pivot_rule: str = 'bland',
            initial_solution: np.ndarray = None,
            iteration_limit: int = None,
            feas_tol: float = 1e-7
            ) -> Tuple[List[np.ndarray], List[List[int]], float, bool]:
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
        - 'manual_select': user selects among possible entering indices

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

        - List[np.ndarray]: Basic feasible solutions at each simplex iteration.
        - List[List[int]]: Corresponding bases at each simplex iteration.
        - float: The current objective value.
        - bool: True if the current objective value is known to be optimal.

    Raises:
        ValueError: Iteration limit must be strictly positive.
        ValueError: initial_solution should have shape (n,1) but was ().
    """
    if iteration_limit is not None and iteration_limit <= 0:
        raise ValueError('Iteration limit must be strictly positive.')

    n,m,A,b,c = equality_form(lp).get_coefficients()
    x,B = phase_one(lp)

    if initial_solution is not None:
        if not initial_solution.shape == (n, 1):
            raise ValueError('initial_solution should have shape (' + str(n)
                             + ',1) but was ' + str(initial_solution.shape))
        x_B = initial_solution
        if (np.allclose(np.dot(A,x_B), b, atol=feas_tol) and
                all(x_B >= np.zeros((n,1)) - feas_tol) and
                len(np.nonzero(x_B)[0]) <= m):
            x = x_B
            B = list(np.nonzero(x_B)[0])
            N = list(set(range(n)) - set(B))
        else:
            print('Initial solution ignored.')

    path = [np.copy(x)]
    bases = [list.copy(B)]
    current_value = float(np.dot(c.transpose(), x))
    optimal = False

    if iteration_limit is not None:
        lim = iteration_limit
    while(not optimal):
        x, B, current_value, optimal = simplex_iteration(lp=lp, x=x, B=B,
                                                         pivot_rule=pivot_rule,
                                                         feas_tol=feas_tol)
        if not optimal:
            path.append(np.copy(x))
            bases.append(list.copy(B))
        if iteration_limit is not None:
            lim = lim - 1
            if lim == 0:
                break

    return path, bases, current_value, optimal


def branch_and_bound(lp: LP,
                     feas_tol: float = 1e-7,
                     int_feas_tol: float = 1e-7
                     ) -> Tuple[np.ndarray, float]:
    """Execute branch and bound on the given LP.

    Execute branch and bound on the given LP assuming that all decision
    variables must be integer. Use a primal feasibility tolerance of feas_tol
    (with default vlaue of 1e-7) and an integer feasibility tolerance of
    int_feas_tol (with default vlaue of 1e-7).

    Args:
        lp (LP): LP on which to run simplex
        feas_tol (float): Primal feasibility tolerance (1e-7 default).
        int_feas_tol (float): Integer feasibility tolerance (1e-7 default).

    Return:
        Tuple:

        - np.ndarray: An optimal all integer solution.
        - float: The optimal value subject to integrality constraints.
    """
    incumbent = None
    best_bound = None
    unexplored = [lp]

    print('-----------------------------')
    while len(unexplored) > 0:
        sub = unexplored.pop()
        try:
            path, bases, value, opt = simplex(sub,feas_tol=feas_tol)
        except Infeasible:
            print('Fathom by infeasibility.')
            print('-----------------------------')
            continue
        x = path[-1]
        print('x:',np.round(x[:lp.n,0],4))
        print('value:',value)
        if best_bound is not None and best_bound > value:
            print('Fathom by bound.')
        else:
            frac_comp = np.abs(x - np.round(x)) > int_feas_tol
            if np.sum(frac_comp) > 0:
                # branch on first fractional component x_i
                i = np.nonzero(frac_comp)[0][0]
                print('Branch on',i+1)
                frac_val = x[i,0]
                lb, ub = math.floor(frac_val), math.ceil(frac_val)

                def create_branch(sub, i, bound, branch):
                    """Create branch off sub LP on fractional variable x_i."""
                    s = {'left': 1, 'right': -1}[branch]
                    n,m,A,b,c = sub.get_coefficients()
                    v = np.zeros(n)
                    v[i] = s
                    A = np.vstack((A,v))
                    b = np.vstack((b,np.array([[s*bound]])))
                    if lp.equality:
                        A = np.hstack((A,np.zeros((len(A),1))))
                        A[-1,-1] = 1
                        c = np.vstack((c,np.array([0])))
                    return LP(A,b,c)

                unexplored.append(create_branch(sub,i,ub,'right'))
                unexplored.append(create_branch(sub,i,lb,'left'))
            else:
                # better all integer solution
                incumbent = x
                best_bound = value
                print('* New Incumbent.')
        print('-----------------------------')

    return incumbent[:lp.n], best_bound
