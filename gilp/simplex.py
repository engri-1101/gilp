"""Linear and integer program solver.

This module defines an LP class. LPs can be solved through an implementation
of phase I and the revised simplex method. Furthermore, integer solutions can
be obtained through an implementation of the branch and bound algorithm.
"""

__author__ = 'Henry Robbins'
__all__ = ['BFS', 'LP', 'simplex', 'branch_and_bound']

from collections import namedtuple
import itertools
from ._geometry import polytope_vertices
import math
import numpy as np
from scipy.linalg import solve, LinAlgError
from typing import Union, List, Tuple
import warnings

BFS = namedtuple('bfs', ['x', 'B', 'obj_val', 'optimal'])
BFS.__doc__ = '''\
Basic feasible solution (BFS) for a linear program (LP).

- x (np.ndarray): Basic feasible solution.
- B (List[int]): Basis for the basic feasible solution.
- obj_val (float): Objective value of the basic feasible solution.
- optimal (bool): True if x is known to be optimal. False otherwise.'''


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

    The LP class maintains the coefficents of a linear program. If initialized
    in standard inequality form, both standard equality and inequality form
    are maintained. Otherwise, only standard equality form is maintained.
    Hence, if equality is True, the attributes A, b, and c are None.

    ::

        inequality        equality
        max c^Tx          max c_eq^Tx
        s.t A x <= b      s.t A_eq x == b_eq
              x >= 0               x >= 0

    Attributes:
        n (int): Number of decision variables (excluding slack variables).
        m (int): Number of constraints (excluding nonnegativity constraints).
        A (np.ndarray): LHS coefficients of LP in standard inequality form.
        A_eq (np.ndarray): LHS coefficients of LP in standard equality form.
        b (np.ndarray): RHS coefficients of LP in standard inequality form.
        b_eq (np.ndarray): RHS coefficients of LP in standard equality form.
        c (np.ndarray): Objective function coefficents for inequality form.
        c_eq (np.ndarray): Objective function coefficents for equality form.
        equality (bool): True iff the LP is in standard equality form.
    """

    def __init__(self,
                 A: Union[np.ndarray, List, Tuple],
                 b: Union[np.ndarray, List, Tuple],
                 c: Union[np.ndarray, List, Tuple],
                 equality: bool = False):
        """Initialize an LP.

        Creates an instance of LP using the given coefficents interpreted as
        either inequality or equality form.

        ::

            inequality        equality
            max c^Tx          max c^Tx
            s.t Ax <= b       s.t Ax == b
                x  >= 0            x >= 0

        Args:
            A (Union[np.ndarray, List, Tuple]): An m*n matrix of coefficients.
            b (Union[np.ndarray, List, Tuple]): Coefficient vector of length m.
            c (Union[np.ndarray, List, Tuple]): Coefficient vector of length n.
            equality (bool): True iff the LP is in standard equality form.

        Raises:
            ValueError: b should have shape (m,1) or (m) but was ().
            ValueError: c should have shape (n,1) or (n) but was ().
        """
        self.equality = equality
        self.m = len(A)
        self.n = len(A[0])

        if self.equality:
            self.A_eq = np.copy(A) if type(A) != np.array else np.array(A)
            self.b_eq = _validate(vector=_vectorize(b), sizes=self.m, name='b')
            self.c_eq = _validate(vector=_vectorize(c), sizes=self.n, name='c')

            A, b, c = (None, None, None)
        else:
            self.A = np.copy(A) if type(A) != np.array else np.array(A)
            self.b = _validate(vector=_vectorize(b), sizes=self.m, name='b')
            self.c = _validate(vector=_vectorize(c), sizes=self.n, name='c')

            self.A_eq = np.hstack((self.A, np.identity(self.m)))
            self.b_eq = np.copy(self.b)
            self.c_eq = np.vstack((self.c, np.zeros((self.m, 1))))

    def get_coefficients(self, equality: bool = True):
        """Returns the coefficents describing this LP.

        If equality is True (defaults to True), then return standard equality
        coefficents. Otherwise, return standard inequality coefficents. Also
        returns the dimensions of m*n matrix A.
        """
        Coefficents = namedtuple('coefficents', ['n', 'm', 'A', 'b', 'c'])
        if equality:
            m, n = self.A_eq.shape
            return Coefficents(n=n,
                               m=m,
                               A=np.copy(self.A_eq),
                               b=np.copy(self.b_eq),
                               c=np.copy(self.c_eq))
        else:
            if self.equality:
                raise ValueError('Equality form LP. No inequality form.')
            m, n = self.A.shape
            return Coefficents(n=n,
                               m=m,
                               A=np.copy(self.A),
                               b=np.copy(self.b),
                               c=np.copy(self.c))

    def get_basic_feasible_sol(self,
                               B: List[int],
                               feas_tol: float = 1e-7) -> BFS:
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
            BFS: Basic feasible solution corresponding to the basis B.

        Raises:
            InvalidBasis: B
            InfeasibleBasicSolution: x_B
        """
        n,m,A,b,c = self.get_coefficients()
        B.sort()
        if len(B) == m and B[-1] < n:
            try:
                x = solve(A[:,B], b)
            except LinAlgError:
                raise InvalidBasis(B)
            x_B = np.zeros((n, 1))
            x_B[B,:] = x
            if all(x_B >= np.zeros((n, 1)) - feas_tol):
                return BFS(x=x_B,
                           B=B,
                           obj_val=float(np.dot(c.transpose(), x_B)),
                           optimal=False)
            else:
                raise InfeasibleBasicSolution(x_B)
        else:
            raise InvalidBasis(B)

    def get_basic_feasible_solns(self) -> List[BFS]:
        """Return all the basic feasible solutions.

        Returns:
            List[BFS]: List of basic feasible solutions.
        """
        n,m,A,b,c = self.get_coefficients()
        bfs = []
        for B in itertools.combinations(range(n), m):
            try:
                bfs.append(self.get_basic_feasible_sol(list(B)))
            except (InvalidBasis, InfeasibleBasicSolution):
                pass
        return bfs

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
        n,m,A,b,c = self.get_coefficients()
        if not _invertible(A[:,B]):
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

    def get_vertices(self) -> np.ndarray:
        """Return the vertices of this inequality LP's feasible region.

        Returns:
            np.ndarray: Vertices of the LP's feasible region.

        Raises:
            ValueError: The LP must be in standard inequality form.
        """
        try:
            n,m,A,b,c = self.get_coefficients(equality=False)
        except ValueError:
            raise ValueError('The LP must be in standard inequality form.')

        # Add non-negativity constraints and return vertices
        A_tmp = np.vstack((A, -np.identity(n)))
        b_tmp = np.vstack((b, np.zeros((n,1))))
        return polytope_vertices(A_tmp, b_tmp)


def _vectorize(array: Union[np.ndarray, List, Tuple]):
    """Vectorize the input array."""
    if type(array) != np.array:
        array = np.array(array)
    if len(array.shape) == 1:
        array = np.array([array]).transpose()
    return array


def _validate(vector: np.ndarray, sizes: List[int], name: str):
    """Validate vector has one of the expected sizes."""
    sizes = [sizes] if type(sizes) == int else sizes
    if vector.shape[0] in sizes:
        return np.copy(vector)
    else:
        sizes_str = ', '.join(["(%d,1), (%d)" % tuple([s]*2) for s in sizes])
        raise ValueError("%s should have one of the following shapes: %s but "
                         "was %s." % (name, sizes_str, str(vector.shape)))


def _invertible(A:np.ndarray) -> bool:
    """Return true if the matrix A is invertible.

    By definition, a matrix A is invertible iff n = m and A has rank n.

    Args:
        A (np.ndarray): An m*n matrix.

    Returns:
        bool: True if the matrix A is invertible. False otherwise.
    """
    return len(A) == len(A[0]) and np.linalg.matrix_rank(A) == len(A)


def _phase_one(lp: LP, feas_tol: float = 1e-7) -> BFS:
    """Execute Phase I of the simplex method.

    Execute Phase I of the simplex method to find an inital basic feasible
    solution to the given LP. Return a basic feasible solution if one exists.
    Otherwise, raise the Infeasible exception.

    Args:
        lp (LP): LP on which phase I of the simplex method will be done.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).

    Returns:
        BFS: Inital basic feasible solution

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

    n,m,A,b,c = lp.get_coefficients()

    # Augment so b is non-negative
    neg = (b < 0)[:,0]
    b[neg] = -b[neg]
    A[neg,:] = -A[neg,:]

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
    obj_val = float(np.dot(c.transpose(),x))
    bfs = BFS(x=x, B=B, obj_val=obj_val, optimal=optimal)

    while(not optimal):
        x, B, obj_val, optimal = _simplex_iteration(lp=aux_lp,
                                                    bfs=bfs,
                                                    feas_tol=feas_tol)
        # Delete appearances of nonbasic artificial variables
        rem = [i for i in list(range(n,aux_lp.n)) if i not in B]
        A,c,x,B = delete_variables(A,c,x,B,rem)
        aux_lp = LP(A,b,c,equality=True)
        bfs = BFS(x=x, B=B, obj_val=obj_val, optimal=optimal)

    # Interpret solution to the auxiliary LP
    if obj_val < -feas_tol:
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
        obj_val = float(np.dot(c.transpose(), x))
        optimal = False
        return BFS(x=x, B=B, obj_val=obj_val, optimal=optimal)


def _simplex_iteration(lp: LP,
                       bfs: BFS,
                       pivot_rule: str = 'bland',
                       feas_tol: float = 1e-7
                       ) -> BFS:
    """Execute a single iteration of the revised simplex method.

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
        bfs (BFS): Basic feasible solution.
        pivot_rule (str): Pivot rule to be used. 'bland' by default.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).

    Returns:
        BFS: Basic feasible solution after pivot.

    Raises:
        ValueError: Invalid pivot rule. Select from (list).
        ValueError: x should have shape (n+m,1) but was ().

    """
    pivot_rules = ['bland','min_index','dantzig','max_reduced_cost',
                   'greatest_ascent','manual', 'manual_select']
    if pivot_rule not in pivot_rules:
        raise ValueError('Invalid pivot rule. Select from ' + str(pivot_rules))

    n,m,A,b,c = lp.get_coefficients()
    x,B = bfs.x, bfs.B
    if not x.shape == (n, 1):
        raise ValueError('x should have shape (%d,1) but was %s'
                         % (n, str(x.shape)))

    B.sort()
    N = list(set(range(n)) - set(B))
    y = solve(A[:,B].transpose(), c[B,:])
    red_costs = c - np.matmul(y.transpose(),A).transpose()
    entering = {k: red_costs[k] for k in N if red_costs[k] > feas_tol}
    if len(entering) == 0:
        current_value = float(np.matmul(c.transpose(), x))
        return BFS(x=x, B=B, obj_val=current_value, optimal=True)
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
        return BFS(x=x, B=B, obj_val=current_value, optimal=False)


def _initial_solution(lp: LP,
                      x: Union[np.ndarray, List, Tuple] = None,
                      feas_tol: float = 1e-7
                      ) -> BFS:
    """Return an initial basic feasible solution for the linear program.

    If an x is provided, check if it is a basic feasible solution. If it is,
    use it as the initial solution. Otherwise, warn the user and proceed as
    though no x was provided. If no x is provided and the LP is in standard
    inequality form, check if the basis [n,n+m] forms a basic feasible
    solution. If it does, use that as the initial solution. Otherwise, use
    Phase I.

    Args:
        lp (LP): LP for which a basic feasible soluiton is given.
        x (Union[np.ndarray, List, Tuple]): Proposed inital solution.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).

    Returns:
        BFS: Initial basic feasible solution.
    """
    n,m,A,b,c = lp.get_coefficients()

    if x is not None:
        x = _vectorize(x).astype(float)
        # Compute slack variables if only decision variables provided.
        if not lp.equality and x.shape == (lp.n, 1):
            slacks = b - np.matmul(lp.A, x)
            x = np.vstack((x, slacks))

        if lp.equality:
            x = _validate(x, n, 'Initial solution')
        else:
            x = _validate(x, [lp.n, n], 'Initial solution')

        if (np.allclose(np.dot(A,x), b, atol=feas_tol)
                and all(x >= np.zeros((n,1)) - feas_tol)
                and len(np.nonzero(x)[0]) <= m):
            B = list(np.nonzero(x)[0])
            N = list(set(range(lp.n+lp.m)) - set(B))
            while len(B) < m:  # if initial solution is degenerate
                B.append(N.pop())
            obj_val = float(np.dot(c.transpose(), x))
            optimal = False
            return BFS(x=x, B=B, obj_val=obj_val, optimal=optimal)
        else:
            warnings.warn("Provided initial solution was not a basic feasible "
                          "solution; ignored.", UserWarning)
    if not lp.equality:
        try:
            B = list(range(lp.n,lp.n+lp.m))
            return lp.get_basic_feasible_sol(B, feas_tol=feas_tol)
        except InfeasibleBasicSolution:
            return _phase_one(lp)
        except InvalidBasis:
            return _phase_one(lp)

    return _phase_one(lp)


def simplex(lp: LP,
            pivot_rule: str = 'bland',
            initial_solution: Union[np.ndarray, List, Tuple] = None,
            iteration_limit: int = None,
            feas_tol: float = 1e-7
            ) -> Tuple[np.ndarray, List[int], float, bool, List[BFS]]:
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
        initial_solution (Union[np.ndarray, List, Tuple]): Initial bfs.
        iteration_limit (int): Simplex iteration limit. None by default.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).

    Return:
        Tuple:

        - x (np.ndarray): Current basic feasible solution.
        - B (List[int]): Corresponding bases of the current best BFS.
        - obj_val (float): The current objective value.
        - optimal (bool): True if x is optimal. False otherwise.
        - path (List[BFS]): Path of simplex.

    Raises:
        ValueError: Iteration limit must be strictly positive.
        ValueError: initial_solution should have shape (n,1) but was ().
    """
    if iteration_limit is not None and iteration_limit <= 0:
        raise ValueError('Iteration limit must be strictly positive.')

    n,m,A,b,c = lp.get_coefficients()
    bfs = _initial_solution(lp=lp, x=initial_solution, feas_tol=feas_tol)
    path = []

    # Print instructions if manual mode is chosen.
    if pivot_rule in ['manual', 'manual_select']:
        s = "INSTRUCTIONS \n\n"
        "At each iteration of simplex, choose one of the variables with a\n"
        "positive coefficent in the objective function. The list of indices\n"
        "for possible variables (also called entering variables) is given.\n"
        print(s)

    i = 0  # number of iterations
    while(not bfs.optimal):
        path.append(BFS(x=np.copy(bfs.x),
                        B=bfs.B.copy(),
                        obj_val=bfs.obj_val,
                        optimal=bfs.optimal))
        bfs = _simplex_iteration(lp=lp,
                                 bfs=bfs,
                                 pivot_rule=pivot_rule,
                                 feas_tol=feas_tol)
        i = i + 1
        if iteration_limit is not None and i >= iteration_limit:
            break
    x, B, obj_val, optimal = bfs
    Simplex = namedtuple('simplex', ['x', 'B', 'obj_val', 'optimal', 'path'])
    return Simplex(x=x, B=B, obj_val=obj_val, optimal=optimal, path=path)


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
    if best_bound is not None and best_bound >= value:
        return BnbIter(fathomed=True, incumbent=incumbent,
                       best_bound=best_bound, left_LP=None, right_LP=None)
    else:
        frac_comp = ~np.isclose(x, np.round(x), atol=int_feas_tol)[:lp.n]
        if np.sum(frac_comp) > 0:
            pos_i = np.nonzero(frac_comp)[0]  # list of indices to branch on
            if manual:
                i = int(input('Pick one of %s' % ([i + 1 for i in pos_i]))) - 1
                if i not in pos_i:
                    raise ValueError('This index can not be branched on.')
            else:
                i = pos_i[0]  # branch on first fractional component x_i
            frac_val = x[i,0]
            lb, ub = math.floor(frac_val), math.ceil(frac_val)

            def create_branch(lp, i, bound, branch):
                """Create branch off LP on fractional variable x_i."""
                s = {'left': 1, 'right': -1}[branch]
                n,m,A,b,c = lp.get_coefficients(equality=lp.equality)
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
