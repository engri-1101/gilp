import numpy as np
from scipy.linalg import solve
import itertools
from typing import List, Tuple

"""Provides an implementation of the revised simplex method.
    
Classes:
    LP: Maintains the coefficents and size of a linear program (LP)
    
Functions:
    invertible: Return true if the matrix A is invertible.
    simplex_iter: Run a single iteration of the revised simplex method.
    simplex: Run the revised simplex method.
"""

class LP:
    """Maintains the coefficents and size of a linear program (LP).
    
    This LP class maintains the coefficents of a linear program in 
    both standard inequality and equality form. A is an m*n matrix 
    describing the linear combination of variables x making up the 
    LHS of each constraint. b is a vector of length m making up 
    the RHS of each constraint. Lastly, c is a vector of length n 
    describing the objective function to be maximized. Both the n 
    decision variables x and m slack variables must be nonnegative. 
    Lastly, the coefficents of b are assumed to be nonnegative.
    
    inequality        equality              
    max c^Tx          max c^Tx              
    s.t Ax <= b       s.t Ax + Is == b
         x >= 0               x,s >= 0       
         
    Attributes:
        n (int): number of decision variables (excludes slacks)
        m (int): number of constraints (excluding nonnegativity ones)
        A (np.ndarray): An m*n matrix of coefficients
        A_I (np.ndarray): An m*(n+m) matrix of coefficients: [A I]
        b (np.ndarray): A nonnegative vector of coefficients of length m
        c (np.ndarray): A vector of coefficients of length n
        c_0 (np.ndarray): A vector of coefficients of length n+m: [c^T 0^T]^T
    """
    
    def __init__(self, A:np.ndarray, b:np.ndarray, c:np.ndarray):
        """Initializes LP.
        
        Creates an instance of LP using the given coefficents. Note the
        coefficents must correspond to an LP in standard INEQUALITY form:
        
        max c^Tx
        s.t Ax <= b
             x >= 0
        
        Args:
            A (np.ndarray): An m*n matrix
            b (np.ndarray): A nonnegative vector of length m
            c (np.ndarray): A vector of length n
            
        Raises:
            ValueError: The shape of the b vector is not (m,1)
            ValueError: The vector b is not nonnegative
            ValueError: The shape of the c vector is not (n,1)
        """
        self.m = len(A)
        self.n = len(A[0])
        if not b.shape == (self.m,1):
            raise ValueError('The shape of the b vector is not (m,1)')
        if not all(b >= np.zeros((self.m,1))):
            raise ValueError('The vector b is not nonnegative')
        if not c.shape == (self.n,1):
            raise ValueError('The shape of the c vector is not (n,1)')
        self.A = A 
        self.A_I = np.hstack((self.A,np.identity(self.m)))
        self.b = b
        self.c = c
        self.c_0 = np.vstack((self.c,np.zeros((self.m,1))))
    
    def get_inequality_form(self):
        '''Returns n,m,A,b,c describing this LP in standard inequality form'''
        return self.n, self.m, self.A, self.b, self.c
    
    def get_equality_form(self):
        '''Returns n,m,A_I,b,c_0 describing this LP in standard equality form'''
        return self.n, self.m, self.A_I, self.b, self.c_0

    def get_basic_feasible_sol(self,B:List[int]) -> np.ndarray:
        """If B is a basis for the LP, return the basic solution if feasible
        
        Be definition, B is a basis iff A_B is invertible. The corresponding
        basic solution x satisfies A_Bx = b. Be definition, x is a basic
        feasible solution iff x satisfies both Ax = b and x > 0.

        Args:
            B (List[int]): A basis for A (A_B must be invertible)

        Returns:
            np.ndarray: The basic feasible solution for basis B (None if infeasible)
        """
        n,m,A,b,c = self.get_equality_form()
        if invertible(A[:,B]):
            x_B = np.zeros((n+m,1))
            x_B[B,:] = np.round(solve(A[:,B],b),7)
            if all(x_B >= np.zeros((n+m,1))): return x_B
        return None

    def get_basic_feasible_solns(self) -> Tuple[List[np.ndarray],List[List[int]],List[float]]:
        """Return all basic feasible solutions and their basis and value for this LP.
            
        Returns:
            (List[np.ndarray]): A list of basic feasible solutions
            (List[List[int]]): The corresponding list of bases
            (List[float]): The corresponding objective values
        """
        bfs, bases, values = [], [], []
        for B in itertools.combinations(range(self.n+self.m),self.m):
            x_B = self.get_basic_feasible_sol(B)
            if x_B is not None:
                bfs.append(x_B) 
                bases.append(B) 
                values.append(np.round(np.dot(x_B[0:self.n,0],self.c)[0],7))
        return (bfs, bases, values)

    def get_tableau(self, B:List[int]) -> np.ndarray:
        """Get the tableau of this LP for the given basis B.
        
        The returned tableau has the following form:
        
        z - (c_N^T - y^TA_N)x_N = y^Tb  where   y^T = c_B^TA_B^(-1)
        x_B + A_B^(-1)A_Nx_N = x_B^*    where   x_B^* = A_B^(-1)b
        
        | z | x_1 | x_2 | ... | = | RHS | <- Header not included in array
        ---------------------------------
        | 1 |  -  |  -  | ... | = |  -  |
        | 0 |  -  |  -  | ... | = |  -  |
                    ...
        | 0 |  -  |  -  | ... | = |  -  |
        
        Args:
            B (List[int]): The basis the tableau corresponds to
            
        Returns:
            (np.ndarray): A numpy array representing the tableau
            
        Raises:
            ValueError: Invalid basis. A_B is not invertible.
        """

        n,m,A,b,c = self.get_equality_form()
        
        if not invertible(A[:,B]):
            raise ValueError('Invalid basis. A_B is not invertible.')
            
        N = list(set(range(n+m)) - set(B))
        A_B_inv = np.linalg.inv(A[:,B])
        yT = np.dot(c[B,:].transpose(),A_B_inv)

        T = np.zeros((m+1,n+m+2))
        T[0,0] = 1
        T[0,1:n+m+1][N] = c[N,:].transpose() - np.dot(yT,A[:,N])
        T[0,n+m+1] = np.dot(yT,b)   
        T[1:,1:n+m+1][:,N] = np.dot(A_B_inv,A[:,N])
        T[1:,1:n+m+1][:,B] = np.identity(len(B))
        T[1:,n+m+1] = np.dot(A_B_inv,b)[:,0]
        
        return np.round(T,7)

def invertible(A:np.ndarray) -> bool:
    """Return true if the matrix A is invertible.
    
    Args:
        A (np.ndarray): An m*n matrix
    
    By definition, a matrix A is invertible iff n = m and A has rank n
    """
    return len(A) == len(A[0]) and np.linalg.matrix_rank(A) == len(A)

def simplex_iter(lp:LP, x:np.ndarray, B:List[int], N:List[int], pivot_rule:str='bland'):
    """Run a single iteration of the revised simplex method.
    
    Starting from the basic feasible solution x with corresponding basis B and 
    non-basic variables N, do one iteration of the revised simplex method. Use
    the given pivot rule. Implemented pivot rules include:
    
    'bland' or 'min_index'
        entering variable: minimum index
        leaving variable: minimum (positive) ratio (minimum index to tie break)
    
    'dantzig' or 'max_reduced_cost'
        entering variable: most positive reduced cost
        leaving variable: minimum (positive) ratio (minimum index to tie break)
    
    'greatest_ascent'
        entering variable: most positive product of minimum ratio and reduced cost
        leaving variable: minimum (positive) ratio (minimum index to tie break)
        
    'manual_select'
        entering variable: user selects among possible entering indices
        leaving variable: minimum (positive) ratio (minimum index to tie break)
        
    Return the new bfs, basis, non-basic variables, and an idication of optimality.
    
    Args:
        lp (LP): The LP on which the simplex iteration is being done
        x (np.ndarray): The initial basic feasible solution
        B (List(int)): The basis corresponding to the bfs x
        N (List(int)): The non-basic variables
        pivot_rule (str): The pivot rule to be used
        
    Returns:
        x (np.ndarray): The new basic feasible solution
        value (float): The value of the new basic feasible solution
        B (List(int)): The basis corresponding to the new bfs x
        N (List(int)): The new non-basic variables
        opt (bool): An idication of optimality. True if optimal.
        
    Raises:
        ValueError: Invalid pivot rule. See simplex_iter? for possible options.
    """
    if pivot_rule not in ['bland','min_index','dantzig','max_reduced_cost',
                          'greatest_ascent','manual_select']:
        raise ValueError('Invalid pivot rule. See simplex_iter? for possible options.')

    n,m,A,b,c = lp.get_equality_form()
    y = solve(A[:,B].transpose(), c[B,:]) # dual cost vector
    red_costs = c - np.dot(y.transpose(),A).transpose() # reduced cost vector
    pos_nonbasic_red = {k : red_costs[k] for k in N if red_costs[k] > 0} # possible entering
    if len(pos_nonbasic_red) == 0:
        # no positive reduced costs -> optimal
        current_value = np.dot(np.dot(c[B,:].transpose(),np.linalg.inv(A[:,B])),b)[0][0]
        return x,current_value,B,N,True
    else:
        if pivot_rule == 'greatest_ascent':
            eligible = {}
            for k in pos_nonbasic_red:
                d = np.zeros((1,n+m))
                d[:,B] = solve(A[:,B], A[:,k])
                ratios = {i : x[i]/d[0][i] for i in B if d[0][i] > 0}
                if len(ratios) == 0:
                    raise ValueError('The LP is unbounded')
                t = min(ratios.values())
                r_pos = [r for r in ratios if ratios[r] == t]
                r =  min(r_pos)
                t = ratios[r]
                eligible[(t*red_costs[k])[0]] = [k,r,t,d]
            k,r,t,d = eligible[max(eligible.keys())]
        else: 
            entering = None
            if pivot_rule == 'manual_select':
                entering = int(input('Pick one of '+str(list(pos_nonbasic_red.keys()))))
            k = {'bland' : min(pos_nonbasic_red.keys()),
                 'min_index' : min(pos_nonbasic_red.keys()),
                 'dantzig' : max(pos_nonbasic_red, key=pos_nonbasic_red.get),
                 'max_reduced_cost' : max(pos_nonbasic_red, key=pos_nonbasic_red.get),
                 'manual_select' : entering}[pivot_rule]
            d = np.zeros((1,n+m))
            d[:,B] = solve(A[:,B], A[:,k])
            ratios = {i : x[i]/d[0][i] for i in B if d[0][i] > 0}
            if len(ratios) == 0:
                raise ValueError('The LP is unbounded')
            t = min(ratios.values())
            r_pos = [r for r in ratios if ratios[r] == t]
            r =  min(r_pos)
            t = ratios[r]
        # update
        x[k] = t
        x[B,:] = x[B,:] - t*(d[:,B].transpose())
        B.append(k); B.remove(r)
        N.append(r); N.remove(k)
        current_value = np.dot(np.dot(c[B,:].transpose(),np.linalg.inv(A[:,B])),b)[0][0]
        return x,current_value,B,N,False

# TODO: fix initial solution being optimal bug
def simplex(lp:LP, pivot_rule:str='bland',
            init_sol:np.ndarray=None,iter_lim:int=None) -> Tuple[bool,float,List[np.ndarray],List[List[int]]]:
    """Run the revised simplex method on the given LP.
    
    Run the revised simplex method on the given LP. If an initial solution is
    given, check if it is a basic feasible solution. If so, start simplex from
    this inital bfs. Otherwise, ignore it. If an iteration limit is given, 
    terminate if the specified limit is reached and output the current solution.
    Indicate that the solution may not be optimal. At each simplex iteration,
    use the given pivot rule. Implemented pivot rules include:
    
    'bland' or 'min_index'
        entering variable: minimum index
        leaving variable: minimum (positive) ratio (minimum index to tie break)
    
    'dantzig' or 'max_reduced_cost'
        entering variable: most positive reduced cost
        leaving variable: minimum (positive) ratio (minimum index to tie break)
    
    'greatest_ascent'
        entering variable: most positive product of minimum ratio and reduced cost
        leaving variable: minimum (positive) ratio (minimum index to tie break)
        
    'manual_select'
        entering variable: user selects among possible entering indices
        leaving variable: minimum (positive) ratio (minimum index to tie break)
        
    Return a list of basic feasible solutions, their bases, and indication of optimality
    
    Args:
        lp (LP): The LP on which to run simplex
        pivot_rule (str): The pivot rule to be used at each simplex iteration
        init_sol (np.ndarray): An n length vector 
        iter_lim (int): The maximum number of simplex iterations to be run
    
    Return:
        opt (bool): True if path[-1] is known to be optimal. False otherwise.
        value (float): The optimal value if opt is True. Otherwise, current value.
        path (List[np,ndarray]): The list of basic feasible solutions (n+m length)
                                 vectors that simplex traces.
        bases (List[List[int]]): The corresponding list of bases that simplex traces
    
    Raises:
        ValueError: Invalid pivot rule. See simplex_iter? for possible options.
        ValueError: Iteration limit must be strictly positive
    """
    
    if pivot_rule not in ['bland','min_index','dantzig','max_reduced_cost',
                          'greatest_ascent','manual_select']:
        raise ValueError('Invalid pivot rule. See simplex_iter? for possible options.')
    if iter_lim is not None and iter_lim <= 0:
        raise ValueError('Iteration limit must be strictly positive.')
    
    n,m,A,b,c = lp.get_equality_form()
    
    # select intital basis and feasible solution
    B = list(range(n,n+m))
    N = list(range(0,n)) 
    x = np.zeros((n+m,1))
    x[B,:] = b

    if init_sol is not None:
        x_B = np.zeros((n+m,1))
        x_B[list(range(n)),:] = init_sol
        for i in range(m):
            x_B[i+2] = b[i]-np.dot(A[i,list(range(n))],init_sol)
        x_B = np.round(x_B,7)
        if (all(np.round(np.dot(A,x_B)) == b) and 
            all(x_B >= np.zeros((n+m,1))) and 
            len(np.nonzero(x_B)[0]) <= n+m-2):
            x = x_B
            B = list(np.nonzero(x_B)[0])
            N = list(set(range(n+m)) - set(B))
            while len(B) < n+m-2:
                B.append(N.pop())
        else:
            print('Initial solution ignored.')
            
    path = [np.copy(x)]
    bases = [np.copy(B)]
                                    
    optimal = False 
    current_value = np.dot(np.dot(c[B,:].transpose(),np.linalg.inv(A[:,B])),b)[0]
    
    if iter_lim is not None: lim = iter_lim
    while(not optimal):
        x,value,B,N,opt = simplex_iter(lp,x,B,N,pivot_rule)
        current_value = value
        # TODO: make a decison about how this should be implemented
        if opt == True:
            optimal = True
        else:
            path.append(np.copy(x))
            bases.append(np.copy(B))
        if iter_lim is not None: lim = lim - 1
        if iter_lim is not None and lim == 0: break
            
    return optimal, current_value, path, bases