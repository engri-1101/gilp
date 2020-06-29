import numpy as np

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
        b (np.ndarray): A vector of coefficients of length m
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