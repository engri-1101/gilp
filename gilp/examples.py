import numpy as np
from .simplex import LP


# Robbins, Henry (2020)
# All basic feasible solutions are integral and have integral tableaus.
ALL_INTEGER_2D_LP = LP(np.array([[2,1],[1,1],[1,0]]),
                       np.array([[20],[16],[7]]),
                       np.array([[5],[3]]))
# Robbins, Henry (2020)
# Demonstrates how the limiting constraint determines the leaving variable.
LIMITING_CONSTRAINT_2D_LP = LP(np.array([[1,0],[0,1],[2,1],[3,2]]),
                               np.array([[4],[6],[9],[15]]),
                               np.array([[2],[1]]))
# Robbins, Henry (2020)
# The (default) intial feasible solution is degenerate.
DEGENERATE_FIN_2D_LP = LP(np.array([[0,1],[1,-1],[1,0],[-2,1]]),
                          np.array([[4],[2],[3],[0]]),
                          np.array([[1],[2]]))
# Klee, Victor; Minty, George J. (1972). "How good is the simplex algorithm?"
# The 'dantzig' pivot rule results in simplex path through every bfs.
KLEE_MINTY_2D_LP = LP(np.array([[1,0],[4,1]]),
                      np.array([[5],[25]]),
                      np.array([[2],[1]]))
# Robbins, Henry (2020)
# Too close to other example
TEST = LP(np.array([[1,0,0],[0,1,0],[1,1,0],[1,0,1]]),
          np.array([[4],[4],[6],[10]]),
          np.array([[2],[2],[1]]))
# Robbins, Henry (2020)
# Highly degenerate. Demonstrates that degeneracy can not be solved by removing
# a seemingly redundant constraint--doing so can alter the feasible region.
SQUARE_PYRAMID_3D_LP = LP(np.array([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1]]),
                          np.array([[4],[0],[4],[0]]),
                          np.array([[0],[0],[1]]))
# Klee, Victor; Minty, George J. (1972). "How good is the simplex algorithm?"
# The 'dantzig' pivot rule results in simplex path through every bfs.
KLEE_MINTY_3D_LP = LP(np.array([[1,0,0],[4,1,0],[8,4,1]]),
                      np.array([[5],[25],[125]]),
                      np.array([[4],[2],[1]]))
