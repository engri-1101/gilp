import numpy as np
from .simplex import LP


ALL_INTEGER_2D_LP = LP(np.array([[2,1],[1,1],[1,0]]),
                       np.array([[20],[16],[7]]),
                       np.array([[5],[3]]))
"""A 2D LP where all basic feasible solutions are integral and have integral
tableaus."""


LIMITING_CONSTRAINT_2D_LP = LP(np.array([[1,0],[0,1],[2,1],[3,2]]),
                               np.array([[4],[6],[9],[15]]),
                               np.array([[5],[3]]))
"""A 2D LP demonstrating how the most limiting constraint determines the
leaving variable."""


DEGENERATE_FIN_2D_LP = LP(np.array([[0,1],[1,-1],[1,0],[-2,1]]),
                          np.array([[4],[2],[3],[0]]),
                          np.array([[1],[2]]))
"""A 2D LP where the (default) intial feasible solution is degenerate."""


KLEE_MINTY_2D_LP = LP(np.array([[1,0],[4,1]]),
                      np.array([[5],[25]]),
                      np.array([[2],[1]]))
'''A 2D LP where the 'dantzig' pivot rule results in a simplex path through
every bfs. Klee, Victor; Minty, George J. (1972). "How good is the simplex
algorithm?"'''


ALL_INTEGER_3D_LP = LP(np.array([[1,0,0],[1,0,1],[0,0,1],[0,1,1]]),
                       np.array([[6],[8],[5],[8]]),
                       np.array([[1],[2],[4]]))
"""A 3D LP where all basic feasible solutions are integral and have integral
tableaus."""


MULTIPLE_OPTIMAL_3D_LP = LP(np.array([[0,1,0],[1,0,1],[1,0,0],[0,1,1]]),
                            np.array([[2],[3],[2],[4]]),
                            np.array([[1],[1],[1]]))
"""A 3D LP demonstrating the geometry of multiple optimal solutions."""


SQUARE_PYRAMID_3D_LP = LP(np.array([[0,-1,1],[-1,0,1],[0,1,1],[1,0,1]]),
                          np.array([[0],[0],[4],[4]]),
                          np.array([[0],[0],[1]]))
"""A 3D LP which is highly degenerate. It demonstrates that degeneracy can not
be solved by removing a seemingly redundant constraint--doing so can alter the
feasible region."""


KLEE_MINTY_3D_LP = LP(np.array([[1,0,0],[4,1,0],[8,4,1]]),
                      np.array([[5],[25],[125]]),
                      np.array([[4],[2],[1]]))
'''A 3D LP where the 'dantzig' pivot rule results in a simplex path through
every bfs. Klee, Victor; Minty, George J. (1972). "How good is the simplex
algorithm?"'''
