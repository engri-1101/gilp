import pytest
import numpy as np
from gilp.simplex import LP

@pytest.fixture
def klee_minty_3d_lp():
    return LP(np.array([[1,0,0],[4,1,0],[8,4,1]]),
              np.array([[5],[25],[125]]),
              np.array([[4],[2],[1]]))


@pytest.fixture
def unbounded_lp():
    return LP(np.array([[-1,1],[1,-1]]),
              np.array([[1],[1]]),
              np.array([[1],[0]]))


@pytest.fixture
def degenerate_lp():
    return LP(np.array([[1,1],[0,1],[1,-1],[1,0],[-2,1]]),
              np.array([[6],[4],[2],[3],[0]]),
              np.array([[1],[2]]))