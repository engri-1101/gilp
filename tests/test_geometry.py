import pytest
import mock
import numpy as np
import gilp.geometry as geo

def test_intersection_bad_inputs():
    with pytest.raises(ValueError, match='.*vector must be length 3.*'):
        n = np.array([4,1,5,6])
        d = 6
        A = np.array([[1,1,3],[0,1,4],[1,-1,2],[1,0,6],[-2,1,1]])
        b = np.array([[6],[4],[2],[3],[0]])
        geo.intersection(n,d,A,b)
    with pytest.raises(ValueError, match='.*must be of shape (n,3)*'):
        n = np.array([4,1,5])
        d = 6
        A = np.array([[1,1],[0,1],[1,-1],[1,0],[-2,1]])
        b = np.array([[6],[4],[2],[3],[0]])
        geo.intersection(n,d,A,b)


@pytest.mark.parametrize("n,d,pts",[
    (np.array([0,0,1]), 0.5,
     [np.array([[1],[1],[0.5]]),
      np.array([[1],[0],[0.5]]),
      np.array([[0],[1],[0.5]]),
      np.array([[0],[0],[0.5]])]),
    (np.array([2,0,1]), 1.5,
     [np.array([[0.25],[1],[1]]),
      np.array([[0.75],[1],[0]]),
      np.array([[0.25],[0],[1]]),
      np.array([[0.75],[0],[0]])])])
def test_intersection_3d(n,d,pts):
    A = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1],
                  [-1,0,0],
                  [0,-1,0],
                  [0,0,-1]])
    b = np.array([[1],[1],[1],[0],[0],[0]])
    actual = geo.intersection(n,d,A,b)
    print(actual)
    assert all(np.allclose(x,y,atol=1e-7) for x,y in zip(actual, pts))