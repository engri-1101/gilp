import pytest
import numpy as np
from collections import deque
from gilp._geometry import (NoInteriorPoint, intersection,
                            polytope_vertices, polytope_facets,
                            halfspace_intersection, interior_point, order)


def test_intersection_bad_inputs():
    with pytest.raises(ValueError, match='.*vector must be length 3.*'):
        n = np.array([4,1,5,6])
        d = 6
        A = np.array([[1,1,3],[0,1,4],[1,-1,2],[1,0,6],[-2,1,1]])
        b = np.array([[6],[4],[2],[3],[0]])
        intersection(n,d,A,b)
    with pytest.raises(ValueError, match='.*must be of shape (n,3)*'):
        n = np.array([4,1,5])
        d = 6
        A = np.array([[1,1],[0,1],[1,-1],[1,0],[-2,1]])
        b = np.array([[6],[4],[2],[3],[0]])
        intersection(n,d,A,b)


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
    actual = intersection(n,d,A,b)
    assert all(np.allclose(x,y,atol=1e-7) for x,y in zip(actual, pts))


@pytest.mark.parametrize("A,b,pt,vertices,facets",[
    (np.array([[2, 1],
               [1, 1],
               [1, 0],
               [-1, -0],
               [-0, -1]]),
     np.array([[20],
               [16],
               [7],
               [0],
               [0]]),
     np.array([2.0,5.0]),
     np.array([[0.0, 0.0],
               [7.0, 0.0],
               [-0.0, 16.0],
               [7.0, 6.0],
               [4.0, 12.0]]),
     [np.array([[7, 6],
                [4, 12]]),
      np.array([[0, 16],
                [4, 12]]),
      np.array([[7, 0],
                [7, 6]]),
      np.array([[0, 0],
                [0, 16]]),
      np.array([[0, 0],
                [7, 0]])]),
    (np.array([[2, 1],
               [1, 1],
               [1, 0],
               [-1, -0],
               [-0, -1]]),
     np.array([[20],
               [16],
               [7],
               [0],
               [0]]),
     None,
     np.array([[0.0, 0.0],
               [7.0, 0.0],
               [-0.0, 16.0],
               [7.0, 6.0],
               [4.0, 12.0]]),
     [np.array([[7, 6],
                [4, 12]]),
      np.array([[0, 16],
                [4, 12]]),
      np.array([[7, 0],
                [7, 6]]),
      np.array([[0, 0],
                [0, 16]]),
      np.array([[0, 0],
                [7, 0]])])])
def test_halfspace_intersection(A,b,pt,vertices,facets):
    hs = halfspace_intersection(A,b,pt)
    v = np.array([list(x[:,0]) for x in hs.vertices])
    fct = [np.array([list(x[:,0]) for x in i]) for i in hs.facets_by_halfspace]
    assert (np.isclose(v,vertices,atol=1e-7)).all()
    assert (np.isclose(fct,facets,atol=1e-7)).all()


def test_no_intersection():
    with pytest.raises(NoInteriorPoint):
        A = np.array([[1,0,0],
                      [0,1,0],
                      [0,0,1],
                      [-1,0,0],
                      [0,-1,0],
                      [0,0,-1],
                      [-1,-1,-1]])
        b = np.array([[1],[1],[1],[0],[0],[0],[-4]])
        interior_point(A,b)


@pytest.mark.parametrize("A,b,x",[
    (np.array([[1,0,0],
               [0,1,0],
               [0,0,1],
               [-1,0,0],
               [0,-1,0],
               [0,0,-1]]),
     np.array([[1],[1],[1],[0],[0],[0]]),
     np.array([0.5,0.5,0.5])),
    (np.array([[-1, 0.],
               [0., -1.],
               [2., 1.],
               [-0.5, 1.]]),
     np.array([[0],[0],[4],[2]]),
     np.array([0.76393202,0.76393202]))])
def test_interior_point(A,b,x):
    assert all(np.isclose(interior_point(A,b),x,atol=1e-7))


def test_order_bad_inputs():
    with pytest.raises(ValueError, match='.*must be in vector form.*'):
        order([np.array([[1,2,3]])])
    with pytest.raises(ValueError, match='.*must be 2 or 3 dimensional'):
        order([np.array([[1],[2],[3],[4]])])


@pytest.mark.parametrize("x_list,pts",[
    ([np.array([[3],[3]]),
      np.array([[2],[4]]),
      np.array([[2],[0]]),
      np.array([[0],[0]]),
      np.array([[3],[1]])],
     np.array([[0,2,3,3,2,0],
               [0,0,1,3,4,0]])),
    ([np.array([[3],[3]]),
      np.array([[2],[4]])],
     np.array([[3,2],
               [3,4]])),
    ([np.array([[1],[4]])],
     [[1],[4]])])
def test_order_2d(x_list,pts):
    test = np.array(order(x_list))
    length = len(x_list)

    # Remove duplicate point appended to end of list
    if length > 2:
        test = test[:,:-1]

    # Check to make sure at least one transformation matches
    transforms = []
    indices = deque(range(len(test[0])))
    for i in range(len(test[0])):
        transforms.append(test[:,list(indices)])
        indices.rotate(1)
    indices.reverse()
    for i in range(len(test[0])):
        transforms.append(test[:,list(indices)])
        indices.rotate(1)

    # Append duplicate point to end of list
    if length > 2:
        transforms = [trans[:,list(range(length))+[0]] for trans in transforms]

    assert any([np.array_equal(trans, pts) for trans in transforms])


@pytest.mark.parametrize("x_list,pts",[
    ([np.array([[-1.7],[0],[0.59]]),
      np.array([[0],[-0.59],[1.7]]),
      np.array([[0],[0.59],[1.7]]),
      np.array([[-1],[-1],[1]]),
      np.array([[-1],[1],[1]])],
     np.array([[-1.0, -1.7, -1.0, 0.0, 0.0, -1.0],
               [1.0, 0.0, -1.0, -0.59, 0.59, 1.0],
               [1.0, 0.59, 1.0, 1.7, 1.7, 1.0]])),
    ([np.array([[0],[1],[0]]),
      np.array([[-0.5],[0],[0.5]]),
      np.array([[0.5],[0],[0.5]])],
     np.array([[0.0, -0.5, 0.5, 0.0],
               [1.0, 0.0, 0.0, 1.0],
               [0.0, 0.5, 0.5, 0.0]]))])
def test_order_3d(x_list,pts):
    test = np.array(order(x_list))
    length = len(x_list)

    # Remove duplicate point appended to end of list
    if length > 2:
        test = test[:,:-1]

    # Check to make sure at least one transformation matches
    transforms = []
    indices = deque(range(len(test[0])))
    for i in range(len(test[0])):
        transforms.append(test[:,list(indices)])
        indices.rotate(1)
    indices.reverse()
    for i in range(len(test[0])):
        transforms.append(test[:,list(indices)])
        indices.rotate(1)

    # Append duplicate point to end of list
    if length > 2:
        transforms = [trans[:,list(range(length))+[0]] for trans in transforms]

    assert any([np.array_equal(trans, pts) for trans in transforms])


@pytest.mark.parametrize("A,b,pt,expected",[
    (np.array([[2, 1],
               [1, 1],
               [1, 0],
               [-1, -0],
               [-0, -1]]),
     np.array([[20],
               [16],
               [7],
               [0],
               [0]]),
     np.array([2.0,5.0]),
     np.array([[0.0, 0.0],
               [7.0, 0.0],
               [-0.0, 16.0],
               [7.0, 6.0],
               [4.0, 12.0]])),
    (np.array([[1., 0., 0.],
               [1., 0., 1.],
               [0., 0., 1.],
               [0., 1., 1.],
               [-1., -0., -0.],
               [-0., -1., -0.],
               [-0., -0., -1.]]),
     np.array([[6.],
               [8.],
               [5.],
               [8.],
               [0.],
               [0.],
               [0.]]),
     None,
     np.array([[0.0, 0.0, 0.0],
               [-0.0, 8.0, -0.0],
               [6.0, 0.0, 2.0],
               [6.0, 0.0, 0.0],
               [6.0, 6.0, 2.0],
               [6.0, 8.0, -0.0],
               [3.0, 0.0, 5.0],
               [0.0, 0.0, 5.0],
               [3.0, 3.0, 5.0],
               [0.0, 3.0, 5.0]])),
    (np.array([[1,0],
               [0,1],
               [-1,0],
               [0,-1]]),
     np.array([[0],[1],[0],[0]]),
     None,
     np.array([[0.0, -0.0],
              [0.0, 1.0]]))])
def test_polytope_vertices(A,b,pt,expected):
    result = polytope_vertices(A,b,pt)
    result = np.array([list(x[:,0]) for x in result])
    assert (result == expected).all()


@pytest.mark.parametrize("A,b,vertices,expected",[
    (np.array([[-1,0,0],
              [0,-1,0],
              [0,0,-1],
              [1,1,1]]),
     np.array([[0],[0],[0],[1]]),
     None,
     [np.array([[0, 0, 1],
                [0, 1, 0],
                [0, 0, 0]]),
      np.array([[0, 0, 1],
                [0, 0, 0],
                [1, 0, 0]]),
      np.array([[0, 1, 0],
                [0, 0, 0],
                [1, 0, 0]]),
      np.array([[0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]])]),
    (np.array([[-1,0,0],
              [0,-1,0],
              [0,0,-1],
              [1,1,1]]),
     np.array([[0],[0],[0],[1]]),
     [np.array([[0], [0], [1]]),
      np.array([[0], [1], [0]]),
      np.array([[0], [0], [0]]),
      np.array([[1], [0], [0]])],
     [np.array([[0, 0, 1],
                [0, 1, 0],
                [0, 0, 0]]),
      np.array([[0, 0, 1],
                [0, 0, 0],
                [1, 0, 0]]),
      np.array([[0, 1, 0],
                [0, 0, 0],
                [1, 0, 0]]),
      np.array([[0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]])])])
def test_polytope_facets(A,b,vertices,expected):
    res = polytope_facets(A,b,vertices)
    res = [np.array([list(x[:,0]) for x in i]) for i in res]
    assert all([(res[i] == expected[i]).all() for i in range(len(expected))])
