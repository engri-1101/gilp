import numpy as np
import itertools
from scipy.optimize import linprog
from pyhull.halfspace import Halfspace, HalfspaceIntersection
from typing import List

"""Contains multiple functions dealing with computational geometry.

Functions:
    intersection: Return the intersection of the plane and convex ployhedron.
    halfspace_intersection: Return the intersection of the given halfspaces.
    interior_point: Return an interior point of the halfspace intersection.
    order: Return the ordered vertices of a non self-intersecting polygon.
"""


def intersection(n: np.ndarray,
                 d: float,
                 A: np.ndarray,
                 b: np.ndarray) -> List[np.ndarray]:
    """Return the intersection of the plane and convex ployhedron.

    Returns the intersection of the plane nx = d and convex ployhedron defined
    by the linear inequalities Ax <= b.

    Args:
        n (np.ndarray): normal vector of the plane.
        d (np.ndarray): offset (or distance) vector of the plane.
        A (np.ndarray): LHS coefficents defining the linear inequalities.
        b (np.ndarray): RHS coefficents defining the linear inequalities.

    Returns:
        List[np.ndarray]: list of vertices defining the intersection (if any)

    Raises:
        ValueError: Normal vector must be length 3.
        ValueError: Matrix A must be of shape (n,3).
    """
    if len(n) != 3:
        raise ValueError('Normal vector must be length 3.')
    if len(A[0]) != 3:
        raise ValueError('Matrix A must be of shape (n,3).')

    pts = []
    n_d = np.hstack((n,d))
    A_b = np.hstack((A,b))
    for indices in itertools.combinations(range(len(A)),2):
        R_c = np.vstack((n,A[list(indices)]))
        R_d = np.vstack((n_d,A_b[list(indices)]))
        if np.linalg.matrix_rank(R_c) == 3 and np.linalg.matrix_rank(R_d) == 3:
            det = np.linalg.det(R_c)
            if det != 0:
                x_1 = np.linalg.det(R_d[:,[3,1,2]])/det
                x_2 = np.linalg.det(R_d[:,[0,3,2]])/det
                x_3 = np.linalg.det(R_d[:,[0,1,3]])/det
                x = np.array([[x_1],[x_2],[x_3]])
                if all(np.matmul(A,x) <= b + 1e-10):
                    pts.append(x)
    return pts


def halfspace_intersection(A: np.ndarray,
                           b: np.ndarray,
                           interior_pt: np.ndarray = None
                           ) -> HalfspaceIntersection:
    """Return the intersection of the given halfspaces.

    Return the halfspace intersection of the halfspaces defined by the linear
    inequalities Ax <= b. If an interior point of the halfspace intersection is
    not given, one is computed using linear programming.

    Args:
        A (np.ndarray): LHS coefficents defining the linear inequalities.
        b (np.ndarray): RHS coefficents defining the linear inequalities.
        interior_pt (np.ndarray): interior point of the halfspace intersection.

    Returns:
        HalfspaceIntersection: object representing the halfspace intersection.
    """
    halfspaces = []
    for i in range(len(A)):
        halfspaces.append(Halfspace(A[i],float(-b[i])))
    if interior_pt is None:
        interior_pt = interior_point(A,b)
    return HalfspaceIntersection(halfspaces, interior_pt)


def interior_point(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return an interior point of the halfspace intersection.

    Given a list of halfspaces in the form of linear inequalities Ax <= b,
    return an interior point of the halfspace intersection. Linear programming
    is used to find the chebyshev center of the halfspace intersection.

    Args:
        A (np.ndarray): LHS coefficents defining the linear inequalities.
        b (np.ndarray): RHS coefficents defining the linear inequalities.

    Returns:
        np.ndarray: An interior point of the halfspace intersection.

    Raises:
        ValueError: The halfspace intersection is empty.
    """
    M = np.hstack((A,-b))
    norm = np.reshape(np.linalg.norm(M[:, :-1], axis=1),(M.shape[0], 1))
    obj_func = np.zeros((M.shape[1],))
    obj_func[-1] = -1
    x = linprog(obj_func,
                A_ub=np.hstack((M[:, :-1], norm)),
                b_ub=-M[:, -1:],
                bounds=(None, None)).x
    if x[-1] <= 0:
        raise ValueError('The halfspace intersection is empty.')
    return x[:-1]


def order(x_list: List[np.ndarray]) -> List[List[float]]:
    """Return the ordered vertices of a non self-intersecting polygon.

    Given a set of vertices, order them such that they form a non
    self-intersecting polygon.

    Args:
        x_list (List[np.ndarray]): list of vertices to order

    Returns:
        List[List[float]]: list of components for non self-intersecting polygon

    Raises:
        ValueError: Points must be represented by column vectors.
        ValueError: Points must be 2 or 3 dimensional.
    """
    n,m = x_list[0].shape
    if not m == 1:
        raise ValueError('Points must be represented by column vectors.')
    if n not in [2,3]:
        raise ValueError('Points must be 2 or 3 dimensional.')

    pts = [tuple(x[0:n,0]) for x in x_list]
    pts = list(set(pts))  # unique points
    pts = np.array(pts)
    p = len(pts)  # number of unique points

    def sort_pts(pts_array):
        """Sort a set of 2d points to form a non-self-intersecting polygon."""
        x = pts_array[:,0]
        y = pts_array[:,1]
        x_center = np.mean(x)
        y_center = np.mean(y)
        return list(np.argsort(np.arctan2(y-y_center, x-x_center)))

    if p > 2:
        if n == 2:
            indices = sort_pts(pts)
        if n == 3:
            b_1 = pts[1] - pts[0]
            b_2 = pts[2] - pts[0]
            b_3 = np.cross(b_1, b_2)
            T = np.linalg.inv(np.array([b_1, b_2, b_3]).transpose())
            pts_T = [list(np.matmul(T,pts[i,:,None])[:2,0]) for i in range(p)]
            pts_T = np.array(pts_T)
            indices = sort_pts(pts_T)
        pts = pts[indices + [indices[0]]]
    components = list(zip(*pts))
    components = [list(component) for component in components]
    return components