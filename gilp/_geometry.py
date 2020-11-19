"""Computational geometry functions.

This module contains various computational geometry functions related to linear
programming.
"""

__author__ = 'Henry Robbins'
__all__ = ['intersection', 'polytope_vertices', 'polytope_facets',
           'halfspace_intersection', 'interior_point', 'order']

from collections import namedtuple
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
from typing import List, Tuple


class NoInteriorPoint(Exception):
    """Raised when the intersection of halfspaces has no interior point."""
    pass


def intersection(n: np.ndarray,
                 d: float,
                 A: np.ndarray,
                 b: np.ndarray) -> List[np.ndarray]:
    """Return the intersection of the plane and convex ployhedron.

    Returns a list of points which define the intersection between the plane
    nx = d and the convex ployhedron defined by linear inequalities Ax <= b.

    Args:
        n (np.ndarray): Normal vector of the plane.
        d (np.ndarray): Offset (or distance) vector of the plane.
        A (np.ndarray): LHS coefficents defining the linear inequalities.
        b (np.ndarray): RHS constants defining the linear inequalities.

    Returns:
        List[np.ndarray]: List of vertices defining the intersection (if any).

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
                    pts.append(np.round(x, 10))
    return pts


def polytope_vertices(A: np.ndarray,
                      b: np.ndarray,
                      interior_pt: np.ndarray = None) -> np.ndarray:
    """Return the vertices of the halfspace intersection Ax <= b.

    Equivalently, return the V-representation of some polytope given the
    H-representation. Provide an interior point to improve computation time.

    Args:
        A (np.ndarray): LHS coefficents of the halfspaces.
        b (np.ndarray): RHS coefficents of the halfspaces.
        interior_pt (np.ndarray): Interior point of the halfspace intersection.

    Returns:
        np.ndarray: Vertices of the halfspace intersection.
    """
    try:
        if interior_pt is None:
            interior_pt = interior_point(A,b)
        interior_pt = interior_pt.astype(float)
        A_b = np.hstack((A,-b))
        vertices = HalfspaceIntersection(A_b, interior_pt).intersections
        vertices = np.round(np.array(vertices), 12)
    except NoInteriorPoint:
        vertices = []
        m, n = A.shape
        for B in itertools.combinations(range(m), n):
            try:
                x = np.linalg.solve(A[B,:], b[B,:])[:,0]
                if all(np.matmul(A,x) <= b[:,0] + 1e-12):
                    vertices.append(x)
            except np.linalg.LinAlgError:
                pass
        vertices = np.round(np.array(vertices), 12)
        vertices = np.unique(vertices, axis=0)
    return [np.array([v]).transpose() for v in vertices]


def polytope_facets(A: np.ndarray,
                    b: np.ndarray,
                    vertices: List[np.ndarray] = None
                    ) -> List[List[np.ndarray]]:
    """Return the facets of the halfspace intersection Ax <= b.

    Provide vertices of the halfspace intersection to improve computation time.

    Args:
        A (np.ndarray): LHS coefficents of the halfspaces.
        b (np.ndarray): RHS coefficents of the halfspaces.
        vertices (List[np.ndarray]): Vertices of the halfspace intersection.

    Returns:
        List[List[np.ndarray]]: List of facets of the halfspace intersection.
    """
    if vertices is None:
        vertices = polytope_vertices(A, b)

    defining_facets = []
    for v in vertices:
        on_facet = np.isclose(a=np.matmul(A, v) - b,
                              b=np.zeros((len(A),1)),
                              atol=1e-10)
        defining_facets.append(np.where(on_facet)[0])
    facets = []
    for i in range(len(A)):
        facet = [j for j in range(len(vertices)) if i in defining_facets[j]]
        facet = [vertices[v] for v in facet]
        facets.append(facet)
    return facets


def halfspace_intersection(A: np.ndarray,
                           b: np.ndarray,
                           interior_pt: np.ndarray = None
                           ) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Return the intersection of the given halfspaces.

    Return the halfspace intersection of the halfspaces defined by the linear
    inequalities Ax <= b. If an interior point of the halfspace intersection is
    not given, one is computed using linear programming.

    Args:
        A (np.ndarray): LHS coefficents defining the linear inequalities.
        b (np.ndarray): RHS constants defining the linear inequalities.
        interior_pt (np.ndarray): Interior point of the halfspace intersection.

    Returns:
        Tuple:

        - vertices (List[np.ndarray]): Vertices of the halfspace intersection.
        - facets_by_halfspace (List[List[int]]): Vertices for each halfspace.
    """
    if interior_pt is None:
        interior_pt = interior_point(A,b)
    vertices = polytope_vertices(A, b, interior_pt=interior_pt)
    facets = polytope_facets(A, b, vertices=vertices)
    HS = namedtuple('hs', ['vertices', 'facets_by_halfspace'])
    return HS(vertices=vertices, facets_by_halfspace=facets)


def interior_point(A: np.ndarray,
                   b: np.ndarray,
                   tol: float = 1e-12) -> np.ndarray:
    """Return an interior point of the halfspace intersection.

    Given a list of halfspaces in the form of linear inequalities Ax <= b,
    return an interior point of the halfspace intersection. Linear programming
    is used to find the chebyshev center of the halfspace intersection.

    Args:
        A (np.ndarray): LHS coefficents defining the linear inequalities.
        b (np.ndarray): RHS constants defining the linear inequalities.
        tol (float) : Tolerance. (Interior radius should be > tol >= 0).

    Returns:
        np.ndarray: An interior point of the halfspace intersection.

    Raises:
        NoInteriorPoint: Halfspace intersection has no interior point.
    """
    M = np.hstack((A,-b))
    norm = np.reshape(np.linalg.norm(M[:, :-1], axis=1),(M.shape[0], 1))
    obj_func = np.zeros((M.shape[1],))
    obj_func[-1] = -1
    x = linprog(obj_func,
                A_ub=np.hstack((M[:, :-1], norm)),
                b_ub=-M[:, -1:],
                bounds=(None,None),
                method='revised simplex').x
    if x[-1] <= tol:
        raise NoInteriorPoint('Halfspace intersection has no interior point.')
    return x[:-1]


def order(x_list: List[np.ndarray]) -> List[List[float]]:
    """Return the ordered vertices of a non self-intersecting polygon.

    Args:
        x_list (List[np.ndarray]): List of vertices (in vector form) to order.

    Returns:
        List[List[float]]: Components for the non self-intersecting polygon.

    Raises:
        ValueError: Points must be in vector form.
        ValueError: Points must be 2 or 3 dimensional.
    """
    n,m = x_list[0].shape
    if not m == 1:
        raise ValueError('Points must be in vector form.')
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
            b_3 = np.cross(b_1, b_2)  # normal vector of plane
            # Change of basis to make z component constant.
            T = np.linalg.inv(np.array([b_1, b_2, b_3]).transpose())
            # Drop z component and use the ordering function for 2d.
            pts_T = [list(np.matmul(T,pts[i,:,None])[:2,0]) for i in range(p)]
            pts_T = np.array(pts_T)
            indices = sort_pts(pts_T)
        pts = pts[indices + [indices[0]]]
    components = list(zip(*pts))
    components = [list(component) for component in components]
    return components
