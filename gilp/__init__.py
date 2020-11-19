"""
GILP
====

GILP (Geometric Interpretation of Linear Programs) provides functions for
visualizing the geomtry of linear programs, the simplex method, and the
branch and bound algorithm for integer programs. The simplex module contains
an LP class which defines a linear program. It also contains implementations of
the revised simplex method (and Phase I) and the branch and bound algorithm.
The visualize module contains three functions for visualizing LP feasible
regions, the simplex method, and branch and bound respectively. Using .show(),
these figures can be viewed inline on a Jupyter Notebook. Alternatively, static
HTML files can be generated via .write_html(). Some linear program examples are
provided in the examples module.
"""

__author__ = 'Henry Robbins'

from .simplex import BFS, LP, simplex, branch_and_bound
from .visualize import lp_visual, simplex_visual, bnb_visual
from . import examples
