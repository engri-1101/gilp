"""Functions to visualize the simplex and branch and bound algorithms.

This moodule uses a custom implementation of the resvised simplex method and
the branch and bound algorithm (simplex module) to create and solve LPs. Using
the graphic module (which provides a high-level interface with the plotly
visualization package) and computational geometry functions from the geometry
module, visualizations of these algorithms are then created to be viewed inline
on a Jupyter Notebook or written to a static HTML file.
"""

__author__ = 'Henry Robbins'
__all__ = ['lp_visual', 'simplex_visual', 'bnb_visual']

import itertools
import math
import networkx as nx
import numpy as np
import plotly.graph_objects as plt
from typing import Union, List, Tuple
from ._constants import (AXIS_2D, AXIS_3D, BFS_SCATTER, BNB_NODE,
                         TABLEAU_TABLE, CONSTRAINT_LINE, CONSTRAINT_POLYGON,
                         DICTIONARY_TABLE, FIG_HEIGHT, FIG_WIDTH,
                         ISOPROFIT_IN_POLYGON, ISOPROFIT_LINE, INTEGER_POINT,
                         ISOPROFIT_OUT_POLYGON, ISOPROFIT_STEPS, LAYOUT,
                         LEGEND_WIDTH, PRIMARY_COLOR, PRIMARY_DARK_COLOR,
                         REGION_2D_POLYGON, REGION_3D_POLYGON, SCATTER,
                         SCATTER_3D, SECONDARY_COLOR, SLIDER, TABLE,
                         TERTIARY_DARK_COLOR, TERTIARY_LIGHT_COLOR, VECTOR)
from ._geometry import (intersection, interior_point, NoInteriorPoint,
                        polytope_vertices, polytope_facets)
from ._graphic import (num_format, equation_string, linear_string, plot_tree,
                       Figure, label, table, vector, scatter, equation,
                       polygon, polytope)
from .simplex import (LP, simplex, branch_and_bound_iteration,
                      UnboundedLinearProgram, Infeasible)


class InfiniteFeasibleRegion(Exception):
    """Raised when an LP is found to have an infinite feasible region and can
    not be accurately visualized."""
    pass


def template_figure(n: int, visual_type: str = 'tableau') -> Figure:
    """Return a figure on which to create a visualization.

    The figure can be for a 2 or 3 dimensional linear program and is either of
    type tableau (in which the tableau of each simplex iteration is on the
    right subplot) or type bnb_tree (in which a branch and bound tree is
    visualized shown on the right subplot).

    Args:
        n (int): Dimension of the LP visualization. Either 2 or 3.
        visual_type (str): Type of visualization. Tableau by default.

    Returns:
        Figure: A figure on which to create a visualization.

    Raises:
        ValueError: Can only visualize 2 or 3 dimensional LPs.
    """
    if n not in [2,3]:
        raise ValueError('Can only visualize 2 or 3 dimensional LPs.')

    # Subplots: plot on left, table/tree on right
    plot_type = {2: 'scatter', 3: 'scene'}[n]
    visual_type = {'tableau': 'table', 'bnb_tree': 'scatter'}[visual_type]
    fig = Figure(subplots=True, rows=1, cols=2,
                 horizontal_spacing=(LEGEND_WIDTH / FIG_WIDTH),
                 specs=[[{"type": plot_type},{"type": visual_type}]])

    layout = LAYOUT.copy()

    # Set axes
    x_domain = [0, (1 - (LEGEND_WIDTH / FIG_WIDTH)) / 2]
    y_domain = [0, 1]
    x = "x<sub>%d</sub>"
    if n == 2:
        layout['xaxis1'] = {**AXIS_2D, **dict(domain=x_domain, title=x % (1))}
        layout['yaxis1'] = {**AXIS_2D, **dict(domain=y_domain, title=x % (2))}
    else:
        layout['scene'] = dict(aspectmode='cube',
                               domain=dict(x=x_domain, y=y_domain),
                               xaxis={**AXIS_3D, **dict(title=x % (1))},
                               yaxis={**AXIS_3D, **dict(title=x % (2))},
                               zaxis={**AXIS_3D, **dict(title=x % (3))})

    # Rotate through 6 line colors
    colors = ['#173D90', '#1469FE', '#65ADFF', '#474849', '#A90C0C', '#DC0000']
    scatter = [plt.Scatter({**SCATTER, **dict(line_color=c)}) for c in colors]

    # Annotation templates for branch and bound tree nodes
    layout['annotations'] = [
        {**BNB_NODE, **dict(name='current', bgcolor='#45568B',
                            font_color=TERTIARY_LIGHT_COLOR)},
        {**BNB_NODE, **dict(name='explored', bgcolor='#D8E4F9')},
        {**BNB_NODE, **dict(name='unexplored', bgcolor=TERTIARY_LIGHT_COLOR)}
    ]

    # Conslidate and construct the template
    template = plt.layout.Template()
    template.layout = layout
    template.data.table = [plt.Table(TABLE)]
    template.data.scatter = scatter
    template.data.scatter3d = [plt.Scatter3d(SCATTER_3D)]
    fig.update_layout(template=template)

    # Right subplot axes
    right_x_axis = dict(domain=[0.5, 1], range=[0,1], visible=False)
    right_y_axis = dict(domain=[0.15, 1], range=[0,1], visible=False)
    if n == 2:
        fig.layout.xaxis2 = right_x_axis
        fig.layout.yaxis2 = right_y_axis
    else:
        fig.layout.xaxis = right_x_axis
        fig.layout.yaxis = right_y_axis

    return fig


def scale_axes(fig: Figure,
               vertices: List[np.ndarray],
               scale: float = 1.3):
    """Scale the axes of the figure to fit the given set of vertices.

    Args:
        fig (Figure): Figure whose axes will get re-scaled.
        vertices (List[np.ndarray]): Set of vertices to be contained.
        scale (float): The factor to multiply the minumum axis lengths by.
    """
    x_list = [list(x[:,0]) for x in vertices]
    limits = [max(i)*scale for i in list(zip(*x_list))]
    fig.set_axis_limits(limits)


def bfs_plot(lp: LP,
             basic_sol: bool = True,
             show_basis: bool = True,
             vertices: List[np.ndarray] = None
             ) -> Union[plt.Scatter, plt.Scatter3d]:
    """Return a scatter trace with hover labels for every basic feasible sol.

    Vertices of LP's feasible region can be given to improve computation time.

    Args:
        lp (LP): LP whose basic feasible solutions will be plotted.
        basic_sol (bool): True if the entire BFS is shown. Default to True.
        show_basis (bool) : True if the basis is shown within the BFS label.
        vertices (List[np.ndarray]): Vertices of the LP's feasible region.

    Returns:
        Union[plt.Scatter, plt.Scatter3d]: Scatter trace for every BFS.
    """
    n,m,A,b,c = lp.get_coefficients(equality=False)
    if vertices is None:
        vertices = lp.get_vertices()

    vertices_arr = np.array([list(v[:,0]) for v in vertices])
    bfs = vertices_arr

    # Add slack variable values to basic feasible solutions
    for i in range(m):
        x_i = -np.matmul(vertices_arr,np.array([A[i]]).transpose()) + b[i]
        bfs = np.hstack((bfs,x_i))
    bfs = [np.array([bfs[i]]).transpose() for i in range(len(bfs))]

    # Get objective values for each basic feasible solution
    values = [np.matmul(c.transpose(),bfs[i][:n]) for i in range(len(bfs))]
    values = [float(val) for val in values]

    # Plot basic feasible solutions with their label
    lbs = []
    for i in range(len(bfs)):
        d = {}
        if basic_sol:
            d['BFS'] = list(bfs[i])
        else:
            d['BFS'] = list(bfs[i][:n])
        if show_basis:
            nonzero = list(np.nonzero(bfs[i])[0])
            zero = list(set(list(range(n + m))) - set(nonzero))
            if len(zero) > n:  # indicates degeneracy
                # add all bases correspondong to this basic feasible solution
                count = 1
                for z in itertools.combinations(zero, len(zero)-n):
                    basis = 'B<sub>' + str(count) + '</sub>'
                    d[basis] = list(np.array(nonzero+list(z)) + 1)
                    count += 1
            else:
                d['B'] = list(np.array(nonzero)+1)  # non-degenerate
        d['Obj'] = float(values[i])
        lbs.append(label(d))

    return scatter(x_list=vertices, text=lbs, template=BFS_SCATTER)


def feasible_region(lp: LP,
                    theme: str = 'light',
                    vertices: List[np.ndarray] = None
                    ) -> List[Union[plt.Scatter, plt.Scatter3d]]:
    """Return traces representing the feasible region of the LP.

    In 2d, a single polygon trace is returned representing a convex shaded
    region in the coordinate plane. In 3d, multiple polygon traces are returned
    defining each facet of a convex polyhedron describing the feasible region.
    Vertices of LP's feasible region can be given to improve computation time.

    Args:
        lp (LP): LP whose feasible region visualization will be returned.
        theme (str): One of light, dark, or outline. Defaults to light.
        vertices (List[np.ndarray]): Vertices of the LP's feasible region.

    Returns:
        List[Union[plt.Scatter, plt.Scatter3d]]: Feasible region viualization.

    Raises:
        InfiniteFeasibleRegion: Can not visualize.
    """
    n,m,A,b,c = lp.get_coefficients(equality=False)
    try:
        simplex(LP(A,b,np.ones((n,1))))
    except UnboundedLinearProgram:
        raise InfiniteFeasibleRegion('Can not visualize.')

    if vertices is None:
        vertices = lp.get_vertices()

    # Add non-negativity constraints
    A_tmp = np.vstack((A, -np.identity(n)))
    b_tmp = np.vstack((b, np.zeros((n,1))))

    # Light theme by default
    opacity = 0.2
    surface_color = PRIMARY_COLOR
    line_color = PRIMARY_DARK_COLOR
    if theme == 'dark':
        surface_color = PRIMARY_DARK_COLOR
        line_color = '#002659'
        opacity = 0.2 + {2: 0.25, 3: 0.1}[lp.n]
    if theme == 'outline':
        surface_color = TERTIARY_LIGHT_COLOR
        line_color = TERTIARY_DARK_COLOR
        opacity = 0.1

    if n == 2:
        return polytope(A=A_tmp, b=b_tmp,
                        vertices=vertices,
                        template=REGION_2D_POLYGON,
                        fillcolor=surface_color,
                        line_color=line_color,
                        opacity=opacity)
    if n == 3:
        return polytope(A=A_tmp, b=b_tmp,
                        vertices=vertices,
                        template=REGION_3D_POLYGON,
                        surfacecolor=surface_color,
                        line_color=line_color,
                        opacity=opacity)


def labeled_feasible_region(lp: LP,
                            theme: str = 'light',
                            basic_sol: bool = True,
                            show_basis: bool = True,
                            vertices: List[np.ndarray] = None
                            ) -> List[Union[plt.Scatter, plt.Scatter3d]]:
    """Return traces representing the feasible region of an LP with bfs labels.

    Vertices of LP's feasible region can be given to improve computation time.

    Args:
        lp (LP): LP whose feasible region visualization will be returned.
        theme (str): One of light, dark, or outline. Defaults to light.
        basic_sol (bool): True if the entire BFS is shown. Default to True.
        show_basis (bool) : True if the basis is shown within the BFS label.
        vertices (List[np.ndarray]): Vertices of the LP's feasible region.

    Returns:
        List[Union[plt.Scatter, plt.Scatter3d]]: Feasible region w/ bfs labels.
    """
    if vertices is None:
        vertices = lp.get_vertices()
    region = feasible_region(lp=lp,
                             theme=theme,
                             vertices=vertices)
    bfs = bfs_plot(lp=lp,
                   basic_sol=basic_sol,
                   show_basis=show_basis,
                   vertices=vertices)
    return region + [bfs]


def feasible_integer_pts(lp: LP, fig: Figure) -> scatter:
    """Return scatter trace representing feasible integer points to the LP.

    Args:
        lp (LP): LP whose integer feasible points will be returned as a trace.
        fig (Figure): Figure this trace will be added to (for axis ranges).

    Returns:
        scatter: Scatter trace representing feasible integer points to the LP.
    """
    limits = fig.get_axis_limits()
    pts = []
    for i in range(math.ceil(limits[0])):
        for j in range(math.ceil(limits[0])):
            if len(limits) == 2:
                x = np.array([[i],[j]])
                if all(np.matmul(lp.A,x) <= lp.b + 1e-10):
                    pts.append(x)
            else:
                for k in range(math.ceil(limits[0])):
                    x = np.array([[i],[j],[k]])
                    if all(np.matmul(lp.A,x) <= lp.b + 1e-10):
                        pts.append(x)
    return scatter(pts, template=INTEGER_POINT)


def constraints(lp: LP,
                limits: List[int]
                ) -> List[Union[plt.Scatter, plt.Scatter3d]]:
    """Return traces for each constraint of the LP.

    Constraints in 2d are represented by a line in the coordinate plane and are
    set to visible by default. Constraints in 3d are represented by planes in
    3d space and are set to invisible by default.

    Args:
        lp (LP): The LP whose constraints will be added to the figure.
        limits (List[int]): Domain on which these constraints will be plotted.

    Returns:
        List[Union[plt.Scatter, plt.Scatter3d]]: List of constraint traces.
    """
    n,m,A,b,c = lp.get_coefficients(equality=False)
    traces = []
    for i in range(m):
        lb = '('+str(i+n+1)+') '+equation_string(A[i],b[i][0])
        template = {2: CONSTRAINT_LINE, 3: CONSTRAINT_POLYGON}[n]
        traces.append(equation(A=A[i],
                               b=b[i][0],
                               domain=limits,
                               name=lb,
                               template=template))
    return traces


def isoprofit_slider(fig: Figure,
                     lp: LP,
                     slider_pos: str = 'bottom') -> plt.layout.Slider:
    """Return a slider iterating through isoprofit lines/planes on the figure.

    Add isoprofits of the LP to the figure and returns a slider to toggle
    between them. The isoprofits show the set of all points with a certain
    objective value (specified by the slider). In 2d, the isoprofit is a line
    and in 3d, the isoprofit is a plane. In 3d, the intersection of the
    isoprofit plane with the feasible region is highlighted.

    Args:
        fig (Figure): Figure to which isoprofits lines/planes are added.
        lp (LP): LP whose isoprofits are added to the figure.
        slider_pos (str): Position (top or bottom) of this slider.

    Return:
        plt.layout.Slider: A slider to toggle between objective values.

    Raises:
        ValueError: The LP must be in standard inequality form.
    """
    if lp.equality:
        raise ValueError('The LP must be in standard inequality form.')
    n,m,A,b,c = lp.get_coefficients(equality=False)

    # Get minimum and maximum value of objective function in plot window
    limits = fig.get_axis_limits()
    obj_at_limits = []
    for pt in itertools.product([0, 1], repeat=n):
        a = np.identity(n)
        np.fill_diagonal(a, pt)
        x = np.dot(a,limits)
        obj_at_limits.append(float(np.dot(c.transpose(),x)))
    max_val = max(obj_at_limits)
    min_val = min(obj_at_limits)

    # Divide the range of objective values into multiple steps
    objectives = list(np.round(np.linspace(min_val,
                                           max_val,
                                           ISOPROFIT_STEPS), 2))

    try:
        opt_val = simplex(lp).obj_val
        objectives.append(round(opt_val,3))
        objectives.sort()
        feas = True
    except Infeasible:
        feas = False
        pass

    # Add the isoprofit traces
    if n == 2:
        for i in range(ISOPROFIT_STEPS + feas):
            trace = equation(A=c[:,0],
                             b=objectives[i],
                             domain=limits,
                             template=ISOPROFIT_LINE)
            fig.add_trace(trace,('isoprofit_'+str(i)))
    if n == 3:
        # If feasible, get the objective values when the isoprofit plane first
        # intersects and last intersects the feasible region respectively
        if feas:
            s_val = -simplex(LP(A,b,-c))[2]
            t_val = opt_val

        # Keep track of an interior point once one is found
        interior_pt = None

        # Add nonnegativity constraints
        A = np.vstack((A,-np.identity(n)))
        b = np.vstack((b,np.zeros((n,1))))

        for i in range(ISOPROFIT_STEPS + feas):
            traces = []
            obj_val = objectives[i]
            traces.append(equation(A=c[:,0],
                                   b=obj_val,
                                   domain=limits,
                                   template=ISOPROFIT_OUT_POLYGON))
            pts = []
            if feas:
                if np.isclose(obj_val, s_val, atol=1e-12):
                    pts = intersection(c[:,0], s_val, A, b)
                elif np.isclose(obj_val, t_val, atol=1e-12):
                    pts = intersection(c[:,0], t_val, A, b)
                elif obj_val >= s_val and obj_val <= t_val:
                    A_tmp = np.vstack((A,c[:,0]))
                    b_tmp = np.vstack((b,obj_val))
                    if interior_pt is None:
                        try:
                            interior_pt = interior_point(A_tmp, b_tmp)
                        except NoInteriorPoint:
                            pass
                    vertices = polytope_vertices(A_tmp, b_tmp, interior_pt)
                    pts = polytope_facets(A_tmp, b_tmp, vertices)[-1]
                if len(pts) != 0:
                    traces.append(polygon(x_list=pts,
                                          template=ISOPROFIT_IN_POLYGON))
            fig.add_traces(traces,('isoprofit_'+str(i)))

    # Create each step of the isoprofit slider
    iso_steps = []
    for i in range(ISOPROFIT_STEPS):
        visible = np.array([fig.data[k].visible for k in range(len(fig.data))])
        visible[fig.get_indices('isoprofit',containing=True)] = False
        visible[fig.get_indices('isoprofit_'+str(i))] = True
        visible[fig.get_indices('tree_edges',containing=True)] = True

        lb = objectives[i]
        step = dict(method="update", label=lb, args=[{"visible": visible}])
        iso_steps.append(step)

    params = {**SLIDER,
              **dict(currentvalue_prefix='Objective Value: ',
                     y={'bottom': 0.01, 'top': 85/FIG_HEIGHT}[slider_pos],
                     steps=iso_steps)}

    return plt.layout.Slider(params)


def lp_strings(lp: LP,
               B: List[int],
               iteration: int,
               form: str) -> Tuple[List[str], List[str]]:
    """Get the string representation of the LP and basis B.

    The LP can be in tableau or dictionary form::

        Tableau:                                   Dictionary:
        ---------------------------------------    (i)
        | (i) z | x_1 | x_2 | ... | x_n | RHS |
        =======================================    max     z  = ... + x_N
        |   1   |  -  |  -  | ... |  -  |  -  |    s.t.   x_i = ... + x_N
        |   0   |  -  |  -  | ... |  -  |  -  |           x_j = ... + x_N
                      ...                                      ...
        |   0   |  -  |  -  | ... |  -  |  -  |           x_k = ... + x_N
        ---------------------------------------

    Raises:
        ValueError: The LP must be in standard inequality form.
    """
    if lp.equality:
        raise ValueError('The LP must be in standard inequality form.')
    n,m = lp.get_coefficients(equality=False)[:2]
    A,b,c = lp.get_coefficients()[2:]
    T = lp.get_tableau(B)
    if form == 'tableau':
        header = ['<b>x<sub>' + str(i) + '</sub></b>' for i in range(n+m+2)]
        header[0] = '<b>('+str(iteration)+') z</b>'
        header[-1] = '<b>RHS</b>'
        content = list(T.transpose())
        content = [[num_format(i,1) for i in row] for row in content]
        content = [['%s' % '<br>'.join(map(str,col))] for col in content]
    if form == 'dictionary':
        B.sort()
        N = list(set(range(n + m)) - set(B))
        header = ['<b>(' + str(iteration) + ')</b>', ' ', ' ']
        content = []
        content.append(['max','s.t.']+[' ' for i in range(m - 1)])

        def x_sub(i: int):
            return 'x<sub>' + str(i) + '</sub>'

        content.append(['z'] + [x_sub(B[i] + 1) for i in range(m)])
        obj_func = ['= ' + linear_string(-T[0,1:n+m+1][N],
                                         list(np.array(N)+1),
                                         T[0,n+m+1])]
        coef = -T[1:,1:n+m+1][:,N]
        const = T[1:,n+m+1]
        eqs = ['= ' + linear_string(coef[i],
                                    list(np.array(N)+1),
                                    const[i]) for i in range(m)]
        content.append(obj_func + eqs)
        content = [['%s' % '<br>'.join(map(str, col))] for col in content]
    return header, content


def simplex_path_slider(fig: Figure,
                        lp: LP,
                        slider_pos: str = 'top',
                        show_lp: bool = True,
                        lp_form: str = 'dictionary',
                        rule: str = 'bland',
                        initial_solution: np.ndarray = None,
                        iteration_limit: int = None,
                        feas_tol: float = 1e-7) -> plt.layout.Slider:
    """Return a slider which toggles through iterations of simplex.

    Plots the path of simplex on the figure as well as the associated lps
    at each iteration. Return a slider to toggle between iterations of simplex.
    Uses the given simplex parameters: rule, initial_solution, iteration_limit,
    and feas_tol. See more about these parameters using help(simplex).

    Args:
        fig (Figure): Figure to add the path of simplex to.
        lp (LP): The LP whose simplex path will be added to the plot.
        slider_pos (str): Position (top or bottom) of this slider.
        show_lp (bool): True if lp should be displayed. Default is True.
        lp_form (str): Displayed lp form: {"dictionary", "tableau"}
        rule (str): Pivot rule to be used. Default is 'bland'
        initial_solution (np.ndarray): An initial solution. Default is None.
        iteration_limit (int): A limit on simplex iterations. Default is None.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).

    Returns:
        plt.layout.Slider: Slider to toggle between simplex iterations.

    Raises:
        ValueError: The LP must be in standard inequality form.
    """
    if lp.equality:
        raise ValueError('The LP must be in standard inequality form.')

    path = simplex(lp=lp, pivot_rule=rule, initial_solution=initial_solution,
                   iteration_limit=iteration_limit,feas_tol=feas_tol).path

    # Add initial lp
    tab_template = {'tableau': TABLEAU_TABLE,
                    'dictionary': DICTIONARY_TABLE}[lp_form]
    if show_lp:
        headerT, contentT = lp_strings(lp, path[0].B, 0, lp_form)
        tab = table(header=headerT, content=contentT, template=tab_template)
        tab.visible = True
        fig.add_trace(tab, ('table0'), row=1, col=2)

    # Iterate through remainder of path
    for i in range(1,len(path)):

        # Add mid-way path and full path
        a = np.round(path[i-1].x[:lp.n],10)
        b = np.round(path[i].x[:lp.n],10)
        m = a+((b-a)/2)
        fig.add_trace(vector(a, m, template=VECTOR),('path'+str(i*2-1)))
        fig.add_trace(vector(a, b, template=VECTOR),('path'+str(i*2)))

        if show_lp:
            # Add mid-way tableau and full tableau
            headerT, contentT = lp_strings(lp, path[i].B, i, lp_form)
            headerB, contentB = lp_strings(lp, path[i-1].B, i-1, lp_form)
            content = []
            for j in range(len(contentT)):
                content.append(contentT[j] + [headerB[j]] + contentB[j])
            mid_tab = table(headerT, content, template=tab_template)
            tab = table(headerT, contentT, template=tab_template)
            fig.add_trace(mid_tab,('table'+str(i*2-1)), row=1, col=2)
            fig.add_trace(tab,('table'+str(i*2)), row=1, col=2)

    # Add initial and optimal solution
    fig.add_trace(scatter(x_list=[path[0].x[:lp.n]]),'path0')
    fig.add_trace(scatter(x_list=[path[-1].x[:lp.n]],
                          marker_symbol='circle',
                          marker_color=SECONDARY_COLOR),'optimal')

    # Create each step of the iteration slider
    steps = []
    for i in range(2*len(path)-1):
        visible = np.array([fig.data[j].visible for j in range(len(fig.data))])

        visible[fig.get_indices('table',containing=True)] = False
        visible[fig.get_indices('path',containing=True)] = False
        visible[fig.get_indices('tree_edges',containing=True)] = True
        visible[fig.get_indices('optimal')] = True
        if show_lp:
            visible[fig.get_indices('table'+str(i))] = True
        for j in range(i+1):
            visible[fig.get_indices('path'+str(j))] = True

        lb = str(int(i / 2)) if i % 2 == 0 else ''
        step = dict(method="update", label=lb, args=[{"visible": visible}])
        steps.append(step)

    params = {**SLIDER,
              **dict(currentvalue_prefix='Iteration: ',
                     y={'bottom': 0.01, 'top': 85/FIG_HEIGHT}[slider_pos],
                     steps=steps)}

    return plt.layout.Slider(params)


def lp_visual(lp: LP,
              basic_sol: bool = True,
              show_basis: bool = True,) -> plt.Figure:
    """Render a figure visualizing the geometry of an LP's feasible region.

    Args:
        lp (LP): LP whose feasible region is visualized.
        basic_sol (bool): True if the entire BFS is shown. Default to True.
        show_basis (bool) : True if the basis is shown within the BFS label.

    Returns:
        plt.Figure: A plotly figure showing the geometry of feasible region.

    Raises:
        ValueError: The LP must be in standard inequality form.
    """
    if lp.equality:
        raise ValueError('The LP must be in standard inequality form.')

    fig = template_figure(lp.n)
    vertices = lp.get_vertices()
    scale_axes(fig, vertices)
    fig.add_traces(labeled_feasible_region(lp=lp,
                                           basic_sol=basic_sol,
                                           show_basis=show_basis,
                                           vertices=vertices))
    fig.add_traces(constraints(lp, fig.get_axis_limits()))
    slider = isoprofit_slider(fig, lp)
    fig.update_layout(sliders=[slider])
    return fig


def simplex_visual(lp: LP,
                   basic_sol: bool = True,
                   show_basis: bool = True,
                   lp_form: str = 'dictionary',
                   rule: str = 'bland',
                   initial_solution: np.ndarray = None,
                   iteration_limit: int = None,
                   feas_tol: float = 1e-7) -> plt.Figure:
    """Render a figure visualizing the geometry of simplex on the given LP.

    Uses the given simplex parameters: rule, initial_solution, iteration_limit,
    and feas_tol. See more about these parameters using help(simplex).

    Args:
        lp (LP): LP on which to run simplex.
        basic_sol (bool): True if the entire BFS is shown. Default to True.
        show_basis (bool) : True if the basis is shown within the BFS label.
        lp_form (str): Displayed lp form: {"dictionary", "tableau"}
        rule (str): Pivot rule to be used. Default is 'bland'
        initial_solution (np.ndarray): An initial solution. Default is None.
        iteration_limit (int): A limit on simplex iterations. Default is None.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).

    Returns:
        plt.Figure: A plotly figure which shows the geometry of simplex.

    Raises:
        ValueError: The LP must be in standard inequality form.
    """
    if lp.equality:
        raise ValueError('The LP must be in standard inequality form.')

    fig = template_figure(lp.n)
    vertices = lp.get_vertices()
    scale_axes(fig, vertices)
    fig.add_traces(labeled_feasible_region(lp=lp,
                                           basic_sol=basic_sol,
                                           show_basis=show_basis,
                                           vertices=vertices))
    fig.add_traces(constraints(lp, fig.get_axis_limits()))
    iso_slider = isoprofit_slider(fig, lp)
    iter_slider = simplex_path_slider(fig=fig,
                                      lp=lp,
                                      lp_form=lp_form,
                                      rule=rule,
                                      initial_solution=initial_solution,
                                      iteration_limit=iteration_limit,
                                      feas_tol=feas_tol)
    fig.update_layout(sliders=[iso_slider, iter_slider])
    fig.update_sliders()
    return fig


def bnb_visual(lp: LP,
               manual: bool = False,
               feas_tol: float = 1e-7,
               int_feas_tol: float = 1e-7) -> List[Figure]:
    """Render figures visualizing the geometry of branch and bound.

    Execute branch and bound on the given LP assuming that all decision
    variables must be integer. Use a primal feasibility tolerance of feas_tol
    (with default vlaue of 1e-7) and an integer feasibility tolerance of
    int_feas_tol (with default vlaue of 1e-7).

    Args:
        lp (LP): LP on which to run the branch and bound algorithm.
        manual (bool): True if the user can choose the variable to branch on.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).
        int_feas_tol (float): Integer feasibility tolerance (1e-7 default).

    Return:
        List[Figure]: A list of figures visualizing the branch and bound.
    """
    figs = []  # ist of figures to be returned
    feasible_regions = [lp]  # list of lps defining remaining feasible region
    incumbent = None
    best_bound = None
    unexplored = [lp]
    lp_to_node = {}  # dictionary from an LP object to the node id

    # Initialize the branch and bound tree
    G = nx.Graph()
    G.add_node(0)
    G.nodes[0]['text'] = ''
    lp_to_node[lp] = 0
    nodes_ct = 1

    # Get the axis limits to be used in all figures
    limits = lp_visual(lp).get_axis_limits()

    # Run the branch and bound algorithm
    while len(unexplored) > 0:
        current = unexplored.pop()

        # Create figure for current iteration
        fig = template_figure(lp.n, visual_type='bnb_tree')
        fig.set_axis_limits(limits)

        # Solve the LP relaxation
        try:
            sol = simplex(lp=current)
            x = sol.x
            value = sol.obj_val
            x_str = ', '.join(map(str, [num_format(i) for i in x[:lp.n]]))
            x_str = 'x* = (%s)' % x_str
            sol_str = '%s<br>%s' % (num_format(value), x_str)
        except Infeasible:
            sol_str = 'infeasible'

        # Update current node with solution and highlight it
        node_id = lp_to_node[current]
        G.nodes[node_id]['text'] += '<br>' + sol_str
        G.nodes[node_id]['template'] = 'current'

        # Plot the branch and bound tree
        plot_tree(fig,G,0)

        # Draw outline of original LP and remaining feasible region
        if current != lp:
            fig.add_traces(labeled_feasible_region(lp=lp,
                                                   theme='outline',
                                                   basic_sol=False,
                                                   show_basis=False))
        for feas_reg in feasible_regions:
            try:
                if current == feas_reg:
                    trace = labeled_feasible_region(lp=feas_reg,
                                                    theme='dark',
                                                    basic_sol=False,
                                                    show_basis=False)
                    fig.add_traces(trace)
                else:
                    trace = labeled_feasible_region(lp=feas_reg,
                                                    basic_sol=False,
                                                    show_basis=False)
                    fig.add_traces(trace)
            except Infeasible:
                pass

        # Show previous branch (constraints) of current node (if not the root)
        if nodes_ct > 1:
            A = current.A[-1]
            b = float(current.b[-1])
            i = int(np.nonzero(A)[0][0])+1
            template = {2: CONSTRAINT_LINE, 3: CONSTRAINT_POLYGON}[lp.n]
            if any(A < 0):
                fig.add_trace(equation(-A,-(b)-1, domain=limits,
                                       name="x<sub>%d</sub> ≤ %d" % (i,-(b+1)),
                                       template=template))
                fig.add_trace(equation(A, b, domain=limits,
                                       name="x<sub>%d</sub> ≥ %d" % (i, -b),
                                       template=template))
            else:
                fig.add_trace(equation(A, b, domain=limits,
                                       name="x<sub>%d</sub> ≤ %d" % (i, b),
                                       template=template))
                fig.add_trace(equation(-A, -(b+1), domain=limits,
                                       name="x<sub>%d</sub> ≥ %d" % (i, (b+1)),
                                       template=template))

        # Add path of simplex for the current node's LP
        try:
            simplex_path_slider(fig=fig,
                                lp=current,
                                slider_pos='bottom',
                                show_lp=False)
            for i in fig.get_indices('path', containing=True):
                fig.data[i].visible = True
        except Infeasible:
            pass

        # Add objective slider
        iso_slider = isoprofit_slider(fig, current)
        fig.update_layout(sliders=[iso_slider])
        fig.update_sliders()

        # Show the figure and add it to the list
        if manual:
            fig.show()
        figs.append(fig)

        # Do an iteration of the branch and bound algorithm
        iteration = branch_and_bound_iteration(lp=current,
                                               incumbent=incumbent,
                                               best_bound=best_bound,
                                               manual=manual,
                                               feas_tol=feas_tol,
                                               int_feas_tol=int_feas_tol)
        fathom = iteration.fathomed
        incumbent = iteration.incumbent
        best_bound = iteration.best_bound
        left_LP = iteration.left_LP
        right_LP = iteration.right_LP

        # If not fathomed, create nodes in the tree for each branch
        if not fathom:
            i = int(np.nonzero(left_LP.A[-1])[0][0])  # branched on index
            lb = int(left_LP.b[-1])
            ub = lb + 1

            # left branch node
            G.add_node(nodes_ct)
            lp_to_node[left_LP] = nodes_ct
            left_str = "x<sub>%d</sub> ≤ %d" % (i+1, lb)
            G.nodes[nodes_ct]['text'] = left_str
            G.add_edge(node_id, nodes_ct)

            # right branch node
            G.add_node(nodes_ct+1)
            lp_to_node[right_LP] = nodes_ct+1
            right_str = "x<sub>%d</sub> ≥ %d" % (i+1, ub)
            G.nodes[nodes_ct+1]['text'] = right_str
            G.add_edge(node_id, nodes_ct+1)
            nodes_ct += 2

            # update unexplored and feasible_regions
            unexplored.append(right_LP)
            unexplored.append(left_LP)
            feasible_regions.remove(current)
            feasible_regions.append(right_LP)
            feasible_regions.append(left_LP)

        # unhighlight the node and and indicate it has been explored
        G.nodes[node_id]['template'] = 'explored'

    return figs
