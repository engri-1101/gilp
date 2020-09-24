import numpy as np
import math
import itertools
import plotly.graph_objects as plt
from plotly.subplots import make_subplots
from .simplex import (LP, simplex, equality_form, UnboundedLinearProgram)
from .style import (format, equation_string, linear_string, label, table,
                    set_axis_limits, get_axis_limits, vector, scatter,
                    equation, polygon, BACKGROUND_COLOR,
                    FIG_HEIGHT, FIG_WIDTH, LEGEND_WIDTH,
                    LEGEND_NORMALIZED_X_COORD, TABLEAU_NORMALIZED_X_COORD)
from .geometry import intersection, halfspace_intersection, interior_point
from typing import List, Tuple

"""A Python module for visualizing the simplex algorithm for LPs.

Classes:
    InfiniteFeasibleRegion: Exception indicating an LP has an infinite feasible
                            region and can not be accurately displayed.

Functions:
    set_up_figure: Return a figure for an n dimensional LP visualization.
    plot_lp: Return a figure visualizing the feasible region of the given LP.
    get_tableau_strings: Get the string representation of the tableau for
                         the LP and basis B.
    add_path: Add vectors for visualizing the simplex path.
              Return vector indices.
"""


ITERATION_STEPS = 2
"""The number of steps each iteration is divided in to."""
ISOPROFIT_STEPS = 25
"""The number of isoprofit lines or plane to render."""


class InfiniteFeasibleRegion(Exception):
    """Raised when an LP is found to have an infinite feasible region and can
    not be accurately displayed."""
    pass


def set_up_figure(n: int) -> plt.Figure:
    """Return a figure for an n dimensional LP visualization."""
    if n not in [2,3]:
        raise ValueError('Can only visualize 2 or 3 dimensional LPs.')

    # Subplots: plot on left, table on right
    plot_type = {2: 'scatter', 3: 'scene'}[n]
    fig = make_subplots(rows=1, cols=2,
                        horizontal_spacing=(LEGEND_WIDTH/FIG_WIDTH),
                        specs=[[{"type": plot_type},{"type": "table"}]])

    # Attributes
    fig.layout.width = FIG_WIDTH
    fig.layout.height = FIG_HEIGHT
    fig.layout.title = dict(text="<b>Geometric Interpretation of LPs</b>",
                            font=dict(size=18, color='#00285F'),
                            x=0, y=0.99, xanchor='left', yanchor='top')
    fig.layout.margin = dict(l=0, r=0, b=0, t=int(FIG_HEIGHT/15))
    fig.layout.font = dict(family='Arial', color='#323232')
    fig.layout.paper_bgcolor = BACKGROUND_COLOR
    fig.layout.plot_bgcolor = '#FAFAFA'

    # Axes
    axis_args = dict(gridcolor='#CCCCCC', gridwidth=1,
                     linewidth=2, linecolor='#4D4D4D',
                     tickcolor='#4D4D4D', ticks='outside',
                     rangemode='tozero', showspikes=False)
    x_domain = [0, (1 - (LEGEND_WIDTH / FIG_WIDTH)) / 2]
    y_domain = [0, 1]
    x_axis_args = {**axis_args, **dict(domain=x_domain)}
    y_axis_args = {**axis_args, **dict(domain=[0,1])}
    fig.layout.xaxis1 = {**x_axis_args, **dict(title='x<sub>1</sub>')}
    fig.layout.yaxis1 = {**y_axis_args, **dict(title='x<sub>2</sub>')}

    def axis(n: int):
        '''Add title x_n to axis attriibutes'''
        return {**axis_args, **dict(title='x<sub>' + str(n) + '</sub>')}

    fig.layout.scene1 = dict(aspectmode='cube',
                             domain=dict(x=x_domain, y=y_domain),
                             xaxis=axis(1), yaxis=axis(2), zaxis=axis(3))

    # Legend
    fig.layout.legend = dict(title=dict(text='<b>Constraint(s)</b>',
                                        font=dict(size=14)),
                             font=dict(size=13),
                             x=LEGEND_NORMALIZED_X_COORD, y=1,
                             xanchor='left', yanchor='top')
    return fig


def plot_lp(lp: LP) -> plt.Figure:
    """Return a figure visualizing the feasible region of the given LP.

    Assumes the LP has 2 or 3 decision variables. Each axis corresponds to a
    single decision variable. The visualization plots each basic feasible
    solution (with their basis and objective value), the feasible region, and
    each of the constraints.

    Args:
        lp (LP): An LP to visualize.

    Returns:
        fig (plt.Figure): A figure containing the visualization.

    Raises:
        InfiniteFeasibleRegion: Can not visualize.
        ValueError: Can only visualize 2 or 3 dimensional LPs.
        ValueError: The LP must be in standard inequality form.
    """
    if lp.equality:
        raise ValueError('The LP must be in standard inequality form.')
    n,m,A,b,c = lp.get_coefficients()
    try:
        simplex(LP(A,b,np.ones((n,1))))
    except UnboundedLinearProgram:
        raise InfiniteFeasibleRegion('Can not visualize.')

    fig = set_up_figure(n)

    A_tmp = np.vstack((A,-np.identity(n)))
    b_tmp = np.vstack((b,np.zeros((n,1))))
    res = halfspace_intersection(A_tmp,b_tmp)
    vertices = res.vertices
    bfs = vertices

    # Add slack variable values to basic feasible solutions
    for i in range(m):
        x_i = -np.matmul(vertices,np.array([A[i]]).transpose()) + b[i]
        bfs = np.hstack((bfs,x_i))

    bfs = [np.array([bfs[i]]).transpose() for i in range(len(bfs))]
    values = [np.matmul(c.transpose(),bfs[i][:n]) for i in range(len(bfs))]
    values = [float(val) for val in values]

    unique = np.unique([list(bfs[i][:,0])
                        + [values[i]] for i in range(len(bfs))], axis=0)
    unique_bfs, unique_val = np.abs(unique[:,:-1]), unique[:,-1]

    # Create labels for each (unique) basic feasible solution
    lbs = []
    for i in range(len(unique_bfs)):
        d = dict(BFS=list(unique_bfs[i]))
        nonzero = list(np.nonzero(unique_bfs[i])[0])
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
        d['Obj'] = float(unique_val[i])
        lbs.append(label(d))

    # Get basic feasible solutions and set axis limits
    pts = np.round([np.array([x[:n]]).transpose() for x in vertices],12)
    set_axis_limits(fig, pts)

    # Get vertices for each face
    facet_vertices_indices = res.facets_by_halfspace
    facet_vertices = {}
    for i in range(n+m):
        facet_vertices[i] = [pts[j] for j in facet_vertices_indices[i]]

    # Plot feasible region
    if n == 2:
        fig.add_trace(polygon(pts,'region'))
    if n == 3:
        for i in range(n+m):
            face_pts = facet_vertices[i]
            if len(face_pts) > 0:
                fig.add_trace(polygon(face_pts,'region',ordered=True))

    # Plot constraints
    limits = get_axis_limits(fig,n)
    for i in range(m):
        lb = '('+str(i+n+1)+') '+equation_string(A[i],b[i][0])
        fig.add_trace(equation(A[i],b[i][0],limits,'constraint',lb))

    # Plot basic feasible solutions with their label
    # (Plot last so they are on the top layer for hovering)
    pts = [np.array([bfs[:n]]).transpose() for bfs in unique_bfs]
    fig.add_trace(scatter(pts,'bfs',lbs))

    return fig


def get_tableau_strings(lp: LP,
                        B: List[int],
                        iteration: int,
                        form: str) -> Tuple[List[str], List[str]]:
    """Get the string representation of the tableau for the LP and basis B.

    The tableau can be in canonical or dictionary form::

        Canonical:                                 Dictionary:
        ---------------------------------------    (i)
        | (i) z | x_1 | x_2 | ... | x_n | RHS |
        =======================================    max          ... + x_N
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
    n,m = lp.get_coefficients()[:2]
    A,b,c = equality_form(lp).get_coefficients()[2:]
    T = lp.get_tableau(B)
    if form == 'canonical':
        header = ['<b>x<sub>' + str(i) + '</sub></b>' for i in range(n+m+2)]
        header[0] = '<b>('+str(iteration)+') z</b>'
        header[-1] = '<b>RHS</b>'
        content = list(T.transpose())
        content = [[format(i,1) for i in row] for row in content]
        content = [['%s' % '<br>'.join(map(str,col))] for col in content]
    if form == 'dictionary':
        B.sort()
        N = list(set(range(n + m)) - set(B))
        header = ['<b>(' + str(iteration) + ')</b>', ' ', ' ']
        content = []
        content.append(['max','s.t.']+[' ' for i in range(m - 1)])
        def x_sub(i: int): return 'x<sub>' + str(i) + '</sub>'
        content.append([' '] + [x_sub(B[i] + 1) for i in range(m)])
        obj_func = [linear_string(-T[0,1:n+m+1][N],
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


def add_isoprofits(fig: plt.Figure, lp: LP) -> Tuple[List[int], List[float]]:
    """Add the set of isoprofit lines/planes which can be toggled over.

    Args:
        fig (plt.Figure): Figure to which isoprofits lines/planes are added
        lp (LP): LP for which the isoprofit lines are being generated

    Returns:
        Tuple:

        - List[int]: Indices of all isoprofit lines/planes
        - List[float]): The corresponding objective values

    Raises:
        ValueError: The LP must be in standard inequality form.
    """
    if lp.equality:
        raise ValueError('The LP must be in standard inequality form.')
    n,m,A,b,c = lp.get_coefficients()
    indices = []

    # Get minimum and maximum value of objective function in plot window
    domain = get_axis_limits(fig,n)
    limits = np.array([domain]).transpose()
    obj_at_limits = []
    for pt in itertools.product([0, 1], repeat=n):
        a = np.identity(n)
        np.fill_diagonal(a, pt)
        x = np.dot(a,limits)
        obj_at_limits.append(float(np.dot(c.transpose(),x)))

    max_val = max(obj_at_limits)
    min_val = min(obj_at_limits)

    objectives = list(np.round(np.linspace(min_val,
                                           max_val,
                                           ISOPROFIT_STEPS), 2))
    opt_val = simplex(lp)[2]
    objectives.append(opt_val)
    objectives.sort()

    if n == 2:
        for obj_val in objectives:
            fig.add_trace(equation(c[:,0], obj_val, domain, 'isoprofit'))
            indices.append(len(fig.data) - 1)
    if n == 3:
        # Get the objective values when the isoprofit plane first intersects
        # and last intersects the feasible region respectively
        s_val = -simplex(LP(A,b,-c))[2]
        t_val = simplex(LP(A,b,c))[2]

        # Keep track of an interior point once one is found
        interior_pt = None

        # Add nonnegativity constraints
        A = np.vstack((A,-np.identity(n)))
        b = np.vstack((b,np.zeros((n,1))))

        for obj_val in objectives:
            fig.add_trace(equation(c[:,0], obj_val, domain, 'isoprofit_out'))
            pts = []
            if np.isclose(obj_val, s_val, atol=1e-12):
                pts = intersection(c[:,0], s_val, A, b)
            elif np.isclose(obj_val, t_val, atol=1e-12):
                pts = intersection(c[:,0], t_val, A, b)
            elif obj_val >= s_val and obj_val <= t_val:
                A_tmp = np.vstack((A,c[:,0]))
                b_tmp = np.vstack((b,obj_val))
                if interior_pt is None:
                    interior_pt = interior_point(A_tmp, b_tmp)
                res = halfspace_intersection(A_tmp,
                                             b_tmp,
                                             interior_pt=interior_pt)
                pts = res.vertices
                pts = pts[res.facets_by_halfspace[-1]]
                pts = [np.array([pt]).transpose() for pt in pts]
            if len(pts) == 0:
                # Add invisible point so two traces are added for each obj val
                fig.add_trace(scatter([np.zeros((n,1))], 'clear'))
            else:
                fig.add_trace(polygon(pts, 'isoprofit_in',ordered=True))
            indices.append([len(fig.data) - 2, len(fig.data) - 1])
    return indices, objectives


def isoprofit_slider(isoprofit_IDs: List[int],
                     objectives: List[float],
                     fig: plt.Figure,
                     n: int) -> plt.layout.Slider:
    '''Create a plotly slider to toggle between isoprofit lines / planes.

    Args:
        isoprofit_IDs (List[int]): IDs of every isoprofit trace.
        objectives (List[float]): Objective values for every isoprofit trace.
        fig (plt.Figure): The figure containing the isoprofit traces.
        n (int): The dimension of the LP the figure visualizes.

    Returns:
        plt.layout.SLider: A plotly slider that can be added to a figure.
    '''
    # Create each step of the isoprofit slider
    iso_steps = []
    for i in range(len(isoprofit_IDs)):
        visible = [fig.data[k].visible for k in range(len(fig.data))]

        # Set isoprofit line / plane visibilities
        for j in isoprofit_IDs:
            if n == 2:
                visible[j] = False
            if n == 3:
                visible[j[0]] = False
                visible[j[1]] = False
        if n == 2:
            visible[isoprofit_IDs[i]] = True
        if n == 3:
            visible[isoprofit_IDs[i][0]] = True
            visible[isoprofit_IDs[i][1]] = True

        lb = objectives[i]
        step = dict(method="update", label=lb, args=[{"visible": visible}])
        iso_steps.append(step)

    # Create the Plotly slider object
    params = dict(x=TABLEAU_NORMALIZED_X_COORD, xanchor="left",
                  y=0.01, yanchor="bottom",
                  pad=dict(l=0, r=0, b=0, t=50),
                  lenmode='fraction', len=0.4, active=0,
                  currentvalue={"prefix": "Objective Value: "},
                  tickcolor='white', ticklen=0, steps=iso_steps)
    return plt.layout.Slider(params)


def add_path(fig: plt.Figure, path: List[np.ndarray]) -> List[int]:
    """Add vectors for visualizing the simplex path. Return vector indices."""
    fig.add_trace(scatter([path[0]], 'initial_sol'))
    indices = []
    for i in range(len(path)-1):
        a = np.round(path[i],7)
        b = np.round(path[i+1],7)
        d = (b-a)/ITERATION_STEPS
        for j in range(ITERATION_STEPS):
            fig.add_trace(vector(a,a+(j+1)*d))
            indices.append(len(fig.data)-1)
    return indices


def add_tableaus(fig: plt.Figure,
                 lp:LP,
                 bases: List[int],
                 tableau_form: str = 'dictionary') -> List[int]:
    """Add the set of tableaus. Return the indices of each table trace."""

    # Create the tables for each tableau
    tables = []
    for i in range(len(bases)):
        headerT, contentT = get_tableau_strings(lp, bases[i], i, tableau_form)
        if i == 0:
            tables.append(table(headerT, contentT, tableau_form))
        else:
            headerB, contentB = get_tableau_strings(lp=lp,
                                                    B=bases[i-1],
                                                    iteration=i - 1,
                                                    form=tableau_form)
            content = []
            for i in range(len(contentT)):
                content.append(contentT[i] + [headerB[i]] + contentB[i])
            tables.append(table(headerT, content, tableau_form))
            tables.append(table(headerT, contentT, tableau_form))

    # Add the tables to the figure
    indices = []
    for i in range(len(tables)):
        tab = tables[i]
        if not i == 0:
            tab.visible = False
        fig.add_trace(tab, row=1, col=2)
        indices.append(len(fig.data) - 1)
    return indices


def iteration_slider(path_IDs: List[int],
                     table_IDs: List[int],
                     fig: plt.Figure,
                     n: int) -> plt.layout.Slider:
    """Create a plotly slider to toggle between iterations of simplex

    Args:
        path_IDs (List[int]): IDs of every simplex path trace.
        table_IDs (List[int]): IDs of every table trace.
        fig (plt.Figure): The figure containing the traces.
        n (int): The dimension of the LP the figure visualizes.

    Returns:
        plt.layout.Slider: A plotly slider that can be added to a figure.
    """
    # Create each step of the iteration slider
    iter_steps = []
    for i in range(len(path_IDs)+1):
        visible = [fig.data[j].visible for j in range(len(fig.data))]

        # Set tableau visibilities
        for j in range(len(table_IDs)):
            visible[table_IDs[j]] = False
        if i % ITERATION_STEPS == 0:
            visible[table_IDs[int(2 * i / ITERATION_STEPS)]] = True
        else:
            visible[table_IDs[2 * math.ceil(i / ITERATION_STEPS) - 1]] = True

        # Set path visibilities
        for j in range(len(path_IDs) + 1):
            if j < len(path_IDs):
                visible[path_IDs[j]] = True if j < i else False

        lb = str(int(i / ITERATION_STEPS)) if i % ITERATION_STEPS == 0 else ''
        step = dict(method="update", label=lb, args=[{"visible": visible}])
        iter_steps.append(step)

    # Create the Plotly slider object
    params = dict(x=TABLEAU_NORMALIZED_X_COORD, xanchor="left",
                  y=(85/FIG_HEIGHT), yanchor="bottom",
                  pad=dict(l=0, r=0, b=0, t=0),
                  lenmode='fraction', len=0.4, active=0,
                  currentvalue={"prefix": "Iteration: "},
                  tickcolor='white', ticklen=0, steps=iter_steps)
    return plt.layout.Slider(params)


def lp_visual(lp: LP) -> plt.Figure:
    """Render a plotly figure visualizing the geometry of an LP."""

    fig = plot_lp(lp)  # Plot feasible region
    isoprofit_IDs, objectives = add_isoprofits(fig, lp)
    iso_slider = isoprofit_slider(isoprofit_IDs, objectives, fig, lp.n)
    fig.update_layout(sliders=[iso_slider])

    return fig


def simplex_visual(lp: LP,
                   tableau_form: str = 'dictionary',
                   rule: str = 'bland',
                   initial_solution: np.ndarray = None,
                   iteration_limit: int = None) -> plt.Figure:
    """Render a figure showing the geometry of simplex.

    Args:
        lp (LP): LP on which to run simplex
        tableau_form (str): Displayed tableau form. Default is 'dictionary'
        rule (str): Pivot rule to be used. Default is 'bland'
        initial_solution (np.ndarray): An initial solution. Default is None.
        iteration_limit (int): A limit on simplex iterations. Default is None.

    Returns:
        plt.Figure: A plotly figure which shows the geometry of simplex.

    Raises:
        ValueError: The LP must be in standard inequality form.
    """

    fig = plot_lp(lp)  # Plot feasible region
    if lp.equality:
        raise ValueError('The LP must be in standard inequality form.')
    n,m,A,b,c = lp.get_coefficients()
    path, bases, value, opt = simplex(lp=lp,
                                      pivot_rule=rule,
                                      initial_solution=initial_solution,
                                      iteration_limit=iteration_limit)

    # Create all traces: isoprofit, path, and table
    isoprofit_IDs, objectives = add_isoprofits(fig, lp)
    path_IDs = add_path(fig, [i[list(range(n)),:] for i in path])
    table_IDs = add_tableaus(fig, lp, bases, tableau_form)

    # Create sliders and add them to figure
    iso_slider = isoprofit_slider(isoprofit_IDs, objectives, fig, n)
    iter_slider = iteration_slider(path_IDs, table_IDs, fig, n)
    fig.update_layout(sliders=[iso_slider, iter_slider])

    return fig
