import numpy as np
import itertools
import plotly.graph_objects as plt
from .simplex import (LP, phase_one, simplex_iteration, simplex,
                      equality_form, UnboundedLinearProgram)
from .style import (format, Figure, equation_string, linear_string, label,
                    table, vector, scatter, equation, polygon,
                    BACKGROUND_COLOR, FIG_HEIGHT, FIG_WIDTH, LEGEND_WIDTH,
                    LEGEND_NORMALIZED_X_COORD, TABLEAU_NORMALIZED_X_COORD)
from .geometry import (intersection, halfspace_intersection, interior_point,
                       NoInteriorPoint)
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


ISOPROFIT_STEPS = 25
"""The number of isoprofit lines or plane to render."""


class InfiniteFeasibleRegion(Exception):
    """Raised when an LP is found to have an infinite feasible region and can
    not be accurately displayed."""
    pass


def set_up_figure(n: int, type: str = 'table') -> Figure:
    """Return a figure for an n dimensional LP visualization.

    Args:
        n (int): Dimension of the LP visualization. Either 2 or 3.
        type (str): Type of the left subplot. Table by default."""
    if n not in [2,3]:
        raise ValueError('Can only visualize 2 or 3 dimensional LPs.')

    # Subplots: plot on left, table on right
    plot_type = {2: 'scatter', 3: 'scene'}[n]
    fig = Figure(subplots=True, rows=1, cols=2,
                 horizontal_spacing=(LEGEND_WIDTH/FIG_WIDTH),
                 specs=[[{"type": plot_type},{"type": type}]])

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
    if type == 'scatter':
        fig.add_shape(dict(type="rect", x0=0, y0=0, x1=1, y1=1,
                           fillcolor="white",
                           opacity=1, layer="below",line_width=0),row=1, col=2)

    # AXES
    # Left Subplot Axes
    axis_args = dict(gridcolor='#CCCCCC', gridwidth=1,
                     linewidth=2, linecolor='#4D4D4D',
                     tickcolor='#4D4D4D', ticks='outside',
                     rangemode='tozero', showspikes=False)
    x_domain = [0, (1 - (LEGEND_WIDTH / FIG_WIDTH)) / 2]
    y_domain = [0, 1]
    if n == 2:
        x_axis_args = {**axis_args, **dict(domain=x_domain)}
        y_axis_args = {**axis_args, **dict(domain=[0,1])}
        fig.layout.xaxis1 = {**x_axis_args, **dict(title='x<sub>1</sub>')}
        fig.layout.yaxis1 = {**y_axis_args, **dict(title='x<sub>2</sub>')}
    else:
        def axis(n: int):
            '''Add title x_n to axis attriibutes'''
            return {**axis_args, **dict(title='x<sub>' + str(n) + '</sub>')}

        fig.layout.scene1 = dict(aspectmode='cube',
                                 domain=dict(x=x_domain, y=y_domain),
                                 xaxis=axis(1), yaxis=axis(2), zaxis=axis(3))

    # Right Subplot Axes
    x_domain = [0.5 + ((LEGEND_WIDTH / FIG_WIDTH) / 2), 1]
    y_domain = [0.15, 1]
    if n == 2:
        fig.layout.xaxis2 = dict(domain=x_domain, range=[0,1], visible=False)
        fig.layout.yaxis2 = dict(domain=y_domain, range=[0,1], visible=False)
    else:
        fig.layout.xaxis = dict(domain=x_domain, range=[0,1], visible=False)
        fig.layout.yaxis = dict(domain=y_domain, range=[0,1], visible=False)

    # Legend
    fig.layout.legend = dict(title=dict(text='<b>Constraint(s)</b>',
                                        font=dict(size=14)),
                             font=dict(size=13),
                             x=LEGEND_NORMALIZED_X_COORD, y=1,
                             xanchor='left', yanchor='top')
    return fig


def add_feasible_region(fig: Figure,
                        lp: LP,
                        set_axes: bool = True,
                        basic_sol: bool = True):
    """Add the feasible region of the LP to the figure.

    Add a visualization of the LP feasible region to the figure. In 2d, the
    feasible region is visualized as a convex shaded region in the coordinate
    plane. In 3d, the feasible region is visualized as a convex polyhedrom.

    Args:
        fig (Figure): Figure on which the feasible region should be added.
        lp (LP): LP whose feasible region will be added to the figure.
        set_axis (bool): True if the figure's axes should be set.
        basic_sol (bool): True if the entire BFS is shown. Default to True.

    Raises:
        ValueError: The LP must be in standard inequality form.
        InfiniteFeasibleRegion: Can not visualize.
    """
    if lp.equality:
        raise ValueError('The LP must be in standard inequality form.')
    n,m,A,b,c = lp.get_coefficients()
    try:
        simplex(LP(A,b,np.ones((n,1))))
    except UnboundedLinearProgram:
        raise InfiniteFeasibleRegion('Can not visualize.')

    try:
        # Get halfspace itersection
        A_tmp = np.vstack((A,-np.identity(n)))
        b_tmp = np.vstack((b,np.zeros((n,1))))
        hs = halfspace_intersection(A_tmp,b_tmp)
        vertices = np.round(hs.vertices,12)
        bfs = vertices

        # Add slack variable values to basic feasible solutions
        for i in range(m):
            x_i = -np.matmul(vertices,np.array([A[i]]).transpose()) + b[i]
            bfs = np.hstack((bfs,x_i))
        bfs = [np.array([bfs[i]]).transpose() for i in range(len(bfs))]

        # Get objective values for each basic feasible solution
        values = [np.matmul(c.transpose(),bfs[i][:n]) for i in range(len(bfs))]
        values = [float(val) for val in values]

        via_hs_intersection = True
    except NoInteriorPoint:
        bfs, bases, values = lp.get_basic_feasible_solns()
        via_hs_intersection = False

    # Get unique basic feasible solutions
    unique = np.unique([list(bfs[i][:,0])
                        + [values[i]] for i in range(len(bfs))], axis=0)
    unique_bfs, unique_val = np.abs(unique[:,:-1]), unique[:,-1]
    pts = [np.array([bfs[:n]]).transpose() for bfs in unique_bfs]

    # Plot feasible region
    if n == 2:
        fig.add_trace(polygon(pts, 'region'), 'feasible_region')
    if n == 3:
        if via_hs_intersection:
            facet_pt_indices = hs.facets_by_halfspace
        traces = []
        for i in range(n+m):
            if via_hs_intersection:
                face_pts = [vertices[j] for j in facet_pt_indices[i]]
                face_pts = [np.array([pt]).transpose() for pt in face_pts]
            else:
                face_pts = [bfs[j][0:n,:] for j in range(len(bfs))
                            if i not in bases[j]]
            if len(face_pts) > 0:
                traces.append(polygon(x_list=face_pts,
                                      style='region',
                                      ordered=via_hs_intersection))
        fig.add_traces(traces,'feasible_region')

    # Plot basic feasible solutions with their label
    lbs = []
    for i in range(len(unique_bfs)):
        d = {}
        if basic_sol:
            d['BFS'] = list(unique_bfs[i])
        else:
            d['BFS'] = list(unique_bfs[i][:n])
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
    fig.add_trace(scatter(pts, 'bfs', lbs), 'basic_feasible_solns')

    if set_axes:
        x_list = [list(x[:,0]) for x in pts]
        limits = [max(i)*1.3 for i in list(zip(*x_list))]
        fig.set_axis_limits(limits)


def add_constraints(fig: Figure, lp: LP):
    """Add the constraints of the LP to the figure.

    Constraints in 2d are represented by a line in the coordinate plane and are
    set to visible by default. Consstraints in 3d are represented by planes in
    3d space and are set to invisible by default.

    Args:
        fig (Figure): Figure for adding the constraints.
        lp (LP): The LP whose constraints will be added to the figure.

    Raises:
        ValueError: The LP must be in standard inequality form.
    """
    if lp.equality:
        raise ValueError('The LP must be in standard inequality form.')
    n,m,A,b,c = lp.get_coefficients()

    # Plot constraints
    limits = fig.get_axis_limits()
    traces = []
    for i in range(m):
        lb = '('+str(i+n+1)+') '+equation_string(A[i],b[i][0])
        traces.append(equation(A[i],b[i][0],limits,'constraint',lb))
    fig.add_traces(traces,'constraints')


def add_isoprofits(fig: Figure, lp: LP) -> plt.layout.Slider:
    """Add isoprofit lines/planes and slider to the figure.

    Add isoprofits of the LP to the figure and returns a slider to toggle
    between them. The isoprofits show the set of all points with a certain
    objective value (specified by the slider). In 2d, the isoprofit is a line
    and in 3d, the isoprofit is a plane. In 3d, the intersection of the
    isoprofit plane with the feasible region is highlighted.

    Args:
        fig (Figure): Figure to which isoprofits lines/planes are added.
        lp (LP): LP whose isoprofits are added to the figure.

    Return:
        plt.layout.Slider: A slider to toggle between objective values

    Raises:
        ValueError: The LP must be in standard inequality form.
    """
    if lp.equality:
        raise ValueError('The LP must be in standard inequality form.')
    n,m,A,b,c = lp.get_coefficients()

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
                                           ISOPROFIT_STEPS-1), 2))
    opt_val = simplex(lp)[2]
    objectives.append(opt_val)
    objectives.sort()

    # Add the isoprofit traces
    if n == 2:
        for i in range(ISOPROFIT_STEPS):
            trace = equation(c[:,0], objectives[i], limits, 'isoprofit')
            fig.add_trace(trace,('isoprofit_'+str(i)))
    if n == 3:
        # Get the objective values when the isoprofit plane first intersects
        # and last intersects the feasible region respectively
        s_val = -simplex(LP(A,b,-c))[2]
        t_val = opt_val

        # Keep track of an interior point once one is found
        interior_pt = None

        # Add nonnegativity constraints
        A = np.vstack((A,-np.identity(n)))
        b = np.vstack((b,np.zeros((n,1))))

        for i in range(ISOPROFIT_STEPS):
            traces = []
            obj_val = objectives[i]
            traces.append(equation(c[:,0], obj_val, limits, 'isoprofit_out'))
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
            if len(pts) != 0:
                traces.append(polygon(pts, 'isoprofit_in',ordered=True))
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

    # Create the slider object
    params = dict(x=TABLEAU_NORMALIZED_X_COORD, xanchor="left",
                  y=0.01, yanchor="bottom",
                  pad=dict(l=0, r=0, b=0, t=50),
                  lenmode='fraction', len=0.4, active=0,
                  currentvalue={"prefix": "Objective Value: "},
                  tickcolor='white', ticklen=0, steps=iso_steps)
    return plt.layout.Slider(params)


def tableau_strings(lp: LP,
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


def add_simplex_path(fig: Figure,
                     lp: LP,
                     tableau_form: str = 'dictionary',
                     rule: str = 'bland',
                     initial_solution: np.ndarray = None,
                     iteration_limit: int = None,
                     feas_tol: float = 1e-7) -> plt.layout.Slider:
    """Add the path of simplex on the given LP to the figure.

    Plots the path of simplex on the figure as well the associated tableaus at
    each iteration. Returns a slider to toggle between iterations of simplex.
    Uses thee given simplex parameters.

    Args:
        fig (Figure): Figure to add the path of simplex to.
        lp (LP): The LP whose simplex path will be added to the plot.
        tableau_form (str): Displayed tableau form. Default is 'dictionary'
        rule (str): Pivot rule to be used. Default is 'bland'
        initial_solution (np.ndarray): An initial solution. Default is None.
        iteration_limit (int): A limit on simplex iterations. Default is None.
        feas_tol (float): Primal feasibility tolerance (1e-7 default).

    Returns:
        plt.layout.Slider: Slider to toggle between simplex iterations.

    Raises:
        ValueError: The LP must be in standard inequality form.
        ValueError: Iteration limit must be strictly positive.
        ValueError: initial_solution should have shape (n,1) but was ().
    """
    if lp.equality:
        raise ValueError('The LP must be in standard inequality form.')
    if iteration_limit is not None and iteration_limit <= 0:
        raise ValueError('Iteration limit must be strictly positive.')

    n,m,A,b,c = equality_form(lp).get_coefficients()
    x,B = phase_one(lp)

    if initial_solution is not None:
        if not initial_solution.shape == (n, 1):
            raise ValueError('initial_solution should have shape (' + str(n)
                             + ',1) but was ' + str(initial_solution.shape))
        x_B = initial_solution
        if (np.allclose(np.dot(A,x_B), b, atol=feas_tol) and
                all(x_B >= np.zeros((n,1)) - feas_tol) and
                len(np.nonzero(x_B)[0]) <= m):
            x = x_B
            B = list(np.nonzero(x_B)[0])
        else:
            print('Initial solution ignored.')

    prev_x = None
    prev_B = None
    current_value = float(np.dot(c.transpose(), x))
    optimal = False

    # Add initial solution and tableau
    fig.add_trace(scatter([x[:lp.n]], 'initial_sol'),'path0')
    headerT, contentT = tableau_strings(lp, B, 0, tableau_form)
    tab = table(headerT, contentT, tableau_form)
    tab.visible = True
    fig.add_trace(tab, ('table0'), row=1, col=2)

    i = 0  # number of iterations
    while(not optimal):
        prev_x = np.copy(x)
        prev_B = np.copy(B)
        x, B, current_value, optimal = simplex_iteration(lp=lp, x=x, B=B,
                                                         pivot_rule=rule,
                                                         feas_tol=feas_tol)
        i = i + 1
        if not optimal:
            # Add mid-way path and full path
            a = np.round(prev_x[:lp.n],10)
            b = np.round(x[:lp.n],10)
            m = a+((b-a)/2)
            fig.add_trace(vector(a,m),('path'+str(i*2-1)))
            fig.add_trace(vector(a,b),('path'+str(i*2)))

            # Add mid-way tableau and full tableau
            headerT, contentT = tableau_strings(lp, B, i, tableau_form)
            headerB, contentB = tableau_strings(lp, prev_B, i-1, tableau_form)
            content = []
            for j in range(len(contentT)):
                content.append(contentT[j] + [headerB[j]] + contentB[j])
            mid_tab = table(headerT, content, tableau_form)
            tab = table(headerT, contentT, tableau_form)
            fig.add_trace(mid_tab,('table'+str(i*2-1)), row=1, col=2)
            fig.add_trace(tab,('table'+str(i*2)), row=1, col=2)

        if iteration_limit is not None and i >= iteration_limit:
            break

    # Create each step of the iteration slider
    steps = []
    iterations = i - 1
    for i in range(2*iterations+1):
        visible = np.array([fig.data[j].visible for j in range(len(fig.data))])

        visible[fig.get_indices('table',containing=True)] = False
        visible[fig.get_indices('path',containing=True)] = False
        visible[fig.get_indices('table'+str(i))] = True
        for j in range(i+1):
            visible[fig.get_indices('path'+str(j))] = True

        lb = str(int(i / 2)) if i % 2 == 0 else ''
        step = dict(method="update", label=lb, args=[{"visible": visible}])
        steps.append(step)

    # Create the slider object
    params = dict(x=TABLEAU_NORMALIZED_X_COORD, xanchor="left",
                  y=(85/FIG_HEIGHT), yanchor="bottom",
                  pad=dict(l=0, r=0, b=0, t=0),
                  lenmode='fraction', len=0.4, active=0,
                  currentvalue={"prefix": "Iteration: "},
                  tickcolor='white', ticklen=0, steps=steps)
    return plt.layout.Slider(params)


def lp_visual(lp: LP) -> plt.Figure:
    """Render a plotly figure visualizing the geometry of an LP."""

    fig = set_up_figure(lp.n)
    add_feasible_region(fig, lp)
    add_constraints(fig, lp)
    slider = add_isoprofits(fig, lp)
    fig.update_layout(sliders=[slider])
    return fig


def simplex_visual(lp: LP,
                   tableau_form: str = 'dictionary',
                   rule: str = 'bland',
                   initial_solution: np.ndarray = None,
                   iteration_limit: int = None,
                   feas_tol: float = 1e-7) -> plt.Figure:
    """Render a figure showing the geometry of simplex on the given LP.

    Args:
        lp (LP): LP on which to run simplex
        tableau_form (str): Displayed tableau form. Default is 'dictionary'
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
    n,m,A,b,c = lp.get_coefficients()

    fig = set_up_figure(lp.n)
    add_feasible_region(fig, lp)
    add_constraints(fig, lp)
    iter_slider = add_simplex_path(fig=fig,
                                   lp=lp,
                                   tableau_form=tableau_form,
                                   rule=rule,
                                   initial_solution=initial_solution,
                                   iteration_limit=iteration_limit,
                                   feas_tol=feas_tol)
    iso_slider = add_isoprofits(fig, lp)
    fig.update_layout(sliders=[iter_slider, iso_slider])
    fig.update_sliders()
    return fig
