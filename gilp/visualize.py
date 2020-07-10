import numpy as np
import math
import itertools
import plotly.graph_objects as plt
from plotly.subplots import make_subplots
from simplex import LP, simplex, UnboundedLinearProgram
from style import (format, equation_string, linear_string, label, table,
                   set_axis_limits, get_axis_limits, vector, scatter,
                   intersection, equation, polygon)
from typing import List, Tuple

# CHANGE BACK .style

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


FIG_HEIGHT = 500
FIG_WIDTH = 950
ITERATION_STEPS = 2
ISOPROFIT_STEPS = 25


class InfiniteFeasibleRegion(Exception):
    """Raised when an LP is found to have an infinite feasible region and can
    not be accurately displayed."""
    pass


def set_up_figure(n: int) -> plt.Figure:
    """Return a figure for an n dimensional LP visualization."""
    if n not in [2,3]:
        raise ValueError('Can only visualize 2 or 3 dimensional LPs.')

    plot_type = {2: 'scatter', 3: 'scene'}[n]
    # Create subplot with the plot on left and the tableaus on right
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,
                        specs=[[{"type": plot_type},{"type": "table"}]])
    # General arguments for each axis
    axis_args = dict(gridcolor='#CCCCCC', gridwidth=1,
                     linewidth=2, linecolor='#4D4D4D',
                     tickcolor='#4D4D4D', ticks='outside',
                     rangemode='tozero')
    # Arguments for the entire figure
    args = dict(width=FIG_WIDTH, height=FIG_HEIGHT,
                xaxis=axis_args, yaxis=axis_args,
                scene=dict(xaxis=axis_args, yaxis=axis_args, zaxis=axis_args),
                margin=dict(l=0, r=0, b=0, t=50), plot_bgcolor='#FAFAFA',
                font=dict(family='Arial', color='#323232'),
                title=dict(text="<b>Simplex Geo</b>",
                           font=dict(size=18, color='#00285F'),
                           x=0, y=0.95, xanchor='left', yanchor='bottom'),
                legend=dict(title=dict(text='<b>Constraint(s)</b>',
                                       font=dict(size=14)),
                            font=dict(size=13),
                            x=0.4, y=1, xanchor='left', yanchor='top'))
    # Set the figure arguments
    fig.update_layout(args)
    # Name each axis appropriately
    if n == 2:
        fig.layout.xaxis.title = 'x<sub>1</sub>'
        fig.layout.yaxis.title = 'x<sub>2</sub>'
    if n == 3:
        fig.layout.scene.xaxis.title = 'x<sub>1</sub>'
        fig.layout.scene.yaxis.title = 'x<sub>2</sub>'
        fig.layout.scene.zaxis.title = 'x<sub>3</sub>'
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
    """

    n,m,A,b,c = lp.get_inequality_form()
    try:
        simplex(LP(A,b,np.ones((n,1))))
    except UnboundedLinearProgram:
        raise InfiniteFeasibleRegion('Can not visualize.')

    fig = set_up_figure(n)

    bfs, bases, values = lp.get_basic_feasible_solns()
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

    # Plot basic feasible solutions with their label
    pts = [np.array([x]).transpose()[0:n] for x in unique_bfs]
    set_axis_limits(fig, pts)
    fig.add_trace(scatter(pts,'bfs',lbs))

    # Plot feasible region
    if n == 2:
        fig.add_trace(polygon(pts,'region'))
    if n == 3:
        for i in range(n+m):
            pts = [bfs[j][0:n,:] for j in range(len(bfs)) if i not in bases[j]]
            fig.add_trace(polygon(pts,'region'))

    # Plot constraints
    for i in range(m):
        lb = '('+str(i+n+1)+') '+equation_string(A[i],b[i][0])
        fig.add_trace(equation(fig,A[i],b[i][0],'constraint',lb))

    return fig


def get_tableau_strings(lp: LP,
                        B: List[int],
                        iteration: int,
                        form: str) -> Tuple[List[str], List[str]]:
    """Get the string representation of the tableau for the LP and basis B.

    The tableau can be in canonical or dictionary form:

    Canonical:                                Dictionary:
    -----------------------------------       Iteration i
    | z | x_1 | x_2 | ... | x_n | RHS |
    ===================================       max               - + x_N
    | 1 |  -  |  -  | ... |  -  |  -  |       subject to  x_i = - + x_N
    | 0 |  -  |  -  | ... |  -  |  -  |                   x_j = - + x_N
                   ...                                         ...
    | 0 |  -  |  -  | ... |  -  |  -  |                   x_k = - + x_N
    -----------------------------------
    """
    n,m,A,b,c = lp.get_inequality_form()
    T = lp.get_tableau(B)
    if form == 'canonical':
        header = ['<b>x<sub>' + str(i) + '</sub></b>' for i in range(n+m+2)]
        header[0] = '<b>z<sub></sub></b>'
        header[-1] = '<b>RHS<sub></sub></b>'
        content = list(T.transpose())
        content = [[format(i,1) for i in row] for row in content]
        content = [['%s' % '<br>'.join(map(str,col))] for col in content]
    if form == 'dictionary':
        B.sort()
        N = list(set(range(n + m)) - set(B))
        header = ['<b>ITERATION ' + str(iteration) + '</b>', ' ', ' ']
        content = []
        content.append(['max','subject to']+['' for i in range(m - 1)])
        def x_sub(i: int): return 'x<sub>' + str(i) + '</sub>'
        content.append([''] + [x_sub(B[i] + 1) for i in range(m)])
        obj_func = [linear_string(T[0,1:n+m+1][N],
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


def add_isoprofits(fig: plt.Figure, lp: LP) -> Tuple[List[int], List[float]]:
    """Add the set of isoprofit lines/planes which can be toggled over.

    Args:
        fig (plt.Figure): Figure to which isoprofits lines/planes are added
        lp (LP): LP for which the isoprofit lines are being generated

    Returns:
        isoprofit_line_IDs (List[int]): Indices of all isoprofit lines/planes
        objectives (List[float]): The corresponding objective values
    """
    n,m,A,b,c = lp.get_inequality_form()
    indices = []

    # Get minimum and maximum value of objective function in plot window
    D = np.identity(n)
    e = np.array([get_axis_limits(fig,n)]).transpose()
    max_val = simplex(LP(D,e,c))[2]
    min_val = -simplex(LP(D,e,-c))[2]

    objectives = list(np.round(np.linspace(min_val,
                                           max_val,
                                           ISOPROFIT_STEPS), 2))
    opt_val = simplex(lp)[2]
    objectives.append(opt_val)
    objectives.sort()

    for obj_val in objectives:
        if n == 2:
            fig.add_trace(equation(fig, c[:,0], obj_val, 'isoprofit'))
            indices.append(len(fig.data) - 1)
        if n == 3:
            fig.add_trace(equation(fig, c[:,0], obj_val, 'isoprofit_out'))
            pts = intersection(c[:,0], obj_val, lp.A, lp.b)
            if len(pts) == 0:
                # Add invisible point so two traces are added for each obj val
                fig.add_trace(scatter([np.zeros((n,1))], 'clear'))
            else:
                fig.add_trace(polygon(pts, 'isoprofit_in'))
            indices.append([len(fig.data) - 2, len(fig.data) - 1])
    return indices, objectives


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
    """

    fig = plot_lp(lp)  # Plot feasible region
    n,m,A,b,c = lp.get_inequality_form()
    path, bases, value, opt = simplex(lp=lp,
                                      pivot_rule=rule,
                                      initial_solution=initial_solution,
                                      iteration_limit=iteration_limit)

    # Keep track of indices for all the different traces
    path_IDs = add_path(fig, [i[list(range(n)),:] for i in path])
    table_IDs = add_tableaus(fig, lp, bases)
    isoprofit_IDs, objectives = add_isoprofits(fig,lp)

    # Add slider for toggling through simplex iterations
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

    iter_slider = dict(x=0.6, xanchor="left", y=0.2, yanchor="bottom",
                       pad={"t": 50}, lenmode='fraction', len=0.4, active=0,
                       currentvalue={"prefix": "Iteration: "},
                       tickcolor='white', ticklen=0, steps=iter_steps)

    # Add slider for toggling through isoprofit lines / planes
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

    iso_slider = dict(x=0.6, xanchor="left", y=0.02, yanchor="bottom",
                      pad={"t": 50}, lenmode='fraction', len=0.4, active=0,
                      currentvalue={"prefix": "Objective Value: "},
                      tickcolor='white', ticklen=0, steps=iso_steps)

    fig.update_layout(sliders=[iter_slider, iso_slider])
    return fig
