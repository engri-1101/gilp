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
import networkx as nx
import numpy as np
import plotly.graph_objects as plt
from typing import List, Tuple
from .geometry import (intersection, halfspace_intersection, interior_point,
                       NoInteriorPoint)
from .graphic import (num_format, equation_string, linear_string, plot_tree,
                      Figure, label, table, vector, scatter, equation, polygon)
from .simplex import (LP, simplex, equality_form, branch_and_bound_iteration,
                      UnboundedLinearProgram, Infeasible)

# Color theme -- Using Google's Material Design Color System
# https://material.io/design/color/the-color-system.html

PRIMARY_COLOR = '#1565c0'
PRIMARY_LIGHT_COLOR = '#5e92f3'
PRIMARY_DARK_COLOR = '#003c8f'
SECONDARY_COLOR = '#d50000'
SECONDARY_LIGHT_COLOR = '#ff5131'
SECONDARY_DARK_COLOR = '#9b0000'
PRIMARY_FONT_COLOR = '#ffffff'
SECONDARY_FONT_COLOR = '#ffffff'
# Grayscale
TERTIARY_COLOR = '#DFDFDF'
TERTIARY_LIGHT_COLOR = 'white'  # Jupyter Notebook: white, Sphinx: #FCFCFC
TERTIARY_DARK_COLOR = '#404040'

# Figure Dimensions

FIG_HEIGHT = 500
"""Default figure height."""
FIG_WIDTH = 950  # Jupyter Notebook: 950, Sphinx: 700
"""Default figure width."""
LEGEND_WIDTH = 200
"""Default legend width."""
COMP_WIDTH = (FIG_WIDTH - LEGEND_WIDTH) / 2
"""Default width of the left and right component of a figure."""
ISOPROFIT_STEPS = 25
"""Number of isoprofit lines or plane to render."""

# Plotly trace templates

CANONICAL_TABLE = dict(header=dict(height=30,
                                   font_size=13,
                                   line=dict(color='black', width=1)),
                       cells=dict(height=25,
                                  font_size=13,
                                  line=dict(color='black',width=1)),
                       columnwidth=[1,0.8])
"""Template attributes for an LP table in canonical tableau form."""

DICTIONARY_TABLE = dict(header=dict(height=25,
                                    font_size=14,
                                    align=['left', 'right', 'left'],
                                    line_color=TERTIARY_LIGHT_COLOR,
                                    line_width=1),
                        cells=dict(height=25,
                                   font_size=14,
                                   align=['left', 'right', 'left'],
                                   line_color=TERTIARY_LIGHT_COLOR,
                                   line_width=1),
                        columnwidth=[50/COMP_WIDTH,
                                     25/COMP_WIDTH,
                                     1 - (75/COMP_WIDTH)])
"""Template attributes for an LP table in dictionary tableau form."""

BFS_SCATTER = dict(marker=dict(size=20, color='gray', opacity=1e-7),
                   hoverinfo='text',
                   hoverlabel=dict(bgcolor=TERTIARY_LIGHT_COLOR,
                                   bordercolor=TERTIARY_DARK_COLOR,
                                   font_family='Arial',
                                   font_color=TERTIARY_DARK_COLOR,
                                   align='left'))
"""Template attributes for an LP basic feasible solutions (BFS)."""

VECTOR = dict(mode='lines', line_color=SECONDARY_COLOR, visible=False)
"""Template attributes for a 2d or 3d vector."""

CONSTRAINT_LINE = dict(mode='lines', showlegend=True,
                       line=dict(width=2, dash='15,3,5,3'))
"""Template attributes for (2d) LP constraints."""

ISOPROFIT_LINE = dict(mode='lines', visible=False,
                      line=dict(color=SECONDARY_COLOR, width=4, dash=None))
"""Template attributes for (2d) LP isoprofit lines."""

REGION_2D_POLYGON = dict(mode="lines", opacity=0.2, fill="toself",
                         line=dict(width=3, color=PRIMARY_DARK_COLOR))
"""Template attributes for (2d) LP feasible region."""

REGION_3D_POLYGON = dict(mode="lines", opacity=0.2,
                         line=dict(width=5, color=PRIMARY_DARK_COLOR))
"""Template attributes for (3d) LP feasible region."""

CONSTRAINT_POLYGON = dict(surfacecolor='gray', mode="none",
                          opacity=0.5, visible='legendonly',
                          showlegend=True)
"""Template attributes for (3d) LP constraints."""

ISOPROFIT_IN_POLYGON = dict(mode="lines+markers",
                            surfacecolor=SECONDARY_COLOR,
                            marker=dict(size=5,
                                        symbol='circle',
                                        color=SECONDARY_COLOR),
                            line=dict(width=5,
                                      color=SECONDARY_COLOR),
                            visible=False)
"""Template attributes for (3d) LP isoprofit plane (interior)."""

ISOPROFIT_OUT_POLYGON = dict(surfacecolor='gray', mode="none",
                             opacity=0.3, visible=False)
"""Template attributes for (3d) LP isoprofit plane (exterior)."""


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

    # Create a default plotly template

    # Intialize the layout attributes dictionary
    layout = dict(width=FIG_WIDTH,
                  height=FIG_HEIGHT,
                  title=dict(text="<b>Geometric Interpretation of LPs</b>",
                             font=dict(size=18,
                                       color=TERTIARY_DARK_COLOR),
                             x=0, y=0.99, xanchor='left', yanchor='top'),
                  legend=dict(title=dict(text='<b>Constraint(s)</b>',
                                         font=dict(size=14)),
                              font=dict(size=13),
                              x=(1 - LEGEND_WIDTH / FIG_WIDTH) / 2, y=1,
                              xanchor='left', yanchor='top'),
                  margin=dict(l=0, r=0, b=0, t=int(FIG_HEIGHT/15)),
                  font=dict(family='Arial', color=TERTIARY_DARK_COLOR),
                  paper_bgcolor=TERTIARY_LIGHT_COLOR,
                  plot_bgcolor=TERTIARY_LIGHT_COLOR,
                  hovermode='closest',
                  clickmode='none',
                  dragmode='turntable')

    # Axes
    axis_args = dict(gridcolor=TERTIARY_COLOR, gridwidth=1, linewidth=2,
                     linecolor=TERTIARY_DARK_COLOR, tickcolor=TERTIARY_COLOR,
                     ticks='outside', rangemode='tozero', showspikes=False,
                     title_standoff=15, automargin=True, zerolinewidth=2)
    scene_axis_args = dict(backgroundcolor=TERTIARY_LIGHT_COLOR,
                           gridcolor=TERTIARY_COLOR,
                           showbackground=True, gridwidth=2, showspikes=False,
                           linecolor=TERTIARY_DARK_COLOR, ticks='',
                           rangemode='tozero', zerolinecolor='white')
    x_domain = [0, (1 - (LEGEND_WIDTH / FIG_WIDTH)) / 2]
    y_domain = [0, 1]
    if n == 2:
        layout['xaxis1'] = {**axis_args, **dict(domain=x_domain,
                                                title='x<sub>1</sub>')}
        layout['yaxis1'] = {**axis_args, **dict(domain=y_domain,
                                                title='x<sub>2</sub>')}
    else:
        def axis(n: int):
            '''Add title x_n to axis attriibutes'''
            return {**scene_axis_args, **dict(title='x<sub>%d</sub>' % (n))}

        layout['scene'] = dict(aspectmode='cube',
                               domain=dict(x=x_domain, y=y_domain),
                               xaxis=axis(1), yaxis=axis(2), zaxis=axis(3))

    # Default table
    table = [plt.Table(header_font_color=[SECONDARY_COLOR, 'black'],
                       header_fill_color=TERTIARY_LIGHT_COLOR,
                       cells_font_color=[['black', SECONDARY_COLOR, 'black'],
                                         ['black', 'black', 'black']],
                       cells_fill_color=TERTIARY_LIGHT_COLOR,
                       visible=False)]

    # Default scatter
    # Rotates through 6 different default line colors.
    default_scatter = dict(mode='markers',
                           visible=True,
                           showlegend=False,
                           name='test',
                           hoverinfo='none',
                           fillcolor=PRIMARY_COLOR,
                           line_color=PRIMARY_DARK_COLOR,
                           line=dict(width=4),
                           marker_line=dict(width=2,
                                            color=SECONDARY_COLOR),
                           marker=dict(size=9,
                                       color=TERTIARY_LIGHT_COLOR,
                                       opacity=0.99))
    scatter = [plt.Scatter({**default_scatter, **dict(line_color='#173D90')}),
               plt.Scatter({**default_scatter, **dict(line_color='#1469FE')}),
               plt.Scatter({**default_scatter, **dict(line_color='#65ADFF')}),
               plt.Scatter({**default_scatter, **dict(line_color='#474849')}),
               plt.Scatter({**default_scatter, **dict(line_color='#A90C0C')}),
               plt.Scatter({**default_scatter, **dict(line_color='#DC0000')})]

    # Default scatter3d
    scatter3d = [plt.Scatter3d(mode='markers',
                               visible=True,
                               showlegend=False,
                               hoverinfo='none',
                               surfacecolor=PRIMARY_LIGHT_COLOR,
                               line=dict(width=6, color=PRIMARY_COLOR),
                               marker_line=dict(width=1,
                                                color=SECONDARY_COLOR),
                               marker=dict(size=5,
                                           symbol='circle-open',
                                           color=SECONDARY_LIGHT_COLOR,
                                           opacity=0.99))]

    # Named annotations templates for branch and bound tree nodes
    layout['annotations'] = [
        dict(name='current', visible=False,
             align="center", bgcolor='#45568B',
             bordercolor=TERTIARY_DARK_COLOR, borderwidth=2, borderpad=3,
             font=dict(size=12, color=TERTIARY_LIGHT_COLOR), ax=0, ay=0),
        dict(name='explored', visible=False,
             align="center", bgcolor='#D8E4F9',
             bordercolor=TERTIARY_DARK_COLOR, borderwidth=2, borderpad=3,
             font=dict(size=12, color=TERTIARY_DARK_COLOR), ax=0, ay=0),
        dict(name='unexplored', visible=False,
             align="center", bgcolor=TERTIARY_LIGHT_COLOR,
             bordercolor=TERTIARY_DARK_COLOR, borderwidth=2, borderpad=3,
             font=dict(size=12, color=TERTIARY_DARK_COLOR), ax=0, ay=0)
    ]

    # Conslidate and construct the template
    template = plt.layout.Template()
    template.layout = layout
    template.data.table = table
    template.data.scatter = scatter
    template.data.scatter3d = scatter3d
    fig.update_layout(template=template)

    # Right Subplot Axes
    if n == 2:
        fig.layout.xaxis2 = dict(domain=[0.5, 1], range=[0,1], visible=False)
        fig.layout.yaxis2 = dict(domain=[0.15, 1], range=[0,1], visible=False)
    else:
        fig.layout.xaxis = dict(domain=[0.5, 1], range=[0,1], visible=False)
        fig.layout.yaxis = dict(domain=[0.15, 1], range=[0,1], visible=False)

    return fig


def add_feasible_region(fig: Figure,
                        lp: LP,
                        set_axes: bool = True,
                        basic_sol: bool = True,
                        show_basis: bool = True,
                        theme: str = 'light'):
    """Add the feasible region of the LP to the figure.

    Add a visualization of the LP feasible region to the figure. In 2d, the
    feasible region is visualized as a convex shaded region in the coordinate
    plane. In 3d, the feasible region is visualized as a convex polyhedron.

    Args:
        fig (Figure): Figure on which the feasible region should be added.
        lp (LP): LP whose feasible region will be added to the figure.
        set_axis (bool): True if the figure's axes should be set.
        basic_sol (bool): True if the entire BFS is shown. Default to True.
        show_basis (bool) : True if the basis is shown within the BFS label.
        theme (str): One of light, dark, or outline. Defaults to light
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

    # Try to get intersection via interior point.
    # If no interior point, compute via brute-force.
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
        bfs_list = lp.get_basic_feasible_solns()
        bfs = [np.round(x, 12) for x in bfs_list.bfs]
        bases = bfs_list.bases
        values = bfs_list.values
        via_hs_intersection = False

    # Get unique basic feasible solutions (remove duplicates from degeneracy)
    unique = np.unique([list(bfs[i][:,0])
                        + [values[i]] for i in range(len(bfs))], axis=0)
    unique_bfs, unique_val = np.abs(unique[:,:-1]), unique[:,-1]
    pts = [np.array([bfs[:n]]).transpose() for bfs in unique_bfs]

    # light theme by default
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

    # Plot feasible region
    if n == 2:
        fig.add_trace(polygon(x_list=pts,
                              fillcolor=surface_color,
                              line_color=line_color,
                              opacity=opacity,
                              template=REGION_2D_POLYGON))
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
                                      ordered=via_hs_intersection,
                                      surfacecolor=surface_color,
                                      line_color=line_color,
                                      opacity=opacity,
                                      template=REGION_3D_POLYGON))
        fig.add_traces(traces)

    # Plot basic feasible solutions with their label
    lbs = []
    for i in range(len(unique_bfs)):
        d = {}
        if basic_sol:
            d['BFS'] = list(unique_bfs[i])
        else:
            d['BFS'] = list(unique_bfs[i][:n])
        if show_basis:
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
    fig.add_trace(scatter(x_list=pts,
                          text=lbs,
                          template=BFS_SCATTER))

    if set_axes:
        x_list = [list(x[:,0]) for x in pts]
        limits = [max(i)*1.3 for i in list(zip(*x_list))]
        fig.set_axis_limits(limits)


def add_constraints(fig: Figure, lp: LP):
    """Add the constraints of the LP to the figure.

    Constraints in 2d are represented by a line in the coordinate plane and are
    set to visible by default. Constraints in 3d are represented by planes in
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
        template = {2: CONSTRAINT_LINE, 3: CONSTRAINT_POLYGON}[n]
        traces.append(equation(A=A[i],
                               b=b[i][0],
                               domain=limits,
                               name=lb,
                               template=template))
    fig.add_traces(traces)


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
            trace = equation(A=c[:,0],
                             b=objectives[i],
                             domain=limits,
                             template=ISOPROFIT_LINE)
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
            traces.append(equation(A=c[:,0],
                                   b=obj_val,
                                   domain=limits,
                                   template=ISOPROFIT_OUT_POLYGON))
            pts = []
            if np.isclose(obj_val, s_val, atol=1e-12):
                pts = intersection(c[:,0], s_val, A, b)
            elif np.isclose(obj_val, t_val, atol=1e-12):
                pts = intersection(c[:,0], t_val, A, b)
            elif obj_val >= s_val and obj_val <= t_val:
                A_tmp = np.vstack((A,c[:,0]))
                b_tmp = np.vstack((b,obj_val))
                # Try to use a previously found interior point or compute a new
                # one. If none exists, this indicate the feasible region is
                # lower dimensional. Use intersection function instead.
                try:
                    if interior_pt is None:
                        interior_pt = interior_point(A_tmp, b_tmp)
                    res = halfspace_intersection(A_tmp,
                                                 b_tmp,
                                                 interior_pt=interior_pt)
                    pts = res.vertices
                    pts = pts[res.facets_by_halfspace[-1]]
                    pts = [np.array([pt]).transpose() for pt in pts]
                except NoInteriorPoint:
                    pts = intersection(c[:,0], obj_val, A, b)
            if len(pts) != 0:
                traces.append(polygon(x_list=pts,
                                      ordered=True,
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

    # Create the slider object
    params = dict(x=0.5 + ((LEGEND_WIDTH / FIG_WIDTH) / 2), xanchor="left",
                  y={'bottom': 0.01, 'top': 85/FIG_HEIGHT}[slider_pos],
                  yanchor="bottom", lenmode='fraction', len=0.4, active=0,
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
    n,m = lp.get_coefficients()[:2]
    A,b,c = equality_form(lp).get_coefficients()[2:]
    T = lp.get_tableau(B)
    if form == 'canonical':
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
        def x_sub(i: int): return 'x<sub>' + str(i) + '</sub>'
        content.append(['z'] + [x_sub(B[i] + 1) for i in range(m)])
        obj_func = ['= ' +  linear_string(-T[0,1:n+m+1][N],
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
                        tableaus: bool = True,
                        tableau_form: str = 'dictionary',
                        rule: str = 'bland',
                        initial_solution: np.ndarray = None,
                        iteration_limit: int = None,
                        feas_tol: float = 1e-7) -> plt.layout.Slider:
    """Return a slider which toggles through iterations of simplex.

    Plots the path of simplex on the figure as well as the associated tableaus
    at each iteration. Return a slider to toggle between iterations of simplex.
    Uses the given simplex parameters: rule, initial_solution, iteration_limit,
    and feas_tol. See more about these parameters using help(simplex).

    Args:
        fig (Figure): Figure to add the path of simplex to.
        lp (LP): The LP whose simplex path will be added to the plot.
        slider_pos (str): Position (top or bottom) of this slider.
        tableaus (bool): True if tableaus should be displayed. Default is True.
        tableau_form (str): Displayed tableau form. Default is 'dictionary'
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

    # Add initial tableau
    tab_template = {'canonical': CANONICAL_TABLE,
                    'dictionary': DICTIONARY_TABLE}[tableau_form]
    if tableaus:
        headerT, contentT = tableau_strings(lp, path[0].B, 0, tableau_form)
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

        if tableaus:
            # Add mid-way tableau and full tableau
            headerT, contentT = tableau_strings(lp, path[i].B, i, tableau_form)
            headerB, contentB = tableau_strings(lp, path[i-1].B,
                                                i-1, tableau_form)
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
        if tableaus:
            visible[fig.get_indices('table'+str(i))] = True
        for j in range(i+1):
            visible[fig.get_indices('path'+str(j))] = True

        lb = str(int(i / 2)) if i % 2 == 0 else ''
        step = dict(method="update", label=lb, args=[{"visible": visible}])
        steps.append(step)

    # Create the slider object
    params = dict(x=0.5 + ((LEGEND_WIDTH / FIG_WIDTH) / 2), xanchor="left",
                  y={'bottom': 0.01, 'top': 85 / FIG_HEIGHT}[slider_pos],
                  yanchor="bottom", lenmode='fraction', len=0.4, active=0,
                  currentvalue={"prefix": "Iteration: "},
                  tickcolor='white', ticklen=0, steps=steps)
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
    """
    fig = template_figure(lp.n)
    add_feasible_region(fig=fig,
                        lp=lp,
                        basic_sol=basic_sol,
                        show_basis=show_basis)
    add_constraints(fig, lp)
    slider = isoprofit_slider(fig, lp)
    fig.update_layout(sliders=[slider])
    return fig


def simplex_visual(lp: LP,
                   basic_sol: bool = True,
                   show_basis: bool = True,
                   tableau_form: str = 'dictionary',
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

    fig = template_figure(lp.n)
    add_feasible_region(fig=fig,
                        lp=lp,
                        basic_sol=basic_sol,
                        show_basis=show_basis)
    add_constraints(fig, lp)
    iter_slider = simplex_path_slider(fig=fig,
                                      lp=lp,
                                      tableau_form=tableau_form,
                                      rule=rule,
                                      initial_solution=initial_solution,
                                      iteration_limit=iteration_limit,
                                      feas_tol=feas_tol)
    iso_slider = isoprofit_slider(fig, lp)
    fig.update_layout(sliders=[iter_slider, iso_slider])
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
            add_feasible_region(fig=fig,
                                lp=lp,
                                theme='outline',
                                set_axes=False,
                                basic_sol=False,
                                show_basis=False)
        for feas_reg in feasible_regions:
            try:
                if current == feas_reg:
                    add_feasible_region(fig, feas_reg,
                                        theme='dark',
                                        set_axes=False, basic_sol=False,
                                        show_basis=False)
                else:
                    add_feasible_region(fig, feas_reg,
                                        set_axes=False, basic_sol=False,
                                        show_basis=False)
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
            iter_slider = simplex_path_slider(fig=fig,
                                              lp=current,
                                              slider_pos='bottom',
                                              tableaus=False)
            fig.update_layout(sliders=[iter_slider])
            fig.update_sliders()
        except Infeasible:
            pass

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
