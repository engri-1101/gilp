import numpy as np
from .geometry import order
import plotly.graph_objects as plt
from typing import List, Dict, Union

"""Provides a higher level interface with plotly. Includes default styles.

Functions:
    format: Return a properly formated string for a number at some precision.
    linear_string: Return the string representation of a linear combination.
    equation_string: Return the string representation of an equation.
    label: Return a styled string representation of the given dictionary.
    table: Return a styled table trace with given headers and content.
    set_axis_limits: Set the axis limits for the given figure.
    get_axis_limits: Return the axis limits for the given figure.
    vector: Return a styled 2d or 3d vector trace from tail to head.
    scatter: Return a styled 2d or 3d scatter trace for given points / labels.
    line: Return a styled 2d line trace.
    equation: Return a styled 2d or 3d trace representing the given equation.
    order: Return an ordered list of points for drawing a 2d or 3d polygon.
    polygon: Return a styled 2d or 3d polygon trace defined by some points.
"""

# Sphinx documentation:
# BACKGROUND_COLOR = '#FCFCFC'
# FIG_WIDTH = 700
# Jupyter Notebook:
# BACKGROUND_COLOR = 'white'
# FIG_WIDTH = 950

BACKGROUND_COLOR = 'white'
"""The background color of the figure"""
FIG_HEIGHT = 500
"""The height of the entire visualization figure."""
FIG_WIDTH = 950
"""The width of the entire visualization figure."""
LEGEND_WIDTH = 200
"""The width of the legend section of the figure."""
LEGEND_NORMALIZED_X_COORD = (1-LEGEND_WIDTH/FIG_WIDTH)/2
"""The normalized x coordinate of the legend (relative to right side)."""
TABLEAU_NORMALIZED_X_COORD = LEGEND_NORMALIZED_X_COORD + LEGEND_WIDTH/FIG_WIDTH
"""The normalized x coordinate of the tableau (relative to right side)."""


def format(num: Union[int,float], precision: int = 3) -> str:
    """Return a properly formated string for a number at some precision."""
    return ('%.*f' % (precision, num)).rstrip('0').rstrip('.')


def linear_string(A: np.ndarray,
                  indices: List[int],
                  constant: float = None) -> str:
    """Return the string representation of a linear combination."""
    def sign(num: float): return {-1: ' - ', 0: ' + ', 1: ' + '}[np.sign(num)]
    s = ''
    if constant is not None:
        s += format(constant)
    for i in range(len(indices)):
        if i == 0:
            if constant is None:
                s += format(A[0]) + 'x<sub>' + str(indices[0]) + '</sub>'
            else:
                s += (sign(A[0]) + format(abs(A[0])) + 'x<sub>'
                      + str(indices[0]) + '</sub>')
        else:
            s += format(abs(A[i])) + 'x<sub>' + str(indices[i]) + '</sub>'
        if i is not len(indices)-1:
            s += sign(A[i+1])
    return s


def equation_string(A: np.ndarray, b: float, comp: str = ' ≤ ') -> str:
    """Return the string representation of an equation.

    The equation is assumed to be in standard form: Ax 'comp' b."""
    return linear_string(A, list(range(1, len(A) + 1))) + comp + format(b)


def label(dic: Dict[str, Union[float, list]]) -> str:
    """Return a styled string representation of the given dictionary."""
    entries = []
    for key in dic.keys():
        s = '<b>' + key + '</b>: '
        value = dic[key]
        if type(value) is float:
            s += format(value)
        if type(value) is list:
            s += '(%s)' % ', '.join(map(str, [format(i) for i in value]))
        entries.append(s)
    return '%s' % '<br>'.join(map(str, entries))


def table(header: List[str], content: List[str], style: str) -> plt.Table:
    """Return a styled table trace with given headers and content."""
    header_colors = ['red', 'black']
    content_colors = [['black', 'red', 'black'],
                      ['black', 'black', 'black']]

    if style == 'canonical':
        return plt.Table(header=dict(values=header,
                                     height=30,
                                     font=dict(color=header_colors, size=13),
                                     fill=dict(color=BACKGROUND_COLOR),
                                     line=dict(color='black', width=1)),
                         cells=dict(values=content,
                                    height=25,
                                    font=dict(color=content_colors, size=13),
                                    fill=dict(color=BACKGROUND_COLOR),
                                    line=dict(color='black',width=1)),
                         columnwidth=[1,0.8])
    elif style == 'dictionary':
        tmp = FIG_WIDTH*LEGEND_NORMALIZED_X_COORD
        return plt.Table(header=dict(values=header,
                                     height=25,
                                     font=dict(color=header_colors, size=14),
                                     align=['left', 'right', 'left'],
                                     fill=dict(color=BACKGROUND_COLOR),
                                     line=dict(color=BACKGROUND_COLOR,
                                               width=1)),
                         cells=dict(values=content,
                                    height=25,
                                    font=dict(color=content_colors, size=14),
                                    align=['left', 'right', 'left'],
                                    fill=dict(color=BACKGROUND_COLOR),
                                    line=dict(color=BACKGROUND_COLOR,
                                              width=1)),
                         columnwidth=[50/tmp, 25/tmp, 1-(75/tmp)])
    else:
        styles = ['canonical', 'dictionary']
        raise ValueError("Invalid style. Currently supports " + styles)


def set_axis_limits(fig: plt.Figure, x_list: List[np.ndarray]):
    """Set the axes limits of fig such that all points in x are visible.

    Given a set of nonnegative 2 or 3 dimensional points, set the axes
    limits such all points are visible within the plot window.
    """
    n = len(x_list[0])
    if n not in [2,3]:
        raise ValueError('x_list is a list of column vectors of length 2 or 3')
    pts = [list(x[:,0]) for x in x_list]
    limits = [max(i)*1.3 for i in list(zip(*pts))]
    if n == 2:
        x_lim, y_lim = limits
        pt = [np.array([[x_lim],[y_lim]])]
        fig.layout.xaxis1.range = [0, x_lim]
        fig.layout.yaxis1.range = [0, y_lim]
        fig.layout.scene1.xaxis.range = [0, x_lim]
        fig.layout.scene1.yaxis.range = [0, y_lim]
    if n == 3:
        x_lim, y_lim, z_lim = limits
        pt = [np.array([[x_lim],[y_lim],[z_lim]])]
        fig.layout.scene1.xaxis.range = [0, x_lim]
        fig.layout.scene1.yaxis.range = [0, y_lim]
        fig.layout.scene1.zaxis.range = [0, z_lim]
    # Add an invisible point at the axes limits to prevent axes from rescaling
    fig.add_trace(scatter(pt, 'clear'))


def get_axis_limits(fig: plt.Figure,n: int) -> List[float]:
    """Return the axis limits for the given figure."""
    if n not in [2,3]:
        raise ValueError('Can only retrieve 2 or 3 axis limits')
    x_lim = fig.layout.scene.xaxis.range[1]
    y_lim = fig.layout.scene.yaxis.range[1]
    if n == 2:
        return x_lim, y_lim
    else:
        z_lim = fig.layout.scene.zaxis.range[1]
        return x_lim, y_lim, z_lim


def vector(tail: np.ndarray,
           head: np.ndarray) -> Union[plt.Scatter, plt.Scatter3d]:
    """Return a styled 2d or 3d vector trace from tail to head."""
    pts = list(zip(*[tail[:,0],head[:,0]]))
    if len(pts) == 2:
        x,y = pts
        z = None
    if len(pts) == 3:
        x,y,z = pts
    args = dict(x=x, y=y, mode='lines',
                line=dict(width=6, color='red'), opacity=1,
                hoverinfo='skip', showlegend=False, visible=False)
    if z is None:
        return plt.Scatter(args)
    else:
        args['z'] = z
        return plt.Scatter3d(args)


def scatter(x_list: List[np.ndarray],
            style: str,
            lbs: List[str] = None) -> plt.Scatter:
    """Return a styled 2d or 3d scatter trace for given points and labels."""
    styles = ['bfs', 'initial_sol', 'clear']
    if style not in styles:
        raise ValueError("Invalid style. Currently supports " + styles)

    pts = list(zip(*[list(x[:,0]) for x in x_list]))
    if len(pts) == 2:
        x,y = pts
        z = None
    if len(pts) == 3:
        x,y,z = pts

    bfs_args = dict(x=x, y=y, text=lbs, mode='markers',
                    marker=dict(size=20, color='gray', opacity=1e-7),
                    showlegend=False, hoverinfo='text',
                    hoverlabel=dict(bgcolor='#FAFAFA',
                                    bordercolor='#323232',
                                    font=dict(family='Arial',
                                              color='#323232')))
    init_args = dict(x=x, y=y, mode='markers',
                     marker=dict(size=5, color='red', opacity=1),
                     hoverinfo='skip', showlegend=False)
    clear_args = dict(x=x, y=y, mode='markers',
                      marker=dict(size=0, color='white', opacity=1e-7),
                      hoverinfo='skip', showlegend=False)

    args = {'bfs': bfs_args,
            'initial_sol': init_args,
            'clear': clear_args}[style]
    if z is None:
        return plt.Scatter(args)
    else:
        args['z'] = z
        return plt.Scatter3d(args)


def line(x_list: List[np.ndarray],
         style: str,
         lb: str = None,
         i=[0]) -> plt.Scatter:
    """Return a 2d line trace in the desired style."""
    styles = ['constraint', 'isoprofit']
    if style not in styles:
        raise ValueError("Invalid style. Currently supports " + styles)

    x,y = list(zip(*[list(x[:,0]) for x in x_list]))
    colors = ['#173D90', '#1469FE', '#65ADFF', '#474849', '#A90C0C', '#DC0000']
    if style == 'constraint':
        i[0] = i[0] + 1 if i[0] + 1 < 6 else 0

    con_args = dict(x=x, y=y, name=lb, mode='lines',
                    line=dict(color=colors[i[0]],
                              width=2,
                              dash='15,3,5,3'),
                    hoverinfo='skip', visible=True, showlegend=True)
    iso_args = dict(x=x, y=y, mode='lines',
                    line=dict(color='red', width=4, dash=None),
                    hoverinfo='skip', visible=False, showlegend=False)
    return plt.Scatter({'constraint': con_args, 'isoprofit': iso_args}[style])


def equation(A: np.ndarray,
             b: float,
             domain: List[float],
             style: str,
             lb: str = None) -> Union[plt.Scatter, plt.Scatter3d]:
    """Return a styled 2d or 3d trace representing the given equation."""
    n = len(A)
    if n not in [2,3]:
        raise ValueError('Only supports equations in 2 or 3 variables')
    if all(A == np.zeros(n)):
        raise ValueError('A must have a nonzero component.')
    if n == 2:
        x_lim, y_lim = domain
        # A[0]x + A[1]y = b
        if A[1] != 0:
            x = np.linspace(0,x_lim,2)
            y = (b - A[0]*x)/A[1]
            x_list = [np.array([[x[i]],[y[i]]]) for i in range(len(x))]
        else:
            x = b/A[0]
            x_list = [np.array([[x],[0]]),np.array([[x],[y_lim]])]
        return line(x_list,style,lb)
    if n == 3:
        x_lim, y_lim, z_lim = domain
        # A[0]x + A[1]y + A[2]z = b
        x_list = []
        if A[2] != 0:
            for x in [0,x_lim]:
                for y in [0,y_lim]:
                    z = (b - A[0]*x - A[1]*y)/A[2]
                    x_list.append(np.array([[x],[y],[z]]))
        elif A[1] != 0:
            for x in [0,x_lim]:
                y = (b - A[0]*x)/A[1]
                for z in [0,z_lim]:
                    x_list.append(np.array([[x],[y],[z]]))
        else:
            x = b/A[0]
            for y in [0,y_lim]:
                for z in [0,z_lim]:
                    x_list.append(np.array([[x],[y],[z]]))
        return polygon(x_list,style,lb=lb)


def polygon(x_list: List[np.ndarray],
            style: str,
            ordered: bool = False,
            lb: str = None) -> Union[plt.Scatter, plt.Scatter3d]:
    """Return a styled 2d or 3d polygon trace defined by some points."""
    if len(x_list) == 0:
        raise ValueError("The list of points was empty.")

    if len(x_list[0]) == 2:
        if not ordered:
            x,y = order(x_list)
        else:
            x_list.append(x_list[0])
            x,y = zip(*[list(x[:,0]) for x in x_list])
        return plt.Scatter(x=x, y=y, mode='lines', fill='toself',
                           fillcolor='#1469FE', opacity=0.3,
                           line=dict(width=2, color='#00285F'),
                           showlegend=False, hoverinfo='none')
    if len(x_list[0]) == 3:
        if not ordered:
            x,y,z = order(x_list)
        else:
            x_list.append(x_list[0])
            x,y,z = zip(*[list(x[:,0]) for x in x_list])
        # When plotting a surface in Plotly, the surface is generated with
        # respect to a chosen axis. If the surface is orthogonal to this
        # axis, then the surface will not appear. This next step ensures
        # that each polygon surface will properly display.
        axis = 2  # default axis
        if len(x) > 2:
            # Get the normal vector of this polygon
            v1 = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]
            v2 = [x[2] - x[0], y[2] - y[0], z[2] - z[0]]
            n = np.round(np.cross(v1,v2), 7)
            for ax in range(3):
                if not np.dot(n,[1 if i == ax else 0 for i in range(3)]) == 0:
                    axis = ax

        if style == 'region':
            return plt.Scatter3d(x=x, y=y, z=z, surfaceaxis=axis,
                                 surfacecolor='#1469FE', mode="lines",
                                 line=dict(width=5, color='#173D90'),
                                 opacity=0.2, hoverinfo='none',
                                 visible=True, showlegend=False)
        elif style == 'constraint':
            return plt.Scatter3d(x=x, y=y, z=z, name=lb, surfaceaxis=axis,
                                 surfacecolor='gray', mode="none",
                                 opacity=0.5, hoverinfo='none',
                                 visible='legendonly', showlegend=True)
        elif style == 'isoprofit_in':
            return plt.Scatter3d(x=x, y=y, z=z, mode="lines+markers",
                                 surfaceaxis=axis, surfacecolor='red',
                                 marker=dict(size=5, color='red', opacity=1),
                                 line=dict(width=5, color='red'),
                                 opacity=1, hoverinfo='none',
                                 visible=False, showlegend=False)
        elif style == 'isoprofit_out':
            return plt.Scatter3d(x=x, y=y, z=z, surfaceaxis=axis,
                                 surfacecolor='gray', mode="none",
                                 opacity=0.3, hoverinfo='none',
                                 visible=False, showlegend=False)
        else:
            styles = ['region', 'constraint', 'isoprofit_in', 'isoprofit_out']
            if style not in styles:
                raise ValueError("Invalid style. Currently supports " + styles)
