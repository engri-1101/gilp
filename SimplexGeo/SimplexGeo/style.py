import numpy as np
import itertools
from scipy.spatial import ConvexHull
import plotly.graph_objects as plt
from simplex import LP, InvalidBasis, InfeasibleBasicSolution
from typing import List, Dict, Union

# CHANGE BACK .style

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
    scatter: Return a styled 2d or 3d scatter trace for given points and labels.
    line: Return a styled 2d line trace.
    intersection: Return the points where Ax = b intersects Dx <= e.
    equation: Return a styled 2d or 3d trace representing the given equation.
    order: Return an ordered list of points for drawing a 2d or 3d polygon.
    polygon: Return a styled 2d or 3d polygon trace defined by some points.
"""


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
    styles = ['canonical','dictionary']
    if style not in styles:
        raise ValueError("Invalid style. Currently supports " + styles)

    canon_args = dict(header=dict(values=header,
                                  height=30,
                                  font=dict(size=13),
                                  fill=dict(color='white'),
                                  line=dict(color='black', width=1)),
                      cells=dict(values=content,
                                 height=25,
                                 font=dict(size=13),
                                 fill=dict(color='white'),
                                 line=dict(color='black', width=1)))
    dict_args = dict(header=dict(values=header,
                                 height=25,
                                 font=dict(size=14),
                                 align=['left', 'right', 'left'],
                                 fill=dict(color='white'),
                                 line=dict(color='white', width=1)),
                     cells=dict(values=content,
                                height=25,
                                font=dict(size=14),
                                align=['left', 'right', 'left'],
                                fill=dict(color='white'),
                                line=dict(color='white', width=1)),
                     columnwidth=[0.3, 0.07, 0.63])
    return plt.Table({'canonical': canon_args, 'dictionary': dict_args}[style])


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
        pts = [np.array([[x_lim],[y_lim]])]
        fig.update_layout(scene=dict(xaxis=dict(range=[0, x_lim]),
                                     yaxis=dict(range=[0, y_lim])))
    if n == 3:
        x_lim, y_lim, z_lim = limits
        pts = [np.array([[x_lim],[y_lim],[z_lim]])]
        fig.update_layout(scene=dict(xaxis=dict(range=[0, x_lim]),
                                     yaxis=dict(range=[0, y_lim]),
                                     zaxis=dict(range=[0, z_lim])))
    # Add an invisible point at the axes limits to prevent axes from rescaling
    fig.add_trace(scatter(pts, 'clear'))


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


def intersection(A: np.ndarray,
                 b: float,
                 D: np.ndarray,
                 e: float) -> List[np.ndarray]:
    """Return the points where Ax = b intersects Dx <= e."""
    n,m = len(A),len(D)
    if n not in [2,3]:
        raise ValueError('Only supports equations in 2 or 3 variables')
    lp = LP(np.vstack((D,A)),np.vstack((e,b)),np.ones((n,1)))
    pts = []
    for B in itertools.combinations(range(n+m),m+1):
        try:
            pts.append(lp.get_basic_feasible_sol(list(B)))
        except (InvalidBasis, InfeasibleBasicSolution):
            pass
    return [pt[0:n,:] for pt in pts]


def equation(fig: plt.Figure,
             A: np.ndarray,
             b: float,
             style: str,
             lb: str = None) -> Union[plt.Scatter, plt.Scatter3d]:
    """Return a styled 2d or 3d trace representing the given equation."""
    n = len(A)
    if n not in [2,3]:
        raise ValueError('Only supports equations in 2 or 3 variables')
    pts = intersection(A,
                       b,
                       np.identity(n),
                       np.array([get_axis_limits(fig, n)]).transpose())
    if n == 2:
        return line(pts,style,lb)
    if n == 3:
        return polygon(pts,style,lb)


def order(x_list: List[np.ndarray]) -> List[List[float]]:
    """Return an ordered list of points for drawing a 2d or 3d polygon."""
    n,m = x_list[0].shape
    if not m == 1:
        raise ValueError('Points must be represented by column vectors')
    if n not in [2,3]:
        raise ValueError('Points must be 2 or 3 dimensional')

    pts = np.array([list(x[0:n,0]) for x in x_list])
    pts = np.unique(pts, axis=0)
    x_list = [np.array([pt]).transpose() for pt in pts]

    if len(pts) > 2:
        if n == 2:
            hull = ConvexHull(pts)
            return pts[hull.vertices,0], pts[hull.vertices,1]
        if n == 3:
            b_1 = pts[1] - pts[0]
            b_2 = pts[2] - pts[0]
            b_3 = np.cross(b_1, b_2)
            T = np.linalg.inv(np.array([b_1, b_2, b_3]).transpose())
            x_list = [list(np.round(np.dot(T,x),7)[0:2,0]) for x in x_list]
            hull = ConvexHull(np.array(x_list))
            pts = list(zip(pts[hull.vertices, 0],
                           pts[hull.vertices, 1],
                           pts[hull.vertices, 2]))
            pts.append(pts[0])
            return list(zip(*pts))
    else:
        return list(zip(*pts))


def polygon(x_list: List[np.ndarray],
            style: str,
            lb: str = None) -> plt.Scatter:
    """Return a styled 2d or 3d polygon trace defined by some points."""
    styles = ['region', 'constraint', 'isoprofit_in', 'isoprofit_out']
    if style not in styles:
        raise ValueError("Invalid style. Currently supports " + styles)

    if len(x_list[0]) == 2:
        x,y = order(x_list)
        return plt.Scatter(x=x, y=y, mode='lines', fill='toself',
                           fillcolor='#1469FE', opacity=0.3,
                           line=dict(width=2, color='#00285F'),
                           showlegend=False, hoverinfo='skip')
    if len(x_list[0]) == 3:
        x,y,z = order(x_list)
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

        region_args = dict(x=x, y=y, z=z, surfaceaxis=axis,
                           surfacecolor='#1469FE', mode="lines",
                           line=dict(width=5, color='#173D90'),
                           opacity=0.2, hoverinfo='skip',
                           visible=True, showlegend=False)
        con_args = dict(x=x, y=y, z=z, name=lb, surfaceaxis=axis,
                        surfacecolor='gray', mode="none",
                        opacity=0.5, hoverinfo='skip',
                        visible='legendonly', showlegend=True)
        iso_in_args = dict(x=x, y=y, z=z, mode="lines+markers",
                           surfaceaxis=axis, surfacecolor='red',
                           marker=dict(size=5, color='red', opacity=1),
                           line=dict(width=5, color='red'),
                           opacity=1, hoverinfo='skip',
                           visible=False, showlegend=False)
        iso_out_args = dict(x=x, y=y, z=z, surfaceaxis=axis,
                            surfacecolor='gray', mode="none",
                            opacity=0.3, hoverinfo='skip',
                            visible=False, showlegend=False)
        return plt.Scatter3d({'region': region_args,
                              'constraint': con_args,
                              'isoprofit_in': iso_in_args,
                              'isoprofit_out': iso_out_args}[style])
