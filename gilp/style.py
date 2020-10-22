import numpy as np
import networkx as nx
from .geometry import order
import plotly.graph_objects as plt
from plotly.subplots import make_subplots
from plotly.basedatatypes import BaseTraceType
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


# TODO: Make the docs better for this class
class Figure(plt.Figure):
    """Extension of the plotly figure to maintain traces.

    The Figure class extends the plotly Figure class. It provides functions to
    adjust axis limits as well as add a traces. When adding trace(s), a name
    is specified so the set of traces can easily be accessed later.

    Attributes:
        trace_indices (Dict): Maintains trace indices for different trace sets.
        axis_limits (List[float]): Current axis limits.
    """
    def __init__(self, subplots: bool, *args, **kwargs):
        """Initialize the figure.

        If subplots is true, use the make_subplots method. Otherwise, use the
        __init__ method in the parent class plt.Figure."""
        if subplots:
            self.__dict__.update(make_subplots(*args, **kwargs).__dict__)
        else:
            super(Figure, self).__init__(*args, **kwargs)
        self.__dict__['trace_indices'] = {}
        self.__dict__['axis_limits'] = None

    def add_trace(self,
                  trace: BaseTraceType,
                  name: str,
                  row: int = None,
                  col: int = None):
        """Add the given trace to the figure with the given name.

        Args:
            trace (BaseTraceType): A trace to be added to the figure.
            name (str): Name to reference the trace by.
            row (int): Row to add this trace to.
            col (int): Column to add this trace to.
        """
        self.add_traces(
            traces=[trace],
            name=name,
            rows=[row] if row is not None else None,
            cols=[col] if col is not None else None)

    def add_traces(self, traces: List[BaseTraceType], name: str, **kwargs):
        """Add the given traces to the figure with the given name.

        Args:
            traces (List[BaseTraceType]): List of traces to add to the figure.
            name (str): Name to reference the traces by.
        """
        # TODO: Probably want to check we are not overriding some other name.
        n = len(self.data)
        self.trace_indices[name] = list(range(n, n+len(traces)))
        for trace in traces:
            super(Figure, self).add_traces(data=trace, **kwargs)

    def set_axis_limits(self, limits: List[float]):
        """Set the axes limits and add extreme point to prevent rescaling.

        Args:
            limits (List[float]): The list of limits to set the axes to.

        Raises:
            ValueError: The list of axis limits is not length 2 or 3.
        """
        n = len(limits)
        if n not in [2,3]:
            raise ValueError('The list of axis limits is not length 2 or 3.')
        self.axis_limits = limits
        if n == 2:
            x_lim, y_lim = limits
            pt = [np.array([[x_lim],[y_lim]])]
            self.layout.xaxis1.range = [0, x_lim]
            self.layout.yaxis1.range = [0, y_lim]
            self.layout.scene1.xaxis.range = [0, x_lim]
            self.layout.scene1.yaxis.range = [0, y_lim]
        if n == 3:
            x_lim, y_lim, z_lim = limits
            pt = [np.array([[x_lim],[y_lim],[z_lim]])]
            self.layout.scene1.xaxis.range = [0, x_lim]
            self.layout.scene1.yaxis.range = [0, y_lim]
            self.layout.scene1.zaxis.range = [0, z_lim]
        self.add_trace(scatter(pt, 'clear'),'extreme_point')

    def get_axis_limits(self) -> List[float]:
        """Return the list of axes limits.

        Returns:
            List[float]: List of axes limits.
        """
        return self.axis_limits

    # TODO: improve the docs for this function
    def get_indices(self, name: str, containing: bool = False) -> List[int]:
        """Return the list of trace indices containing the given name.

        Args:
            name (str): Name of the set of indices.
            containing (bool): Include indices under key containing name.

        Returns:
            List[int]: List of indices.
        """
        if containing:
            keys = [key for key in self.trace_indices if name in key]
            indices = [self.trace_indices[key] for key in keys]
            indices = [item for sublist in indices for item in sublist]
        else:
            indices = self.trace_indices[name]
        return indices

    def update_sliders(self):
        """Update the sliders of this figure.

        If a trace is added after a slider is created, the visibility of that
        trace in the steps of the slider is not specified. This method sets
        the visibility of these traces to False.
        """
        n = len(self.data)
        for slider in self.layout.sliders:
            for step in slider.steps:
                tmp = list(step.args[0]['visible'])
                step.args[0]['visible'] = tmp + [False]*(n-len(tmp))


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
                         columnwidth=[1,0.8], visible=False)
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
                         columnwidth=[50/tmp, 25/tmp, 1-(75/tmp)],
                         visible=False)
    else:
        styles = ['canonical', 'dictionary']
        raise ValueError("Invalid style. Currently supports " + styles)


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
                      hoverinfo='skip', showlegend=False, visible=False)

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


def plot_tree(fig:Figure,
              T:nx.classes.graph.Graph,
              root:Union[str,int],
              row:int = 1,
              col:int = 2):
    """Plot the tree on the figure.

    Args:
        fig (Figure): The figure to which the network should be plotted.
        T (nx.classes.graph.Graph): Tree to be plotted.
        root (Union[str,int]): Root node of the tree.
        row (int, optional): Subplot row of the figure. Defaults to 1.
        col (int, optional): Subplot col of the figure. Defaults to 2.
    """
    # Generate the positions for each node.
    node_to_level = nx.single_source_shortest_path_length(T, root)
    levels = {}
    for i in node_to_level:
        l = node_to_level[i]
        if l in levels:
            levels[l].append(i)
        else:
            levels[l] = [i]

    level_count = max(levels.keys())+1
    level_heights = np.linspace(0.9,0.1,level_count)
    for i in range(max(levels.keys())+1):
        level_widths = np.linspace(0,1,len(levels[i])+2)[1:-1]
        for j in range(len(levels[i])):
            T.nodes[(levels[i][j])]['pos'] = (level_widths[j],level_heights[i])

    # Plot on Figure
    edge_x = []
    edge_y = []
    for edge in T.edges():
        x0, y0 = T.nodes[edge[0]]['pos']
        x1, y1 = T.nodes[edge[1]]['pos']
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = plt.Scatter(x=edge_x, y=edge_y,
                             line=dict(width=1, color='#262626'),
                             hoverinfo='none', showlegend=False, mode='lines')
    fig.add_trace(edge_trace, 'tree_edges', row, col)

    for node in T.nodes():
        if 'text' in T.nodes[node]:
            text = T.nodes[node]['text']
        else:
            text = node
        if 'color' in T.nodes[node]:
            color = T.nodes[node]['color']
        else:
            color = 'white'
        if 'text_color' in T.nodes[node]:
            text_color = T.nodes[node]['text_color']
        else:
            text_color = "#262626"
        x,y = T.nodes[node]['pos']
        fig.add_annotation(x=x, y=y, text=text, align="center", bgcolor=color,
                           bordercolor="#262626", borderwidth=2, borderpad=3,
                           font=dict(size=12, color=text_color),
                           ax=0, ay=0, row=row, col=col)
