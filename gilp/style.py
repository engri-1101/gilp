import numpy as np
import networkx as nx
from .geometry import order
import plotly.graph_objects as plt
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.basedatatypes import BaseTraceType
from typing import List, Dict, Union

"""Provides a higher level interface with plotly.

Classes:
    Figure: Extension of the plotly figure to maintain traces.

Functions:
    format: Return a properly formated string for a number at some precision.
    linear_string: Return the string representation of a linear combination.
    equation_string: Return the string representation of an equation.
    label: Return a styled string representation of the given dictionary.
    table: Return a table trace with given headers and content.
    vector: Return a styled 2d or 3d vector trace from tail to head.
    scatter: Return a scatter trace for the given set of points.
    line: Return a scatter trace representing a 2d line.
    equation: Return a 2d or 3d trace representing the given equation.
    polygon: Return a 2d or 3d polygon trace defined the given points.
    plot_tree: Plot the tree on the figure.
"""


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
        self.add_trace(scatter(pt, visible=False),'extreme_point')

    def get_axis_limits(self) -> List[float]:
        """Return the list of axes limits.

        Returns:
            List[float]: List of axes limits.
        """
        return self.axis_limits

    def show(self, **kwargs):
        """Show the figure using default configuration settings."""
        kwargs['config'] = dict(doubleClick=False,
                                displayModeBar=False,
                                editable=False,
                                responsive=False,
                                showAxisDragHandles=False,
                                showAxisRangeEntryBoxes=False)
        plt.Figure.show(self, **kwargs)

    def write_html(self, file: str, **kwargs):
        """ Write a figure to an HTML file representation."""
        kwargs['config'] = dict(doubleClick=False,
                                displayModeBar=False,
                                editable=False,
                                responsive=False,
                                showAxisDragHandles=False,
                                showAxisRangeEntryBoxes=False)
        pio.write_html(self, file, **kwargs)

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

    def _ipython_display_(self):
        """Handle rich display of figures in ipython contexts."""
        if pio.renderers.render_on_display and pio.renderers.default:
            self.show()
        else:
            print(repr(self))


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


def table(header: List[str],
          content: List[str],
          template: Dict = None,
          **kwargs) -> plt.Table:
    """Return a table trace with given headers and content.

    Note: keyword arguments given outside of template are given precedence.

    Args:
        header (List[str]): Column titles for the table.
        content (List[str]): Content in each column of the table.
        template (Dict): Dictionary of trace attributes. Defaults to None.
        *kwargs: Arbitrary keyword arguments for plt.Table.

    Returns:
        plt.Table: A table trace with given headers and content.
    """
    if template is None:
        return plt.Table(header_values=header, cells_values=content)
    else:
        template = dict(template)
        template.update(kwargs)
        template['header']['values'] = header
        template['cells']['values'] = content
        return plt.Table(template)


def vector(tail: np.ndarray,
           head: np.ndarray,
           template: Dict = None,
           **kwargs) -> Union[plt.Scatter, plt.Scatter3d]:
    """Return a 2d or 3d vector trace from tail to head.

    Note: keyword arguments given outside of template are given precedence.

    Args:
        tail (np.ndarray): Point of the vector tail (in vector form).
        head (np.ndarray): Point of the vector head (in vector form).
        template (Dict): Dictionary of scatter attributes. Defaults to None.
        *kwargs: Arbitrary keyword arguments for plt.Scatter or plt.Scatter3d.
    """
    pts = list(zip(*[tail[:,0],head[:,0]]))
    if len(pts) == 2:
        x,y = pts
        z = None
    if len(pts) == 3:
        x,y,z = pts

    if template is None:
        if z is None:
            return plt.Scatter(x=x, y=y, **kwargs)
        else:
            return plt.Scatter3d(x=x, y=y, z=z, **kwargs)
    else:
        template = dict(template)
        template.update(kwargs)
        template['x'] = x
        template['y'] = y
        if z is None:
            return plt.Scatter(template)
        else:
            template['z'] = z
            return plt.Scatter3d(template)


def scatter(x_list: List[np.ndarray],
            template: Dict = None,
            **kwargs) -> Union[plt.Scatter, plt.Scatter3d]:
    """Return a scatter trace for the given set of points.

    Note: keyword arguments given outside of template are given precedence.

    Args:
        x_list (List[np.ndarray]): List of points in the form of vectors.
        template (Dict): Dictionary of scatter attributes. Defaults to None.
        *kwargs: Arbitrary keyword arguments for plt.Scatter or plt.Scatter3d.

    Returns:
        Union[plt.Scatter, plt.Scatter3d]: A scatter trace.
    """
    pts = list(zip(*[list(x[:,0]) for x in x_list]))
    if len(pts) == 2:
        x,y = pts
        z = None
    if len(pts) == 3:
        x,y,z = pts

    if template is None:
        if z is None:
            return plt.Scatter(x=x, y=y, **kwargs)
        else:
            return plt.Scatter3d(x=x, y=y, z=z, **kwargs)
    else:
        template = dict(template)
        template.update(kwargs)
        template['x'] = x
        template['y'] = y
        if z is None:
            return plt.Scatter(template)
        else:
            template['z'] = z
            return plt.Scatter3d(template)


def line(x_list: List[np.ndarray],
         template: Dict = None,
         **kwargs) -> plt.Scatter:
    """Return a scatter trace representing a 2d line.

    Note: keyword arguments given outside of template are given precedence.

    Args:
        x_list (List[np.ndarray]): List of points in the form of vectors.
        template (Dict): Dictionary of scatter attributes. Defaults to None.
        *kwargs: Arbitrary keyword arguments for plt.Scatter.

    Returns:
        plt.Scatter: A scatter trace representing a 2d line.
    """
    x,y = list(zip(*[list(x[:,0]) for x in x_list]))
    if template is None:
        return plt.Scatter(x=x, y=y)
    else:
        template = dict(template)
        template.update(kwargs)
        template['x'] = x
        template['y'] = y
        return plt.Scatter(template)


def equation(A: np.ndarray,
             b: float,
             domain: List[float],
             template: Dict = None,
             **kwargs) -> Union[plt.Scatter, plt.Scatter3d]:
    """Return a 2d or 3d trace representing the given equation.

    Note: keyword arguments given outside of template are given precedence.

    Args:
        A (np.ndarray): LHS coefficents of the constraint.
        b (float): RHS coefficent of the constraint.
        domain (List[float]): Domain on which to plot this constraint.
        template (Dict): Dictionary of scatter attributes. Defaults to None.
        *kwargs: Arbitrary keyword arguments for plt.Scatter or plt.Scatter3d.

    Raises:
        ValueError: Only supports equations in 2 or 3 variables.
        ValueError: A must have a nonzero component.

    Returns:
        Union[plt.Scatter, plt.Scatter3d]: A trace representing the equation.
    """
    n = len(A)
    if n not in [2,3]:
        raise ValueError('Only supports equations in 2 or 3 variables.')
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
        return line(x_list=x_list, template=template, **kwargs)
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
        return polygon(x_list=x_list, template=template, **kwargs)


def polygon(x_list: List[np.ndarray],
            ordered: bool = False,
            template: Dict = None,
            **kwargs) -> Union[plt.Scatter, plt.Scatter3d]:
    """Return a 2d or 3d polygon trace defined the given points.

    Note: keyword arguments given outside of template are given precedence.

    Args:
        x_list (List[np.ndarray]): List of points in the form of vectors.
        ordered (bool): True if given points are ordered. Defaults to False.
        template (Dict): Dictionary of scatter attributes. Defaults to None.
        *kwargs: Arbitrary keyword arguments for plt.Scatter or plt.Scatter3d.

    Returns:
        Union[plt.Scatter, plt.Scatter3d]: A 2d or 3d polygon trace.
    """
    if len(x_list) == 0:
        raise ValueError("The list of points was empty.")

    if len(x_list[0]) == 2:
        if not ordered:
            x,y = order(x_list)
        else:
            x_list.append(x_list[0])
            x,y = zip(*[list(x[:,0]) for x in x_list])
        if template is None:
            return plt.Scatter(x=x, y=y, **kwargs)
        else:
            template = dict(template)
            template.update(kwargs)
            template['x'] = x
            template['y'] = y
            return plt.Scatter(template)

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

        if template is None:
            return plt.Scatter3d(x=x, y=y, z=z, surfaceaxis=axis, **kwargs)
        else:
            template = dict(template)
            template.update(kwargs)
            template['x'] = x
            template['y'] = y
            template['z'] = z
            template['surfaceaxis'] = axis
            return plt.Scatter3d(template)


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
    T.nodes[0]['pos'] = (0.5,0.9)  # root position
    node_to_level = nx.single_source_shortest_path_length(T, root)
    levels = {}
    for i in node_to_level:
        l = node_to_level[i]
        if l in levels:
            levels[l].append(i)
        else:
            levels[l] = [i]

    level_count = max(levels.keys())+1
    level_heights = np.linspace(1.1,-0.1,level_count+2)[1:-1]
    for i in range(1,max(levels.keys())+1):
        # if 4 nodes in level, spread evenly;
        # otherwise, try to put nodes under their parent
        if len(levels[i]) <= 4:
            # get parents of every pair of children in the level
            children = {}
            for node in levels[i]:
                parent = [i for i in list(T.neighbors(node)) if i < node][0]
                if parent in children:
                    children[parent].append(node)
                else:
                    children[parent] = [node]

            # initial attempt at positioning
            pos = {}
            for parent in children:
                x = T.nodes[parent]['pos'][0]
                d = max((1/2)**(i+1), 0.1)
                pos[children[parent][0]] = [x-d, level_heights[i]]
                pos[children[parent][1]] = [x+d, level_heights[i]]

            # perturb if needed
            keys = list(pos.keys())
            x = [p[0] for p in pos.values()]
            n = len(x) - 1
            while (any(np.array([x[i+1] - x[i] for i in range(n)]) < 0.195)):
                for i in range(len(x)-1):
                    if abs(x[i+1] - x[i]) < 0.2:
                        shift = (0.2 - abs(x[i+1] - x[i]))/2
                        x[i] -= shift
                        x[i+1] += shift
            # shift to be within width
            x = np.array(x) + (max(0.05 - x[0], 0)) - (max(x[-1] - 0.95, 0))

            for i in range(len(x)):
                pos[keys[i]][0] = x[i]

            # set position
            for node in pos:
                T.nodes[node]['pos'] = pos[node]
        else:
            level_widths = np.linspace(-0.1,1.1,len(levels[i])+2)[1:-1]
            for j in range(len(levels[i])):
                T.nodes[(levels[i][j])]['pos'] = (level_widths[j],
                                                  level_heights[i])

    # Plot on Figure
    edge_x = []
    edge_y = []
    for edge in T.edges():
        x0, y0 = T.nodes[edge[0]]['pos']
        x1, y1 = T.nodes[edge[1]]['pos']
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = plt.Scatter(x=edge_x, y=edge_y,
                             line=dict(width=1, color='black'),
                             hoverinfo='none', showlegend=False, mode='lines')
    fig.add_trace(edge_trace, 'tree_edges', row, col)

    for node in T.nodes():
        if 'text' in T.nodes[node]:
            text = T.nodes[node]['text']
        else:
            text = node
        if 'template' in T.nodes[node]:
            template = T.nodes[node]['template']
        else:
            template = 'unexplored'
        x,y = T.nodes[node]['pos']
        fig.add_annotation(x=x, y=y, visible=True, text=text,
                           templateitemname=template, row=row, col=col)
