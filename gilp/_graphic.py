"""High-level plotly interface.

This module contains functions for creating various graphical components such
as tables, vectors, 3d polygons, etc. It also contains functions for nicely
formatting numbers and equations. The module serves as a high-level interface
to the expansive plotly visualization package.
"""

__author__ = 'Henry Robbins'
__all__ = ['Figure', 'num_format', 'linear_string', 'equation_string',
           'label', 'table', 'vector', 'scatter', 'line', 'equation',
           'polygon', 'polytope', 'plot_tree']

from ._geometry import order, polytope_vertices, polytope_facets
import networkx as nx
import numpy as np
from plotly.basedatatypes import BaseTraceType
import plotly.graph_objects as plt
import plotly.io as pio
from plotly.subplots import make_subplots
from typing import List, Dict, Union


class Figure(plt.Figure):
    """Extension of the plotly Figure which maintains trace names.

    This class extends the plotly Figure class. It provides the abiliity to
    give trace(s) a name. A map from trace names to their indices is maintained
    so that traces can easily be accessed later. Furthermore, it overrides the
    show and write_html functions by passing configuration settings.

    Attributes:
        _trace_name_to_indices (Dict): Map of trace names to trace indices.
        _axis_limits (List[float]): Axis limits of this figure.
    """

    _config = dict(doubleClick=False,
                   displayModeBar=False,
                   editable=False,
                   responsive=False,
                   showAxisDragHandles=False,
                   showAxisRangeEntryBoxes=False)
    """Configuration settings to be used by show and write_html functions."""

    def __init__(self, subplots: bool, *args, **kwargs):
        """Initialize the figure.

        If subplots is true, the args and kwargs are passed to make_subplots
        to generate a subplot; otherwise, the args and kwargs are passed to
        the parent class plotly.graph_objects.Figure __init__ method.

        Args:
            subplots (bool): True if arguments are intended for make_subplots.
        """
        if subplots:
            self.__dict__.update(make_subplots(*args, **kwargs).__dict__)
        else:
            super(Figure, self).__init__(*args, **kwargs)
        self.__dict__['_trace_name_to_indices'] = {}
        self.__dict__['_axis_limits'] = None

    def add_trace(self,
                  trace: BaseTraceType,
                  name: str = None,
                  row: int = None,
                  col: int = None):
        """Add a trace to the figure.

        If no name argument is passed, there will be no name mapping to this
        trace. It must be accessed by its index directly.

        Args:
            trace (BaseTraceType): A trace to be added to the figure.
            name (str, optional): Name to reference the trace by.
            row (int): Row to add this trace to.
            col (int): Column to add this trace to.
        """
        self.add_traces(traces=[trace], name=name, rows=row, cols=col)

    def add_traces(self,
                   traces: List[BaseTraceType],
                   name: str = None, **kwargs):
        """Add traces to the figure.

        If no name argument is passed, there will be no name mapping to these
        traces. They must be accessed by their indices directly.

        Args:
            traces (List[BaseTraceType]): List of traces to add to the figure.
            name (str, optional): Name to reference the traces by.

        Raises:
            ValueError: This trace name is already in use.
        """
        if name is not None:
            if name in self._trace_name_to_indices.keys():
                raise ValueError('This trace name is already in use.')
            n = len(self.data)
            self._trace_name_to_indices[name] = list(range(n, n+len(traces)))
        # Time trials revealed adding traces one at a time to be quicker than
        # using the add_traces function.
        for trace in traces:
            super(Figure, self).add_traces(data=trace, **kwargs)

    def get_indices(self, name: str, containing: bool = False) -> List[int]:
        """Return the list of trace indices with given name.

        If containing is False, find trace indices whose trace name is exactly
        as given; otherwise, find all trace indices whose trace name at least
        contains the given name.

        Args:
            name (str): Name of traces to be accessed.
            containing (bool): True if trace names containing name returned.

        Returns:
            List[int]: List of trace indices.
        """
        if containing:
            keys = [key for key in self._trace_name_to_indices if name in key]
            indices = [self._trace_name_to_indices[key] for key in keys]
            indices = [item for sublist in indices for item in sublist]
        else:
            indices = self._trace_name_to_indices[name]
        return indices

    def set_axis_limits(self, limits: List[float]):
        """Set axis limits and add extreme point to prevent rescaling.

        Args:
            limits (List[float]): The list of axis limits.

        Raises:
            ValueError: The list of axis limits is not length 2 or 3.
        """
        n = len(limits)
        if n not in [2,3]:
            raise ValueError('The list of axis limits is not length 2 or 3.')
        self._axis_limits = limits
        if n == 2:
            x_lim, y_lim = limits
            pt = [np.array([[x_lim],[y_lim]])]
            self.layout.xaxis1.range = [0, x_lim]
            self.layout.yaxis1.range = [0, y_lim]
        if n == 3:
            x_lim, y_lim, z_lim = limits
            pt = [np.array([[x_lim],[y_lim],[z_lim]])]
            self.layout.scene1.xaxis.range = [0, x_lim]
            self.layout.scene1.yaxis.range = [0, y_lim]
            self.layout.scene1.zaxis.range = [0, z_lim]
        self.add_trace(scatter(pt, visible=False))

    def get_axis_limits(self) -> List[float]:
        """Return the list of axis limits.

        Returns:
            List[float]: List of axis limits.
        """
        return self._axis_limits.copy()

    def update_sliders(self, default: bool = False):
        """Update the sliders of this figure.

        If a trace is added after a slider is created, the visibility of that
        trace in the steps of the slider is not specified. This method sets
        the visibility of these traces to False.

        Args:
            default (bool): Default visibility if unknown. Defaults to False.
        """
        n = len(self.data)
        for slider in self.layout.sliders:
            for step in slider.steps:
                tmp = list(step.args[0]['visible'])
                step.args[0]['visible'] = tmp + [default]*(n-len(tmp))

    def show(self, **kwargs):
        """Show the figure using default configuration settings."""
        kwargs['config'] = Figure._config
        plt.Figure.show(self, **kwargs)

    def write_html(self, file: str, **kwargs):
        """ Write a figure to an HTML file representation.

        Args:
            file (str): name of the file to write the HTML to."""
        kwargs['config'] = Figure._config
        pio.write_html(self, file, **kwargs)

    def _ipython_display_(self):
        """Handle rich display of figures in ipython contexts."""
        if pio.renderers.render_on_display and pio.renderers.default:
            self.show()
        else:
            print(repr(self))


def num_format(num: Union[int,float], precision: int = 3) -> str:
    """Return a properly formated string for a number at some precision.

    Formats a number to some precesion with trailing 0 and . removed.

    Args:
        num (Union[int,float]): Number to be formatted.
        precision (int, optional): Precision to use. Defaults to 3.

    Returns:
        str: String representation of the number."""
    return ('%.*f' % (precision, num)).rstrip('0').rstrip('.')


def linear_string(A: np.ndarray,
                  indices: List[int],
                  constant: float = None) -> str:
    """Return the string representation of a linear combination.

    For A = [a1,..,an] and indices = [i1,..,in], returns the linear combination
    a1 * x_(i1) + ... + an * x_(in) with a1,..,an formatted correctly. If a
    constant b is provided, then returns b + a1 * x_(i1) + ... + an * x_(in).

    Args:
        A (np.ndarray): List of coefficents for the linear combination.
        indices (List[int]): List of indices of the x variables.
        constant (float, optional): Constant of the linear combination.

    Returns:
        str: String representation of the linear combination.
    """
    # This function returns the correct sign (+ or -) prefix for a number
    def sign(num: float):
        return {-1: ' - ', 0: ' + ', 1: ' + '}[np.sign(num)]

    s = ''
    if constant is not None:
        s += num_format(constant)
    for i in range(len(indices)):
        if i == 0:
            if constant is None:
                s += num_format(A[0]) + 'x<sub>' + str(indices[0]) + '</sub>'
            else:
                s += (sign(A[0]) + num_format(abs(A[0])) + 'x<sub>'
                      + str(indices[0]) + '</sub>')
        else:
            s += num_format(abs(A[i])) + 'x<sub>' + str(indices[i]) + '</sub>'
        if i is not len(indices)-1:
            s += sign(A[i+1])
    return s


def equation_string(A: np.ndarray, b: float, rel: str = ' ≤ ') -> str:
    """Return the string representation of an equation.

    For A = [a1,..,an], b, and rel returns the string form of the equation
    a1 * x_(1) + ... + an * x_(n) rel b where rel represents some equality
    symbol = or inequality symbol <, >, ≥, ≤, ≠.

    Args:
        A (np.ndarray): Coefficents of the equation's LHS.
        b (float): Constant on the RHS of the equation.
        comp (str): Relation symbol: =, <, >, ≥, ≤, ≠.

    Returns:
        str: String representation of the equation.
    """
    return linear_string(A, list(range(1, len(A) + 1))) + rel + num_format(b)


def label(dic: Dict[str, Union[float, list]]) -> str:
    """Return a styled string representation of the given dictionary.

    Every key, value pair in the dictionary is on its own line with the key
    name bolded followed by the formatted value it maps to.

    Args:
        dic (Dict[str, Union[float, list]]): Dictionary to create string for.

    Returns:
        str: String representation of the given dictionary.
    """
    entries = []
    for key in dic.keys():
        s = '<b>' + key + '</b>: '
        value = dic[key]
        if type(value) is float:
            s += num_format(value)
        if type(value) is list:
            s += '(%s)' % ', '.join(map(str, [num_format(i) for i in value]))
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
    return scatter(x_list=[tail,head], template=template, **kwargs)


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
    pts = pts + [None]*(3 - len(pts))
    x,y,z = pts

    if template is None:
        template = kwargs
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
    return scatter(x_list=x_list, template=template, **kwargs)


def equation(A: np.ndarray,
             b: float,
             domain: List[float],
             template: Dict = None,
             **kwargs) -> Union[plt.Scatter, plt.Scatter3d]:
    """Return a 2d or 3d trace representing the given equation.

    Note: keyword arguments given outside of template are given precedence.

    Args:
        A (np.ndarray): LHS coefficents of the equation.
        b (float): RHS coefficent of the equation.
        domain (List[float]): Domain on which to plot this equation.
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
    """Return a 2d or 3d polygon trace defined by the given points.

    Note: keyword arguments given outside of template are given precedence.

    Args:
        x_list (List[np.ndarray]): List of points in the form of vectors.
        ordered (bool): True if given points are ordered. Defaults to False.
        template (Dict): Dictionary of scatter attributes. Defaults to None.
        *kwargs: Arbitrary keyword arguments for plt.Scatter or plt.Scatter3d.

    Returns:
        Union[plt.Scatter, plt.Scatter3d]: A 2d or 3d polygon trace.

    Raises:
        ValueError: The list of points was empty.
        ValueError: The points are not 2 or 3 dimensional.
    """
    if len(x_list) == 0:
        raise ValueError("The list of points was empty.")
    if len(x_list[0]) not in [2,3]:
        raise ValueError("The points are not 2 or 3 dimensional.")

    if len(x_list[0]) == 2:
        if not ordered:
            x,y = order(x_list)
        else:
            x_list.append(x_list[0])
            x,y = zip(*[list(x[:,0]) for x in x_list])
        z = None
    else:
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
        template = kwargs
    else:
        template = dict(template)
        template.update(kwargs)

    if z is None:
        x_list = [np.array([i]).transpose() for i in zip(x,y)]
        return scatter(x_list=x_list, template=template)
    else:
        x_list = [np.array([i]).transpose() for i in zip(x,y,z)]
        template['surfaceaxis'] = axis
        return scatter(x_list=x_list, template=template)


def polytope(A: np.ndarray,
             b: np.ndarray,
             vertices: List[np.ndarray] = None,
             template: Dict = None,
             **kwargs) -> List[Union[plt.Scatter, plt.Scatter3d]]:
    """Return a 2d or 3d polytope defined by the list of halfspaces Ax <= b.

    Returns a plt.Scatter polygon in the case of a 2d polytope and returns a
    list of plt.Scatter3d polygons in the case of a 3d polytope. The vertices
    of the halfspace intersection can be provided to improve computation time.

    Note: keyword arguments given outside of template are given precedence.

    Args:
        A (np.ndarray): LHS coefficents of halfspaces
        b (np.ndarray): RHS coefficents of halfspaces
        vertices (List[np.ndarray]): Vertices of the halfspace intersection.
        template (Dict): Dictionary of scatter attributes. Defaults to None.
        *kwargs: Arbitrary keyword arguments for plt.Scatter or plt.Scatter3d.

    Returns:
        List[Union[plt.Scatter, plt.Scatter3d]]: 2d or 3d polytope.
    """
    if vertices is None:
        vertices = polytope_vertices(A,b)

    if A.shape[1] == 2:
        return [polygon(x_list=vertices,
                        template=template,
                        **kwargs)]
    if A.shape[1] == 3:
        facets = polytope_facets(A, b, vertices=vertices)
        polygons = []
        for facet in facets:
            if len(facet) > 0:
                polygons.append(polygon(x_list=facet,
                                        template=template,
                                        **kwargs))
        return polygons


def tree_positions(T:nx.classes.graph.Graph,
                   root:Union[str,int]) -> Dict[int, List[float]]:
    """Get positions for every node in the tree T with the given root.

    Args:
        T (nx.classes.graph.Graph): Tree graph.
        root (Union[str,int]): Root of the tree graph

    Returns:
        Dict[int, List[float]]: Dictionary from nodes in T to positions.
    """
    PAD = 0.1
    HORIZONTAL_SPACE = 0.2

    position = {}
    position[root] = (0.5, 1-PAD)  # root position

    node_to_level = nx.single_source_shortest_path_length(T, root)
    level_count = max(node_to_level.values()) + 1
    levels = {}
    for l in range(level_count):
        levels[l] = [i for i in node_to_level if node_to_level[i] == l]

    level_heights = np.linspace(1.1, -0.1, level_count + 2)[1:-1]
    for l in range(1, level_count):
        # If there are more than 5 nodes in level, spread evenly across width;
        # otherwise, try to put nodes under their parent.
        if len(levels[l]) <= 4:
            # get parents of every pair of children in the level
            children = {}
            for node in levels[l]:
                parent = [i for i in list(T.neighbors(node)) if i < node][0]
                if parent in children:
                    children[parent].append(node)
                else:
                    children[parent] = [node]

            # initial attempt at positioning
            pos = {}
            for parent in children:
                x = position[parent][0]
                d = max((1/2)**(l+1), HORIZONTAL_SPACE / 2)
                pos[children[parent][0]] = [x-d, level_heights[l]]
                pos[children[parent][1]] = [x+d, level_heights[l]]

            # perturb if needed
            keys = list(pos.keys())
            x = [p[0] for p in pos.values()]
            n = len(x) - 1
            while any([x[i+1]-x[i]+0.05 < HORIZONTAL_SPACE for i in range(n)]):
                for i in range(len(x)-1):
                    if abs(x[i+1] - x[i]) < HORIZONTAL_SPACE:
                        shift = (HORIZONTAL_SPACE - abs(x[i+1] - x[i]))/2
                        x[i] -= shift
                        x[i+1] += shift

            # shift to be within width
            x[0] = x[0] + (max(PAD - x[0], 0))
            for i in range(1,len(x)):
                x[i] = x[i] + max(HORIZONTAL_SPACE - (x[i] - x[i-1]), 0)

            x[-1] = x[-1] - (max(x[-1] - (1-PAD), 0))
            for i in reversed(range(len(x)-1)):
                x[i] = x[i] - max(HORIZONTAL_SPACE - (x[i+1] - x[i]), 0)

            # update the position dictionary with new x values
            for i in range(len(x)):
                pos[keys[i]][0] = x[i]

            # set position
            for node in pos:
                position[node] = pos[node]
        else:
            level_widths = np.linspace(-0.1, 1.1, len(levels[l]) + 2)[1:-1]
            for j in range(len(levels[l])):
                position[(levels[l][j])] = (level_widths[j], level_heights[i])

    return position


def plot_tree(fig:Figure,
              T:nx.classes.graph.Graph,
              root:Union[str,int],
              row:int = 1,
              col:int = 2):
    """Plot the tree on the figure.

    This function assumes the type of subplot at the given row and col is of
    type scatter plot and has both x and y range of [0,1].

    Args:
        fig (Figure): The figure to which the tree should be plotted.
        T (nx.classes.graph.Graph): Tree to be plotted.
        root (Union[str,int]): Root node of the tree.
        row (int, optional): Subplot row of the figure. Defaults to 1.
        col (int, optional): Subplot col of the figure. Defaults to 2.
    """
    nx.set_node_attributes(T, tree_positions(T, root), 'pos')

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
    fig.add_trace(trace=edge_trace, row=row, col=col)

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
