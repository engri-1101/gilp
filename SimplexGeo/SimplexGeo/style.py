import numpy as np
from simplex import LP
import itertools
import plotly.graph_objects as plt
from scipy.spatial import ConvexHull
from typing import List, Dict, Union

# CHANGE BACK .style

"""Provides a higher level interface with plotly in a desired style.

Functions:
    format: Round to specified precision. Return string with trailing zeros and decimal removed.
    linear_string: Return a string representation of the given linear combination.
    equation_string: Return a string representation of the given equation.
    label: Return a styled string representation of the given dictionary.
    table: Return a styled table trace with given headers and content.
    set_axis_limits: Set the axis limits for the given figure.
    get_axis_limits: Return the axis limits for the given figure.
    vector: Return a styled 2d or 3d vector trace from tail to head.
    scatter: Return a styled 2d or 3d scatter trace of the given points and labels.
    line: Return a 2d line trace in the desired style.
    intersection: Return the points where Ax = b intersects Dx <= e.
    equation: Return a styled 2d or 3d trace representing the given equation.
    order: Return the given points in the correct order for drawing a 2d or 3d polygon.
    polygon: Return a styled 2d or 3d polygon trace defined by the given points.
"""

def format(num:Union[int,float],precision:int=3) -> str:
    """Round to specified precision. Return string with trailing zeros and decimal removed."""
    return ('%.*f' % (precision, num)).rstrip('0').rstrip('.')

def linear_string(A:np.ndarray, indices:List[int], const:float=None) -> str:
    """Return a string representation of the given linear combination.
    
    The const argument identifies if A[0] is a constant or not."""
    def sign(num:float) -> str:
        return {-1 : ' - ', 0 : ' + ', 1: ' + '}[np.sign(num)]
    s = ''
    if const is not None: s+=format(const)
    for i in range(len(indices)):
        if i == 0: 
            if const is None: s+=format(A[0])+'x<sub>'+format(indices[0])+'</sub>'
            else: s+=sign(A[0])+format(abs(A[0]))+'x<sub>'+format(indices[0])+'</sub>'
        else: s+=format(abs(A[i]))+'x<sub>'+format(indices[i])+'</sub>'
        if i is not len(indices)-1: s+=sign(A[i+1])
    return s

def equation_string(A:np.ndarray,b:float,comp:str='≤') -> str:
    """Return a string representation of the given equation.
    
    The equation is assumed to be in standard form: Ax 'comp' b."""
    return linear_string(A,list(range(1,len(A)+1)))+' '+comp+' '+format(b)

def label(dic:Dict[str,Union[float,list]]) -> str:
    """Return a styled string representation of the given dictionary."""
    entries = []
    for key in dic.keys():
        s = '<b>'+key+'</b>: '
        value = dic[key]
        if type(value) is float: s += format(value)
        if type(value) is list: s += '(%s)'%', '.join(map(str,[format(i) for i in value]))
        entries.append(s)
    return '%s'%'<br>'.join(map(str,entries))

def table(header:List[str],content:Union[np.ndarray,List[str]],style:str) -> plt.Table:
    """Return a styled table trace with given headers and content."""
    if style not in ['canonical','dictionary']:
        raise ValueError("Invalid style. Currently supports 'canonical' and 'dictionary'")
    canon_args = dict(header= dict(values=header, height=30,fill=dict(color='white'),
                                   line=dict(color='black',width=1), font=dict(size=13)),
                      cells= dict(values=content, height=25,fill=dict(color='white'),
                                   line=dict(color='black',width=1), font=dict(size=13)))
    dict_args = dict(header= dict(values=header, height=25,fill=dict(color='white'),
                                   align=['left','right','left'], font=dict(size=14),
                                   line=dict(color='white',width=1)),
                     cells= dict(values=content, height=25,fill=dict(color='white'),
                                   align=['left','right','left'], font=dict(size=14),
                                   line=dict(color='white',width=1)),
                     columnwidth=[0.3,0.07,0.63])
    return plt.Table({'canonical':canon_args, 'dictionary':dict_args}[style])

def set_axis_limits(fig:plt.Figure, x_list:List[np.ndarray]):
    """Set the axes limits of fig such that all points in x are visible.
    
    Given a set of nonnegative 2 or 3 dimensional points, set the axes 
    limits such all points are visible within the plot window. 
    
    Args:
        fig (plt.Figure): The plot for which the axes limits will be set
        List[np.ndarray] : A set of nonnegative 2 or 3 dimensional points
        
    Raises:
        ValueError: The points in x must be 2 or 3 dimensional
        ValueError: The points in x must all be nonnegative
    """
    n = len(x_list[0]) 
    if not n in [2,3]:
        raise ValueError('The points in x must be in 2 or 3 dimensions')
    pts = [list(x[:,0]) for x in x_list]
    if not all((i >= 0 for i in pt) for pt in pts):
        raise ValueError('The points in x must all be nonnegative')
    limits = [max(i)*1.3 for i in list(zip(*pts))]
    if n == 2: 
        x_lim, y_lim = limits
        pts = [np.array([[x_lim],[y_lim]])]
    if n == 3: 
        x_lim, y_lim, z_lim = limits
        pts = [np.array([[x_lim],[y_lim],[z_lim]])]
    fig.update_layout(scene = dict(xaxis = dict(range=[0,x_lim]),
                                  yaxis = dict(range=[0,y_lim])))
    if n == 3: fig.layout.scene.zaxis= dict(range=[0,z_lim])
    fig.add_trace(scatter(pts,'bfs')) # Fixes bug with changing axes

def get_axis_limits(fig:plt.Figure,n:int) -> List[float]:
    """Return the axis limits for the given figure."""
    if n not in [2,3]:
        raise ValueError('Can only retrieve 2 or 3 axes')
    x_lim = fig.layout.scene.xaxis.range[1]
    y_lim = fig.layout.scene.yaxis.range[1] 
    if n == 2: return x_lim, y_lim
    z_lim = fig.layout.scene.zaxis.range[1] 
    return x_lim, y_lim, z_lim

def vector(tail:np.ndarray,head:np.ndarray) -> Union[plt.Scatter,plt.Scatter3d]:
    """Return a styled 2d or 3d vector trace from tail to head."""
    pts = list(zip(*[tail[:,0],head[:,0]]))
    if len(pts) == 2: x,y = pts; z = None
    if len(pts) == 3: x,y,z = pts
    args = dict(x=x, y=y, mode='lines',line=dict(width=6,color='red'),
                opacity=1, hoverinfo='skip', showlegend=False, visible=False)
    if z is None: return plt.Scatter(args)
    else:
        args['z'] = z
        return plt.Scatter3d(args)

def scatter(x_list:List[np.ndarray],style:str,lbs:List[str]=None) -> plt.Scatter:
    """Return a styled 2d or 3d scatter trace of the given points and labels."""
    if style not in ['bfs','initial_sol']:
        raise ValueError("Invalid style. Currently supports 'bfs' and 'initial_sol'")
    pts = list(zip(*[list(x[:,0]) for x in x_list]))
    if len(pts) == 2: x,y = pts; z = None
    if len(pts) == 3: x,y,z = pts
    bfs_args = dict(x=x, y=y, text=lbs, mode='markers', 
                    marker=dict(size=20, color='gray', opacity=0.00001),
                    showlegend=False, hoverinfo='text', 
                    hoverlabel=dict(bgcolor='#FAFAFA', bordercolor='#323232',
                                font=dict(family='Arial',color='#323232')))
    init_args = dict(x=x, y=y, mode='markers', hoverinfo='skip', showlegend=False,
                     marker=dict(size=5, color='red', opacity=1))
    args = {'bfs' : bfs_args, 'initial_sol' : init_args}[style]
    if z is None: 
        return plt.Scatter(args)
    else:
        args['z'] = z
        return plt.Scatter3d(args)

def line(x_list:List[np.ndarray],style:str,lb:str=None,i=[0]) -> plt.Scatter:
    """Return a 2d line trace in the desired style."""
    if style not in ['constraint', 'isoprofit']:
        raise ValueError("Invalid style. Currently supports 'constraint' and 'isoprofit'")
    x,y = list(zip(*[list(x[:,0]) for x in x_list]))

    colors= ['#173D90', '#1469FE', '#65ADFF', '#474849', '#A90C0C', '#DC0000']
    if style == 'constraint': i[0] = i[0]+1 if i[0]+1<6 else 0

    con_args = dict(x=x, y=y, mode='lines', hoverinfo='skip', visible= True,
                    line= dict(color=colors[i[0]], width=2, dash='15,3,5,3'),
                    showlegend= True, name=lb)
    iso_args = dict(x=x, y=y, mode='lines', hoverinfo='skip', visible=False,
                    line= dict(color='red', width=4, dash=None), showlegend= False)
    return plt.Scatter({'constraint' : con_args, 'isoprofit' : iso_args}[style])

def intersection(A:np.ndarray, b:float, D:np.ndarray, e:float) -> List[np.ndarray]:
    """Return the points where Ax = b intersects Dx <= e."""
    n,m = len(A),len(D)
    if n not in [2,3]:
        raise ValueError('Only supports equations in 2 or 3 variables')
    lp = LP(np.vstack((D,A)),np.vstack((e,b)),np.ones((n,1)))
    pts = []
    for B in itertools.combinations(range(n+m),m+1):
        pt = lp.get_basic_feasible_sol(B)
        if pt is not None: pts.append(pt)
    return [pt[0:n,:] for pt in pts]

def equation(fig:plt.Figure, A:np.ndarray, b:float, style:str, lb:str=None) -> Union[plt.Scatter,plt.Scatter3d]:
    """Return a styled 2d or 3d trace representing the given equation."""
    n = len(A)
    if n not in [2,3]:
        raise ValueError('Only supports equations in 2 or 3 variables')
    pts = intersection(A,b,np.identity(n),np.array([get_axis_limits(fig,n)]).transpose())
    if n == 2: return line(pts,style,lb)
    if n == 3: return polygon(pts,style,lb)

def order(x_list:List[np.ndarray]) -> List[List[float]]:
    """Correctly order the points for drawing a polygon."""
    n,m = x_list[0].shape
    if not m == 1:
        raise ValueError('Points must be in vector form')
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
            b_3 = np.cross(b_1,b_2)
            T = np.linalg.inv(np.array([b_1,b_2,b_3]).transpose())
            x_list = np.array([list(np.round(np.dot(T,x),7)[0:2,0]) for x in x_list])
            hull = ConvexHull(x_list) 
            pts = list(zip(pts[hull.vertices,0], pts[hull.vertices,1], pts[hull.vertices,2]))
            pts.append(pts[0])
            return list(zip(*pts)) 
    else:
        return list(zip(*pts)) 

def polygon(x_list:List[np.ndarray],style:str,lb:str=None) -> plt.Scatter:
    """Return a styled 2d or 3d polygon trace defined by the given points."""
    styles = ['region','constraint','isoprofit_in', 'isoprofit_out']
    if style not in styles:
        raise ValueError("Invalid style. Currently supports " + styles)
    components = order(x_list)
    
    if len(components) == 2:
        x,y = components
        return plt.Scatter(x=x, y=y, mode='lines', fill='toself', fillcolor = '#1469FE',
                           line= dict(width=2, color='#00285F'), opacity=0.3, 
                           showlegend=False, hoverinfo='skip')
    if len(components) == 3:
        pts = order(x_list)
        # TODO: Describe this issue..
        axis=2 
        for i in range(3):
            if len(set(pts[i])) == 1: axis=i
        x,y,z = pts
        region_args = dict(x=x, y=y, z=z, surfaceaxis=axis, mode="lines", hoverinfo='skip',
                           surfacecolor= '#1469FE', line= dict(width=5, color='#173D90'), 
                           opacity= 0.2, visible=True, showlegend= False)
        con_args = dict(x=x, y=y, z=z, surfaceaxis=axis, mode="none", hoverinfo='skip', name=lb,
                        surfacecolor= 'gray', opacity= 0.5, visible='legendonly', showlegend= True)
        iso_in_args = dict(x=x, y=y, z=z, surfaceaxis=axis, mode="lines+markers", hoverinfo='skip',
                        surfacecolor= 'red', marker= dict(size=5,color='red',opacity=1), 
                        line = dict(width=5, color='red'), opacity= 1, 
                        visible=False, showlegend= False)
        iso_out_args = dict(x=x, y=y, z=z, surfaceaxis=axis, mode="none", hoverinfo='skip',
                            surfacecolor= 'gray', opacity= 0.3, 
                            visible=False, showlegend= False,)
        return plt.Scatter3d({styles[0]:region_args, styles[1]:con_args, 
                              styles[2]:iso_in_args, styles[3]:iso_out_args}[style])