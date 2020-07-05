import numpy as np
import plotly.graph_objects as plt
from scipy.spatial import ConvexHull
from typing import List, Union, Dict

"""Provides a higher level interface with plotly in a desired style"""

def format(x:Union[int,float],precision:int=3) -> str:
    """Return x rounded to 3 decimal places with trailing zeros and decimal removed."""
    return ('%.*f' % (precision, x)).rstrip('0').rstrip('.')

def equation_string(A:np.array,b:float,comp:str='≤') -> str:
    """Return a string representation of the equation Ax 'type' b"""
    lb = format(A[0])+'x<sub>1</sub> + '+format(A[1])+'x<sub>2</sub> '
    if len(A) == 3: lb = lb + '+ '+format(A[2])+'x<sub>3</sub> '
    return lb+comp+' '+format(b)

def label(dic:Dict[str,Union[float,list]]) -> str:
    """Return a string form of the given dictionary"""
    entries = []
    for key in dic.keys():
        s = '<b>'+key+'</b>: '
        value = dic[key]
        if type(value) is float: s += format(value)
        if type(value) is list: s += '(%s)'%', '.join(map(str,[format(i) for i in value]))
        entries.append(s)
    return '%s'%'<br>'.join(map(str,entries))

def table(header=List[str],content=List[Union[int,float]]) -> plt.Table:
    """Return a default styled table trace"""
    content= [[format(i,1) for i in row] for row in content] # format content
    return plt.Table(header= dict(values=header, height=30,fill=dict(color='#5B8ECC'),
                                  font=dict(color='white', size=14),
                                  line=dict(color='white',width=2)),
                     cells= dict(values=content, height=25,fill=dict(color='#E9F3FF'),
                                  line=dict(color='white',width=2)))

# Adjust docstring for both plane and line -- more general and have specific styles
def line(x:List[float],y:List[float],con:bool,lb:str=None,i=[0]) -> plt.Scatter:
    """Return a default styled line defined by the given points 

    Args:
        x,y (List[float]): The respective components of the points defining the line
        con (bool): True if the "constraint" style should be used. False otherwise.
        lb (str, optional): The label in the legend. Defaults to None.
        i (list, optional): Used to iterate through constraint colors.

    Returns:
        plt.Scatter: A correctly styled line trace
    """
    colors= ['#173D90', '#1469FE', '#65ADFF', '#474849', '#A90C0C', '#DC0000']
    if con: i[0] = i[0]+1 if i[0]+1<6 else 0
    l = plt.Scatter(x=x, y=y, mode='lines', hoverinfo='skip', visible= con,
                    line= dict(color=colors[i[0]], width=2, dash='15,3,5,3') if con
                          else dict(color='red', width=4, dash=None),
                    name= lb, showlegend= con)
    return l

def order(x_list:List[np.array]) -> List[List[float]]:
    """Correctly order the points for drawing a polygon.

    Args:
        x_list (List[np.array]): A list of 2 or 3 dimensional points (vectors)

    Raises:
        ValueError: Points must be in vector form
        ValueError: Points must be 2 or 3 dimensional

    Returns:
        List[Tuple[float]]: An ordered list of each component
    """
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
            pts = [[pt[0]+0.0001,pt[1]+0.0001,pt[2]] if not pt[2] == 0 else pt for pt in pts]
            return list(zip(*pts)) 
    else:
        return list(zip(*pts)) 

def polygon(x_list:List[np.array]) -> plt.Scatter:
    """Return a polygon trace defined by the given points"""
    x,y = order(x_list)
    return plt.Scatter(x=x, y=y, mode='markers', marker=dict(size=0,opacity=0.00001), 
                       fill='toself', fillcolor = '#1469FE',opacity=0.3,
                       showlegend=False, hoverinfo='skip')

# Adjust docstring for both plane and line -- more general and have specific styles
def plane(x_list:List[np.array],face:bool,lb:str=None) -> plt.Scatter3d:
    """Return a default styled plane defined by the given points 

    Args:
        x_list (List[np.array]): The points defining the plane
        face (bool): True if the "polytope face" style should be used. False otherwise.
        lb (str, optional): The label in the legend. Defaults to None.

    Returns:
        plt.Scatter3d: A correctly styled plane trace
    """
    x,y,z = order(x_list)
    args = dict(x=x, y=y, z=z, surfaceaxis=2, mode="markers+lines", hoverinfo='skip',
                    surfacecolor= '#1469FE' if face else 'red',
                    marker= dict(size=0,color='white',opacity=0) if face
                            else dict(size=5,color='red',opacity=1),
                    line= dict(width=5, color='#173D90' if face else 'red'),
                    opacity= 0.2 if face else 1.0, visible= face,
                    showlegend= face, name= lb if face else None)

    p=plt.Scatter3d(args)
    return p

def scatter(x_list:List[np.array],lbs:List[str]) -> plt.Scatter:
    """Return a scatter trace of the given points with the given labels"""

    pts = list(zip(*[list(x[:,0]) for x in x_list]))
    if len(pts) == 2: x,y = pts; z = None
    if len(pts) == 3: x,y,z = pts

    args = dict(x=x, y=y, text=lbs, mode='markers', 
                marker=dict(size=20, color='gray', opacity=0.00001),
                showlegend=False, hoverinfo='text', 
                hoverlabel=dict(bgcolor='#FAFAFA', bordercolor='#323232',
                                font=dict(family='Arial',color='#323232')))
    if z is None: 
        return plt.Scatter(args)
    else:
        args['z'] = z
        return plt.Scatter3d(args)