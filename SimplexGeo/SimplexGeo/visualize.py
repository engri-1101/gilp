import numpy as np
from typing import List, Tuple
import itertools
from scipy.linalg import solve
import plotly.graph_objects as plt
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from simplex import LP, simplex, invertible
from style import line, plane, table, scatter, polygon,label, equation_string

def set_axis_limits(fig:plt.Figure, x:List[List[float]]):
    """Set the axes limits of fig such that all points in x are visible.
    
    Given a set of nonnegative 2 or 3 dimensional points, set the axes 
    limits such all points are visible within the plot window. 
    
    Args:
        fig (plt.Figure): The plot for which the axes limits will be set
        x (List[List[float]]): A set of nonnegative 2 or 3 dimensional points
        
    Raises:
        ValueError: The points in x must be 2 or 3 dimensional
        ValueError: The points in x must all be nonnegative
    """
    
    n = len(x[0]) 
    if not n in [2,3]:
        raise ValueError('The points in x must be in 2 or 3 dimensions')
    if not (x >= np.zeros((len(x),len(x[0])))).all():
        raise ValueError('The points in x must all be nonnegative')
    limits = [max(i)*1.3 for i in list(zip(*x))]
    if n == 2: x_lim, y_lim = limits
    if n == 3: x_lim, y_lim, z_lim = limits
    fig.update_layout(scene = dict(xaxis = dict(range=[0,x_lim]),
                                  yaxis = dict(range=[0,y_lim])))
    if n == 3: fig.layout.scene.zaxis= dict(range=[0,z_lim])

def get_axis_limits(fig:plt.Figure,n:int) -> List[float]:
    """Get the axis limits for the chosen figure

    Args:
        fig (plt.Figure): The figure whose axes limits are in question
        n (int): The number of axes

    Raises:
        ValueError: There must be 2 or 3 axes   

    Returns:
        List[float]: List of axes limits 
    """

    if n not in [2,3]:
        raise ValueError('There must be 2 or 3 axes')
    x_lim = fig.layout.scene.xaxis.range[1]
    y_lim = fig.layout.scene.yaxis.range[1] 
    if n == 2: return x_lim, y_lim
    z_lim = fig.layout.scene.zaxis.range[1] 
    return x_lim, y_lim, z_lim

# TODO: Consider unbounded scenario
def plot_feasible_region(fig:plt.Figure, lp:LP):
    """Render the LP's feasible region on the given figure.
    
    Depict each of the LP's basic feasible solutions and set the plot axes
    such that all basic feasible solutions are visible. Shade the feasible
    region for 2d plot and construct feasible region polytope for 3d plots.
    
    Args:
        fig (plt.Figure): The plot on which the feasible region will be rendered
        lp (LP): The LP which will have its feasible region rendered
        
    Raises:
        ValueError: The LP must have 2 or 3 decision variables
    """
    
    n,m,A,b,c = lp.get_inequality_form()
    if not n in [2,3]:
        raise ValueError('The LP must have 2 or 3 decision variables')
    bfs, bases, values = lp.get_basic_feasible_solns()
    set_axis_limits(fig, [list(x[list(range(n)),0]) for x in bfs])

    lbs = [label(dict(x=list((bfs[i])[0:n,0]), 
                      Obj=float(values[i]), 
                      BFS=list((bfs[i])[:,0]), 
                      B=list(np.array(bases[i])+1))) for i in range(len(bfs))]
    pts = [i[0:n,:] for i in bfs]
    fig.add_trace(scatter(pts,lbs))

    if n == 2:
        fig.add_trace(polygon(pts))
    if n == 3:
        for i in range(n+m):
            pts = [bfs[j][0:n,:] for j in range(len(bfs)) if i not in bases[j]]
            if i < 3: 
                G = [-1 if j == i else 0 for j in range(3)]
                lb = '('+str(i+1)+') '+equation_string(G,0)
            else: 
                lb = '('+str(i+1)+') '+equation_string(A[i-3],b[i-3])
            fig.add_trace(plane(pts,True,lb))  

def xy_from_equation(fig:plt.Figure, A:np.array,b:float) -> Tuple[List[float],List[float]]:
    """Get the x and y components of the pts making up Ax <= b
    
    Args:
        fig (plt.Figure): The figure on which the constraint will be plotted
        A (np.ndarray): A list of 2 coefficents for the LHS of the constraint
        b (float): Represents the RHS of the constraint
    
    Raises:
        ValueError: A must be 2 dimensional
        ValueError: A must be nonzero
    """
    if len(A) is not 2:
        raise ValueError('A must be 2 dimensional')
    if len(np.nonzero(A)[0]) == 0:
        raise ValueError('A must be nonzero')
    
    x_lim, y_lim = get_axis_limits(fig,2)
    if A[1] == 0:
        x_loc = b/A[0]
        return [x_loc,x_loc],[0,y_lim]
    else:
        if A[0] == 0: 
            x = np.linspace(0, x_lim, 2)
        else:
            # possible window intersection points
            pts = [(0,b/A[1]),(b/A[0],0),(x_lim,(b-A[0]*x_lim)/A[1]),((b-A[1]*y_lim)/A[0],y_lim)]
            pts = [pt for pt in pts if 0 <= pt[0] <= x_lim and 0 <= pt[1] <= y_lim] # in window
            if len(pts) < 2: return # constraint does not appear in window
            pts = list(set(pts)) # get rid of repeats
            if len(pts) == 1: 
                x = np.linspace(pts[0][0],pts[0][0],2)
            else:
                x = np.linspace(min(pts[0][0],pts[1][0]),max(pts[0][0],pts[1][0]),2)
        y = (b - A[0]*x)/(A[1])
    return x,y

def plot_constraints(fig:plt.Figure, lp:LP):
    """Render each of the LP's constraints on the given plot.
    
    Args:
        fig (plt.Figure): The plot on which the constraints will be rendered
        lp (LP): The LP which will have its constraints rendered (2 decision variables)
        
    Raises:
        ValueError: The LP must have 2 or 3 decision variables
    """
    n,m,A,b,c = lp.get_inequality_form()
    if not n in [2,3]:
        raise ValueError('The LP must have 2 or 3 decision variables')
    if n == 2:
        for i in range(m):
            lb = '('+str(i+3)+') '+equation_string(A[i],b[i][0])
            x,y = xy_from_equation(fig,A[i],b[i][0]) 
            fig.add_trace(line(x,y,True,lb))

def plot_lp(lp:LP) -> plt.Figure:
    """Render a visualization of the given LP.
    
    For some LP with 2 or 3 decision variables, visualize the feasible region,
    basic feasible solutions, and constraints.
    
    Args:
        lp (LP): The LP which will be visualized
        
    Returns:
        fig (plt.Figure): A figure which displays the visualization
    
    Raises:
        ValueError: The LP must have 2 or 3 decision variables
    """
    
    n,m,A,b,c = lp.get_inequality_form()
    if not lp.n in [2,3]:
        raise ValueError('The LP must have 2 or 3 decision variables')
        
    if n == 2:
        plot_type = 'scatter'
        scene= dict(xaxis= dict(title= 'x<sub>1</sub>'),
                    yaxis= dict(title= 'x<sub>2</sub>'))
    if n == 3:
        plot_type = 'scene'
        scene= dict(xaxis= dict(title= 'x<sub>1</sub>'),
                    yaxis= dict(title= 'x<sub>2</sub>'),
                    zaxis= dict(title= 'x<sub>3</sub>'))
        
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2, specs=[[{"type": plot_type},{"type": "table"}]])
    fig.update_layout(title=dict(text= "<b>Simplex Geo</b>", x=0, y=0.95, xanchor= 'left', yanchor='bottom',
                                 font=dict(size=18,color='#00285F')), plot_bgcolor='#FAFAFA',
                     scene= scene, legend=dict(title=dict(text='<b>Constraint(s)</b>',font=dict(size=14)), font=dict(size=13),
                     x=0.4, y=1, xanchor='left', yanchor='top'), font=dict(family='Arial',color='#323232'),
                     width=950,height=500,margin=dict(l=0, r=0, b=0, t=50),
                     xaxis=dict(title='x<sub>1</sub>',rangemode='tozero',fixedrange=True,
                                gridcolor='#CCCCCC',gridwidth=1,linewidth=2,linecolor='#4D4D4D',tickcolor='#4D4D4D',ticks='outside'),
                     yaxis=dict(title='x<sub>2</sub>',rangemode='tozero',fixedrange=True,
                                gridcolor='#CCCCCC',gridwidth=1,linewidth=2,linecolor='#4D4D4D',tickcolor='#4D4D4D',ticks='outside'))
    
    plot_feasible_region(fig,lp)
    plot_constraints(fig,lp)   

    return fig

# TODO: scale 2d and 3d arrow heads correctly
def add_path(fig:plt.Figure, path:List[List[float]]):
    """Render the path of ordered points on the given plot.
    
    Draw arrows between consecutive points to trace the path. 
    
    Args:
        fig (plt.Figure): The plot on which the path will be rendered
        path (List[List[float]]): The path of points to be traced
        
    Returns:
        arrows (List[Arrow]): An ordered list of Arrow glyphs (N/A for 3d)
        
    Raises:
        ValueError: The points in the path must be 2 or 3 dimensional
    """
    arrows = []
    n = len(path[0])
    if not n in [2,3]:
        raise ValueError('The points in the path must be 2 or 3 dimensional')
    for i in range(len(path)-1):
        a = np.round(path[i],7)
        b = np.round(path[i+1],7)
        p = list(zip(*[a,b]))
        ratio = 1/20
        if n == 2:
            x_lim, y_lim = get_axis_limits(fig,2)
            line = plt.Scatter(x=p[0], y=p[1],mode="lines",
                                opacity=1,hoverinfo='skip',showlegend=False,
                                line=dict(width=6,color='#FF0000'),visible=False)
            d = (b-a)*ratio
            arrow = ff.create_quiver(hoverinfo='skip',showlegend=False,visible=False,
                                     x=[(b-d)[0]], y=[(b-d)[1]], u=[d[0]], v=[d[1]],
                                     scale = 1, scaleratio = 1).data[0]
            arrow.line.color='red'
            arrow.line.width=4
            fig.add_trace(line)
            fig.add_trace(arrow)
            s = len(fig.data)
            arrows.append((s-2,s-1))
        if n == 3:
            x_lim, y_lim, z_lim = get_axis_limits(fig,3)
            line = plt.Scatter3d(x=p[0], y=p[1], z=p[2],mode="lines",
                                opacity=1,hoverinfo='skip',showlegend=False,
                                line=dict(width=10,color='#FF0000'),visible=False)
            d = (b-a)/np.linalg.norm(b-a)*3
            head = plt.Cone(hoverinfo='skip',showscale=False,colorscale=['#FF0000','#FF0000'],
                           x=[b[0]], y=[b[1]], z=[b[2]], u=[d[0]], v=[d[1]], w=[d[2]],visible=False)
            fig.add_trace(line)
            fig.add_trace(head)
            s = len(fig.data)
            arrows.append((s-2,s-1))
    return arrows

def plot_intersection(fig:plt.Figure, lp:LP, G:np.array, h:float):
    """Plot the intersection of the LP's feasible region and Gx = h

    Args:
        fig (plt.Figure): The figure on which to plot the intersection
        lp (LP): The LP
        G (np.array): The LHS coefficents of the standard form plane equation
        h (float): The RHS of the standard form plane equation
    """
    n,m,A,b,c = lp.get_inequality_form()
    lp = LP(np.vstack((A,G)),np.vstack((b,h)),c)
    pts = []
    for B in itertools.combinations(range(n+m),m+1):
        pt = lp.get_basic_feasible_sol(B)
        if pt is not None: pts.append(pt)
    fig.add_trace(plane([pt[0:n,:] for pt in pts],False)) 

def add_isoprofits(fig:plt.Figure, lp:LP) -> Tuple[List[int],List[float]]:
    """Render all the isoprofit lines/planes which can be toggled over.
    
    Args:
        fig (plt.Figure): The figure to which these isoprofits lines/planes are added
        lp (LP): The LP for which the isoprofit lines are being generated
        
    Returns:
        isoprofit_line_IDs (List[int]): The ID of all the isoprofit lines/planes
        objectives (List[float]): The corresponding objectives
    """
    n,m,A,b,c = lp.get_inequality_form()
    isoprofit_line_start = len(fig.data)
    x_lim, y_lim = get_axis_limits(fig,2)
    if n == 2:
        obj1=(simplex(LP(np.identity(2),np.array([[x_lim],[y_lim]]),c))[1])
        obj2=-(simplex(LP(np.identity(2),np.array([[x_lim],[y_lim]]),-c))[1])
        objectives = list(np.round(np.linspace(min(obj1,obj2),max(obj1,obj2),25),2))
        objectives.append(simplex(lp)[1])
        objectives.sort()
        for obj in objectives:
            x,y = xy_from_equation(fig,c[:,0],obj) 
            fig.add_trace(line(x,y,False))
    if n == 3:
        objectives = np.round(np.linspace(0,simplex(lp)[1]),2)
        for obj in objectives:
            plot_intersection(fig,lp,c[:,0],obj)
    isoprofit_line_end = len(fig.data)
    isoprofit_line_IDs = list(range(isoprofit_line_start,isoprofit_line_end))
    fig.data[isoprofit_line_IDs[0]].visible=True
    return isoprofit_line_IDs, objectives

def tableau_table(T:np.ndarray) -> plt.Table:
    """Return a table trace representing the given tableau
    
    Args:
        T (np.ndarray): An array representing the tableau 
    
    Returns:
        table (plt.Table): A trace for the tableau
    """
    header = []
    for j in range(len(T[0])):
        if j == 0: header.append('z<sub></sub>')
        elif j == len(T[0])-1: header.append('RHS<sub></sub>')
        else: header.append("x<sub>"+str(j)+"</sub>")
    content = list(T.transpose())
    return table(header,content)

# TODO: Arrow heads set to always be invisible
def simplex_visual(lp:LP,rule:str='dantzig',init_sol:np.ndarray=None,iter_lim:int=None):
    """Return a plot showing the geometry of simplex.
    
    Args:
        lp (LP): The LP on which to run simplex
        rule (str): The pivot rule to be used at each simplex iteration
        init_sol (np.ndarray): An n length vector 
        iter_lim (int): The maximum number of simplex iterations to be run"""
    
    n,m,A,b,c = lp.get_inequality_form()
    
    fig = plot_lp(lp)
    opt, value, path, bases = simplex(lp,rule,init_sol)
    arrows = add_path(fig,[i[list(range(n)),0] for i in path])
    tables = [tableau_table(lp.get_tableau(basis)) for basis in bases]
    
    table_IDs = []
    for i in range(len(tables)):
        table = tables[i]
        if not i == 0: table.visible=False
        fig.add_trace(table,row=1,col=2)
        table_IDs.append(len(fig.data)-1)
    
    isoprofit_line_IDs, objectives = add_isoprofits(fig,lp)
        
    iter_steps = []
    for i in range(len(arrows)+1):
        visible = [fig.data[j].visible for j in range(len(fig.data))]

        for j in range(len(table_IDs)):
            visible[table_IDs[j]] = False
        visible[table_IDs[i]] = True 
        
        for j in range(len(arrows)+1):
            if j < len(arrows):
                visible[arrows[j][0]] = True if j < i else False
                visible[arrows[j][1]] = False if j < i else False # TODO heads always invisible
        step = dict(method="update", label = i, args=[{"visible": visible}])
        iter_steps.append(step)

    iter_slider = dict(x=0.6, xanchor="left", y=0.22, yanchor="bottom", len= 0.4, lenmode='fraction',
                    pad={"t": 50}, active=0, currentvalue={"prefix":"Iteration: "}, tickcolor= 'white',
                    ticklen = 0, steps=iter_steps)
    
    iso_steps = []
    for i in range(len(isoprofit_line_IDs)):
        visible = [fig.data[k].visible for k in range(len(fig.data))]
        for j in isoprofit_line_IDs:
            visible[j]= False
        visible[isoprofit_line_IDs[i]]= True
        step = dict(method="update", label = objectives[i], args=[{"visible": visible}])
        iso_steps.append(step)
    iso_slider = dict(x=0.6, xanchor="left", y=0.02, yanchor="bottom", len= 0.4, lenmode='fraction',
                    pad={"t": 50}, active=0, currentvalue={"prefix":"Objective: "}, tickcolor= 'white',
                    ticklen = 0, steps=iso_steps)

    fig.update_layout(sliders=[iter_slider,iso_slider])
    return fig