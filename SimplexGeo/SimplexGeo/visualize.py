import numpy as np
from typing import List, Tuple
import itertools
from scipy.linalg import solve
import plotly.graph_objects as plt
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from simplex import LP, simplex, invertible
from style import equation_string, label, table
from style import axis_limits, vector, scatter, line, intersection, equation, polygon

def set_axis_limits(fig:plt.Figure, x_list:List[np.array]):
    """Set the axes limits of fig such that all points in x are visible.
    
    Given a set of nonnegative 2 or 3 dimensional points, set the axes 
    limits such all points are visible within the plot window. 
    
    Args:
        fig (plt.Figure): The plot for which the axes limits will be set
        List[np.array] : A set of nonnegative 2 or 3 dimensional points
        
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
    fig.add_trace(scatter(pts,None)) # Fixes bug with changing axes

def plot_lp(lp:LP) -> plt.Figure:
    """Return a figure visualizing the feasible region of the given LP
    
    For some LP with 2 or 3 decision variables, label the basic feasible
    solutions (with their objective value and basis), and plot the 
    feasible region and constraints.
    
    Args:
        lp (LP): The LP to visualize
        
    Returns:
        fig (plt.Figure): A figure containing the visualization
    
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
    
    # Plot basic feasible solutions
    bfs, bases, values = lp.get_basic_feasible_solns()
    lbs = [label(dict(x=list((bfs[i])[0:n,0]), 
                      Obj=float(values[i]), 
                      BFS=list((bfs[i])[:,0]), 
                      B=list(np.array(bases[i])+1))) for i in range(len(bfs))]
    pts = [i[0:n,:] for i in bfs]
    set_axis_limits(fig, pts) 
    fig.add_trace(scatter(pts,lbs)) 

    # Plot feasible region
    if n == 2: fig.add_trace(polygon(pts,'region')) # convex ploygon 
    if n == 3: # convex ploytope
        for i in range(n+m):
            pts = [bfs[j][0:n,:] for j in range(len(bfs)) if i not in bases[j]]
            fig.add_trace(polygon(pts,'region'))  

    # Plot constraints
    for i in range(m):
        lb = '('+str(i+n+1)+') '+equation_string(A[i],b[i][0])
        fig.add_trace(equation(fig,A[i],b[i][0],'constraint',lb))

    return fig

def add_path(fig:plt.Figure, path:List[List[float]]) -> List[int]:
    """Add each vector in the path to the figure. Return the index of each vector."""
    indices = []
    for i in range(len(path)-1):
        a = np.round(path[i],7)
        b = np.round(path[i+1],7)
        fig.add_trace(vector(a,b))
        indices.append(len(fig.data)-1)
    return indices

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
    indices = []
    x_lim, y_lim = axis_limits(fig,2)
    if n == 2:
        obj1=(simplex(LP(np.identity(2),np.array([[x_lim],[y_lim]]),c))[1])
        obj2=-(simplex(LP(np.identity(2),np.array([[x_lim],[y_lim]]),-c))[1])
        objectives = list(np.round(np.linspace(min(obj1,obj2),max(obj1,obj2),25),2))
        objectives.append(simplex(lp)[1])
        objectives.sort()
        for obj in objectives:
            fig.add_trace(equation(fig,c[:,0],obj,'isoprofit'))
            indices.append(len(fig.data)-1)
    if n == 3:
        objectives = np.round(np.linspace(0,simplex(lp)[1]),2)
        for obj in objectives:
            fig.add_trace(equation(fig,c[:,0],obj,'isoprofit_out'))
            pts = intersection(c[:,0],obj,lp.A,lp.b)
            fig.add_trace(polygon(pts,'isoprofit_in'))
            indices.append([len(fig.data)-2,len(fig.data)-1])
    return indices, objectives

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
    path_IDs = add_path(fig,[i[list(range(n)),0] for i in path])
    tables = []
    for basis in bases:
        header, content = lp.get_tableau(basis,'dictionary')
        tables.append(table(header, content))
    
    table_IDs = []
    for i in range(len(tables)):
        tab = tables[i]
        if not i == 0: tab.visible=False
        fig.add_trace(tab,row=1,col=2)
        table_IDs.append(len(fig.data)-1)
    
    isoprofit_IDs, objectives = add_isoprofits(fig,lp)
        
    iter_steps = []
    for i in range(len(path_IDs)+1):
        visible = [fig.data[j].visible for j in range(len(fig.data))]

        for j in range(len(table_IDs)):
            visible[table_IDs[j]] = False
        visible[table_IDs[i]] = True 
        
        for j in range(len(path_IDs)+1):
            if j < len(path_IDs):
                visible[path_IDs[j]] = True if j < i else False
        step = dict(method="update", label = i, args=[{"visible": visible}])
        iter_steps.append(step)

    iter_slider = dict(x=0.6, xanchor="left", y=0.22, yanchor="bottom", len= 0.4, lenmode='fraction',
                    pad={"t": 50}, active=0, currentvalue={"prefix":"Iteration: "}, tickcolor= 'white',
                    ticklen = 0, steps=iter_steps)
    
    iso_steps = []
    for i in range(len(isoprofit_IDs)):
        visible = [fig.data[k].visible for k in range(len(fig.data))]
        for j in isoprofit_IDs:
            if n == 2: visible[j]= False
            if n == 3: visible[j[0]]= False; visible[j[1]]= False
        if n == 2: visible[isoprofit_IDs[i]]= True
        if n == 3:
            visible[isoprofit_IDs[i][0]]= True
            visible[isoprofit_IDs[i][1]]= True
        step = dict(method="update", label = objectives[i], args=[{"visible": visible}])
        iso_steps.append(step)
    iso_slider = dict(x=0.6, xanchor="left", y=0.02, yanchor="bottom", len= 0.4, lenmode='fraction',
                    pad={"t": 50}, active=0, currentvalue={"prefix":"Objective: "}, tickcolor= 'white',
                    ticklen = 0, steps=iso_steps)

    fig.update_layout(sliders=[iter_slider,iso_slider])
    return fig