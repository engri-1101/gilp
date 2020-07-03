import numpy as np
from typing import List, Tuple
from scipy.spatial import ConvexHull
import plotly.graph_objects as plt
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from simplex import LP, simplex

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
    bfs, bases = lp.get_basic_feasible_solns()
    set_axis_limits(fig, [list(x[list(range(n)),0]) for x in bfs])
    
    lbs = ["x: "+str(tuple((bfs[i])[0:n,0]))+"<br>"+
           "Obj: "+str(np.dot((bfs[i])[0:n,0],c)[0])+"<br>"+
           "BFS: "+str(tuple((bfs[i])[:,0]))+"<br>"+
           "B: "+str(tuple(np.array(bases[i])+1)) for i in range(len(bfs))]
    
    pts = list(zip(*[list(i[0:n,0]) for i in bfs])) 
    
    
    if n == 2: # plot convex polygon 
        x,y = pts
        bfs_scatter = plt.Scatter(mode='markers', marker=dict(size=8, color='gray', opacity=0.6),
                                 x=x, y=y, showlegend=False, text=lbs,
                                hoverinfo='text',hoverlabel=dict(bgcolor='white', bordercolor='black'))
        fig.add_trace(bfs_scatter)
        bfs_list = np.array([list(x[list(range(n)),0]) for x in bfs])
        hull = ConvexHull(bfs_list) 
        feas_region = plt.Scatter(mode='markers', marker=dict(size=0,opacity=0), opacity=0.3,
                                 x=bfs_list[hull.vertices,0], y=bfs_list[hull.vertices,1],
                                 text=lbs, fill='toself', fillcolor = '#1469FE', showlegend=False,
                                 hoverinfo='skip')
        fig.add_trace(feas_region)
    if n == 3: # plot convex polytope
        x,y,z = pts 
        bfs_scatter = plt.Scatter3d(mode='markers', marker=dict(size=5, color='gray', opacity=0.3),
                                   x=x, y=y, z=z, text=lbs, showlegend=False,
                                   hoverinfo='text', hoverlabel=dict(bgcolor='white', bordercolor='black'))
        fig.add_trace(bfs_scatter)
        for i in range(n+m):
            bfs_subset = [list(bfs[j][list(range(n)),0]) for j in range(len(bfs)) if i not in bases[j]]
            pts = bfs_subset
            if len(pts) >= 3:
                if i < 3:
                    b_1 = np.zeros(3)
                    b_1[i] = -1
                    p,q,r = b_1
                else:
                    b_1 = p,q,r = A[i-3]
                perp = np.array([[q,-p,0], [r,0,-p], [0,r,-q]])
                b_2 = perp[np.nonzero(perp)[0][0]]
                b_3 = np.cross(b_1,b_2)
                T = np.linalg.inv(np.array([b_1,b_2,b_3]).transpose())
                pts = [list(np.round(np.dot(T,x),7))[1:3] for x in pts]
                pts = np.array(pts)
                hull = ConvexHull(pts) # get correct order
                pts = np.array(bfs_subset) # go back to bfs
                pts = list(zip(pts[hull.vertices,0], pts[hull.vertices,1], pts[hull.vertices,2]))
                # plot polygon 
                pts.append(pts[0])
                pts = [[pt[0]+0.0001,pt[1]+0.0001,pt[2]] if not pt[2] == 0 else pt for pt in pts]
                x,y,z = list(zip(*pts)) 
                def label(nums:list):
                    return('('+str(nums[0])+') '+
                            str(nums[1])+'x<sub>1</sub> + '+
                            str(nums[2])+'x<sub>2</sub> + '+
                            str(nums[3])+'x<sub>3</sub> ≤' +
                            str(nums[4]))
                if i < 3:
                    a = [-1 if j == i else 0 for j in range(3)]
                    lb = label([i+1,a[0],a[1],a[2],0])
                else:
                    lb = label([i+1,A[i-3][0],A[i-3][1],A[i-3][2],b[i-3][0]])
                face = plt.Scatter3d(mode="lines", x=x, y=y, z=z, surfaceaxis=2, surfacecolor='#1469FE',
                                    line=dict(width=5, color='#173D90'), opacity=0.2, 
                                    hoverinfo='skip', showlegend=True, name=lb)
                fig.add_trace(face)   

def plot_constraint(fig:plt.Figure, A:np.ndarray, b:float, label:str=None, 
                    color:str='black', dash:str=None, show:bool=False, visible:bool=True):
    """Plot the constraint on the given figure.
    
    Assumes the constraint is given in standard form: Ax <= b
    
    Args:
        fig (plt.Figure): The figure on which the constraint will be plotted
        A (np.ndarray): A list of 2 coefficents for the LHS of the constraint
        b (float): Represents the RHS of the constraint
        label (str): text label for this constraint (if it appears in the legend)
        color (str): color of the constraint
        dash (str): dash pattern for plotting the constraint 
        show (bool): True if this constraint appears in the legend. False otherwise
        visible (bool): True if this constraint is visible. False otherwise
    
    Raises:
        ValueError: A must be 2 dimensional
        ValueError: A must be nonzero
    """
    if not len(A) == 2:
        raise ValueError('A must be 2 dimensional')
    if len(np.nonzero(A)[0]) == 0:
        raise ValueError('A must be nonzero')
    
    x_lim, y_lim = fig.layout.scene.xaxis.range[1], fig.layout.scene.yaxis.range[1] # get limits
    if len(A) == 2:
        if A[1] == 0:
            x_loc = b/A[0]
            fig.add_trace(plt.Scatter(x=[x_loc,x_loc], y=[0,y_lim], hoverinfo='skip',visible=visible,
                                      name = label, showlegend=show, mode='lines', 
                                      line = dict(color=color, width=2, dash=dash))) 
        else:
            if A[0] == 0: 
                x = np.linspace(0, x_lim, 2)
            else:
                x = np.linspace(max(0,(b - y_lim*A[1])/A[0]), min(x_lim,b/A[0]), 2)
            y = (b - A[0]*x)/(A[1])
            fig.add_trace(plt.Scatter(x=x, y=y,hoverinfo='skip',visible=visible, mode='lines',
                                      name = label, showlegend=show, 
                                      line = dict(color=color, dash=dash,width=2)))

# TODO: constraint drawing for 3d
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
        color = ['#173D90', '#1469FE', '#65ADFF', '#474849', '#A90C0C', '#DC0000']
        c = 0 # current color
        x_lim, y_lim = fig.layout.scene.xaxis.range[1], fig.layout.scene.yaxis.range[1]
        x = np.linspace(0, x_lim, 10) 
        for i in range(m):
            lb = '('+str(i+3)+') '+str(A[i][0])+'x<sub>1</sub> + '+str(A[i][1])+'x<sub>1</sub> ≤ '+str(b[i][0])
            plot_constraint(fig,A[i],b[i][0],label=lb,color=color[c],dash='15,3,5,3',show=True,visible=True)
            c = c+1 if c+1<6 else 0

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
    fig.update_layout(title=dict(text= "Simplex Geo", x=0, y=0.98, xanchor= 'left', yanchor='top'),
                     scene= scene, legend=dict(title='Constraint(s)', 
                     x=0.4, y=1, xanchor='left', yanchor='top'),
                     width=950,height=500,margin=dict(l=0, r=0, b=0, t=30),
                     xaxis=dict(title='x<sub>1</sub>',rangemode='tozero',fixedrange=True),
                     yaxis=dict(title='x<sub>2</sub>',rangemode='tozero',fixedrange=True))
    
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
        if n == 2:
            line = plt.Scatter(x=p[0], y=p[1],mode="lines",
                                opacity=1,hoverinfo='skip',showlegend=False,
                                line=dict(width=4,color='#FF0000'),visible=False)
            d = (b-a)/(np.linalg.norm(b-a)*2)
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
            line = plt.Scatter3d(x=p[0], y=p[1], z=p[2],mode="lines",
                                opacity=1,hoverinfo='skip',showlegend=False,
                                line=dict(width=6,color='#FF0000'),visible=False)
            d = (b-a)/np.linalg.norm(b-a)*3
            head = plt.Cone(hoverinfo='skip',showscale=False,colorscale=['#FF0000','#FF0000'],
                           x=[b[0]], y=[b[1]], z=[b[2]], u=[d[0]], v=[d[1]], w=[d[2]],visible=False)
            fig.add_trace(line)
            fig.add_trace(head)
            s = len(fig.data)
            arrows.append((s-2,s-1))
    return arrows

# TODO: Fix long decimal steps
def add_isoprofit_lines(fig:plt.Figure, lp:LP) -> Tuple[List[int],List[float]]:
    """Render all the isoprofit lines which can be toggled over.
    
    Args:
        fig (plt.Figure): The figure to which these isoprofit lines are added
        lp (LP): The LP for which the isoprofit lines are being generated
        
    Returns:
        isoprofit_line_IDs (List[int]): The ID of all the isoprofit lines
        objectives (List[float]): The corresponding objectives
    """
    n,m,A,b,c = lp.get_inequality_form()
    isoprofit_line_start = len(fig.data)
    max_obj = np.round(np.dot(lp.c_0.transpose(),simplex(lp)[1][-1])[0][0],7)
    objectives = np.linspace(0,max_obj,10)
    for obj in objectives:
        plot_constraint(fig,c[:,0],obj,label=None,color='red',show=False,visible=False)
    isoprofit_line_end = len(fig.data)
    isoprofit_line_IDs = list(range(isoprofit_line_start,isoprofit_line_end))
    fig.data[isoprofit_line_IDs[0]].visible=True
    return isoprofit_line_IDs, objectives

def tableau_table(T:np.ndarray) -> plt.Table:
    """Generate a plt.Table trace for the given tableau.
    
    Args:
        T (np.ndarray): An array representing the tableau 
    
    Returns:
        table (plt.Table): A plt.Table trace for the tableau
    """
    rows = len(T)
    cols = len(T[0])
    header_values = []
    for j in range(cols):
        if j == 0:
            header_values.append('z')
        elif j == cols-1:
            header_values.append('RHS')
        else:
            header_values.append("x<sub>"+str(j)+"</sub>")
    cell_values = T.transpose()
    return plt.Table(header=dict(values=header_values),cells=dict(values=cell_values))

def simplex_visual(lp:LP,rule:str='dantzig',init_sol:np.ndarray=None,iter_lim:int=None):
    """Return a plot showing the geometry of simplex.
    
    Args:
        lp (LP): The LP on which to run simplex
        rule (str): The pivot rule to be used at each simplex iteration
        init_sol (np.ndarray): An n length vector 
        iter_lim (int): The maximum number of simplex iterations to be run"""
    
    n,m,A,b,c = lp.get_inequality_form()
    
    fig = plot_lp(lp)
    path, bases = simplex(lp,rule,init_sol)[1:3]
    arrows = add_path(fig,[i[list(range(n)),0] for i in path])
    tables = [tableau_table(lp.get_tableau(basis)) for basis in bases]
    
    table_IDs = []
    for i in range(len(tables)):
        table = tables[i]
        if not i == 0: table.visible=False
        fig.add_trace(table,row=1,col=2)
        table_IDs.append(len(fig.data)-1)
    
    if n == 2:
        isoprofit_line_IDs, objectives = add_isoprofit_lines(fig,lp)
        
    iter_steps = []
    for i in range(len(arrows)+1):
        visible = [fig.data[j].visible for j in range(len(fig.data))]

        for j in range(len(table_IDs)):
            visible[table_IDs[j]] = False
        visible[table_IDs[i]] = True 
        
        for j in range(len(arrows)+1):
            if j < len(arrows):
                visible[arrows[j][0]] = True if j < i else False
                visible[arrows[j][1]] = True if j < i else False
        step = dict(method="update", label = i, args=[{"visible": visible}])
        iter_steps.append(step)

    iter_slider = dict(x=0.6, xanchor="left", y=0.22, yanchor="bottom", len= 0.4, lenmode='fraction',
                    pad={"t": 50}, active=0, currentvalue={"prefix":"Iteration: "}, tickcolor= 'white',
                    ticklen = 0, steps=iter_steps)
    
    if n == 2:
    
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
    
    if n == 3: fig.update_layout(sliders=[iter_slider])
    return fig