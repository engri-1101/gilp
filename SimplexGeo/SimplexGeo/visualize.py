import numpy as np
from typing import List, Tuple
import itertools
from scipy.spatial import ConvexHull
from scipy.linalg import solve
import plotly.graph_objects as plt
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from simplex import LP, simplex, invertible

def disp(num:float,precision:int=3) -> str:
    """Return a string representation of the num to a given precision with 
    trailing zeros and decimal (if applicable) removed."""
    return ('%.*f' % (precision, num)).rstrip('0').rstrip('.')

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
    bfs, bases = lp.get_basic_feasible_solns()
    set_axis_limits(fig, [list(x[list(range(n)),0]) for x in bfs])
    
    lbs = ["<b>x</b>: "+'(%s)'%', '.join(map(str,[disp(f) for f in (bfs[i])[0:n,0]]))+"<br>"+
           "<b>Obj</b>: "+disp(np.dot((bfs[i])[0:n,0],c)[0]) +"<br>"+
           "<b>BFS</b>: "+'(%s)'%', '.join(map(str,[disp(f)  for f in (bfs[i])[:,0]]))+"<br>"+
           "<b>B</b>: "+str(tuple(np.array(bases[i])+1)) for i in range(len(bfs))]
    
    pts = list(zip(*[list(i[0:n,0]) for i in bfs])) 

    if n == 2: # plot convex polygon 
        x,y = pts
        bfs_scatter = plt.Scatter(mode='markers', marker=dict(size=8, color='gray', opacity=0.6),
                                 x=x, y=y, showlegend=False, text=lbs,
                                hoverinfo='text',hoverlabel=dict(bgcolor='#FAFAFA', bordercolor='#323232',
                                font=dict(family='Arial',color='#323232')))
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
                                   hoverinfo='text', hoverlabel=dict(bgcolor='#FAFAFA', bordercolor='black',
                                   font=dict(family='Arial',color='#323232')))
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
                            disp(nums[1])+'x<sub>1</sub> + '+
                            disp(nums[2])+'x<sub>2</sub> + '+
                            disp(nums[3])+'x<sub>3</sub> ≤ ' +
                            disp(nums[4]))
                if i < 3:
                    a = [-1 if j == i else 0 for j in range(3)]
                    lb = label([i+1,a[0],a[1],a[2],0])
                else:
                    lb = label([i+1,A[i-3][0],A[i-3][1],A[i-3][2],b[i-3][0]])
                face = plt.Scatter3d(mode="lines", x=x, y=y, z=z, surfaceaxis=2, surfacecolor='#1469FE',
                                    line=dict(width=5, color='#173D90'), opacity=0.2, 
                                    hoverinfo='skip', showlegend=True, name=lb)
                fig.add_trace(face)   

def plot_constraint(fig:plt.Figure, A:np.ndarray, b:float, label:str=None, width:int=2,
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
        ValueError: A must be 2 or 3 dimensional
        ValueError: A must be nonzero
    """
    if len(A) not in [2,3]:
        raise ValueError('A must be 2 or 3 dimensional')
    if len(np.nonzero(A)[0]) == 0:
        raise ValueError('A must be nonzero')
    
    if len(A) == 2:
        x_lim, y_lim = get_axis_limits(fig,2)
        if A[1] == 0:
            x_loc = b/A[0]
            fig.add_trace(plt.Scatter(x=[x_loc,x_loc], y=[0,y_lim], hoverinfo='skip',visible=visible,
                                      name = label, showlegend=show, mode='lines', 
                                      line = dict(color=color, width=width, dash=dash))) 
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
            fig.add_trace(plt.Scatter(x=x, y=y,hoverinfo='skip',visible=visible, mode='lines',
                                      name = label, showlegend=show, 
                                      line = dict(color=color, dash=dash,width=width)))
    if len(A) == 3:
        x_lim, y_lim, z_lim = get_axis_limits(fig,3)
        pts = []
        for i,j in itertools.product(np.linspace(0,x_lim,10),np.linspace(0,y_lim,10)):
            pts.append((i,j,np.round((b-A[0]*i-A[1]*j)/A[2],7)))
        pts = [pt for pt in pts if 0 <= pt[0] <= x_lim and 0 <= pt[1] <= y_lim and 0 <= pt[2] <= z_lim]
        x,y,z= zip(*pts)
        plane = plt.Mesh3d(x=x,y=y,z=z, color=color, opacity=1, visible=visible, hoverinfo='skip', showlegend=show)
        fig.add_trace(plane)

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
        for i in range(m):
            lb = '('+str(i+3)+') '+disp(A[i][0])+'x<sub>1</sub> + '+disp(A[i][1])+'x<sub>1</sub> ≤ '+disp(b[i][0])
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
            plot_constraint(fig,c[:,0],obj,label=None,color='red',show=False,visible=False,width=4)
    if n == 3:

        objectives = np.round(np.linspace(0,simplex(lp)[1]),2)
        for obj in objectives:
            pts = []
            G = np.vstack((A,-1*np.identity(3)))
            G = np.vstack((G,c.transpose()))
            h = np.vstack((b,np.zeros((3,1))))
            h = np.vstack((h,obj))
            for pair in itertools.combinations(range(n+m),2):
                B = [pair[0],pair[1],n+m]
                if invertible(G[B,:]):
                    x_B = np.round(solve(G[B,:],h[B,:]),7)
                    if all(np.round(np.dot(A,x_B) <= b + 0.0000001)) and all((x_B >= np.zeros((3,1)))[:,0]): 
                        pts.append(x_B)
            pts = list(set([tuple((pt[:,0])) for pt in pts]))
            
            if len(pts) >= 3:
                b_1 = p,q,r = c[:,0]
                perp = np.array([[q,-p,0], [r,0,-p], [0,r,-q]])
                b_2 = perp[np.nonzero(perp)[0][0]]
                b_3 = np.cross(b_1,b_2)
                T = np.linalg.inv(np.array([b_1,b_2,b_3]).transpose())
                pts_T = [list(np.round(np.dot(T,np.array([x]).transpose()),7)[1:3,0]) for x in pts]
                pts_T = np.array(pts_T)
                hull = ConvexHull(pts_T)
                pts = np.array(pts)
                pts = list(zip(pts[hull.vertices,0], pts[hull.vertices,1], pts[hull.vertices,2]))
                pts.append(pts[0])
                pts = [[pt[0]+0.0001,pt[1]+0.0001,pt[2]] if not pt[2] == 0 else pt for pt in pts]
            x,y,z = list(zip(*pts)) 
            surface = plt.Scatter3d(mode="markers+lines", x=x, y=y, z=z, surfaceaxis=2, surfacecolor='red',
                                    line=dict(width=5, color='red'), opacity=1, visible=False,
                                    hoverinfo='skip', showlegend=False, marker=dict(size=5, color='red', opacity=1))
            fig.add_trace(surface) 
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
            header_values.append('z<sub></sub>')
        elif j == cols-1:
            header_values.append('RHS<sub></sub>')
        else:
            header_values.append("x<sub>"+str(j)+"</sub>")
    cell_values = T.transpose()
    cell_values = [[disp(cell_values[i,j],1) for j in range(len(cell_values[0]))] for i in range(len(cell_values))]
    return plt.Table(header=dict(values=header_values,line=dict(color='white',width=2),fill=dict(color='#5B8ECC'),align='center',
                                 font=dict(color='white', size=14), height=30),
                     cells=dict(values=cell_values,line=dict(color='white',width=2),fill=dict(color='#E9F3FF'), height=25))

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