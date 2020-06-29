import numpy as np
import math, itertools
from typing import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import solve
from scipy.spatial import ConvexHull
from bokeh.plotting import *
from bokeh.io import *
from bokeh.models import *
from bokeh.layouts import *
from bokeh.events import *
from bokeh.models.widgets import *
from SimplexGeo.LP import LP

output_notebook()

def invertible(A:np.ndarray) -> bool:
    """Return true if the matrix A is invertible.
    
    Args:
        A (np.ndarray): An m*n matrix
    
    By definition, a matrix A is invertible iff n = m and A has rank n
    """
    
    return len(A) == len(A[0]) and np.linalg.matrix_rank(A) == len(A)

def basic_feasible_solns(lp:LP) -> Tuple[List[np.ndarray],List[np.ndarray]]:
    """Return all basic feasible solutions and their basis for the LP.
    
    By definition, x is a basic feasible solution of an LP in standard 
    equality form iff x is a basic solution and both Ax = b and x > 0.
    A basic solution x is the solution to A_Bx = b for some basis B
    such that A_B is invertible.
           
    Returns:
        bfs (List[np.ndarray]): A list of basic feasible solutions
        bases (List[np.ndarray]): The corresponding list of bases
    """
    
    n,m,A,b,c = lp.get_equality_form()
    bfs = []
    bases = []
    for B in itertools.combinations(range(n+m),m):
        if invertible(A[:,B]):
            x_B = np.zeros((n+m,1))
            x_B[B,:] = np.round(solve(A[:,B],b),7)
            if all(x_B >= np.zeros((n+m,1))): 
                bfs.append(x_B) 
                bases.append(B) 
    return (bfs, bases)

def set_axis_limits(ax, x:List[List[float]]):
    """Set the axes limits of ax such that all points in x are visible.
    
    Given a set of nonnegative 2 or 3 dimensional points, set the axes 
    limits such all points are visible within the plot window. 
    
    Args:
        ax (TODO): The plot for which the axes limits will be set
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
    if n == 2:   
        x_lim, y_lim = limits
        ax.x_range=Range1d(0, x_lim)
        ax.y_range=Range1d(0, y_lim)
    if n == 3:
        x_lim, y_lim, z_lim = limits
        ax.set_xlim(0, x_lim)
        ax.set_ylim(0, y_lim)
        ax.set_zlim(0, z_lim)
        
# TODO: decide how to visualize given the scaling issues
# TODO: revert 3d vector field scaling issue
def add_vector_field(ax, d:np.ndarray, k:int):
    """Add a vector field to the plot with k vectors in direction x.
    
    Args:
        ax (TODO): The plot to add the vector feild to
        d (np.ndarray): A 2 or 3 dimensional vector defining the direction of field 
        k (int): The approximate number of vectors to be drawn
        
    Raises:
        ValueError: The direction d must be 2 or 3 dimensional
    """
    
    n = len(list(d[:,0])) 
    if not n in [2,3]:
        raise ValueError('The direction d must be 2 or 3 dimensional')
    vectors_per_axis = round(k**(1./n))
    if n == 2:
        x_lim, y_lim = ax.x_range.end, ax.y_range.end
        X = np.arange(0, x_lim, x_lim/vectors_per_axis)
        Y = np.arange(0, y_lim, y_lim/vectors_per_axis)
        grid = itertools.product(X,Y)
        d = (d/np.linalg.norm(d))*(np.sqrt(x_lim**2+y_lim**2)/20)
        for x,y in grid:
            arrow = Arrow(end=NormalHead(size=8,
                                         line_color='gray',line_alpha = 0.2,fill_color="gray",fill_alpha=0.2),
                          line_width=1.5, line_color='gray', line_alpha=0.2,
                          x_start=x,x_end=x+d[0][0],
                          y_start=y,y_end=y+d[1][0])
            ax.add_layout(arrow)
    if n == 3:
        x_lim, y_lim, z_lim = ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]
        X = np.arange(0, x_lim, x_lim/vectors_per_axis)
        Y = np.arange(0, y_lim, y_lim/vectors_per_axis) 
        Z = np.arange(0, z_lim, z_lim/vectors_per_axis) 
        X,Y,Z = np.meshgrid(X, Y, Z) 
        d = d/np.linalg.norm(d)
        d = [d[0]*x_lim,d[1]*y_lim,d[2]*z_lim]
        ax.quiver(X, Y, Z, d[0], d[1], d[2] ,color='gray',alpha=0.2, length=0.1) 
        
# TODO: Consider unbounded scenario
# TODO: implement basis hovering (2d done)
def plot_feasible_region(ax, lp:LP):
    """Render the LP's feasible region on the given plot.
    
    Depict each of the LP's basic feasible solutions and set the plot axes
    such that all basic feasible solutions are visible. Shade the feasible
    region for 2d plot and construct feasible region polytope for 3d plots.
    
    Args:
        ax (TODO): The plot on which the feasible region will be rendered
        lp (LP): The LP which will have its feasible region rendered
        
    Raises:
        ValueError: The LP must have 2 or 3 decision variables
    """
    
    n,m,A,b,c = lp.get_inequality_form()
    if not n in [2,3]:
        raise ValueError('The LP must have 2 or 3 decision variables')
    bfs, bases = basic_feasible_solns(lp)
    set_axis_limits(ax, [list(x[list(range(n)),0]) for x in bfs])
    if n == 2: # plot convex polygon
        bfs_list = np.array([list(x[list(range(n)),0]) for x in bfs])
        hull = ConvexHull(bfs_list) 
        ax.patch(bfs_list[hull.vertices,0], 
                 bfs_list[hull.vertices,1], 
                 line_width=0, fill_color='gray', alpha=0.3)
        x,y = list(zip(*bfs_list)) 
        obj_val = [np.dot(i[[0,1],0],c) for i in bfs]
        bases = [list(np.array(basis)+1) for basis in bases]
        source = ColumnDataSource(dict(x=x, y=y, bfs=bfs, obj_val=obj_val, B=bases))
        ax.circle('x', 'y', size=7, source=source, name='bfs',color='gray')
    if n == 3: # plot convex polytope
        for i in range(n+m):
            bfs_subset = [list(bfs[j][list(range(n)),0]) for j in range(len(bfs)) if i not in bases[j]]
            pts = bfs_subset
            if len(pts) >= 3:
                # need to order basic feasible solutions to draw polygon
                if i < 3:
                    b_1 = np.zeros(3)
                    b_1[i] = -1
                    a,b,c = b_1
                else:
                    b_1 = a,b,c = A[i-3]
                perp = np.array([[b,-a,0], [c,0,-a], [0,c,-b]])
                b_2 = perp[np.nonzero(perp)[0][0]]
                b_3 = np.cross(b_1,b_2)
                T = np.linalg.inv(np.array([b_1,b_2,b_3]).transpose())
                pts = [list(np.round(np.dot(T,x),7))[1:3] for x in pts]
                pts = np.array(pts)
                hull = ConvexHull(pts) # get correct order
                pts = np.array(bfs_subset) # go back to bfs
                pts = list(zip(pts[hull.vertices,0], pts[hull.vertices,1], pts[hull.vertices,2]))
                # plot polygon 
                poly = Poly3DCollection([pts])
                poly.set_edgecolor('#173D90')
                poly.set_facecolor('#1469FE')
                poly.set_alpha(0.2)
                ax.add_collection3d(poly)
                
# TODO: constraint drawing for 3d
def plot_constraints(ax, lp:LP):
    """Render each of the LP's constraints on the given plot.
    
    Args:
        ax (TODO): The plot on which the constraints will be rendered
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
        x = np.linspace(ax.x_range.start, ax.x_range.end, 10) 
        dash = [15,3,5,3]
        for i in range(m):
            lb = '('+str(i+3)+') '+str(A[i][0])+'x₁ + '+str(A[i][1])+'x₂ ≤ '+str(b[i][0])
            if A[i][1] == 0:
                x_loc = b[i][0]/A[i][0]
                y_min, y_max = ax.y_range.start,ax.y_range.end 
                ax.line([x_loc,x_loc], [y_min,y_max], 
                        line_width=2, line_color=color[c], line_alpha=1, 
                        line_dash=dash,muted_alpha=0.2, legend_label=lb)
            else:
                y = (b[i] - A[i][0]*x)/(A[i][1])
                ax.line(x, y,
                        line_width=2, line_color=color[c], line_alpha=1,
                        line_dash=dash, muted_alpha=0.2, legend_label=lb)
            c = c+1 if c+1<6 else 0
    # TODO: constraint drawing for 3d

def plot_lp(lp:LP):
    """Render a visualization of the given LP.
    
    For some LP with 2 or 3 decision variables, visualize the feasible region,
    basic feasible solutions, and constraints.
    
    Args:
        lp (LP): The LP which will be visualized
    
    Raises:
        ValueError: The LP must have 2 or 3 decision variables
    """
    
    if not lp.n in [2,3]:
        raise ValueError('The LP must have 2 or 3 decision variables')
    
    if lp.n == 2:
        hover = HoverTool(names=['bfs'],tooltips=[("x", "(@x,@y)"),
                                                  ("Obj", "@obj_val"),
                                                  ("BFS", "(@bfs)"),
                                                  ("Basis", "[@B]")])
        ax = figure(plot_height=400,plot_width=600,tools=[hover],title="Simplex Geometry")
        ax.toolbar.logo = None
        ax.toolbar_location = None
    if lp.n == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    plot_feasible_region(ax,lp)
    add_vector_field(ax,lp.c,100)
    plot_constraints(ax,lp)   
    
    if lp.n==2:
        ax.xaxis.axis_label = "x₁"
        ax.yaxis.axis_label = "x₂"
        ax.legend.visible=False
        items = ax.legend.items
        legend = Legend(items=items, location="top_right",click_policy="mute",title='Constraint(s)')
        ax.add_layout(legend, 'right')
        
    if lp.n==3:
        ax.set_xlabel(r'$x₁$')
        ax.set_ylabel(r'$x₂$')
        ax.set_zlabel(r'$x₃$')
        
    return ax

# TODO: arrows rather than lines (2d done)
def add_path(ax, path:List[List[float]]) -> List[Arrow]:
    """Render the path of ordered points on the given plot.
    
    Draw arrows between consecutive points to trace the path. 
    
    Args:
        ax (TODO): The plot on which the path will be rendered
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
            arrow = Arrow(end=NormalHead(line_color='red',fill_color="red",size=15),line_width=3,
                   x_start=a[0], y_start=a[1], x_end=b[0], y_end=b[1],line_color='red')
            ax.add_layout(arrow)
            arrows.append(arrow)
        if n == 3:
            ax.plot(p[0],p[1],p[2],color='r',linewidth=3)
            ax.scatter(b[0],b[1],b[2],'ro',color='r') 
            lb = str(i+1)+': ('+str(b[0])+','+str(b[1])+','+str(b[2])+')'
            d = list((np.array([[ax.get_xlim()[1],ax.get_ylim()[1],ax.get_zlim()[1]]])/15)[0])
            ax.text(b[0]+d[0], b[1]+d[1], b[2]+d[2],lb,color='r')
    return arrows

def simplex_iter(lp:LP, x:np.ndarray, B:List[int], N:List[int], pivot_rule:str='bland'):
    """Run a single iteration of the revised simplex method.
    
    Starting from the basic feasible solution x with corresponding basis B and 
    non-basic variables N, do one iteration of the revised simplex method. Use
    the given pivot rule. Implemented pivot rules include:
    
    'bland' or 'min_index'
        entering variable: minimum index
        leaving variable: minimum (positive) ratio (minimum index to tie break)
    
    'dantzig' or 'max_reduced_cost'
        entering variable: most positive reduced cost
        leaving variable: minimum (positive) ratio (minimum index to tie break)
    
    'greatest_ascent'
        entering variable: most positive product of minimum ratio and reduced cost
        leaving variable: minimum (positive) ratio (minimum index to tie break)
        
    'manual_select'
        entering variable: user selects among possible entering indices
        leaving variable: minimum (positive) ratio (minimum index to tie break)
        
    Return the new bfs, basis, non-basic variables, and an idication of optimality.
    
    Args:
        lp (LP): The LP on which the simplex iteration is being done
        x (np.ndarray): The initial basic feasible solution
        B (List(int)): The basis corresponding to the bfs x
        N (List(int)): The non-basic variables
        pivot_rule (str): The pivot rule to be used
        
    Returns:
        x (np.ndarray): The new basic feasible solution
        B (List(int)): The basis corresponding to the new bfs x
        N (List(int)): The new non-basic variables
        opt (bool): An idication of optimality. True if optimal.
        
    Raises:
        ValueError: Invalid pivot rule. See simplex_iter? for possible options.
    """
    if pivot_rule not in ['bland','min_index','dantzig','max_reduced_cost',
                          'greatest_ascent','manual_select']:
        raise ValueError('Invalid pivot rule. See simplex_iter? for possible options.')

    n,m,A,b,c = lp.get_equality_form()
    y = solve(A[:,B].transpose(), c[B,:]) # dual cost vector
    red_costs = c - np.dot(y.transpose(),A).transpose() # reduced cost vector
    pos_nonbasic_red = {k : red_costs[k] for k in N if red_costs[k] > 0} # possible entering
    if len(pos_nonbasic_red) == 0:
        # no positive reduced costs -> optimal
        return x,B,N,True
    else:
        if pivot_rule == 'greatest_ascent':
            eligible = {}
            for k in pos_nonbasic_red:
                d = np.zeros((1,n+m))
                d[:,B] = solve(A[:,B], A[:,k])
                ratios = {i : x[i]/d[0][i] for i in B if d[0][i] > 0}
                if len(ratios) == 0:
                    raise ValueError('The LP is unbounded')
                t = min(ratios.values())
                r_pos = [r for r in ratios if ratios[r] == t]
                r =  min(r_pos)
                t = ratios[r]
                eligible[(t*red_costs[k])[0]] = [k,r,t,d]
            k,r,t,d = eligible[max(eligible.keys())]
        else: 
            entering = None
            if pivot_rule == 'manual_select':
                entering = int(input('Pick one of '+str(list(pos_nonbasic_red.keys()))))
            k = {'bland' : min(pos_nonbasic_red.keys()),
                 'min_index' : min(pos_nonbasic_red.keys()),
                 'dantzig' : max(pos_nonbasic_red, key=pos_nonbasic_red.get),
                 'max_reduced_cost' : max(pos_nonbasic_red, key=pos_nonbasic_red.get),
                 'manual_select' : entering}[pivot_rule]
            d = np.zeros((1,n+m))
            d[:,B] = solve(A[:,B], A[:,k])
            ratios = {i : x[i]/d[0][i] for i in B if d[0][i] > 0}
            if len(ratios) == 0:
                raise ValueError('The LP is unbounded')
            t = min(ratios.values())
            r_pos = [r for r in ratios if ratios[r] == t]
            r =  min(r_pos)
            t = ratios[r]
        # update
        x[k] = t
        x[B,:] = x[B,:] - t*(d[:,B].transpose())
        B.append(k); B.remove(r);
        N.append(r); N.remove(k);
        return x,B,N,False
    
# TODO: fix initial solution being optimal bug
# TODO: what if infeasible?
def simplex(lp:LP, pivot_rule:str='bland',
            init_sol:np.ndarray=None,iter_lim:int=None) -> Tuple[bool,List[np.ndarray],List[List[int]]]:
    """Run the revised simplex method on the given LP.
    
    Run the revised simplex method on the given LP. If an initial solution is
    given, check if it is a basic feasible solution. If so, start simplex from
    this inital bfs. Otherwise, ignore it. If an iteration limit is given, 
    terminate if the specified limit is reached and output the current solution.
    Indicate that the solution may not be optimal. At each simplex iteration,
    use the given pivot rule. Implemented pivot rules include:
    
    'bland' or 'min_index'
        entering variable: minimum index
        leaving variable: minimum (positive) ratio (minimum index to tie break)
    
    'dantzig' or 'max_reduced_cost'
        entering variable: most positive reduced cost
        leaving variable: minimum (positive) ratio (minimum index to tie break)
    
    'greatest_ascent'
        entering variable: most positive product of minimum ratio and reduced cost
        leaving variable: minimum (positive) ratio (minimum index to tie break)
        
    'manual_select'
        entering variable: user selects among possible entering indices
        leaving variable: minimum (positive) ratio (minimum index to tie break)
        
    Return a list of basic feasible solutions, their bases, and indication of optimality
    
    Args:
        lp (LP): The LP on which to run simplex
        pivot_rule (str): The pivot rule to be used at each simplex iteration
        init_sol (np.ndarray): An n length vector 
        iter_lim (int): The maximum number of simplex iterations to be run
    
    Return:
        opt (bool): True if path[-1] is known to be optimal. False otherwise.
        path (List[np,ndarray]): The list of basic feasible solutions (n+m length)
                                 vectors that simplex traces.
        bases (List[List[int]]): The corresponding list of bases that simplex traces
    
    Raises:
        ValueError: Invalid pivot rule. See simplex_iter? for possible options.
        ValueError: Iteration limit must be strictly positive
    """
    
    if pivot_rule not in ['bland','min_index','dantzig','max_reduced_cost',
                          'greatest_ascent','manual_select']:
        raise ValueError('Invalid pivot rule. See simplex_iter? for possible options.')
    if iter_lim is not None and iter_lim <= 0:
        raise ValueError('Iteration limit must be strictly positive.')
    
    n,m,A,b,c = lp.get_equality_form()
    
    # select intital basis and feasible solution
    B = list(range(n,n+m))
    N = list(range(0,n)) 
    x = np.zeros((n+m,1))
    x[B,:] = b

    if init_sol is not None:
        x_B = np.zeros((n+m,1))
        x_B[list(range(n)),:] = init_sol
        for i in range(m):
            x_B[i+2] = b[i]-np.dot(A[i,list(range(n))],init_sol)
        x_B = np.round(x_B,7)
        if (all(np.round(np.dot(A,x_B)) == b) and 
            all(x_B >= np.zeros((n+m,1))) and 
            len(np.nonzero(x_B)[0]) <= n+m-2):
            x = x_B
            B = list(np.nonzero(x_B)[0])
            N = list(set(range(n+m)) - set(B))
            while len(B) < n+m-2:
                B.append(N.pop())
        else:
            print('Initial solution ignored.')
            
    path = [np.copy(x)]
    bases = [np.copy(B)]
                                    
    optimal = False 
    
    if iter_lim is not None: lim = iter_lim
    while(not optimal):
        x,B,N,opt = simplex_iter(lp,x,B,N,pivot_rule)
        # TODO: make a decison about how this should be implemented
        if opt == True:
            optimal = True
        else:
            path.append(np.copy(x))
            bases.append(np.copy(B))
        if iter_lim is not None: lim = lim - 1
        if iter_lim is not None and lim == 0: break;
            
    return optimal, path, bases

def tableau(lp:LP, B:list) -> np.ndarray:
    """Get the tableau of the LP for the given basis B.
    
    The returned tableau has the following form:
    
    z - (c_N^T - y^TA_N)x_N = y^Tb
    x_B + A_B^(-1)A_Nx_N = x_B^*
    
    y^T = c_B^TA_B^(-1)
    x_B^* = A_B^(-1)b
    
    | z | x_1 | x_2 | ... | = | RHS |
    ---------------------------------
    | 1 |  -  |  -  | ... | = |  -  |
    | 0 |  -  |  -  | ... | = |  -  |
                  ...
    | 0 |  -  |  -  | ... | = |  -  |
    
    
    Args:
        lp (LP): The LP for which a tableau will be returned
        B (list): The basis the tableau corresponds to
        
    Returns:
        T (np.ndarray): A numpy array representing the tableau
        
    Raises:
        ValueError: Invalid basis. A_B is not invertible.
        
    
    """
    
    n,m,A,b,c = lp.get_equality_form()
    
    if not invertible(A[:,B]):
        raise ValueError('Invalid basis. A_B is not invertible.')
        
    N = list(set(range(n+m)) - set(B))
    
    A_B_inv = np.linalg.inv(A[:,B])
    yT = np.dot(c[B,:].transpose(),A_B_inv)
    
    top_row_coef = np.zeros(n+m+1)
    top_row_coef[N] = c[N,:].transpose() - np.dot(yT,A[:,N])
    top_row_coef[-1] = np.dot(yT,b)    
    
    con_row_coef = np.zeros((m,n+m))
    con_row_coef[:,N] = np.dot(A_B_inv,A[:,N])
    con_row_coef[:,B] = np.identity(len(B))
    
    rhs_coef = np.dot(A_B_inv,b)
    
    T = np.hstack((con_row_coef,rhs_coef))
    T = np.vstack((top_row_coef,T))
    T = np.hstack((np.zeros((m+1,1)),T))
    T[0,0] = 1
    
    return np.round(T,7)

def tableau_table(lp:LP) -> DataTable:
    '''Return a DataTable to display a tableau for the LP in Bokeh'''
    cols = lp.n+lp.m+2
    columns = []
    for j in range(cols):
        if j == 0:
            columns.append(TableColumn(field="z", title="z"))
        elif j == cols-1:
            columns.append(TableColumn(field="rhs", title="RHS"))
        else:
            columns.append(TableColumn(field="x_"+str(j), title="<MATH>x<sub>"+str(j)+"</sub></MATH>"))
          
    return DataTable(columns=columns, width=300, height=170,index_position=None, 
                     editable=False, reorderable=False, sortable=False, selectable=False) 

def tableau_data(T:np.ndarray) -> ColumnDataSource:
    '''Return a ColumnDataSource for the given tableau to be passsed to a DataTable'''
    rows = len(T)
    cols = len(T[0])
    data = {}
    for j in range(cols):
        if j == 0:
            data['z'] = [T[i][j] for i in range(rows)]
        elif j == cols-1:
            data['rhs'] = [T[i][j] for i in range(rows)]
        else:
            data['x_'+str(j)] = [T[i][j] for i in range(rows)]
    return ColumnDataSource(data) 

# TODO: Fix unexpected in tableau table 
def web_demo(lp:LP, name:str):
    """Create an interactive HTML webpage for the given LP (Prototype)
    
    Args:
        lp (LP): The LP to be featured on the webpage
        name (str): The name of the .html file
        
    Raises:
        ValueError: Only supports LPs with 2 decision variables      
    """
    n,m,A,b,c = lp.get_inequality_form()
    
    if not n == 2:
        ValueError('Only supports LPs with 2 decision variables')
    
    ax = plot_lp(lp)
    
    rules_arrows = []
    rules_table_data = []
    for rule in ['bland','dantzig','greatest_ascent']:
        path, bases = simplex(lp,rule)[1:3]
        n = len(lp.A[0])
        path = [i[list(range(n)),0] for i in path]
        arrows = add_path(ax,path)
        for arrow in arrows:
            arrow.visible=False
        rules_arrows.append(arrows)
        table_data = [tableau_data(tableau(lp,basis)) for basis in bases]
        rules_table_data.append(table_data)

    table = tableau_table(lp)
    table.source= rules_table_data[0][0]
    pivot_rule_btn = RadioButtonGroup(labels=["Bland", "Dantzig", "Greatest Ascent"],active=0)
    slider = Slider(start=0, end=len(rules_arrows[0]), value=0, step=1, title="Iteration")
    
    # JavaScript code for bokeh 2d plot interaction
    update_window = """
    // Current selection
    var rule = pivot_rule_btn.active
    var path = rules_arrows[rule]
    var iter = slider.value
            
    // Adjust for selected pivot rule
    var total_iter = rules_arrows[rule].length
    slider.end = total_iter
    if (iter > total_iter) {
        iter = total_iter
        slider.value = total_iter
    }
            
    // Adjust current tableau
    table.source = rules_table_data[rule][iter]
    table.change.emit()
            
    // Adjust current path on plot
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < rules_arrows[i].length; j++) {
            rules_arrows[i][j].visible=false
        }      
    }
    for (let i = 0; i < path.length; i++) {
        if (i < iter) {
            path[i].visible=true
        } else {
            path[i].visible=false
        }
    }"""

    callback = CustomJS(args=dict(table=table, pivot_rule_btn=pivot_rule_btn, slider=slider,
                                  rules_arrows=rules_arrows, rules_table_data=rules_table_data), 
                                  code=update_window)
        
    slider.js_on_change('value',callback)
    pivot_rule_btn.js_on_click(callback)
        
    title_text = Div(height=30,text=
                     """<p style = "font-family:helvetica;font-size:18px;"> <b>Simplex Lab Prototype</b> </p>""")
    descrip_text = Div(text="""<p style = "font-family:helvetica;font-size:14px;"> 
        Hover over basic feasible solutions. Select a pivot rule. Look at the iterations of simplex.</p>""")

    layout = row(ax,column(title_text,descrip_text,table,pivot_rule_btn,slider))
    output_file(name+'.html', title="Simplex Lab")
    show(layout)
    
# TODO: Fix unexpected in tableau table 
def simplex_visual(lp:LP,rule:str='dantzig',init_sol:np.ndarray=None,iter_lim:int=None):
    """Return a plot showing the geometry of simplex.
    
    Args:
        lp (LP): The LP on which to run simplex
        rule (str): The pivot rule to be used at each simplex iteration
        init_sol (np.ndarray): An n length vector 
        iter_lim (int): The maximum number of simplex iterations to be run"""
    
    ax = plot_lp(lp)
    
    path, bases = simplex(lp,rule,init_sol,iter_lim)[1:3]
    n = len(lp.A[0])
    path = [i[list(range(n)),0] for i in path]
    arrows = add_path(ax,path)
    for arrow in arrows:
        arrow.visible=False
    tables = [tableau_data(tableau(lp,basis)) for basis in bases]

    if n==2: 
        table = tableau_table(lp)
        table.source= tables[0]
        slider = Slider(start=0, end=len(arrows), value=0, step=1, title="Iteration")
        
        # JavaScript code for bokeh 2d plot interaction
        update_window = """
        var iter = slider.value
        for (let i = 0; i < arrows.length; i++) {
            if (i < iter) {
                arrows[i].visible=true
            } else {
                arrows[i].visible=false 
            }
        }
        table.source = tables[iter]
        table.change.emit()"""
        
        callback = CustomJS(args=dict(table=table,slider=slider,
                                      arrows=arrows,tables=tables), code=update_window)
        
        slider.js_on_change('value',callback)
        
        title_text = Div(text=
                         """<p style = "font-family:helvetica;font-size:18px;"> <b>Simplex Lab Prototype</b></p>""",height=30)
        descrip_text = Div(text="""<p style = "font-family:helvetica;font-size:14px;"> 
        Hover over basic feasible solutions. Look at the iterations of simplex.</p>""")
        layout = row(ax,column(title_text,descrip_text,table,slider))
        show(layout)