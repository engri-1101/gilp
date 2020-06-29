SimplexGeo is being developed as a part of course development for ENGRI 1101:
Engineering Applications of Operations Research at Cornell University over
Summer 2020. The aim is to create a versatile python package for visualizing
the simplex algorithm on linear programs. The code closely resembles the
structure of ORIE 3300: Optimization I lecture slides. This should allow for
those students to comfortably examine the source code.

Current Functionality

The LP class allows one to create linear programs and stores them in both
standard equality and inequality forms. The current simplex implementation
is the revised simplex method. It includes 4 pivot rules: Bland, Dantzig,
greatest ascent, and manual selection by the user. Furthermore, one can
set an initial basic feasible solution and terminate early by an iteration
limit. The simplex function returns basic feasible solutions, their
corresponding basis, and an indication of optimality.

Miscellaneous functions include:
    invertible(A): returns True if A is invertible
    basic_feasible_solns(LP): returns basic feasible solutions of LP
    tableau(LP,B): returns a tableau for the LP and basis B

Miscellaneous plotting functions include:
    set_axis_limits: sets axis limits so that all points are visible
    add_vector_field: adds vector field in some direction
    plot_feasible_region: plots the feasible region of the LP
    plot_constraints: plots the constraints of the LP (2D only)
    plot_lp(LP): does all of the above for the given LP
    add_path: creates a set of arrows that trace the simplex path

Miscellaneous Bokeh functions include:
    tableau_table: create an HTML table to display the LP tableau
    tableau_data(T): for tableau T, create data for the HTML table

Currently, there are two main visualization tools. The first is designed to
be used with a Jupyter Notebook. One can define an LP using numpy and the
LP class. Then, they can run simplex using a chosen pivot rule, iteration
limit, and initial basic feasible solution. Afterwards, they can view the
path of simplex geometrically and the tableau at each iteration of simplex.
This uses the function simplex_visual. Alternatively, one can use the
function web_demo to generate an interactive HTML page for some LP. The self
contained web-page allows the user to toggle between non-manual pivot rules
and view each iteration of simplex. Similarly, both the geometric visual and
tableau are shown for each iteration of simplex.
