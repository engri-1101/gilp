Tutorial
========

First, open up a Jupyter Notebook or Google Colab enviroment. Reminder: if you
are using a Google Colab enviroment, you will have to reinstall gilp every time
by running the cell :code:`!pip install gilp`.

Example LPs
-----------

GILP comes with many LP examples. Before we use them, we must import them.

.. code-block:: python

    from gilp import examples as ex

We can now access the LP examples using :code:`ex.NAME` where :code:`NAME`
is the name of the example LP. For example, consider:

.. math::

    \begin{align*}
    \text{maximize}  \quad & 5x_1 + 3x_2\\
    \text{subject to} \quad & 2x_1 + 1x_2 \leq 20 \\
    & 1x_1 + 1x_2 \leq 16 \\
    & 1x_1 + 0x_2 \leq 7 \\
    & x_1, x_2 \geq 0
    \end{align*}

This example LP is called :code:`ALL_INTEGER_2D_LP`. Let us assign this LP to a
variable called :code:`lp`.

.. code-block:: python

    lp = ex.ALL_INTEGER_2D_LP

Now, we can begin visualizing LPs. We import the visualization function below.

.. code-block:: python

    from gilp.visualize import simplex_visual

The function :code:`simplex_visual()` takes an LP and returns a plotly figure.
The figure can then be viewed on a Jupyter Notebook inline using

.. code-block:: python

    simplex_visual(lp).show()

If :code:`.show()` is run outside a Jupyter Notebook enviroment, the
visualization will open up in the browser. Alternatively, the HTML file can be
written and then opened.

.. code-block:: python

    simplex_visual(lp).write_html('name.html')

Here is the resulting visualization from running
:code:`simplex_visual(lp).show()`

.. raw:: html
   :file: ../visualizations/ALL_INTEGER_2D_LP.html

The resulting visualization has the following components.

* **Plot**:
    On the left, a plot shows the feasible region of the LP shaded in blue. You
    can hover over the corner points to see the feasible solution, dictionary,
    and objective value asscociated with that point.
* **Constraints**:
    In the middle, there is a list of constraints (not including the
    nonnegativity constraints). You can click on a constraint to mute it and
    click again to bring it back.
* **Dictionary Form LP**:
    The dictionary form for the current iteration of simplex is shown in the top
    right. If the slider is between iterations, the dictionary form for both the
    previous and next iteration are shown.
* **Sliders**:
    The iteration slider allows you to toggle through iterations of simplex. You
    can see the path of simplex on the plot and the updating corresponding
    dictionary LPs. The objective slider allows you to see the isoprofit line or
    plane for various objective values.

Defining LPs
------------

We can also create our own LPs! First, we must import the :code:`LP` class.

.. code-block:: python

    from gilp.simplex import LP

The :code:`LP` class creates linear programs from their standard inequality
form. We can represent a standard inequality form LP in terms of three
matrices.

.. math::

    \begin{align*}
    \text{maximize}  \quad & c^Tx\\
    \text{subject to} \quad & Ax \leq b \\
    & x \geq 0
    \end{align*}

For example, consider the following LP in standard inequality form.

.. math::

    \begin{align*}
    \text{maximize}  \quad & 1x_1 + 2x_2\\
    \text{subject to} \quad & 0x_1 + 1x_2 \leq 4 \\
    & 1x_1 - 1x_2 \leq 2 \\
    & 1x_1 + 0x_2 \leq 3 \\
    & -2x_1 + 1x_2 \leq 0 \\
    & x \geq 0
    \end{align*}

In this example, we have :math:`A = \begin{bmatrix} 0 & 1 \\ 1 & -1 \\ 1 & 0 \\
-2 & 1\end{bmatrix}`, :math:`b = \begin{bmatrix} 4 \\ 2 \\ 3 \\ 0
\end{bmatrix}`, and :math:`c = \begin{bmatrix} 1 \\ 2 \end{bmatrix}`. Note
:math:`x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}`

We will use these three matrices to create an instance of :code:`LP`. First, we
will import NumPy to create the matrices.

.. code-block:: python

   import numpy as np

Now, using NumPy, we create the matrices and create the :code:`LP` instance.

.. code-block:: python

   from gilp.simplex import LP

   A = np.array([[0, 1],
                 [1, -1],
                 [1, 0],
                 [-2, 1]])
   b = np.array([[4],
                 [2],
                 [3],
                 [0]])
   c = np.array([[1],
                 [2]])
   # Alternatively
   b = np.array([4,2,3,0])
   c = np.array([1,2])

   lp = LP(A,b,c)

Now, we can visualize it like before!

.. code-block:: python

    simplex_visual(lp).show()

.. raw:: html
   :file: ../visualizations/DEGENERATE_FIN_2D_LP.html

The complete code for defining the LP and visualizing it is given below.

.. code-block:: python
    :linenos:

    import numpy as np
    from gilp.simplex import LP
    from gilp.visualize import simplex_visual

    A = np.array([[0, 1],
                 [1, -1],
                 [1, 0],
                 [-2, 1]])
    b = np.array([4,2,3,0])
    c = np.array([1,2])
    lp = LP(A,b,c)

    simplex_visual(lp).show()


Solver Parameters
-----------------

The :code:`simplex_visual()` function has some optional solver parameters that
can be set. These include an initial solution, iteration limit, and pivot rule.
We go over each in more detail using :code:`ex.KLEE_MINTY_3D_LP` as an example.
For reference, here is the visualization of the Klee Minty Cube with no solver
parameters set.

.. code-block:: python

    simplex_visual(ex.KLEE_MINTY_3D_LP).show()


.. raw:: html
   :file: ../visualizations/init_sol_origin.html

|

Setting an Initial Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the intial solution is always set at the origin. However, one can
choose from any corner point to be the initial solution. For those with
previous experience with LPs, the initial solution must be a *basic feasible
solution*. An initial solution is set as follows:

.. code-block:: python

    simplex_visual(lp, initial_solution=x).show()

where :code:`x` is a NumPy vector representing the initial solution. Above,
you can see the default initial feasible solution is the origin. Let us try
setting a different initial solution.

.. code-block:: python

    x = np.array([[0],[25],[25]])
    simplex_visual(ex.KLEE_MINTY_3D_LP, initial_solution=x).show()

.. raw:: html
   :file: ../visualizations/init_sol_set.html

|

Iteration Limits
~~~~~~~~~~~~~~~~

By default, the simplex algorithm will run simplex iterations until an optimal
solution is found. Alternatively, an iteration limit can be set:

.. code-block:: python

    simplex_visual(lp, iteration_limit=l).show()

where :code:`l` is an integer iteration limit. Above, you can see it takes 5
simplex iterations to reach the optimal solution. Let's set the iteration limit
to be 3.

.. code-block:: python

    simplex_visual(ex.KLEE_MINTY_3D_LP, iteration_limit=3).show()

.. raw:: html
   :file: ../visualizations/iter_lim.html

|

Setting a Pivot Rule
~~~~~~~~~~~~~~~~~~~~

Be default, the simplex algorithm uses Bland's pivot. In addition to Bland's
rule, three other pivot rules are implemented. In an iteration of simplex, the
leaving variable is always the minimum (positive) ratio (minimum index to tie
break) regardless of the chosen pivot rule. Of the eligible entering variables
(those with positive coefficients in the objective function), each pivot rule
determines the entering variable as follows:

- **Bland's Rule** (reference as :code:`bland` or :code:`min_index`) Minimum index.
- **Dantzig's Rule** (reference as :code:`dantzig` or :code:`max_reduced_cost`) Most positive reduced cost.
- **Greatest Ascent** (reference as :code:`greatest_ascent`) Most positive (minimum ratio) x (reduced cost).
- **Manual Select** (reference as :code:`manual_select`) Selected by user.

A desired pivot rule is specified as follows.

.. code-block:: python

    simplex_visual(lp, rule=r).show()

where :code:`r` is a string representing the chosen rule. Let us try some other
pivot rules on :code:`ex.KLEE_MINTY_3D_LP`!

.. code-block:: python

    simplex_visual(ex.KLEE_MINTY_3D_LP, rule='dantzig').show()

.. raw:: html
   :file: ../visualizations/rule_dantzig.html

|

.. code-block:: python

    simplex_visual(ex.KLEE_MINTY_3D_LP, rule='greatest_ascent').show()

.. raw:: html
   :file: ../visualizations/rule_greatest_ascent.html

|

.. code-block:: python

    simplex_visual(ex.KLEE_MINTY_3D_LP, rule='manual_select').show()

For this visualization, the chosen entering variables were 2,3, and then 5.

.. raw:: html
   :file: ../visualizations/rule_manual.html

|

This concludes the quickstart tutorial! See the :ref:`dev` section for
information on developing for GILP.