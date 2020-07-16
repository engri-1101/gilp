Tutorial
========

.. raw:: html

    <style> .red {color:red} </style>
    <p>
        <span class="red">
            Currently copied from README. To be expanded soon.
        </span>
    </p>

The LP class creates linear programs from (3) NumPy arrays: A, b, and c
which define the LP in standard inequality form.

.. tabularcolumns:: ll

+----------------------+-------------------+
| :math:`\max`         | :math:`c^Tx`      |
+----------------------+-------------------+
| :math:`\text{s.t.}`  | :math:`Ax \leq b` |
+----------------------+-------------------+
|                      | :math:`x \geq 0`  |
+----------------------+-------------------+

For example, consider the following LP:

.. tabularcolumns:: ll

+----------------------+-----------------------------+
| :math:`\max`         | :math:`5x_1 + 3x_2`         |
+----------------------+-----------------------------+
| :math:`\text{s.t.}`  | :math:`2x_1 + 1x_2 \leq 20` |
+----------------------+-----------------------------+
|                      | :math:`1x_1 + 1x_2 \leq 16` |
+----------------------+-----------------------------+
|                      | :math:`1x_1 + 0x_2 \leq 7`  |
+----------------------+-----------------------------+
|                      | :math:`x_1, x_2 \geq 0`     |
+----------------------+-----------------------------+

The LP instance is created as follows.

.. code-block:: python

   from gilp.simplex import LP

   A = np.array([[2, 1],
                 [1, 1],
                 [1, 0]])
   b = np.array([[20],
                 [16],
                 [7]])
   c = np.array([[5],
                 [3]])
   lp = LP(A,b,c)

After creating an LP, one can run simplex and generate a visualization with

.. code-block:: python

    from gilp.visualize import simplex_visual
    simplex_visual(lp)

where :code:`simplex_visual()` returns a plotly figure. The figure can then be viewed
on a Jupyter Notebook inline using

.. code-block:: python

    simplex_visual(lp).show()

If :code:`.show()` is run outside a Jupyter Notebook enviroment, the
visualization will open up in the browser. Alternatively, the HTML file can be
written and then opened.

.. code-block:: python

    simplex_visual(lp).write_html('name.html')

Below is the visualization for the example LP. The plot on the left shows the
feasible region and constraints of the LP. Hovering over an extreme point shows
the basic feasible solution, basis, and objective value. The iteration slider
allows one to toggle through the iterations of simplex and see the updating
tableaus. The objective slider lets one see the objective line or plane for
some range of objective values.

.. raw:: html
   :file: ../examples/ALL_INTEGER_2D_LP.html