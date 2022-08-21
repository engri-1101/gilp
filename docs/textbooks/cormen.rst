Introduction to Algorithms
==========================

This page includes several visualizations for linear programs used in the third
edition of *Introduction to Algorithms*, by Cormen, Leiserson, Rivest, and
Stein.

Chapter 29 of *Cormen, Leiserson, Rivest, and Stein* uses the following LP to
motivate fundamental LP concepts and introduce the graphical method for
solving 2-dimensional LPs.  See Equations (29.11-29.15).

.. math::

    \begin{align*}
    \text{maximize}  \quad & x_1+x_2\\
    \text{subject to} \quad & 4x_1-x_2 \leq 8 \\
    & 2x_1+x_2 \leq 10 \\
    & 5x_1 -2x_2 \geq -2 \\
    & x_1, x_2 \geq 0
    \end{align*}

Note that we have rewritten the third constraint as :math:`-5x_1+2x_2\leq 2`
when defining the LP in :code:`gilp` below

.. code-block:: python

    from gilp import examples
    from gilp.visualize import simplex_visual

    simplex_visual(examples.CORMEN_INTEGER_2D_LP).show()


.. raw:: html
   :file: ../visualizations/CORMEN_INTEGER_2D_LP.html

Section 29.3 of *Cormen, Leiserson, Rivest, and Stein* uses the following LP as
an extended example of the Simplex algorithm.  See Equations (29.53-29.57).

.. math::

    \begin{align*}
    \text{maximize}  \quad & 3x_1 + x_2 + 2x_3\\
    \text{subject to} \quad & x_1 + x_2 + 3x_3\leq 30 \\
    & 2x_1+2x_2+5x_3 \leq 24 \\
    & 4x_1+x_2+2x_3 \leq 36 \\
    & x_1, x_2, x_3 \geq 0
    \end{align*}

.. code-block:: python

    simplex_visual(examples.CORMEN_SIMPLEX_EX_3D_LP).show()

.. raw:: html
   :file: ../visualizations/CORMEN_SIMPLEX_EX_3D_LP.html

Section 29.3 of *Cormen, Leiserson, Rivest, and Stein* uses the following LP to
explain the phenomenon of **degeneracy**.

.. math::

    \begin{align*}
    \text{maximize}  \quad & x_1 + x_2 + x_3\\
    \text{subject to} \quad & x_1 + x_2 \leq 8 \\
    & -x_2+x_3 \leq 0 \\
    & x_1, x_2, x_3 \geq 0
    \end{align*}

Note that *Cormen, Leiserson, Rivest, and Stein* originally write this LP in
dictionary form as shown below.

.. math::

    \begin{align*}
    \text{maximize}  \quad & z= x_1 + x_2 + x_3\\
    \text{subject to} \quad & z_4 = 8 - x_1 - x_2 \\
    & x_5 = 0 +x_2 - x_3 \\
    & x_1, x_2, x_3, x_4, x_5\geq 0
    \end{align*}

When you run the following visualization, observe that iteration 0 of simplex
matches this dictionary form. Observe also that iterations 1 and 2 correspond
to the same corner point, with the same objective value. Using Bland's pivot
rule prevents cycling.

.. code-block:: python

    simplex_visual(examples.CORMEN_DEGENERATE_3D_LP).show()

.. raw:: html
   :file: ../visualizations/CORMEN_DEGENERATE_3D_LP.html
