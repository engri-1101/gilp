.. GILP documentation master file, created by
   sphinx-quickstart on Wed Jul 15 11:28:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: branding/gilp_color.png
  :height: 150
  :alt: GILP

|

Visualizing Linear Programs with GILP
=====================================

GILP (Geometric Interpretation of Linear Programs) is a Python package that
utilizes `Plotly <https://plotly.com/python/>`_ for visualizing the geometry of
linear programs (LPs) and the simplex algorithm. It was developed for the course
`ENGRI 1101: Engineering Applications of Operations Research
<https://classes.cornell.edu/search/roster/SP20?q=engri+1101&days-type
=any&crseAttrs-type=any&breadthDistr-type=any&pi=>`_ at Cornell University.
`GILP: An Interactive Tool for Visualizing the Simplex Algorithm
<https://arxiv.org/abs/2210.15655>`_ describes the development and usage of the
tool, and reports feedback from its use in the course.
Furthermore, GILP is part of the forthcoming book by David B. Shmoys, Samuel C.
Gutekunst, Frans Schalekamp, and David P. Williamson, entitled Data Science and
Decision Making: An Elementary Introduction to Modeling and Optimization.

This site contains multiple tutorials as well as the full GILP :ref:`docs`. If
you are new to `linear programming <https://en.wikipedia.org/wiki
/Linear_programming>`_ and the `simplex algorithm <https://
en.wikipedia.org/wiki/Simplex_algorithm>`_, we provide a brief :ref:`intro`. It
is recommended to start with the :ref:`quick` which includes installation
instructions and a tutorial. If you are interested in developing on GILP, see
:ref:`dev`. Lastly, :ref:`ex` contains mutliple example visualizations created
using GILP.

.. Potentially put some examples here.

.. toctree::
   :maxdepth: 2

   quickstart_guide/index
   development/index
   examples/index
   textbooks/index
   modules
