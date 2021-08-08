Package Overview
================

This package overview serves as a source of necessary and helpful information
for developing gilp. First, we will discuss the structure of the gilp package
at a high level.


Package Structure
-----------------

The gilp package contains 4 modules: :code:`simplex`, :code:`style`,
:code:`visualize`, and :code:`examples`. The :code:`simplex` module contains the
:code:`LP` class definition as well as an implementation of the revised simplex
method. Additionally, it contains some custom exception classes that can be
thrown by the :code:`LP` methods and simplex functions. The :code:`style` module
mainly serves as a higher level interface with the `Plotly Graphing Library
<https://plotly.com/python/>`_. Furthermore, it contains some additional
functions for styling text and numbers. The height, width, and background color
of the generated visualizations are set with constants in this module. The
:code:`visualize` drives most of the gilp package. This module utilizes the
:code:`simplex` and :code:`style` modules to generate interactive
visualizations. Additionally, it contains a custom exception class and
constants which specify properties of the visualization. Lastly, the
:code:`examples` module contains 8 example LPs in the form of 8 constants. Now,
we will go into each module in more detail.

Simplex Module
~~~~~~~~~~~~~~

The main components of the



Style Module
~~~~~~~~~~~~


Visualize Module
~~~~~~~~~~~~~~~~


Examples Module
~~~~~~~~~~~~~~~





