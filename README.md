# GILP (Geometric Interpretation of Linear Programming)

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/henryrobbins/gilp)
[![PyPI download month](https://img.shields.io/pypi/dm/gilp.svg)](https://pypi.python.org/pypi/gilp/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/gilp.svg)](https://pypi.python.org/pypi/gilp/)


## Installation

The quickest way to start using gilp is with a pip install

```pip install gilp```

To develop and run tests, you will need to pip install with extra dependencies

```pip install gilp[dev]```

## Example

The LP class creates linear programs from (3) numpy arrays: A, b, and c which define the LP in standard inequality form.

max  c^Tx<br/>
s.t. Ax <= b<br/>
      x >= 0<br/>

Consider the following input.

```A = np.array([[1,0], [1, 2]])```<br/>
```b = np.array([[2],[4]])```<br/>
```c = np.array([[1],[1]])```<br/>
```lp = LP(A,b,c)```<br/>

The corresponding LP is:

max  1x_1 + 1x_2<br/>
s.t  1x_1 + 0x_2 <= 2<br/>
     1x_1 + 2x_2 <= 4<br/>
      x_1,   x_2 >= 0<br/>

To visualize the simplex algorithm on an LP, first create a plotly figure
and then use ```.show()``` to open up an HTML file or ```.write_html()```
to write an HTML file with a given name.

```fig = simplex_visual(lp)```<br/>
```fig.show()```<br/>
```fig.write_html('example.html)```<br/>
