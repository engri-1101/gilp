import pytest
import gilp.examples as ex
from gilp.simplex import LP
from gilp.visualize import (InfiniteFeasibleRegion, template_figure,
                            lp_strings, feasible_region, simplex_visual,
                            lp_visual, bnb_visual, feasible_integer_pts)


# The following functions are not tested since they create visual objects:
# set_up_fig, plot_lp, add_path, add_isoprofits, add_tableaus, simplex_visual


def test_template_figure():
    with pytest.raises(ValueError, match='.*visualize 2 or 3 dimensional.*'):
        template_figure(4)


def test_feasible_region(unbounded_lp):
    with pytest.raises(InfiniteFeasibleRegion):
        feasible_region(unbounded_lp)


def test_lp_strings(degenerate_lp):
    B = [0,1,4,5,6]
    canonical_head = ['<b>(3) z</b>', '<b>x<sub>1</sub></b>',
                      '<b>x<sub>2</sub></b>', '<b>x<sub>3</sub></b>',
                      '<b>x<sub>4</sub></b>', '<b>x<sub>5</sub></b>',
                      '<b>x<sub>6</sub></b>', '<b>x<sub>7</sub></b>',
                      '<b>RHS</b>']
    canonical_cont = [['1<br>0<br>0<br>0<br>0<br>0'],
                      ['0<br>1<br>0<br>0<br>0<br>0'],
                      ['0<br>0<br>1<br>0<br>0<br>0'],
                      ['1<br>1<br>0<br>-1<br>-1<br>2'],
                      ['1<br>-1<br>1<br>2<br>1<br>-3'],
                      ['0<br>0<br>0<br>1<br>0<br>0'],
                      ['0<br>0<br>0<br>0<br>1<br>0'],
                      ['0<br>0<br>0<br>0<br>0<br>1'],
                      ['10<br>2<br>4<br>4<br>1<br>0']]
    dictionary_head = ['<b>(3)</b>',' ', ' ']
    dictionary_cont = [['max<br>s.t.<br> <br> <br> <br> '],
                       ['z<br>x<sub>1</sub><br>x<sub>2</sub><br>x<sub>5</sub>'
                        + '<br>x<sub>6</sub><br>x<sub>7</sub>'],
                       ['= 10 - 1x<sub>3</sub> - 1x<sub>4</sub><br>'
                        + '= 2 - 1x<sub>3</sub> + 1x<sub>4</sub><br>'
                        + '= 4 + 0x<sub>3</sub> - 1x<sub>4</sub><br>'
                        + '= 4 + 1x<sub>3</sub> - 2x<sub>4</sub><br>'
                        + '= 1 + 1x<sub>3</sub> - 1x<sub>4</sub><br>'
                        + '= 0 - 2x<sub>3</sub> + 3x<sub>4</sub>']]

    actual = lp_strings(degenerate_lp,B,3,'tableau')
    assert canonical_head == actual[0]
    assert canonical_cont == actual[1]
    actual = lp_strings(degenerate_lp,B,3,'dictionary')
    assert dictionary_head == actual[0]
    assert dictionary_cont == actual[1]


def test_lp_visual():
    # Does not check for correctness but ensures no errors
    tests = [ex.ALL_INTEGER_3D_LP,
             ex.ALL_INTEGER_2D_LP,
             ex.DEGENERATE_FIN_2D_LP,
             ex.KLEE_MINTY_2D_LP,
             ex.KLEE_MINTY_3D_LP,
             ex.LIMITING_CONSTRAINT_2D_LP,
             ex.MULTIPLE_OPTIMAL_3D_LP,
             ex.SQUARE_PYRAMID_3D_LP]
    for test in tests:
        lp_visual(test)


def test_simplex_visual():
    # Does not check for correctness but ensures no errors
    tests = [ex.ALL_INTEGER_3D_LP,
             ex.ALL_INTEGER_2D_LP,
             ex.DEGENERATE_FIN_2D_LP,
             ex.KLEE_MINTY_2D_LP,
             ex.KLEE_MINTY_3D_LP,
             ex.LIMITING_CONSTRAINT_2D_LP,
             ex.MULTIPLE_OPTIMAL_3D_LP,
             ex.SQUARE_PYRAMID_3D_LP]
    for test in tests:
        simplex_visual(test)


def test_bnb_visual():
    # Does not check for correctness but ensures no errors
    tests = [ex.DODECAHEDRON_3D_LP,
             ex.STANDARD_2D_IP,
             ex.EVERY_FATHOM_2D_IP,
             ex.VARIED_BRANCHING_3D_IP]
    for test in tests:
        bnb_visual(test)


def test_integer_points():
    lp = LP([[1,1],[2,0],[-1,2]],
            [3,5,2],
            [2,1])
    fig = lp_visual(lp)
    fig.add_trace(feasible_integer_pts(lp, fig))

    fig = lp_visual(ex.DODECAHEDRON_3D_LP)
    fig.add_trace(feasible_integer_pts(ex.DODECAHEDRON_3D_LP, fig))
