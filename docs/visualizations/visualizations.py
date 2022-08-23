from gilp import lp_visual, simplex_visual
import gilp.examples as ex

# Examples to include in "Examples" section
# =========================================

simplex_visual(ex.ALL_INTEGER_2D_LP).write_html("ALL_INTEGER_2D_LP.html")
simplex_visual(ex.LIMITING_CONSTRAINT_2D_LP).write_html("LIMITING_CONSTRAINT_2D_LP.html")
simplex_visual(ex.DEGENERATE_FIN_2D_LP).write_html("DEGENERATE_FIN_2D_LP.html")
simplex_visual(ex.KLEE_MINTY_2D_LP, rule="dantzig").write_html("KLEE_MINTY_2D_LP.html")
simplex_visual(ex.ALL_INTEGER_3D_LP).write_html("ALL_INTEGER_3D_LP.html")
simplex_visual(ex.MULTIPLE_OPTIMAL_3D_LP).write_html("MULTIPLE_OPTIMAL_3D_LP.html")
simplex_visual(ex.SQUARE_PYRAMID_3D_LP).write_html("SQUARE_PYRAMID_3D_LP.html")
simplex_visual(ex.DODECAHEDRON_3D_LP).write_html("DODECAHEDRON_3D_LP.html")
simplex_visual(ex.KLEE_MINTY_3D_LP, rule="dantzig").write_html("KLEE_MINTY_3D_LP.html")
simplex_visual(ex.CLRS_INTEGER_2D_LP).write_html("CLRS_INTEGER_2D_LP.html")
simplex_visual(ex.CLRS_SIMPLEX_EX_3D_LP).write_html("CLRS_SIMPLEX_EX_3D_LP.html")
simplex_visual(ex.CLRS_DEGENERATE_3D_LP).write_html("CLRS_DEGENERATE_3D_LP.html")

# Examples to include in "Introduction" section
# =============================================

lp_visual(ex.ALL_INTEGER_2D_LP).write_html("feasible_region.html")

# Examples to include in "Tutorial" section
# =========================================

# Setting an Initial Solution

simplex_visual(ex.KLEE_MINTY_3D_LP).write_html("init_sol_origin.html")
simplex_visual(ex.KLEE_MINTY_3D_LP, initial_solution=[0,25,25]).write_html("init_sol_set.html")

# Iteration Limits
simplex_visual(ex.KLEE_MINTY_3D_LP, iteration_limit=3).write_html("iter_lim.html")

# Setting a Pivot Rule
simplex_visual(ex.KLEE_MINTY_3D_LP, rule="dantzig").write_html("rule_dantzig.html")
simplex_visual(ex.KLEE_MINTY_3D_LP, rule="greatest_ascent").write_html("rule_greatest_ascent.html")
print("Select 2 -> 3 -> 5")
simplex_visual(ex.KLEE_MINTY_3D_LP, rule="manual_select").write_html("rule_manual.html")
