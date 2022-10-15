"""Constants.

This module contains all constants for gilp (except for LP examples).
"""

__author__ = 'Henry Robbins'

import json

try:
    with open("gilp_style.json", "r") as f:
        style = json.load(f)
except FileNotFoundError:
    style = {}

# Color Theme -- Using Google's Material Design Color System
# https://material.io/design/color/the-color-system.html

PRIMARY_COLOR = style.get("PRIMARY_COLOR", "#1565c0")
PRIMARY_LIGHT_COLOR = style.get("PRIMARY_LIGHT_COLOR", "#5e92f3")
PRIMARY_DARK_COLOR = style.get("PRIMARY_DARK_COLOR", "#003c8f")
SECONDARY_COLOR = style.get("SECONDARY_COLOR", "#d50000")
SECONDARY_LIGHT_COLOR = style.get("SECONDARY_LIGHT_COLOR", "#ff5131")
SECONDARY_DARK_COLOR = style.get("SECONDARY_DARK_COLOR", "#9b0000")

# Grayscale
GRAY_COLOR = style.get("GRAY_COLOR", "#DFDFDF")
LIGHT_GRAY_COLOR = style.get("LIGHT_GRAY_COLOR", "#ffffff")
DARK_GRAY_COLOR = style.get("DARK_GRAY_COLOR", "#404040")

# Font Colors
LIGHT_FONT_COLOR = style.get("LIGHT_FONT_COLOR", "#ffffff")
DARK_FONT_COLOR = style.get("DARK_FONT_COLOR", "#404040")

# Branch and Bound Tree
BNB_CURRENT_COLOR = style.get("BNB_CURRENT_COLOR", "#45568B")
BNB_EXPLORED_COLOR = style.get("BNB_EXPLORED_COLOR", "#d8e4f9")
BNB_UNEXPLORED_COLOR = style.get("BNB_UNEXPLORED_COLOR", "#ffffff")

# 2D Constraint Colors
# NOTE: The fourth color in the list is the first color used in 2D simplex
# visuals. The first color is used in 2D branch and bound visuals.
CONSTRAINT_COLORS = \
    style.get("CONSTRAINT_COLORS",
              ['#1469FE', '#9495A6', '#DC0000', '#173D90', '#65ADFF'])

# Figure Dimensions
FIG_HEIGHT = style.get("FIG_HEIGHT", 500)
FIG_WIDTH = style.get("FIG_WIDTH", 950)
LEGEND_WIDTH = style.get("LEGEND_WIDTH", 200)
COMP_WIDTH = (FIG_WIDTH - LEGEND_WIDTH) / 2

ISOPROFIT_STEPS = 25
"""Number of isoprofit planes/lines plotted to the figure."""

# Plotly Default Attributes

LAYOUT = dict(width=FIG_WIDTH,
              height=FIG_HEIGHT,
              legend=dict(title=dict(text='<b>Constraint(s)</b>',
                                     font=dict(size=14)),
                          font=dict(size=13),
                          x=(1 - LEGEND_WIDTH / FIG_WIDTH) / 2, y=1,
                          xanchor='left', yanchor='top'),
              margin=dict(l=0, r=0, b=0, t=int(FIG_HEIGHT/15)),
              font=dict(family='Arial', color=DARK_FONT_COLOR),
              paper_bgcolor=LIGHT_GRAY_COLOR,
              plot_bgcolor=LIGHT_GRAY_COLOR,
              hovermode='closest',
              clickmode='none',
              dragmode='turntable')
"""Layout attributes."""

AXIS_2D = dict(gridcolor=GRAY_COLOR, gridwidth=1, linewidth=2,
               linecolor=DARK_GRAY_COLOR, tickcolor=GRAY_COLOR,
               ticks='outside', rangemode='tozero', showspikes=False,
               title_standoff=15, automargin=True, zerolinewidth=1,
               layer="below traces")
"""2d axis attributes."""

AXIS_3D = dict(backgroundcolor=LIGHT_GRAY_COLOR, showbackground=True,
               gridcolor=GRAY_COLOR, gridwidth=2, showspikes=False,
               linecolor=DARK_GRAY_COLOR, zerolinecolor='white',
               rangemode='tozero', ticks='')
"""3d axis attributes."""

SLIDER = dict(x=0.5 + ((LEGEND_WIDTH / FIG_WIDTH) / 2), xanchor="left",
              yanchor="bottom", lenmode='fraction', len=COMP_WIDTH / FIG_WIDTH,
              active=0, tickcolor='white', ticklen=0)
"""slider attributes."""

TABLE = dict(header_font_color=[SECONDARY_COLOR, 'black'],
             header_fill_color=LIGHT_GRAY_COLOR,
             cells_font_color=[['black', SECONDARY_COLOR, 'black'],
                               ['black', 'black', 'black']],
             cells_fill_color=LIGHT_GRAY_COLOR,
             visible=False)
"""table attributes."""

SCATTER = dict(mode='markers',
               hoverinfo='none',
               visible=True,
               showlegend=False,
               cliponaxis=False,
               fillcolor=PRIMARY_COLOR,
               line=dict(width=4,
                         color=PRIMARY_DARK_COLOR),
               marker_line=dict(width=2,
                                color=SECONDARY_COLOR),
               marker=dict(size=9,
                           color=LIGHT_GRAY_COLOR,
                           opacity=0.99))
"""2d scatter attributes."""

SCATTER_3D = dict(mode='markers',
                  hoverinfo='none',
                  visible=True,
                  showlegend=False,
                  surfacecolor=PRIMARY_LIGHT_COLOR,
                  line=dict(width=6,
                            color=PRIMARY_COLOR),
                  marker_line=dict(width=1,
                                   color=SECONDARY_COLOR),
                  marker=dict(size=5,
                              symbol='circle-open',
                              color=SECONDARY_LIGHT_COLOR,
                              opacity=0.99))
"""3d scatter attributes."""

# Plotly Template Attributes

TABLEAU_TABLE = dict(header=dict(height=30,
                                 font_size=13,
                                 line=dict(color='black', width=1)),
                     cells=dict(height=25,
                                font_size=13,
                                line=dict(color='black',width=1)),
                     columnwidth=[1,0.8])
"""Template attributes for an LP in tableau form."""

DICTIONARY_TABLE = dict(header=dict(height=25,
                                    font_size=14,
                                    align=['left', 'right', 'left'],
                                    line_color=LIGHT_GRAY_COLOR,
                                    line_width=1),
                        cells=dict(height=25,
                                   font_size=14,
                                   align=['left', 'right', 'left'],
                                   line_color=LIGHT_GRAY_COLOR,
                                   line_width=1),
                        columnwidth=[50/COMP_WIDTH,
                                     25/COMP_WIDTH,
                                     1 - (75/COMP_WIDTH)])
"""Template attributes for an LP in dictionary form."""

BFS_SCATTER = dict(marker=dict(size=20, color='gray', opacity=1e-7),
                   hoverinfo='text',
                   hoverlabel=dict(bgcolor=LIGHT_GRAY_COLOR,
                                   bordercolor=DARK_GRAY_COLOR,
                                   font_family='Arial',
                                   font_color=DARK_FONT_COLOR,
                                   align='left'))
"""Template attributes for an LP basic feasible solutions (BFS)."""

VECTOR = dict(mode='lines', line_color=SECONDARY_COLOR, line_width=5,
              visible=False)
"""Template attributes for a 2d or 3d vector."""

CONSTRAINT_LINE = dict(mode='lines', showlegend=True,
                       line=dict(width=3, dash='12,3'))
"""Template attributes for (2d) LP constraints."""

ISOPROFIT_LINE = dict(mode='lines', visible=False,
                      line=dict(color=SECONDARY_COLOR, width=4, dash=None))
"""Template attributes for (2d) LP isoprofit lines."""

REGION_2D_POLYGON = dict(mode="lines", opacity=0.2, fill="toself",
                         line=dict(width=3, color=PRIMARY_DARK_COLOR))
"""Template attributes for (2d) LP feasible region."""

REGION_3D_POLYGON = dict(mode="lines", opacity=0.2,
                         line=dict(width=5, color=PRIMARY_DARK_COLOR))
"""Template attributes for (3d) LP feasible region."""

INTEGER_POINT = dict(marker=dict(color=DARK_GRAY_COLOR,
                                 line_color=DARK_GRAY_COLOR,
                                 symbol='circle'),
                     opacity=0.4)
"""Template attributes for a 2d or 3d integer point."""

CONSTRAINT_POLYGON = dict(surfacecolor='gray', mode="none",
                          opacity=0.5, visible='legendonly',
                          showlegend=True)
"""Template attributes for (3d) LP constraints."""

ISOPROFIT_IN_POLYGON = dict(mode="lines+markers",
                            surfacecolor=SECONDARY_COLOR,
                            marker=dict(size=5,
                                        symbol='circle',
                                        color=SECONDARY_COLOR),
                            line=dict(width=5,
                                      color=SECONDARY_COLOR),
                            visible=False)
"""Template attributes for (3d) LP isoprofit plane (interior)."""

ISOPROFIT_OUT_POLYGON = dict(surfacecolor='gray', mode="none",
                             opacity=0.3, visible=False)
"""Template attributes for (3d) LP isoprofit plane (exterior)."""

BNB_NODE = dict(visible=False, align="center",
                bordercolor=DARK_GRAY_COLOR, borderwidth=2, borderpad=3,
                font=dict(size=12, color=DARK_FONT_COLOR), ax=0, ay=0)
"""Template attributes for a branch and bound node."""
