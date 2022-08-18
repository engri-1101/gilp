"""Constants.

This module contains all constants for gilp (except for LP examples).
"""

__author__ = 'Henry Robbins'

# Color Theme -- Using Google's Material Design Color System
# https://material.io/design/color/the-color-system.html

PRIMARY_COLOR = '#1565c0'
PRIMARY_LIGHT_COLOR = '#5e92f3'
PRIMARY_DARK_COLOR = '#003c8f'
SECONDARY_COLOR = '#d50000'
SECONDARY_LIGHT_COLOR = '#ff5131'
SECONDARY_DARK_COLOR = '#9b0000'
PRIMARY_FONT_COLOR = '#ffffff'
SECONDARY_FONT_COLOR = '#ffffff'
# Grayscale
TERTIARY_COLOR = '#DFDFDF'
TERTIARY_LIGHT_COLOR = 'white'  # Jupyter Notebook: white, Sphinx: #FCFCFC
TERTIARY_DARK_COLOR = '#404040'

# Figure Dimensions
FIG_HEIGHT = 500
FIG_WIDTH = 950  # Jupyter Notebook: 950, Sphinx: 700
LEGEND_WIDTH = 200
COMP_WIDTH = (FIG_WIDTH - LEGEND_WIDTH) / 2

ISOPROFIT_STEPS = 25
"""Number of isoprofit planes/lines plotted to the figure."""

# Plotly Default Attributes

LAYOUT = dict(width=FIG_WIDTH,
              height=FIG_HEIGHT,
              title=dict(text="<b>Geometric Interpretation of LPs</b>",
                         font=dict(size=18,
                                   color=TERTIARY_DARK_COLOR),
                         x=0, y=0.99, xanchor='left', yanchor='top'),
              legend=dict(title=dict(text='<b>Constraint(s)</b>',
                                     font=dict(size=14)),
                          font=dict(size=13),
                          x=(1 - LEGEND_WIDTH / FIG_WIDTH) / 2, y=1,
                          xanchor='left', yanchor='top'),
              margin=dict(l=0, r=0, b=0, t=int(FIG_HEIGHT/15)),
              font=dict(family='Arial', color=TERTIARY_DARK_COLOR),
              paper_bgcolor=TERTIARY_LIGHT_COLOR,
              plot_bgcolor=TERTIARY_LIGHT_COLOR,
              hovermode='closest',
              clickmode='none',
              dragmode='turntable')
"""Layout attributes."""

AXIS_2D = dict(gridcolor=TERTIARY_COLOR, gridwidth=1, linewidth=2,
               linecolor=TERTIARY_DARK_COLOR, tickcolor=TERTIARY_COLOR,
               ticks='outside', rangemode='tozero', showspikes=False,
               title_standoff=15, automargin=True, zerolinewidth=2)
"""2d axis attributes."""

AXIS_3D = dict(backgroundcolor=TERTIARY_LIGHT_COLOR, showbackground=True,
               gridcolor=TERTIARY_COLOR, gridwidth=2, showspikes=False,
               linecolor=TERTIARY_DARK_COLOR, zerolinecolor='white',
               rangemode='tozero', ticks='')
"""3d axis attributes."""

SLIDER = dict(x=0.5 + ((LEGEND_WIDTH / FIG_WIDTH) / 2), xanchor="left",
              yanchor="bottom", lenmode='fraction', len=COMP_WIDTH / FIG_WIDTH,
              active=0, tickcolor='white', ticklen=0)
"""slider attributes."""

TABLE = dict(header_font_color=[SECONDARY_COLOR, 'black'],
             header_fill_color=TERTIARY_LIGHT_COLOR,
             cells_font_color=[['black', SECONDARY_COLOR, 'black'],
                               ['black', 'black', 'black']],
             cells_fill_color=TERTIARY_LIGHT_COLOR,
             visible=False)
"""table attributes."""

SCATTER = dict(mode='markers',
               hoverinfo='none',
               visible=True,
               showlegend=False,
               fillcolor=PRIMARY_COLOR,
               line=dict(width=4,
                         color=PRIMARY_DARK_COLOR),
               marker_line=dict(width=2,
                                color=SECONDARY_COLOR),
               marker=dict(size=9,
                           color=TERTIARY_LIGHT_COLOR,
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
                                    line_color=TERTIARY_LIGHT_COLOR,
                                    line_width=1),
                        cells=dict(height=25,
                                   font_size=14,
                                   align=['left', 'right', 'left'],
                                   line_color=TERTIARY_LIGHT_COLOR,
                                   line_width=1),
                        columnwidth=[50/COMP_WIDTH,
                                     25/COMP_WIDTH,
                                     1 - (75/COMP_WIDTH)])
"""Template attributes for an LP in dictionary form."""

BFS_SCATTER = dict(marker=dict(size=20, color='gray', opacity=1e-7),
                   hoverinfo='text',
                   hoverlabel=dict(bgcolor=TERTIARY_LIGHT_COLOR,
                                   bordercolor=TERTIARY_DARK_COLOR,
                                   font_family='Arial',
                                   font_color=TERTIARY_DARK_COLOR,
                                   align='left'))
"""Template attributes for an LP basic feasible solutions (BFS)."""

VECTOR = dict(mode='lines', line_color=SECONDARY_COLOR, visible=False)
"""Template attributes for a 2d or 3d vector."""

CONSTRAINT_LINE = dict(mode='lines', showlegend=True,
                       line=dict(width=2, dash='15,3,5,3'))
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

INTEGER_POINT = dict(marker=dict(color=TERTIARY_DARK_COLOR,
                                 line_color=TERTIARY_DARK_COLOR,
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
                bordercolor=TERTIARY_DARK_COLOR, borderwidth=2, borderpad=3,
                font=dict(size=12, color=TERTIARY_DARK_COLOR), ax=0, ay=0)
"""Template attributes for a branch and bound node."""
