"""style.py
Contains standard style for figures.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from matplotlib import rcParams
try:
    import seaborn as sns
    sns.set(style='ticks', palette='Set1') 
except:
    print('seaborn not installed')

plot_style  = ["pdf", "print", "presentation", "poster"][0]

# Figure size
# Only specify height in inch. Width is calculated from golden ratio.
height = 3.4   # inch
linewidth  = 1.0
cross_size = 9 # pt, size of cross markers

# Choose parameters for pdf or print
if plot_style == "pdf":
    figure_path = os.path.join(".", "figures")
#     axes_color = "#636363" 
#     axes_color = "#bdbdbd" 
    axes_color = "#959595" 
#     text_color = "#636363"
    text_color = "#363636"
    font_family         = 'serif'
elif plot_style == "print":
    figure_path = os.path.join(".", "figures")
    axes_color = "#959595" 
    text_color = "#363636"
    # Font
    latex_preamble      = [r'\usepackage[T1,small,euler-digits]{eulervm}']
    font_family         = 'serif'
elif plot_style == "presentation":
    figure_path = os.path.join("..", "presentation", "figures")
    axes_color = "#959595" 
    text_color = "#363636"
    # Font
    latex_preamble      = [r'\usepackage[cmbright]{sfmath}']
    rcParams['text.latex.preamble'] = latex_preamble
    font_family         = 'sans-serif'
elif plot_style == "poster":
    figure_path = os.path.join("..", "poster", "figures") 
    axes_color = "#959595" 
    text_color = "#363636"
    # Font
    latex_preamble      = [r'\usepackage[cmbright]{sfmath}']
    rcParams['text.latex.preamble'] = latex_preamble
    font_family         = 'sans-serif'
    # Figure size
    #        cm  / 2.54 cm * 1.0 inch  
    height = 9.5 / 2.54     # inch
    linewidth  = 0.8
    cross_size = 9 # pt, size of cross markers

# Figure size
from  scipy.constants import golden_ratio as gr
figsize  = (gr * height, 1. * height)

fontsize_labels         = 12    # pt, size used in latex document
fontsize_labels_axes    = fontsize_labels
fontsize_labels_title   = fontsize_labels
fontsize_plotlabel      = fontsize_labels       # for labeling plots with 'A', 'B', etc.
legend_ms  = 2  # scale of markers in legend

# # Use sans-serif with latex in matplotlib
# # https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlibhttps://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
# rcParams['text.latex.preamble'] = [
#     r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#     r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
#     r'\usepackage{helvet}',    # set the normal font here
#     r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#     r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
# ]

# Adapt the matplotlib.rc
rcParams['font.family']         = font_family
# rcParams['font.serif']          = 'Computer Modern'
rcParams['text.usetex']         = True
rcParams['figure.figsize']      = figsize
rcParams['font.weight']         = "light"
rcParams['font.size']           = fontsize_labels
rcParams['xtick.labelsize']     = fontsize_labels
rcParams['ytick.labelsize']     = fontsize_labels
rcParams['legend.fontsize']     = fontsize_labels
rcParams['axes.labelsize']      = fontsize_labels_axes
rcParams['axes.titlesize']      = fontsize_labels_title
rcParams['legend.markerscale']  = legend_ms
rcParams['text.color']          = text_color
rcParams['xtick.color']         = text_color
rcParams['ytick.color']         = text_color
rcParams['axes.labelcolor']     = text_color
rcParams['axes.edgecolor']      = axes_color
rcParams['axes.grid']           = False
rcParams['lines.linewidth']     = linewidth
rcParams['lines.markersize']    = 3
# rcParams['figure.autolayout']   = True # this disables tight layout!

tick_params = {
                ## TICKS
                # see http://matplotlib.org/api/axis_api.html#matplotlib.axis.Tick
                'xtick.major.size'     : 3,      # major tick size in points
                'xtick.minor.size'     : 2,      # minor tick size in points
                'xtick.major.width'    : 0.5,    # major tick width in points
                'xtick.minor.width'    : 0.5,    # minor tick width in points
                'xtick.major.pad'      : 4,      # distance to major tick label in points
                'xtick.minor.pad'      : 4,      # distance to the minor tick label in points
                'xtick.direction'      : 'out',    # direction: in, out, or inout
                
                'ytick.major.size'     : 3,      # major tick size in points
                'ytick.minor.size'     : 2,      # minor tick size in points
                'ytick.major.width'    : 0.5,    # major tick width in points
                'ytick.minor.width'    : 0.5,    # minor tick width in points
                'ytick.major.pad'      : 4,      # distance to major tick label in points
                'ytick.minor.pad'      : 4,      # distance to the minor tick label in points
                'ytick.direction'      : 'out'    # direction: in, out, or inout
                }
rcParams.update(tick_params)


# Convenience functions
def tick_manager(ax, max_tick_num=[5, 4]):
    """ Reduce the number of ticks to a bare minimum. """
    for max_n_ticks, ticks, lim, tick_setter in zip(
        max_tick_num,
        [ax.get_xticks(), ax.get_yticks()],
        [ax.get_xlim(), ax.get_ylim()],
        [lambda t: ax.set_xticks(t), lambda t: ax.set_yticks(t)]):
        # Reduce ticks to those anyways visible on the lower side
        ticks = ticks[(ticks >= lim[0]) * (ticks <= lim[1])]
        
        n_ticks = len(ticks)
        while n_ticks > max_n_ticks:
            ticks = ticks[::2]
            n_ticks = len(ticks)
        # Set new ticks
        tick_setter(ticks)
    return None

def fixticks(fig_or_ax, fix_spines=True, max_tick_num=[5, 4]):
    """ Polishes graphs.
    Input: figure, list of axes or single axes. 
    """
    if type(fig_or_ax) is matplotlib.figure.Figure:  
        axes = fig_or_ax.axes
    elif type(fig_or_ax) is list:
        axes = fig_or_ax
    else:
        axes = [fig_or_ax]
    for ax in axes:
        ax.grid(False)      # Turn off grid (distracts!)
        # Set spines to color of axes
        for t in ax.xaxis.get_ticklines(): t.set_color(axes_color)
        for t in ax.yaxis.get_ticklines(): t.set_color(axes_color)
        if fix_spines:
            # Remove top axes & spines
            ax.spines['top'].set_visible(False)
            #ax.xaxis.set_ticks_position('bottom') # this resets spines in case of sharex=True...
            # Remove axes & spines on the side not used
            active_side = ax.yaxis.get_ticks_position()
            if active_side in ['default', 'left']:
                inactive_side = 'right'
            elif active_side == 'right':
                inactive_side = 'left'
            if active_side in ['default', 'left', 'right']:
                ax.spines[inactive_side].set_visible(False)
                ax.yaxis.set_ticks_position(active_side)
            # Note: otherwise active_side == 'unknown'
        # Take care of tick number
        tick_manager(ax, max_tick_num)
    return None

def saving_fig(fig, figure_path, fig_name, data_type="both", verbose=True, dpi=1000):
    # DPI = 1000 is a simple recommendation for publication plots
    from pathlib import Path
    if not Path(figure_path).is_dir():
        os.makedirs(figure_path)
        print("Made new directory ", figure_path)        
    if verbose:
        print("Save figure to " + os.path.join(figure_path, fig_name) + "." + data_type)
    if data_type == "png":
        fig.savefig(os.path.join(figure_path, fig_name + ".png"), 
                dpi=dpi, 
                bbox_inches='tight', format="png") 
    elif data_type == "pdf":
        fig.savefig(os.path.join(figure_path, fig_name + ".pdf"), 
                dpi=dpi, 
                bbox_inches='tight', format="pdf")
    elif data_type=="both":
        # Both
        fig.savefig(os.path.join(figure_path, fig_name + ".png"), 
                dpi=dpi, 
                bbox_inches='tight', format="png") 
        fig.savefig(os.path.join(figure_path, fig_name + ".pdf"), 
                dpi=dpi, 
                bbox_inches='tight', format="pdf")
    elif data_type == "svg":
        fig.savefig(os.path.join(figure_path, fig_name + "." + data_type),
                format="svg")
    else:
        fig.savefig(os.path.join(figure_path, fig_name + "." + data_type),
                format=data_type)
