import yt
from matplotlib import rc_context
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import argparse
import numpy as np
import os
import gc 

parser = argparse.ArgumentParser()
parser.add_argument(
    "directory", type=str, help="Directory containing the simulation output"
)
parser.add_argument("prefix", type=str, help="Prefix of the athdf5 files")
parser.add_argument(
    "resolution_x", type=int, help="Resolution in the horizontal direction"
)
parser.add_argument(
    "resolution_y", type=int, help="Resolution in the vertical direction"
)
parser.add_argument(
    "--field", type=str, help="Field to plot (density, temperature, etc.)"
)
parser.add_argument("--axis", type=str, help="Axis to slice along (x, y, z)")
parser.add_argument(
    "--norm",
    type=bool,
    default=True,
    help="Normalization for the colorbar, True if Log, False if Linear. Defaults to True.",
)
args = parser.parse_args()

axis = args.axis
field = args.field
buffer = (args.resolution_x, args.resolution_y)

if args.norm:
    norm = mcolors.LogNorm(vmin=1e-2, vmax=1)
else:
    norm = mcolors.Normalize(vmin=1e-2, vmax=1)

import glob 
fns = glob.glob(f"{args.directory}/{args.prefix}.*.npy")
fns.sort()

T_hot = np.load('global_max.npy')

first = yt.load(f"kh_custom.out1.00000.athdf")

# Initial plot using the preloaded data
plot = yt.SlicePlot(first, axis, ("gas", field), buff_size=buffer)
plot.set_norm(("gas", field), norm)
plot.set_cmap(("gas", field), "RdBu")

# Lock the color scale by setting a fixed normalization
plot.plots[("gas", field)].cb.norm = norm

plot.plots[("gas", field)].cb.set_label("$T/T_{\mathrm{hot}}$")

# Extract the figure to be used in the animation
fig = plot.plots[("gas", field)].figure

# Get the axis object directly
ax = plot.plots[("gas", field)].axes

# Define the animation function
def animate(i):
    data = np.load(fns[i]) / T_hot
    ax.images[0].set_array(data)  # Update the plot
    ax.images[0].set_norm(norm)  # Update the normalization
    ax.images[0].set_cmap("RdBu")  # Update the colormap
    fig.canvas.draw_idle()
    # storage[i] = None
    gc.collect()

# Create the animation object
animation = FuncAnimation(fig, animate, frames=len(fns), interval=10, save_count=None)

# Save the animation with locked colorbars
with rc_context({"mathtext.fontset": "stix"}):
    animation.save(f"{args.prefix}_{args.field}.mp4")

# Clean up temp files
os.system(f'rm {args.directory}/*.npy')