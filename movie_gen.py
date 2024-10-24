'''
	What: create movie using timeseries of .npy files
	How: python movie_gen.py . kh_custom.out1 2048 1024 --field temperature --axis z
		!!!Do not use MPI on this!!!
	Note: Right now it naively creates plots using the first athdf file and YT. 
		Will need to be changed. 
'''

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


import glob 
fns = glob.glob(f"{args.directory}/{args.prefix}.*.npy")
fns.sort()

T_hot = np.load(fns[0]).max()
T_min = np.load(fns[0]).min()

if args.norm:
    norm = mcolors.LogNorm(vmin=T_min/T_hot, vmax=1)
else:
    norm = mcolors.Normalize(vmin=T_min/T_hot, vmax=1)

# TODO: Remove dependency on first athdf block
first = yt.load(f"kh_custom.out1.00000.athdf")

# Initial plot using the preloaded data
plot = yt.SlicePlot(first, axis, ("gas", field), buff_size=buffer)
plot.set_norm(("gas", field), norm)
plot.set_cmap(("gas", field), "RdBu_r")

# Lock the color scale by setting a fixed normalization
plot.plots[("gas", field)].cb.norm = norm

plot.plots[("gas", field)].cb.set_label("$T/T_{\mathrm{hot}}$")

# Extract the figure to be used in the animation
fig = plot.plots[("gas", field)].figure

# Get the axis object directly
ax = plot.plots[("gas", field)].axes

# Define the animation function
def animate(i):
    data = np.load(fns[i])/T_hot	# TODO: Generalization (eventually)
    ax.images[0].set_array(data)  # Update the plot
    ax.images[0].set_norm(norm)  # Update the normalization
    ax.images[0].set_cmap("RdBu_r")  # Update the colormap
    fig.canvas.draw_idle()
    # storage[i] = None
    gc.collect()

# Create the animation object
animation = FuncAnimation(fig, animate, frames=len(fns), interval=100, save_count=None)

# Save the animation with locked colorbars
with rc_context({"mathtext.fontset": "stix"}):
    animation.save(f"{args.prefix}_{args.field}.mp4")

# Clean up temp files
# os.system(f'rm {args.directory}/*.npy')
