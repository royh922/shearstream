import yt
from matplotlib import rc_context
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import argparse
import numpy as np
import os
import gc
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Parse the command line arguments
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

if comm.rank == 0:

    import glob 
    fns = glob.glob(f"{args.directory}/{args.prefix}.*.athdf")
    fns.sort()

    # Define the unit base for the simulation
    unit_base = {
        "length_unit": (1.0, "pc"),
        "time_unit": (1.0, "s*pc/km"),
        "mass_unit": (2.38858753789e-24, "g/cm**3*pc**3"),
    }

    # Define the field and axis
    axis = args.axis
    field = args.field

    # Buffer size
    buffer = (args.resolution_x, args.resolution_y)

    axis_map = {"x": 0, "y": 1, "z": 2}

    # Load the first dataset to get the domain center
    first = yt.load(fns[0], units_override=unit_base)
    # first = yt.load(fns[0])
    axis_center = first.domain_center[axis_map[axis]]

    perp_axis_1 = (axis_map[axis] + 1) % 3
    perp_axis_2 = (axis_map[axis] + 2) % 3

else: 
    fns = None
    unit_base = None
    axis = None
    field = None
    perp_axis_1 = None
    perp_axis_2 = None
    axis_center = None
    storage = None
    buffer = None


# Broadcast the necessary variables to all ranks
fns = comm.bcast(fns, root=0)
unit_base = comm.bcast(unit_base, root=0)
axis = comm.bcast(axis, root=0)
field = comm.bcast(field, root=0)
perp_axis_1 = comm.bcast(perp_axis_1, root=0)
perp_axis_2 = comm.bcast(perp_axis_2, root=0)
axis_center = comm.bcast(axis_center, root=0)
buffer = comm.bcast(buffer, root=0)


# ONLY ENABLE PARALLELISM AFTER BROADCASTING VARIABLES BECAUSE YT IS ASS AND SUCKS AT ACTUALLY DOING SHIT
yt.enable_parallelism()

# Storage for the DataSets
storage = {"min": np.inf, "max": -np.inf}

for sto, fn in yt.parallel_objects(fns, -1, storage=storage, dynamic=True):

    ds = yt.load(fn, units_override=unit_base)
    slc = ds.slice(axis, axis_center)
    frb = yt.FixedResolutionBuffer(
        slc,
        (
            ds.domain_left_edge[perp_axis_1],
            ds.domain_right_edge[perp_axis_1],
            ds.domain_left_edge[perp_axis_2],
            ds.domain_right_edge[perp_axis_2],
        ),
        buffer
    )  # Resolution

    field_data = frb[field].d

    np.save(f'{fn}.npy', field_data)

    # Find min/max values
    # storage["min"] = min(storage["min"], field_data.min())
    storage["max"] = max(storage["max"], field_data.max())

    # Check for NaN or Inf in data
    if np.isnan(field_data).any() or np.isinf(field_data).any():
        print(f"Invalid data (NaN or Inf) found in {ds}")
        quit(2)
    
    ds = None

if args.norm:
    norm = mcolors.LogNorm(vmin=1e-2, vmax=1)
else:
    norm = mcolors.Normalize(vmin=1e-2, vmax=1)

local_max = storage["max"]
global_max = comm.reduce(local_max, op=MPI.MAX, root=0)

if yt.is_root():

    T_hot = global_max

    # del storage["min"]
    # del storage["max"]

    # storage = [value for key, value in sorted(storage.items())]

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
        data = np.load(f'{fns[i]}.npy') / T_hot
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
