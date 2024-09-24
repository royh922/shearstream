import yt
from matplotlib import rc_context
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import argparse
import numpy as np
from mpi4py import MPI

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

# Define the unit base for the simulation
unit_base = {
    "length_unit": (1.0, "pc"),
    "time_unit": (1.0, "s*pc/km"),
    "mass_unit": (2.38858753789e-24, "g/cm**3*pc**3"),
}
yt.enable_parallelism()

ts = yt.load(f"{args.directory}/{args.prefix}.*.athdf", units_override=unit_base)

# Define the field and axis
field = args.field
axis = args.axis

axis_map = {"x": 0, "y": 1, "z": 2}
axis_center = ts[0].domain_center[axis_map[axis]]

perp_axis_1 = (axis_map[axis] + 1) % 3
perp_axis_2 = (axis_map[axis] + 2) % 3

# Storage for the DataSets
storage = {"min": np.inf, "max": -np.inf}

# Buffer size
buffer = (args.resolution_x, args.resolution_y)


for sto, ds in ts.piter(storage=storage):
    slc = ds.slice(axis, axis_center)
    frb = yt.FixedResolutionBuffer(
        slc,
        (
            ds.domain_left_edge[perp_axis_1],
            ds.domain_right_edge[perp_axis_1],
            ds.domain_left_edge[perp_axis_2],
            ds.domain_right_edge[perp_axis_2],
        ),
        buffer,
    )  # Resolution

    # Store the preloaded buffer
    sto.result = frb[field]
    sto.result_id = str(ds)

    # Find min/max values
    field_data = frb[field].d
    storage["min"] = min(storage["min"], field_data.min())
    storage["max"] = max(storage["max"], field_data.max())

    # Check for NaN or Inf in data
    if np.isnan(field_data).any() or np.isinf(field_data).any():
        print(f"Invalid data (NaN or Inf) found in {ds}")
        quit(2)


if args.norm:
    norm = mcolors.LogNorm(vmin=storage["min"], vmax=storage["max"])
else:
    norm = mcolors.Normalize(vmin=storage["min"], vmax=storage["max"])

if MPI.COMM_WORLD.rank == 0:

    del storage["min"]
    del storage["max"]

    storage = [value for key, value in sorted(storage.items())]

    # Initial plot using the preloaded data
    plot = yt.SlicePlot(ts[0], axis, ("gas", field), buff_size=buffer)
    plot.set_norm(("gas", field), norm)

    # Lock the color scale by setting a fixed normalization
    plot.plots[("gas", field)].cb.norm = norm
    plot.plots[("gas", field)].cb._draw_all()

    # Extract the figure to be used in the animation
    fig = plot.plots[("gas", field)].figure

    # Get the axis object directly
    ax = plot.plots[("gas", field)].axes

    # Define the animation function
    def animate(i):
        frb = storage[i]  # Use preloaded data
        ax.images[0].set_array(frb)  # Update the plot
        ax.images[0].set_norm(norm)  # Update the normalization
        fig.canvas.draw_idle()

    # Create the animation object
    animation = FuncAnimation(fig, animate, frames=len(storage), interval=10)

    # Save the animation with locked colorbars
    with rc_context({"mathtext.fontset": "stix"}):
        animation.save(f"{args.prefix}_{args.field}_{args.axis}.mp4")
