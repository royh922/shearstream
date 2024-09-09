import yt
from matplotlib import rc_context
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import argparse
import numpy as np

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("directory", type=str, help="Directory containing the simulation output")
parser.add_argument("prefix", type=str, help="Prefix of the athdf5 files")
parser.add_argument("--field", type=str, help="Field to plot (density, temperature, etc.)")
parser.add_argument("--axis", type=str, help="Axis to slice along (x, y, z)")
parser.add_argument("--norm", type=bool, default=True, help="Normalization for the colorbar, True if Log, False if Linear. Defaults to True.")
args = parser.parse_args()

yt.enable_parallelism()

# Define the unit base for the simulation
unit_base = {"length_unit": (1.0, "pc"), "time_unit": (1.0, "s*pc/km"), "mass_unit": (2.38858753789e-24, "g/cm**3*pc**3")}
ts = yt.load(f"{args.directory}/{args.prefix}.*.athdf", units_override=unit_base)

# Define the field and axis
field = args.field
axis = args.axis

# Preload data and calculate min/max values in one pass
preloaded_data = []
vmin, vmax = np.inf, -np.inf

for ds in ts:
    slc = ds.slice(axis, 0.5)
    frb = yt.FixedResolutionBuffer(slc, (ds.domain_left_edge[0], ds.domain_right_edge[0],
                                         ds.domain_left_edge[1], ds.domain_right_edge[1]), (800, 800))  # Resolution
    
    print(ds.domain_left_edge, ds.domain_right_edge)

    # Store the preloaded buffer
    preloaded_data.append(frb)
    
    # Find min/max values
    field_data = frb["gas", field].d
    val_min = min(vmin, field_data.min())
    val_max = max(vmax, field_data.max())

    # Check for NaN or Inf in data
    if np.isnan(field_data).any() or np.isinf(field_data).any():
        print(f"Invalid data (NaN or Inf) found in {ds}")
        quit(2)

if args.norm:
    norm = mcolors.LogNorm(vmin=val_min, vmax=val_max)
else:
    norm = mcolors.Normalize(vmin=val_min, vmax=val_max)

# Initial plot using the preloaded data
plot = yt.SlicePlot(ts[0], axis, ("gas", field))
plot.set_norm(('gas', field), norm)

# Lock the color scale by setting a fixed normalization
plot.plots[("gas", field)].cb.norm = norm
plot.plots[("gas", field)].cb._draw_all()

# Extract the figure to be used in the animation
fig = plot.plots[("gas", field)].figure

# Get the axis object directly
ax = plot.plots[("gas", field)].axes

# Define the animation function
def animate(i):
    frb = preloaded_data[i]  # Use preloaded data
    ax.images[0].set_array(frb["gas", field].d)  # Update the plot
    ax.images[0].set_norm(norm)  # Update the normalization
    fig.canvas.draw_idle()

# Create the animation object
animation = FuncAnimation(fig, animate, frames=len(preloaded_data), interval=100)

# Save the animation with locked colorbars
with rc_context({"mathtext.fontset": "stix"}):
    animation.save(f"animation_{args.field}_{args.axis}.mp4")
