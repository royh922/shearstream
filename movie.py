import yt
from matplotlib import rc_context
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import argparse
import numpy as np

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("dir", type=str)
parser.add_argument("--field", type=str)
parser.add_argument("--axis", type=str)
args = parser.parse_args()

# Define the unit base for the simulation
unit_base = {"length_unit": (1.0, "pc"), "time_unit": (1.0, "s*pc/km"), "mass_unit": (2.38858753789e-24, "g/cm**3*pc**3")}
ts = yt.load(f"{args.dir}/kh_custom.out1.*.athdf", units_override=unit_base)

# Define the field to be plotted
field = args.field
axis = args.axis


# Find the minimum and maximum values of the field
vmin = ts[0].all_data().min(field)
vmax = ts[0].all_data().max(field)
for ds in ts:
    vmin = min(vmin, ds.all_data().min(field))
    vmax = max(vmax, ds.all_data().max(field))
    temp = ds.r[('gas', args.field)].v
    if np.isnan(temp).any():
        print(f"NaN found in {ds}")
        quit()
    if np.isinf(temp).any():
        print(f"Inf found in {ds}")
        quit()

# Initial plot
plot = yt.SlicePlot(ts[0], args.axis, ("gas", field))
plot.set_zlim(("gas", field), vmin, vmax)

# Lock the color scale by setting a fixed normalization
# norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) # Log scale
norm = mcolors.Normalize(vmin=vmin, vmax=vmax) # Linear scale
plot.plots[("gas", field)].cb.norm = norm
plot.plots[("gas", field)].cb._draw_all()

# Extract the figure to be used in the animation
fig = plot.plots[("gas", field)].figure

# Get the axis object directly
ax = plot.plots[("gas", field)].axes

# Calculate the shape of the image from the resolution of the plot
xres = ax.images[0].get_array().shape[1]
yres = ax.images[0].get_array().shape[0]

# Find bounds
xylim = (plot.xlim[0].d, plot.xlim[1].d, plot.ylim[0].d, plot.ylim[1].d)

# Define the animation function
def animate(i):
    ds = ts[i]
    slc = ds.slice(args.axis, 0.5)
    # slc.set_log(('gas', field), False) # Linear scale
    frb = yt.FixedResolutionBuffer(slc, xylim, (xres, yres))
    
    ax.images[0].set_array(frb["gas", field].d)
    fig.canvas.draw_idle()

# Create the animation object
animation = FuncAnimation(fig, animate, frames=len(ts), interval=100)

# Save the animation with locked colorbars
with rc_context({"mathtext.fontset": "stix"}):
    animation.save(f"animation_{args.field}_{args.axis}.mp4")
