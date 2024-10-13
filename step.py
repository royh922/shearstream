import yt
from matplotlib import rc_context
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import argparse
import numpy as np
import os
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
    # unit_base = {
    #     "length_unit": (1.0, "pc"),
    #     "time_unit": (1.0, "s*pc/km"),
    #     "mass_unit": (2.38858753789e-24, "g/cm**3*pc**3"),
    # }

    # Define the field and axis
    axis = args.axis
    field = args.field

    # Storage for the DataSets
    storage = {"min": np.inf, "max": -np.inf}

    # Buffer size
    buffer = (args.resolution_x, args.resolution_y)

    axis_map = {"x": 0, "y": 1, "z": 2}

    # Load the first dataset to get the domain center
    # first = yt.load(fns[0], units_override=unit_base)
    first = yt.load(fns[0])
    axis_center = first.domain_center[axis_map[axis]]

    perp_axis_1 = (axis_map[axis] + 1) % 3
    perp_axis_2 = (axis_map[axis] + 2) % 3

else: 
    fns = None
    # unit_base = None
    # axis = None
    # field = None
    # perp_axis_1 = None
    # perp_axis_2 = None
    # axis_center = None
    # storage = None


print(f'Rank {comm.rank} has {fns}')

fns = comm.bcast(fns, root=0)

print(f'Rank {comm.rank} has {fns}')
# comm.Barrier()
# print('Test')
