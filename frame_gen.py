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
args = parser.parse_args()

if comm.rank == 0:

    import glob 
    fns = glob.glob(f"{args.directory}/{args.prefix}.*.athdf")
    fns.sort()
    # fns = fns[:400]
    # Define the unit base for the simulation
    unit_base = {
        "length_unit": (1e18, "cm"),
        "time_unit": (1e10, "s"),
        "mass_unit": (1e26, "g"),
    }

    # Define the field and axis
    axis = args.axis
    field = args.field

    # Buffer size
    buffer = (args.resolution_x, args.resolution_y)

    axis_map = {"x": 0, "y": 1, "z": 2}
    
    # Load the first dataset to get the domain center
    first = yt.load(fns[0], units_override=unit_base, unit_system="cgs")
    axis_center = first.domain_center[axis_map[axis]]

    perp_axis_1 = (axis_map[axis] + 1) % 3
    perp_axis_2 = (axis_map[axis] + 2) % 3

    # Iterate through time series to find min/max values
    Max = -np.inf
    Min = np.inf
    ts = yt.load(fns, units_override=unit_base, unit_system="cgs")
    for ds in ts:
        temp_Min, temp_Max = ds.r[field].min(), ds.r[field].max()
        Min = min(Min, temp_Min)
        Max = max(Max, temp_Max)

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
    Max = None
    Min = None


# Broadcast the necessary variables to all ranks
fns = comm.bcast(fns, root=0)
unit_base = comm.bcast(unit_base, root=0)
axis = comm.bcast(axis, root=0)
field = comm.bcast(field, root=0)
perp_axis_1 = comm.bcast(perp_axis_1, root=0)
perp_axis_2 = comm.bcast(perp_axis_2, root=0)
axis_center = comm.bcast(axis_center, root=0)
buffer = comm.bcast(buffer, root=0)
Max = comm.bcast(Max, root=0)
Min = comm.bcast(Min, root=0)


# ONLY ENABLE PARALLELISM AFTER BROADCASTING VARIABLES BECAUSE YT IS ASS AND SUCKS AT ACTUALLY DOING SHIT
yt.enable_parallelism()

# Storage for the DataSets
storage = {"min": np.inf, "max": -np.inf}

for sto, fn in yt.parallel_objects(fns, -1, storage=storage, dynamic=True):

    ds = yt.load(fn, units_override=unit_base, unit_system='cgs')
    
    # Check for NaN or Inf in data
    if np.isnan(ds.r[field]).any() or np.isinf(ds.r[field]).any():
        print(f"Invalid data (NaN or Inf) found in {ds}")
        quit(2)

    plot = ds.proj([field], str(axis))
    plot.save()
#    slc = ds.proj([('athena_pp', 'press'), ('athena_pp', 'rho')], str(axis))    # For Projection Plot
#    slc = ds.slice(axis, axis_center)  # For Slice Plot

    # Find min/max values
    # storage["min"] = min(storage["min"], field_data.min())
    # storage["max"] = max(storage["max"], field_data.max())


#local_max = storage["max"]
#global_max = comm.reduce(local_max, op=MPI.MAX, root=0)

#np.save('global_max.npy', global_max)