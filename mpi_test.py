import numpy as np
from mpi4py import MPI
import yt
import argparse


yt.enable_parallelism()

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
    var = glob.glob(f"{args.directory}/{args.prefix}.*.athdf")
    var2 = 2
    var3 = 3
    var4 = 4
    var5 = 5
    var.sort()
else:
    var = None
    var2 = None
    var3 = None
    var4 = None
    var5 = None

print(f'Rank {comm.rank} has {var}')

var = comm.bcast(var, root=0)
var2 = comm.bcast(var2, root=0)
var3 = comm.bcast(var3, root=0)
var4 = comm.bcast(var4, root=0)
var5 = comm.bcast(var5, root=0)

print(f'Rank {comm.rank} has {var}')