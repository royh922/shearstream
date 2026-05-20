import yt
import unyt
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Plot a field from a dataset")
parser.add_argument("prefix", type=str, help="Prefix for the dataset series")
parser.add_argument("field", type=str, help="Field to plot")
parser.add_argument(
    "--plot-type",
    type=str,
    choices=["projection", "slice"],
    default="projection",
    help="Whether to generate ProjectionPlot or SlicePlot",
)
parser.add_argument("--zmin", type=float, default=None, help="Manual lower color limit")
parser.add_argument("--zmax", type=float, default=None, help="Manual upper color limit")
parser.add_argument(
    "--recompute-extrema",
    action="store_true",
    help="Force recomputing global extrema even if cache exists",
)
args = parser.parse_args()

parallel_enabled = False
try:
    yt.enable_parallelism()
    parallel_enabled = True
except Exception as exc:
    print(f"MPI unavailable ({exc.__class__.__name__}), running in serial mode.")

@yt.derived_field(
    name="temperature", units="K", display_name="Temperature", sampling_type="local", force_override=True
)
def temperature(field, data):
    return data[('gas', 'pressure')] / data[('gas', 'density')] / yt.units.kb * unyt.physical_constants.mp * 0.6  # Mean molecular weight = 0.6

# Load all of the DD*/output_* files into a DatasetSeries object
# in this case it is a Time Series
prefix = args.prefix
field = args.field
plot_type = args.plot_type
plot_field = ("gas", field)
ts = yt.load(f"./{prefix}/KH-*/KH*.block_list")

# Store computed min/max per field so future runs can skip scan.
cache_path = Path(prefix) / "field_extrema.json"
global_extrema = None
cache_key = f"{plot_type}:{field}"

def build_plot(ds):
    if plot_type == "projection":
        return yt.ProjectionPlot(ds, "z", plot_field, weight_field=("gas", "density"))
    return yt.SlicePlot(ds, "z", plot_field)

if args.zmin is not None or args.zmax is not None:
    if args.zmin is None or args.zmax is None:
        raise ValueError("If setting manual limits, both --zmin and --zmax are required.")
    global_extrema = [args.zmin, args.zmax]
elif cache_path.exists() and not args.recompute_extrema:
    with cache_path.open("r") as f:
        cache = json.load(f)
    if cache_key in cache:
        global_extrema = cache[cache_key]
        print(f"Using cached extrema for {cache_key}: {global_extrema}")

if global_extrema is None:
    # First pass: parallel scan for exact displayed (2D buffer) extrema.
    storage = {}
    extrema = [float("inf"), float("-inf")]
    for sto, ds in ts.piter(storage=storage):
        p = build_plot(ds)
        frame = p.frb[plot_field]
        sto.result = [frame.min(), frame.max()]
        sto.result_id = str(ds)

    if yt.is_root():
        for vals in storage.values():
            extrema[0] = min(extrema[0], float(vals[0]))
            extrema[1] = max(extrema[1], float(vals[1]))
        global_extrema = [extrema[0], extrema[1]]
        print(f"Computed extrema for {field}: {global_extrema}")

        cache = {}
        if cache_path.exists():
            with cache_path.open("r") as f:
                cache = json.load(f)
        cache[cache_key] = global_extrema
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(cache, f, indent=2)

    if parallel_enabled:
        global_extrema = ts.comm.mpi_bcast(global_extrema, root=0)

# Use piter() to iterate over the time series, one proc per dataset
# and plot with fixed color limits.
for ds in ts.piter():
    p = build_plot(ds)
    p.set_zlim(plot_field, zmin=global_extrema[0], zmax=global_extrema[1])
    p.set_cmap(plot_field, 'RdBu')
    if field == 'temperature':
        p.set_cmap(plot_field, 'RdBu_r')
    # p.annotate_scale(corner='lower_left')
    p.annotate_timestamp(time_unit="Myr",corner='upper_right', time_format='t = {time:.3f} {units}')
    p.annotate_title(f"{field} {plot_type.title()}")
    p.save(name=f"{prefix}/{field}/{plot_type}/")