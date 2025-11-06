import yt
import unyt
import argparse

parser = argparse.ArgumentParser(description="Plot a field from a dataset")
parser.add_argument("prefix", type=str, help="Prefix for the dataset series")
parser.add_argument("field", type=str, help="Field to plot")
args = parser.parse_args()

yt.enable_parallelism()

@yt.derived_field(
    name="temperature", units="K", display_name="Temperature", sampling_type="local", force_override=True
)
def temperature(field, data):
    return data[('gas', 'pressure')] / data[('gas', 'density')] / yt.units.kb * unyt.physical_constants.mp * 0.6  # Mean molecular weight = 0.6

# Load all of the DD*/output_* files into a DatasetSeries object
# in this case it is a Time Series
prefix = args.prefix
field = args.field
ts = yt.load(f"./{prefix}/KH-*/KH*.block_list")

# Define an empty storage dictionary for collecting information
# in parallel through processing
storage = {}
global_extrema = [float('inf'), float('-inf')]

# Use piter() to iterate over the time series, one proc per dataset
# and store the resulting information from each dataset in
# the storage dictionary
for sto, ds in ts.piter(storage=storage):
    p = yt.ProjectionPlot(ds, "z", ('gas', f'{field}'), weight_field=('gas', 'density'))
    # p.set_zlim(field, zmin=1e4)
    p.set_cmap(field, 'RdBu')
    if field == 'temperature':
        p.set_cmap(field, 'RdBu_r')
    # p.annotate_scale(corner='lower_left')
    p.annotate_timestamp(time_unit="Myr",corner='upper_right', time_format='t = {time:.3f} {units}')
    p.annotate_title(f"{field} Projection")
    p.save(name=f'{prefix}/{field}/')
    dd = ds.all_data()
    sto.result = dd.quantities.extrema(("gas", f"{field}"))
    sto.result_id = str(ds)

if yt.is_root():
    units = next(iter(storage.values())).units
    global_extrema *= units
    for name, vals in storage.items():
        global_extrema[0] = min(global_extrema[0], vals[0])
        global_extrema[1] = max(global_extrema[1], vals[1])
    print(global_extrema)
#for ds in ts.piter():
#    p = yt.SlicePlot(ds, "z", ('gas', f'{field}'))
#    p.set_cmap(field, 'RdBu')
#    p.set_zlim(field, zmin=global_extrema[0], zmax=global_extrema[1])
#    p.save(name=f'{field}/')  