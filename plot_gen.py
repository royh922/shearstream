import yt

# yt.enable_parallelism()

# Load all of the DD*/output_* files into a DatasetSeries object
# in this case it is a Time Series
ts = yt.load("~/enzo-e/KH-MHD-*/KH-MHD-*.block_list")

# Define an empty storage dictionary for collecting information
# in parallel through processing
storage = {}

# Use piter() to iterate over the time series, one proc per dataset
# and store the resulting information from each dataset in
# the storage dictionary
for sto, ds in ts.piter(storage=storage):
    p = yt.ProjectionPlot(ds, "z", "density")
    p.save()