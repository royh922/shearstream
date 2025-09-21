import yt
import numpy as np
import sys

# Read args
frame = sys.argv[1]

def _log_relative_density(field, data):
    return np.log(data[('gas', 'density')] / (1.0037999999999999e-26 * yt.units.gram / yt.units.cm**3))  # Density of stream in cgs

yt.add_field(('gas', 'log_relative_density'), function=_log_relative_density, units="", display_name="Log Relative Density", sampling_type="cell", force_override=True)

ds = yt.load(f"./data/KH-00{frame}/KH-00{frame}.block_list")  
ds.index
ds.field_list
ds.derived_field_list

# Density of stream in cgs
rho_s = 1.0037999999999999e-26
plot = yt.SlicePlot(ds, 'z', ('gas', 'log_relative_density'), center='c', buff_size=(512, 512))
plot.set_cmap(('gas', 'log_relative_density'), 'nipy_spectral')
plot.set_zlim(('gas', 'log_relative_density'), zmin='min', zmax='max')
plot.set_log(('gas', 'log_relative_density'), False)

# Save the plot
plot.save(f"KH-00{frame}_log_relative_density.png")