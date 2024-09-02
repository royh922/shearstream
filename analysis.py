import yt

ts = yt.load("kh_custom.out1.*.athdf")

field = "temperature"

vmin = ts[0].all_data().min(field)
vmax = ts[0].all_data().max(field)
for ds in ts:
    vmin = min(vmin, ds.all_data().min(field))
    vmax = max(vmax, ds.all_data().max(field))


print(vmin, vmax)