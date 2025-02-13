import xarray as xr

ds = xr.open_dataset("data_stream-oper_stepType-instant.nc", engine="netcdf4")

print(list(ds.variables.keys()))