from osgeo import gdal
import numpy as np

def combine_tiffs(folder, input_files, output_file):
    datasets = [gdal.Open(folder + input_file) for input_file in input_files]

    cols = datasets[0].RasterXSize
    rows = datasets[0].RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_file, cols, rows, len(datasets), gdal.GDT_Float32)

    for i, dataset in enumerate(datasets):
        band_data = dataset.ReadAsArray()
        out_band = out_dataset.GetRasterBand(i + 1)
        out_band.WriteArray(band_data)

    out_dataset.SetProjection(datasets[0].GetProjection())
    out_dataset.SetGeoTransform(datasets[0].GetGeoTransform())

    print(f"Output file {output_file} created successfully!")


folder = "E:/Geomatics/thesis/_amsterdamset/location_6/3d/"

input_files = ['dsm_0.tif', 'dsm_1.tif', 'dsm_2.tif']
output_file = "E:/Geomatics/thesis/_amsterdamset/location_6/3d/dsms.tif"

combine_tiffs(folder, input_files, output_file)