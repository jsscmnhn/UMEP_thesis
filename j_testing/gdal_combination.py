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


folder = "D:/Geomatics/thesis/gaptesting_database/case2/"

input_files = ['case2_0.tif', 'case2_1gap_1.tif', 'case2_2gap_2.tif', 'case2_2gap_3.tif', 'case2_2gap_4.tif']  # List files in the desired order
output_file = 'case2_5layers.tif'

combine_tiffs(folder, input_files, output_file)