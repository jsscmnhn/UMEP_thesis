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

for numb in [500, 1000, 1500, 2000, 3000]:
    start = "D:/Geomatics/optimization_tests"
    folder = f"{start}/{numb}/"

    input_files = ['final_dsm_0.tif', 'final_dsm_1.tif', 'final_dsm_2.tif']
    output_file = f"{start}/{numb}/dsms.tif"

    combine_tiffs(folder, input_files, output_file)