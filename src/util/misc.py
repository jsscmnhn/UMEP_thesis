__author__ = 'xlinfr'

import numpy as np
from osgeo import gdal, osr
from osgeo.gdalconst import GDT_Float32

def saveraster(gdal_data, filename, raster):
    '''
    Unchanged function to save raster data to a GeoTIFF file, using another opened in GDAL .TIFF file as basis.

    Input:
        gdal_data (gdal.Open):  Basis gdal dataset
        filename (str):  filename to save to
        raster (np.ndarray): raster array
    '''
    rows = gdal_data.RasterYSize
    cols = gdal_data.RasterXSize

    outDs = gdal.GetDriverByName("GTiff").Create(filename, cols, rows, int(1), GDT_Float32)
    outBand = outDs.GetRasterBand(1)

    # write the data
    outBand.SetNoDataValue(-9999)
    outBand.WriteArray(raster, 0, 0)
    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()

    # georeference the image and set the projection
    outDs.SetGeoTransform(gdal_data.GetGeoTransform())
    outDs.SetProjection(gdal_data.GetProjection())