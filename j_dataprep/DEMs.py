import requests
from io import BytesIO
import gzip
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import mapping
from rasterio.features import geometry_mask
from scipy.interpolate import NearestNDInterpolator
import startinpy
from rasterio import Affine
from shapely.geometry import shape
from rasterio.crs import CRS
import pyproj
from shapely.geometry import box

import laspy
import time
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter
from pathlib import Path
import matplotlib.pyplot as plt


def fetch_ahn_wcs(bbox, output_file="output/dtm.tif", coverage="dtm_05m", resolution=0.5):
    # Calculate width and height from bbox and resolution
    width = int((bbox[2] - bbox[0]) / resolution)
    height = int((bbox[3] - bbox[1]) / resolution)

    # WCS Service URL
    WCS_URL = "https://service.pdok.nl/rws/ahn/wcs/v1_0"

    # Construct query parameters
    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCoverage",
        "FORMAT": "GEOTIFF",
        "COVERAGE": coverage,
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "CRS": "EPSG:28992",
        "RESPONSE_CRS": "EPSG:28992",
        "WIDTH": str(width),
        "HEIGHT": str(height)
    }

    # Send GET request to fetch the data
    response = requests.get(WCS_URL, params=params, headers={"User-Agent": "Mozilla/5.0"})

    # getting correct crs data (using rasterio directly for crs fails on my laptop...)
    pyproj_crs = pyproj.CRS.from_epsg(28992)
    wkt_string = pyproj_crs.to_wkt()
    crs = CRS.from_wkt(wkt_string)

    if response.status_code == 200:
        with open("temp.tif", "wb") as f:
            f.write(response.content)

        with rasterio.open("temp.tif", "r") as f:
            print(f.crs)
        # Step 1: Clean the file using GDAL
        gdal_translate_command = f"gdal_translate -of GTiff -a_srs EPSG:28992 temp.tif {output_file}"
        os.system(gdal_translate_command)  # This runs the GDAL command

        # Step 2: Open the cleaned raster and modify NoData value
        try:
            with rasterio.open(output_file) as dataset:
                # Read the array
                array = dataset.read(1)

                # Get the current NoData value
                old_nodata = dataset.nodata

                # Set the new NoData value
                new_nodata = -9999

                # Replace old NoData values with the new NoData value
                array[array == old_nodata] = new_nodata

                # Step 3: Save the modified raster with the new NoData value
                # Open the file in write mode and update the NoData value
                with rasterio.open(output_file, 'r+') as dst:
                    dst.write(array, 1)
                    dst.nodata = new_nodata

        except Exception as e:
            print(f"Error reading or modifying raster: {e}")
            return None

        # Step 4: Delete the temporary file after use
        if os.path.exists("temp.tif"):
            os.remove("temp.tif")
            print("Temporary file 'temp.tif' has been deleted.")

        return dst, array

    else:
        print(f"Failed to fetch AHN data: HTTP {response.status_code}")
        return None

def download_wfs_data(bbox, gpkg_name, output_folder, output_name):
    """
    Download data from a WFS server in batches and save it to a GeoPackage.
    -----------------------------------------------------
    Input:
    -   wfs_url (str): URL of the WFS service.
    -   layer_name (str): The layer name to download.
    -   bbox (tuple): Bounding box as (minx, miny, maxx, maxy).
    -   gpkg_name (str): Name for the output GeoPackage file.
    -   tile_name (str): Layer name for saving in the GeoPackage.
    Output:
    -   None: saves a GeoPackage file to the given {output_gpkg} at layer {tile_name}.
    """
    # Initialize variables for feature collection, max requestable amount from server is 10000
    all_features = []
    start_index = 0
    count = 10000

    wfs_url = "https://data.3dbag.nl/api/BAG3D/wfs"
    layer_name = "BAG3D:lod13"

    while True:
        params = {
            "SERVICE": "WFS",
            "REQUEST": "GetFeature",
            "VERSION": "2.0.0",
            "TYPENAMES": layer_name,
            "SRSNAME": "urn:ogc:def:crs:EPSG::28992",
            "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},urn:ogc:def:crs:EPSG::28992",
            "COUNT": count,
            "STARTINDEX": start_index
        }

        # Mimicking a QGIS request
        headers = {
            "User-Agent": "Mozilla/5.0 QGIS/33411/Windows 11 Version 2009"
        }

        response = requests.get(wfs_url, params=params, headers=headers)

        # Check if the request was successful & download data
        if response.status_code == 200:
            if response.headers.get('Content-Encoding', '').lower() == 'gzip' and response.content[:2] == b'\x1f\x8b':
                data = gzip.decompress(response.content)
            else:
                data = response.content

            with BytesIO(data) as f:
                gdf = gpd.read_file(f)

            all_features.append(gdf)

            # Check if the number of features retrieved is less than the requested count: then we can stop
            if len(gdf) < count:
                break

                # Start index for next request
            start_index += count

        else:
            print(f"Failed to download WFS data. Status code: {response.status_code}")
            print(f"Error message: {response.text}")
            break

            # Concatenate all features into a single GeoDataFrame
    if all_features:
        full_gdf = gpd.GeoDataFrame(pd.concat(all_features, ignore_index=True))

        os.makedirs(output_folder, exist_ok=True)
        output_gpkg = os.path.join(output_folder, f"{gpkg_name}.gpkg")

        full_gdf.to_file(output_gpkg, layer=output_name, driver="GPKG")
    else:
        print("No features were downloaded.")


def load_buildings(buildings_path, layer):
    """
    Load in the building shapes from a geopackage file.
    ----
    Input:
    - buildings_path (string):   path to the geopackage file.
    - layer (string):            (Tile) name of the layer of buildings to be used

    Output:
    - List of dictionaries: A list of dictionaries containing:
      - "geometry": building geometry in GeoJSON-like format.
      - "parcel_id": corresponding parcel ID.
    """
    buildings_gdf = gpd.read_file(buildings_path, layer=layer)

    if 'identificatie' not in buildings_gdf.columns:
        raise ValueError("Column 'identificatie' not found in the dataset")

    return [{"geometry": mapping(geom), "parcel_id": identificatie} for geom, identificatie in zip(buildings_gdf.geometry, buildings_gdf["identificatie"])]


def extract_center_cells(geo_array, no_data=-9999):
    """
    Extract the values of each cell in the input data and save these with the x and y (row and col)
    indices. Thereby, make sure that the corners of the dataset are filled for a full coverage triangulation
    in the next step.
    ----
    Input:
    - (2d numpy array): raster data.
    - no_data (int, optional): no_data value to replace source no data value with.

    Output:
    - xyz_filled (list): list containing x, y and z coordinates of the cells.
    """
    # Get the indices of the rows and columns
    rows, cols = np.indices(geo_array.shape)

    # Identify corner coordinates
    corners = {
        "top_left": (0, 0),
        "top_right": (0, geo_array.shape[1] - 1),
        "bottom_left": (geo_array.shape[0] - 1, 0),
        "bottom_right": (geo_array.shape[0] - 1, geo_array.shape[1] - 1)
    }

    # Mask for valid center cells (non-no_data)
    valid_center_cells = (geo_array != no_data)

    # Extract x, y, z values for valid cells
    x_valid = cols[valid_center_cells]
    y_valid = rows[valid_center_cells]
    z_valid = geo_array[valid_center_cells]

    # Create interpolator from valid points
    interpolator = NearestNDInterpolator(list(zip(x_valid, y_valid)), z_valid)

    # Check each corner for no data and interpolate if necessary
    for corner_name, (row, col) in corners.items():
        if geo_array[row, col] == no_data:
            # Interpolate the nearest valid value
            geo_array[row, col] = interpolator((col, row))

    # Extract non-no_data and center cells again after filling corners
    valid_center_cells = (geo_array != no_data)

    # Extract final x, y, z values after filling corners
    x_filled = cols[valid_center_cells]
    y_filled = rows[valid_center_cells]
    z_filled = geo_array[valid_center_cells]

    # Prepare final list of [x, y, z]
    xyz_filled = []
    for x_i, y_i, z_i in zip(x_filled, y_filled, z_filled):
        xyz_filled.append([x_i, y_i, z_i])

    return xyz_filled


def fill_raster(geo_array, nodata_value, transform):
    """
    Fill the no data values of a given raster using Laplace interpolation.
    ----
    Input:
    - geo_array (2d numpy array): cropped raster data.
    - nodata_value (int): nodata value to replace NAN after interplation with.
    - transform (rasterio transform): affine transform matrix.

    Output:
    - new_data[0, 1:-1, 1:-1] (2d numpy array): filled raster data with first and last rows and columns remove to ensure
                                                there are no nodata values.
    - new_transform (rasterio transform): affine transform matrix reflecting the one column one row removal shift.
    """

    # creating delaunay
    points = extract_center_cells(geo_array, no_data=nodata_value)
    dt = startinpy.DT()
    dt.insert(points, "BBox")

    # now interpolation
    new_data = np.copy(geo_array)

    # for interpolation, grid of all column and row positions, excluding the first and last rows/cols
    cols, rows = np.meshgrid(
        np.arange(1, geo_array.shape[1] - 1),
        np.arange(1, geo_array.shape[0] - 1)

    )

    # flatten the grid to get a list of all (col, row) locations
    locs = np.column_stack((cols.ravel(), rows.ravel()))
    interpolated_values = dt.interpolate({"method": "Laplace"}, locs)

    # reshape interpolated grid back to original
    interpolated_grid = np.reshape(interpolated_values, (geo_array.shape[0] - 2, geo_array.shape[1] - 2))

    # fill new_data with interpolated values
    new_data[1:-1, 1:-1] = interpolated_grid
    new_data = np.where(np.isnan(new_data), nodata_value, new_data)

    new_transform = transform * Affine.translation(1, 1)

    return new_data[1:-1, 1:-1], new_transform


def chm_finish(chm_array, dtm_array, transform, min_height=2, max_height=40):
    """
    Finish the CHM file by first removing the ground height. Then remove vegetation height
    below and above a certain range to ensure effective shade and remove noise.
    ----
    Input:
    - chm_array (2d numpy array):       cropped raster array of the CHM.
    - dtm_array (2d numpy array):       cropped raster array of the filled DSM.
    - transform (rasterio transform):   affine transform matrix.
    - min_height (float, optional):     minimal height for vegetation to be included.
    - max_height (float, optional):     maximum height for vegetation to be included.

    Output:
    - result_array (2d numpy array):    Array of the CHM with normalized height and min and max heights removed.
    - new_transform (rasterio transform): affine transform matrix reflecting the one column one row removal shift.
    """

    result_array = chm_array[1:-1, 1:-1] - dtm_array
    result_array[(result_array < min_height) | (result_array > max_height)] = 0
    result_array[np.isnan(result_array)] = 0

    new_transform = transform * Affine.translation(1, 1)

    return result_array, new_transform


def replace_buildings(filled_dtm, dsm_buildings, buildings_geometries, transform):
    """
    Replace the values of the filled dtm with the values of the filled dsm, if there is a building.
    ----
    Input:
    - filled_dtm (2d np array):         filled array of the cropped AHN dtm.
    - dsm_buildings (2d np array):      Filled array of the cropped AHN dsm.
    - building_geometries (list):       A list of the building geometries
    - transform (rasterio transform):   affine transform matrix.

    Output:
    - final_dsm (2d numpy array):   a np array representing the final dsm, containing only ground and building
                                    heights.

    """
    geometries = [shape(building['geometry']) for building in buildings_geometries]

    # Ensure mask has same shape as filled_dtm
    building_mask = geometry_mask(geometries, transform=transform, invert=False, out_shape=filled_dtm.shape)

    # Get shape differences
    dtm_shape = filled_dtm.shape
    dsm_shape = dsm_buildings.shape

    if dtm_shape != dsm_shape:
        # Compute the cropping offsets
        row_diff = dsm_shape[0] - dtm_shape[0]
        col_diff = dsm_shape[1] - dtm_shape[1]

        # Ensure even cropping from all sides (center alignment)
        row_start = row_diff // 2
        col_start = col_diff // 2
        row_end = row_start + dtm_shape[0]
        col_end = col_start + dtm_shape[1]

        # Crop dsm_buildings to match filled_dtm
        dsm_buildings = dsm_buildings[row_start:row_end, col_start:col_end]

    # Apply the mask
    final_dsm = np.where(building_mask, filled_dtm, dsm_buildings)

    return final_dsm


def load_buildings(buildings_path, layer):
    """
    Load in the building shapes from a geopackage file.
    ----
    Input:
    - buildings_path (string):   path to the geopackage file.
    - layer (string):            (Tile) name of the layer of buildings to be used

    Output:
    - List of dictionaries: A list of geometries in GeoJSON-like dictionary format.
      Each dictionary represents a building geometry with its spatial coordinates.
    """
    buildings_gdf = gpd.read_file(buildings_path, layer=layer)
    return [mapping(geom) for geom in buildings_gdf.geometry]


def write_output(dataset, crs, output, transform, name, change_nodata=False):
    """
    Write grid to .tiff file.
    ----
    Input:
    - dataset: Can be either a rasterio dataset (for rasters) or laspy dataset (for point clouds)
    - output (Array): the output grid, a numpy grid.
    - name (String): the name of the output file.
    - transform:
      a user defined rasterio Affine object, used for the transforming the pixel coordinates
      to spatial coordinates.
    - change_nodata (Boolean): true: use a no data value of -9999, false: use the datasets no data value
    """
    output_file = name

    print(type(output))
    output = np.squeeze(output)
    print(output.shape)
    print(output.dtype)

    # Set the nodata value: use -9999 if nodata_value is True or dataset does not have nodata.
    if change_nodata:
        nodata_value = -9999
    else:
        try:
            nodata_value = dataset.nodata
            if nodata_value is None:
                raise AttributeError("No no data value found in dataset.")
        except AttributeError as e:
            print(f"Warning: {e}. Defaulting to -9999.")
            nodata_value = -9999

    # output the dataset
    with rasterio.open(output_file, 'w',
                       driver='GTiff',
                       height=output.shape[0],
                       width=output.shape[1],
                       count=1,
                       dtype=np.float32,
                       crs=crs,
                       nodata=nodata_value,
                       transform=transform) as dst:
        dst.write(output, 1)
    print("File written to '%s'" % output_file)


def fill_dems(buildings, dsm, dsm_array, dtm, dtm_array):
    transform = dtm.transform
    filled_dtm, _ = fill_raster(dtm_array, dtm.nodata, transform)
    filled_dsm, new_transform = fill_raster(dsm_array, dsm.nodata, transform)
    final_dsm = replace_buildings(filled_dtm, filled_dsm, buildings, new_transform)
    print(type(final_dsm))

    pyproj_crs = pyproj.CRS.from_epsg(28992)
    wkt_string = pyproj_crs.to_wkt()
    crs = CRS.from_wkt(wkt_string)

    write_output(dsm, crs, filled_dsm, new_transform, "output/filled_final_dsm.tif")
    write_output(dsm, crs, final_dsm, new_transform, "output/final_dsm.tif")
    write_output(dtm, crs, filled_dtm, new_transform, "output/final_dtm.tif")

def find_tiles(x_min, y_min, x_max, y_max):
    query_geom = box(x_min, y_min, x_max, y_max)
    matches = gdf.sindex.query(query_geom) # ,  predicate="overlaps": tricky i want to still get something if it is all contained in one
    return gdf.iloc[matches]["GT_AHNSUB"].tolist()


if __name__ == "__main__":
    pyproj_crs = pyproj.CRS.from_epsg(28992)
    wkt_string = pyproj_crs.to_wkt()
    crs = CRS.from_wkt(wkt_string)

    shapefile_path = "geotiles\AHN_lookup.geojson"
    gdf = gpd.read_file(shapefile_path)