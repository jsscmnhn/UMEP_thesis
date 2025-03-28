import requests
from io import BytesIO
import gzip
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import mapping
from rasterio.features import geometry_mask, shapes
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
import startinpy
from rasterio import Affine
from shapely.geometry import shape,box
from rasterio.crs import CRS
from pathlib import Path
import laspy
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter, label
import uuid
from rtree import index


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

    output = np.squeeze(output)
    # Set the nodata value: use -9999 if nodata_value is True or dataset does not have nodata.
    if change_nodata:
        nodata_value = -9999
    else:
        try:
            # TO DO: CHANGE THIS TO JUST INPUTTING A NODATA VALUE, NO NEED FOR THE WHOLE DATASET IN THIS FUNCTION
            nodata_value = dataset.nodata
            if nodata_value is None:
                raise AttributeError("No no data value found in dataset.")
        except AttributeError as e:
            print(f"Warning: {e}. Defaulting to -9999.")
            nodata_value = -9999

    # output the dataset
    with rasterio.open(output_file, 'w',
                       driver='GTiff',
                       height=output.shape[0],  # Assuming output is (rows, cols)
                       width=output.shape[1],
                       count=1,
                       dtype=np.float32,
                       crs=crs,
                       nodata=nodata_value,
                       transform=transform) as dst:
        dst.write(output, 1)
    print("File written to '%s'" % output_file)


class Buildings:
    def __init__(self, bbox, wfs_url="https://data.3dbag.nl/api/BAG3D/wfs", layer_name="BAG3D:lod13", gpkg_name="buildings", output_folder = "output", output_layer_name="buildings"):
        self.bbox = bbox
        self.wfs_url = wfs_url
        self.layer_name = layer_name
        self.data = self.download_wfs_data(gpkg_name, output_folder, output_layer_name)
        self.building_geometries = self.load_buildings(self.data)
        self.removed_buildings = []
        self.user_buildings = []
        self.user_buildings_higher = []
        self.removed_user_buildings = []
        self.is3D = False


    def download_wfs_data(self, gpkg_name, output_folder, layer_name):
        all_features = []
        start_index = 0
        count = 10000

        while True:
            params = {
                "SERVICE": "WFS",
                "REQUEST": "GetFeature",
                "VERSION": "2.0.0",
                "TYPENAMES": self.layer_name,
                "SRSNAME": "urn:ogc:def:crs:EPSG::28992",
                "BBOX": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]},urn:ogc:def:crs:EPSG::28992",
                "COUNT": count,
                "STARTINDEX": start_index
            }
            headers = {"User-Agent": "Mozilla/5.0 QGIS/33411/Windows 11 Version 2009"}
            response = requests.get(self.wfs_url, params=params, headers=headers)

            if response.status_code == 200:
                if response.headers.get('Content-Encoding', '').lower() == 'gzip' and response.content[
                                                                                      :2] == b'\x1f\x8b':
                    data = gzip.decompress(response.content)
                else:
                    data = response.content

                with BytesIO(data) as f:
                    gdf = gpd.read_file(f)
                all_features.append(gdf)
                if len(gdf) < count:
                    break
                start_index += count
            else:
                print(f"Failed to download WFS data. Status code: {response.status_code}")
                print(f"Error message: {response.text}")
                return gpd.GeoDataFrame()

        if all_features:
            full_gdf = gpd.GeoDataFrame(pd.concat(all_features, ignore_index=True))
            os.makedirs(output_folder, exist_ok=True)
            output_gpkg = os.path.join(output_folder, f"{gpkg_name}.gpkg")
            full_gdf.to_file(output_gpkg, layer=layer_name, driver="GPKG")
            print("loaded")
            return full_gdf
        else:
            print("No features were downloaded.")
            return None

    @staticmethod
    def load_buildings(buildings_gdf, buildings_path=None, layer=None):
        if buildings_gdf is None:
            if buildings_path is not None:
                buildings_gdf = gpd.read_file(buildings_path, layer=layer)
            else: return None

        return [{"geometry": mapping(geom), "parcel_id": identificatie} for geom, identificatie in
                zip(buildings_gdf.geometry, buildings_gdf["identificatie"])]

    def remove_buildings(self, identification):
        self.removed_buildings.append(identification)

    def retrieve_buildings(self, identification):
        self.removed_buildings.remove(identification)

    def insert_user_buildings(self, highest_array, transform, footprint_array=None):
        self.is3D = footprint_array is not None
        self.removed_user_buildings = []
        self.user_buildings_higher = []

        labeled_array, num_clusters = label(highest_array > 0)

        shapes_highest = shapes(labeled_array.astype(np.uint8), mask=(labeled_array > 0), transform=transform)

        highest_buildings = [
            {"geometry": mapping(shape(geom)), "parcel_id": str(uuid.uuid4())[:8]}
            for geom, value in shapes_highest
        ]

        if footprint_array is not None:
            rtree_index = index.Index()
            for idx, building in enumerate(highest_buildings):
                geom = shape(building['geometry'])
                rtree_index.insert(idx, geom.bounds)

            labeled_footprint_array, num_clusters_fp = label(footprint_array > 0)

            shapes_fp = shapes(labeled_footprint_array.astype(np.uint8), mask=(labeled_footprint_array > 0),
                                   transform=transform)

            footprint_buildings = [
                {"geometry": mapping(shape(geom)), "parcel_id": str(uuid.uuid4())[:8]}
                for geom, value in shapes_fp
            ]

            for footprint_building in footprint_buildings:
                footprint_geom = shape(footprint_building['geometry'])

                possible_matches = list(
                    rtree_index.intersection(footprint_geom.bounds))

                for match_idx in possible_matches:
                    highest_building = highest_buildings[match_idx]
                    highest_geom = shape(highest_building['geometry'])

                    if footprint_geom.intersects(highest_geom) or footprint_geom.within(highest_geom):
                        footprint_building['parcel_id'] = highest_building['parcel_id']
                        break
            self.user_buildings = footprint_buildings
            self.user_buildings_higher = highest_buildings
        else:
            self.user_buildings = highest_buildings



    def remove_user_buildings(self, identification):
        self.removed_user_buildings.append(identification)

    def retrieve_user_buildings(self, identification):
        self.removed_user_buildings.remove(identification)



class DEMS:
    def __init__(self, bbox, building_data, bridge=False):
        self.bbox = bbox
        self.building_data = building_data
        self.user_building_data = []
        self.bridge = bridge
        self.crs = (CRS.from_epsg(28992))
        self.dtm, self.dsm, self.transform = self.create_dem(bbox)
        self.og_dtm, self.og_dsm = self.dtm, self.dsm
        self.is3D = False

    @staticmethod
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

        if response.status_code == 200:
            with open("temp.tif", "wb") as f:
                f.write(response.content)

            with rasterio.open("temp.tif", "r") as f:
                # TO DO: test if this is still needed after fixing the libraries
                gdal_translate_command = f"gdal_translate -of GTiff -a_srs EPSG:28992 temp.tif {output_file}"
                os.system(gdal_translate_command)

            try:
                with rasterio.open(output_file) as dataset:
                    array = dataset.read(1)
                    old_nodata = dataset.nodata
                    new_nodata = -9999
                    array[array == old_nodata] = new_nodata

                    with rasterio.open(output_file, 'r+') as dst:
                        dst.write(array, 1)
                        dst.nodata = new_nodata

            except Exception as e:
                print(f"Error reading or modifying raster: {e}")
                return None

            # Delete the temporary file after use
            if os.path.exists("temp.tif"):
                os.remove("temp.tif")

            return dst, array

        else:
            print(f"Failed to fetch AHN data: HTTP {response.status_code}")
            return None

    @staticmethod
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

    def fill_raster(self, geo_array, nodata_value, transform):
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
        points = self.extract_center_cells(geo_array, no_data=nodata_value)
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

    def replace_buildings(self, filled_dtm, dsm_buildings, buildings_geometries, transform, bridge):
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
        geometries = [shape(building['geometry']) for building in buildings_geometries if 'geometry' in building]
        bridging_geometries = []
        if bridge is True:
            bridge_crs = "http://www.opengis.net/def/crs/EPSG/0/28992"
            url = f"https://api.pdok.nl/lv/bgt/ogc/v1/collections/overbruggingsdeel/items?bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}&bbox-crs={bridge_crs}&crs={bridge_crs}&limit=1000&f=json"
            response = requests.get(url)
            if response.status_code == 200:
                bridging_data = response.json()
                if "features" in bridging_data:  # Ensure data contains geometries
                    bridging_geometries = [shape(feature['geometry']) for feature in bridging_data["features"] if
                                           'geometry' in feature]
            else:
                print(f"Error fetching bridges: {response.status_code}, {response.text}")

        # Ensure mask has same shape as filled_dtm
        all_geometries = bridging_geometries + geometries
        building_mask = geometry_mask(all_geometries, transform=transform, invert=False, out_shape=filled_dtm.shape)

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

    def create_dem(self, bbox):
        dtm_dst, dtm_array = self.fetch_ahn_wcs(bbox, output_file="archive/outputs/outputs2/dtm.tif", coverage="dtm_05m", resolution=0.5)
        transform = dtm_dst.transform

        filled_dtm, new_transform = self.fill_raster(dtm_array, dtm_dst.nodata, transform)
        write_output(dtm_dst, self.crs, filled_dtm, new_transform, "output/final_dtm.tif")

        if self.building_data:
            dsm_dst, dsm_array = self.fetch_ahn_wcs(bbox, output_file="archive/outputs/outputs2/dsm.tif",
                                                    coverage="dsm_05m", resolution=0.5)
            filled_dsm, _ = self.fill_raster(dsm_array, dsm_dst.nodata, transform)
            final_dsm = self.replace_buildings(filled_dtm, filled_dsm, self.building_data, new_transform, self.bridge)
            write_output(dsm_dst, self.crs, final_dsm, new_transform, "output/final_dsm_over.tif")
        else:
            write_output(dtm_dst, self.crs, filled_dtm, new_transform, "output/final_dsm_over.tif")

        return filled_dtm, final_dsm, transform

    def remove_buildings(self, remove_list, remove_user_list, user_buildings_higher=None):
        remove_set = set(remove_list)
        remove_user_set = set(remove_user_list)

        # Find buildings to remove from both datasets
        to_remove = [building for building in self.building_data if building['identificatie'] in remove_set]
        to_remove_user = [building for building in self.user_building_data if
                          building['identificatie'] in remove_user_set]

        remove_all = to_remove + to_remove_user

        # Extract geometries for mask creation
        geometries = [shape(building['geometry']) for building in remove_all if 'geometry' in building]

        # Create the removal mask if there are geometries
        if geometries:
            remove_building_mask = geometry_mask(geometries, transform=self.transform, invert=False,
                                                 out_shape=self.dtm.shape)

            if not self.is3D:
                self.dsm[...] = np.where(remove_building_mask, self.dsm, self.dtm)
            else:
                self.dsm[0][...] = np.where(remove_building_mask, self.dsm[0], self.dtm)

                if user_buildings_higher:
                    remove_other_layers = [building for building in user_buildings_higher if
                                           building['identificatie'] in remove_user_set]
                    other_geometries = [shape(building['geometry']) for building in remove_other_layers if
                                        'geometry' in building]

                    if other_geometries:
                        remove_others_mask = geometry_mask(other_geometries, transform=self.transform, invert=False,
                                                           out_shape=self.dtm.shape)

                        for i in range(1, len(self.dsm)):
                            self.dsm[i][...] = np.where(remove_others_mask, self.dsm[i], np.nan)

    def update_dsm(self, user_buildings, user_array=None, user_arrays=None, higher_buildings=None):
        dsm = self.dsm
        self.is3D = user_arrays is not None

        if self.is3D:
            if isinstance(self.dsm, np.ndarray):
                self.dsm = [self.dsm]

        for building in user_buildings:
            if 'geometry' in building:
                geom = shape(building['geometry'])

                mask = geometry_mask([geom], transform=self.transform, invert=True, out_shape=self.dtm.shape)
                # Find the minimum value within the mask
                min_value = np.fmin(self.dtm[mask])

                if not self.is3D:
                    dsm[mask] = user_array[mask] + min_value
                else:
                    dsm[0][mask] = user_array[mask] + min_value

                    if higher_buildings:
                        new_build = next(
                            (b for b in higher_buildings if b['identificatie'] == building["identificatie"]),
                            None
                        )

                        if new_build and 'geometry' in new_build:
                            new_geom = shape(new_build["geometry"])
                            new_mask = geometry_mask([new_geom], transform=self.transform, invert=True,
                                                     out_shape=self.dtm.shape)

                            for i in range(1, len(user_array)):
                                while len(self.dsm) <= i:
                                    self.dsm.append(np.full_like(self.dtm, np.nan))

                                dsm[i][new_mask] = user_array[i][new_mask] + min_value








class CHM:
    def __init__(self, bbox, dtm, trunk_height, output_folder, input_folder):
        self.bbox = bbox
        self.crs = (CRS.from_epsg(28992))
        self.dtm = dtm
        self.gdf = gpd.read_file("geotiles/AHN_lookup.geojson")
        self.chm, self.polygons, self.transform = self.init_chm(bbox, output_folder=output_folder, input_folder=input_folder)
        self.trunk_array = self.chm * trunk_height
        self.original_chm, self.og_polygons, self.original_trunk = self.chm, self.polygons, self.trunk_array



    def find_tiles(self, x_min, y_min, x_max, y_max):
        query_geom = box(x_min, y_min, x_max, y_max)
        matches = self.gdf.sindex.query(
            query_geom)  # predicate="overlaps": tricky i want to still get something if it is all contained in one
        return self.gdf.iloc[matches]["GT_AHNSUB"].tolist()

    @staticmethod
    def filter_points_within_bounds(las_data, bounds):
        """Filter points within the given bounding box."""
        x_min, y_min, x_max, y_max = bounds
        mask = (
                (las_data.x >= x_min) & (las_data.x <= x_max) &
                (las_data.y >= y_min) & (las_data.y <= y_max)
        )
        return las_data[mask]

    @staticmethod
    def extract_vegetation_points(LasData, ndvi_threshold=0.1, pre_filter=False):
        """
        Extract vegetation points based on classification and NDVI threshold.
        ------
        Input:
        - LasData (laspy.LasData): Input point cloud data in LAS format.
        - ndvi_threshold (float): The NDVI threshold for identifying vegetation points.
                                  NDVI values greater than this threshold are considered vegetation.
        - pre_filter (bool): If True, applies an additional filter to remove vegetation points below a certain height
                             threshold (1.5 meters above the lowest vegetation point).
        Output:
        - laspy.LasData: A new LasData object containing only the filtered vegetation points based on the specified criteria.
        """

        # Filter points based on classification (vegetation-related classes), note: vegetation classes are empty in AHN4
        possible_vegetation_points = LasData[(LasData.classification == 1) |  # Unclassified
                                             (LasData.classification == 3) |  # Low vegetation
                                             (LasData.classification == 4) |  # Medium vegetation
                                             (LasData.classification == 5)]  # High vegetation

        # Calculate NDVI
        red = possible_vegetation_points.red
        nir = possible_vegetation_points.nir
        ndvi = (nir.astype(float) - red) / (nir + red)

        # Filter the points whose NDVI is greater than the threshold
        veg_points = possible_vegetation_points[ndvi > ndvi_threshold]

        # Option: already filter away the points with a height below 1.5m from the lowest veg point, introduced because
        # of one very large tile (25GN2_24.LAZ)
        if pre_filter:
            heights = veg_points.z
            min_height = heights.min()

            # Filter out points with heights between the minimum height and 1.5 meters
            filtered_veg_points = veg_points[(heights <= min_height) | (heights > 1.5)]
            return filtered_veg_points
        return veg_points

    @staticmethod
    def raster_center_coords(min_x, max_x, min_y, max_y, resolution):
        """
        Compute the center xy coordinates of a grid.
        ----
        Input:
        - min_x, max_x, min_y, max_y(float): Minimum and maximum x and y coordinates of the grid.
        - resolution (float): The length of each cell, function can only be used for square cells.

        Output:
        - grid_center_x: a grid where each cell contains the value of its center point's x coordinates.
        - grid_center_y: a grid where each cell contains the value of its center point's y coordinates.
        """
        # create coordinates for the x and y border of every cell.
        x_coords = np.arange(min_x, max_x, resolution)  # x coordinates expand from left to right.
        y_coords = np.arange(max_y, min_y, -resolution)  # y coordinates reduce from top to bottom.

        # create center point coordinates for evey cell.
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        grid_center_x = grid_x + resolution / 2
        grid_center_y = grid_y - resolution / 2
        return grid_center_x, grid_center_y

    @staticmethod
    def median_filter_chm(chm_array, nodata_value=-9999, size=3):
        """
        Apply a median filter to a CHM, handling NoData values.
        -----
        Parameters:
        - chm_array (np.ndarray): The array representing the height values of the CHM.
        - nodata_value (float): Value representing NoData in the input raster.
        - size (int): Size of the median filter. It defines the footprint of the filter.

        Returns:
        - smoothed_chm (np.ndarray): The smoothed CHM array.
        """
        # Create a mask for valid data
        valid_mask = chm_array != nodata_value

        # Pad the data with nodata_value
        pad_width = size // 2
        padded_chm = np.pad(chm_array, pad_width, mode='constant', constant_values=nodata_value)

        # Apply median filter to padded data
        filtered_padded = median_filter(padded_chm.astype(np.float32), size=size)

        # Remove padding
        smoothed_chm = filtered_padded[pad_width:-pad_width, pad_width:-pad_width]

        # Only keep valid data in smoothed result
        smoothed_chm[~valid_mask] = nodata_value

        return smoothed_chm

    def interpolation_vegetation(self, LasData, veg_points, resolution, no_data_value=-9999):
        """
        Create a vegetation raster using Laplace interpolation.

        InpurL
        - LasData (laspy.LasData):          Input LiDAR point cloud data.
        - veg_points (laspy.LasData):       Vegetation points to be interpolated.
        - resolution (float):               Resolution of the raster.
        - no_data_value (int, optional):    Value for no data

        Returns:
        - interpolated_grid (np.ndarray): Generated raster for vegetation.
        - grid_center_xy (tuple): Grid of x, y center coordinates for each raster cell.
        """

        # Extents of the pc
        min_x, max_x = round(LasData.x.min()), round(LasData.x.max())
        min_y, max_y = round(LasData.y.min()), round(LasData.y.max())

        # Define size of the region
        x_length = max_x - min_x
        y_length = max_y - min_y

        # Number of rows and columns
        cols = round(x_length / resolution)
        rows = round(y_length / resolution)

        # Initialize raster grid
        veg_raster = np.full((rows, cols), no_data_value, dtype=np.float32)

        # Calculate center coords for each grid cell
        grid_center_xy = self.raster_center_coords(min_x, max_x, min_y, max_y, resolution)

        if veg_points.x.shape[0] == 0:
            print("There are no vegetation points in the current area.")
            veg_raster = np.full((rows, cols), -200, dtype=np.float32)
            return veg_raster, grid_center_xy

        # create the delaunay triangulation
        dt = startinpy.DT()
        dt.insert(veg_points.xyz, "BBox")

        # Flatten the grid to get a list of all center coords
        locs = np.column_stack((grid_center_xy[0].ravel(), grid_center_xy[1].ravel()))

        vegetation_points = np.column_stack((veg_points.x, veg_points.y))
        tree = cKDTree(vegetation_points)

        # Find the distance to the nearest vegetation point for each grid cell
        distances, _ = tree.query(locs, k=1)

        distance_threshold = 1
        # masking cells that exceed threshold
        within_threshold_mask = distances <= distance_threshold
        # Interpolation only for those near
        valid_locs = locs[within_threshold_mask]

        # laplace interpolation
        interpolated_values = dt.interpolate({"method": "Laplace"}, valid_locs)

        # reshape interpolated grid back to og
        interpolated_grid = np.full_like(veg_raster, no_data_value, dtype=np.float32)  # Start with no_data
        interpolated_grid.ravel()[within_threshold_mask] = interpolated_values

        return interpolated_grid, grid_center_xy

    def download_las_tiles(self, matching_tiles, output_folder):
        base_url = "https://geotiles.citg.tudelft.nl/AHN5_T"
        os.makedirs(output_folder, exist_ok=True)

        for full_tile_name in matching_tiles:
            # Extract tile name and sub-tile number
            if '_' in full_tile_name:
                tile_name, sub_tile = full_tile_name.split('_')
            else:
                print(f"Skipping invalid tile entry: {full_tile_name}")
                continue

            # Construct the URL and file path
            sub_tile_str = f"_{int(sub_tile):02}"
            url = f"{base_url}/{tile_name}{sub_tile_str}.LAZ"
            filename = f"{tile_name}{sub_tile_str}.LAZ"
            file_path = os.path.join(output_folder, filename)

            # Check if the file already exists to avoid re-downloading
            if os.path.exists(file_path):
                print(f"File {file_path} already exists, skipping download.")
                continue

            # Attempt to download the file
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded and saved {file_path}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {url}: {e}")

    def merge_las_files(self, laz_files, bounds, merged_output):
        merged_output = Path(merged_output)

        las_merged = None
        all_points = []
        merged_scales = None
        merged_offset = None

        if merged_output.exists():
            with laspy.open(merged_output) as las:
                las_merged = las.read()
            return las_merged

        for file in laz_files:
            with laspy.open(file) as las:
                las_data = las.read()
                cropped_las = self.filter_points_within_bounds(las_data, bounds)

                if las_merged is None:
                    # Initialize merged LAS file using the first input file
                    las_merged = laspy.LasData(las_data.header)
                    las_merged.points = cropped_las.points

                    merged_scales = las_merged.header.scales
                    merged_offset = las_merged.header.offset
                else:

                    scale = las_data.header.scales
                    offset = las_data.header.offsets
                    # Convert integer coordinates to real-world values & Transform into merged coordinate system
                    new_x = ((cropped_las.X * scale[0] + offset[0]) - merged_offset[0]) / merged_scales[0]
                    new_y = ((cropped_las.Y * scale[1] + offset[1]) - merged_offset[1]) / merged_scales[1]
                    new_z = ((cropped_las.Z * scale[2] + offset[2]) - merged_offset[2]) / merged_scales[2]

                    # Copy points and update X, Y, Z
                    new_points = cropped_las.points
                    new_points["X"] = new_x.astype(np.int32)
                    new_points["Y"] = new_y.astype(np.int32)
                    new_points["Z"] = new_z.astype(np.int32)

                    all_points.append(new_points.array)

                    # Final merge step
        if las_merged is not None:
            if all_points:
                all_points.append(las_merged.points.array)

                merged_array = np.concatenate(all_points, axis=0)
                las_merged.points = laspy.ScaleAwarePointRecord(merged_array, las_merged.header.point_format,
                                                                las_merged.header.scales, las_merged.header.offsets)

            las_merged.write(str(merged_output))

        return las_merged

    @staticmethod
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

    def chm_creation(self, LasData, vegetation_data, output_filename, resolution=0.5, smooth=False, nodata_value=-9999,
                     filter_size=3):
        """
        Create a CHM from LiDAR vegetation data and save it as a raster.
        -------
        Input:
        - LasData (laspy.LasData):      Input LiDAR point cloud data used for metadata and output CRS.
        - vegetation_data (tuple):      A tuple containing:
                            - veg_raster (numpy.ndarray): The array representing the height values of vegetation.
                            - grid_centers (tuple of numpy.ndarrays): Contains two arrays (x, y) with the coordinates
                              of the center points of each grid cell.
        - output_filename (str): The name of the output .tif file for saving the CHM.
        - resolution (float, optional): The spatial resolution of the output raster in the same units as the input data
                                        (default: 0.5).
        - smooth (bool, optional): If True, applies a median filter to smooth the CHM.
        - nodata_value (float, optional): The value for NoData pixels (default: -9999).
        - filter_size (int, optional): Size of the median filter (default: 3).

        Output:
        - None: The function saves the CHM as a raster file (.tif) to the specified output path.
        """
        print(resolution)
        veg_raster = vegetation_data[0]
        grid_centers = vegetation_data[1]
        top_left_x = grid_centers[0][0, 0] - resolution / 2
        top_left_y = grid_centers[1][0, 0] + resolution / 2

        transform = Affine.translation(top_left_x, top_left_y) * Affine.scale(resolution, -resolution)

        if smooth:
            veg_raster = self.median_filter_chm(veg_raster, nodata_value=nodata_value, size=filter_size)

        veg_raster, new_transform = self.chm_finish(veg_raster, self.dtm, transform)

        write_output(LasData, self.crs, veg_raster, new_transform, output_filename, True)

        # create the polygons
        labeled_array, num_clusters = label(veg_raster > 0)
        shapes_gen = shapes(labeled_array.astype(np.uint8), mask=(labeled_array > 0), transform=transform)
        polygons = [
            {"geometry": shape(geom), "polygon_id": int(value)}
            for geom, value in shapes_gen if value > 0
        ]

        gdf = gpd.GeoDataFrame(geometry=polygons, crs=CRS.from_epsg(28992))
        gdf.to_file("output/tree_clusters.geojson", driver="GeoJSON")

        return veg_raster, polygons, new_transform

    def init_chm(self, bbox, output_folder="output", input_folder="temp",  merged_output="output/pointcloud.las",  smooth_chm=True, resolution=0.5, ndvi_threshold=0.05, filter_size=3):

        matching_tiles = self.find_tiles(*bbox)
        print("Tiles covering the area:", matching_tiles)

        existing_tiles = {
            os.path.splitext(file)[0] for file in os.listdir(input_folder) if file.endswith(".LAZ")
        }

        missing_tiles = [tile for tile in matching_tiles if tile not in existing_tiles]

        if missing_tiles:
            print("Missing tiles:", missing_tiles)
            self.download_las_tiles(missing_tiles, input_folder)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        laz_files = [
            os.path.join(input_folder, file)
            for file in os.listdir(input_folder)
            if file.endswith(".LAZ") and os.path.splitext(file)[0] in matching_tiles
        ]

        if not laz_files:
            print("No relevant LAZ files found in the input folder or its subfolders.")
            return

        las_data = self.merge_las_files(laz_files, bbox, merged_output)

        if las_data is None:
            print("No valid points found in the given boundary.")
            return

        # Extract vegetation points
        veg_points = self.extract_vegetation_points(las_data, ndvi_threshold=ndvi_threshold, pre_filter=False)

        vegetation_data = self.interpolation_vegetation(las_data, veg_points, 0.5)
        output_filename = os.path.join(output_folder, f"CHM.TIF")

        # Create the CHM and save it
        chm, polygons, transform = self.chm_creation(las_data, vegetation_data, output_filename, resolution=resolution, smooth=smooth_chm, nodata_value=-9999,
                     filter_size=filter_size)

        return chm, polygons, transform

    def remove_trees(self, tree_id):
        target_polygons = [tree["geometry"] for tree in self.polygons if tree["polygon_id"] == tree_id]

        if not target_polygons:
            print(f"No trees found with ID: {tree_id}")
            return

        tree_mask = geometry_mask(
            geometries=target_polygons,
            transform=self.transform,
            invert=True,
            out_shape=self.chm.shape
        )

        self.chm = np.where(tree_mask, 0, self.chm)
        self.trunk_array = np.where(tree_mask, 0, self.trunk_array)
        write_output(None, self.crs, self.chm, self.transform, "output/updated_chm.tif")

    def insert_tree(self, position, height, crown_radius, resolution=0.5, trunk_height=0.0, type='parabolic', randomness=0.8):
        '''
        Function

        Inputs:
        array (2d-numpy array):         Canopy Height Model Array (CHM)
        trunk_array (2d-numpy array):   Array of trunk heights
        position (tuple):               (row, col) coordinates for tree center.
        height (float):                 Total height of the tree.
        crown_radius (float):           Radius of the crown in real-world units.
        trunk_height (float):           Height of the trunk.
        type (str):                     Canopy shape type ('gaussian', 'cone', etc.).
        randomness (float):             Randomness/noise factor.
        resolution (float)              Real-world units per pixel (default = 1.0).

        Output:
        new_array (2d-numpy array):         Modified CHM array.
        new_trunk_array (2d-numpy array):   Modified Trunk height array
        '''
        new_array = np.copy(self.chm)
        new_trunk_array = np.copy(self.trunk_array)

        crown_radius_px = crown_radius / resolution
        size = int(crown_radius_px * 2.5)

        # Calculate the distance from surrounding cells to the tree center
        x = np.arange(-size//2, size//2 +1)
        y = np.arange(-size//2, size//2 + 1)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)

        # Create canopy shape
        if type == 'gaussian':
            canopy = (height - trunk_height) * np.exp(-distance**2 / (2 * (crown_radius_px / 2)**2)) + trunk_height
            print(canopy)
        elif type == 'cone':
            canopy = np.clip((height - trunk_height) * (1 - distance / crown_radius_px), 0, height - trunk_height) + trunk_height
        elif type == 'parabolic':
            canopy = (height - trunk_height) * (1 - (distance / crown_radius_px)**2)
            canopy = np.clip(canopy, 0, height - trunk_height) + trunk_height
        elif type == 'hemisphere':
            canopy = np.sqrt(np.clip(crown_radius_px**2 - distance**2, 0, None)) / crown_radius_px * (height - trunk_height) + trunk_height
        else:
            raise ValueError("Unsupported tree type.")

        mask = (distance <= crown_radius_px) & (canopy >= trunk_height)

        noise = np.random.normal(0, randomness, canopy.shape)
        canopy[mask] += noise[mask]

        canopy[~mask] = 0
        canopy = np.clip(canopy, 0, None)

        # Define insertion window
        row, col = position
        half_size = size // 2
        r_start = max(0, row - half_size)
        r_end = min(self.chm.shape[0], row + half_size)
        c_start = max(0, col - half_size)
        c_end = min(self.chm.shape[1], col + half_size)

        # Calculate actual insertion indices
        canopy_r_start = half_size - (row - r_start)
        canopy_r_end = canopy_r_start + (r_end - r_start)
        canopy_c_start = half_size - (col - c_start)
        canopy_c_end = canopy_c_start + (c_end - c_start)

        # Blend
        new_array[r_start:r_end, c_start:c_end] = np.maximum(
            self.chm[r_start:r_end, c_start:c_end],
            canopy[canopy_r_start:canopy_r_end, canopy_c_start:canopy_c_end]
        )

        new_trunk_array[r_start:r_end, c_start:c_end] = np.minimum(
        self.trunk_array[r_start:r_end, c_start:c_end],
        trunk_height
        )

        tree_mask = (new_array > self.chm)
        shapes_gen = shapes(tree_mask.astype(np.uint8), mask=tree_mask, transform=self.transform)
        tree_polygons = [
            {"geometry": mapping(shape(geom)), "tree_id": str(uuid.uuid4())[:8]}
            for geom, value in shapes_gen if value > 0
        ]

        self.tree_polygons.extend(tree_polygons)
        self.chm, self.trunk_array = new_array, new_trunk_array


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


if __name__ == "__main__":
    # bbox = (94500, 469500, 95000, 470000)
    bbox = (122630, 487150, 123030, 487550)


    buildings = Buildings(bbox).building_geometries
    DEMS = DEMS(bbox, buildings, bridge=True)
    dtm = DEMS.dtm
    # chm = CHM(bbox, dtm, 0.25, "output", "temp")

    # buildings = Buildings(bbox).data
    # # buildings_data = load_buildings("temp/buildings_test.gpkg", "buildings")
    #
    # # DEMS = DEMS(bbox, buildings_data)
    # # dtm = DEMS.dtm
    # # dsm = DEMS.dsm
    # with rasterio.open("output/final_dtm_test.tif") as src:
    #     dtm = src.read(1)
    # chm = CHM(bbox, dtm, "output", "temp2").chm


    # def plot_raster(filepath, title="Raster Data"):
    #     with rasterio.open(filepath) as src:
    #         array = src.read(1)
    #         plt.figure(figsize=(10, 8))
    #         plt.imshow(array, cmap="viridis", origin="upper")
    #         plt.colorbar(label="Elevation (m)")
    #         plt.title(title)
    #         plt.show()

    # plot_raster("output/final_dtm_test.tif", "Final Filled DTM")
    # plot_raster("output/final_dsm_test.tif", "Final filled DSM")
    # plot_raster("output/CHM_test.tif", "Final CHM")
    #
    # with rasterio.open("output/CHM_test.tif") as src:
    #     tree_height = src.read(1)
    #     transform = src.transform
    #     crs = src.crs

