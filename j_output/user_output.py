import h5py
import numpy as np
from osgeo import gdal
from rasterio import features
import geopandas as gpd
from shapely.geometry import mapping, shape
from affine import Affine
import os
import re

class TmrtOutput:
    def __init__(self, output_folder, building_data=None, buildings_path=None, layer=None):
        self.output_folder = output_folder
        self.building_data = building_data
        self.buildings_path = buildings_path
        self.gdal_dataset = None
        self.layer = layer
        self.building_mask = self.create_building_mask()

        self.tmrt_arrays_by_time = self.calc_arrays(output_folder)
        self.time_groups = self.group_by_time_of_day()
        self.averaged_tmrt = self.average_time_groups()

        self.pet_arrays_by_time = {}
        self.classified_pet_by_time = {}
        self.averaged_pet = {}
        self.averaged_class_pet = {}

    @staticmethod
    def get_pet_raster_from_lookup(tmrt_raster, wind_speed, air_temp, rh, body_type, lookup_file):
        with h5py.File(lookup_file, "r") as f:
            pet_dataset = f[body_type]

            wind_speeds = np.array([0.1, 2.0, 6.0])
            rhs = np.arange(100, -1, -20)
            tmrts = np.arange(65, -1, -1)
            temps = np.arange(40, -1, -1)

            def find_nearest_index(array, value):
                return np.abs(array - value).argmin()

            ws_idx = find_nearest_index(wind_speeds, wind_speed)
            rh_idx = find_nearest_index(rhs, rh)
            ta_idx = find_nearest_index(temps, air_temp)

            print(ws_idx)
            print(rh_idx)
            print(ta_idx)

            # Clamp and map tmrt values to lookup indices (reversed axis)
            tmrt_clipped = np.clip(tmrt_raster, 0, 65)
            tmrt_rounded = np.round(tmrt_clipped).astype(int)  # Round to nearest integer
            # Map tmrt values to lookup indices (reversed axis)
            valid_mask = ~np.isnan(tmrt_clipped)
            tmrt_indices = np.full_like(tmrt_raster, -1, dtype=np.int32)
            tmrt_indices[valid_mask] = (65 - tmrt_rounded[valid_mask]).astype(np.int32)


            pet_raster = np.take(pet_dataset[ws_idx, rh_idx, :, ta_idx], tmrt_indices)
            pet_raster[~valid_mask] = np.nan
            return pet_raster

    def create_building_mask(self):
        """Create building mask using the first matching TMRT file"""
        pattern = re.compile(r'^Tmrt_\d{4}_\d{3}_(\d{4})D\.tif$')

        first_tmrt_file = None
        for filename in os.listdir(self.output_folder):
            if pattern.match(filename):
                first_tmrt_file = os.path.join(self.output_folder, filename)
                break

        if not first_tmrt_file:
            print("No TMRT .tif files found in the folder.")
            return None

        self.gdal_dataset = gdal.Open(first_tmrt_file)

        if self.building_data is not None:
            building_shapes = [shape(b['geometry']) for b in self.building_data if 'geometry' in b]
        else:
            gdf = gpd.read_file(self.buildings_path, layer=self.layer)
            building_shapes = [geom for geom in gdf.geometry]

        transform = Affine.from_gdal(*self.gdal_dataset.GetGeoTransform())
        raster_shape = self.gdal_dataset.GetRasterBand(1).ReadAsArray().shape

        mask = features.rasterize(
            ((mapping(geom), 1) for geom in building_shapes),
            out_shape=raster_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        return mask


    def calc_arrays(self, output_folder):
        tmrt_arrays_by_time = {}

        pattern = re.compile(r'^Tmrt_\d{4}_\d{3}_(\d{4})D\.tif$')

        for filename in os.listdir(output_folder):
            match = pattern.match(filename)
            if match:
                time_key = match.group(1)
                file_path = os.path.join(output_folder, filename)

                # Open file using gdal
                dataset = gdal.Open(file_path)
                if dataset is None:
                    print(f"Could not open {file_path}")
                    continue

                band = dataset.GetRasterBand(1)
                array = band.ReadAsArray()

                masked_array = np.where(self.building_mask == 1, np.nan, array)

                tmrt_arrays_by_time[time_key] = masked_array

        return tmrt_arrays_by_time

    def get_time_group(self, time_str):
        time_val = int(time_str)
        if 600 <= time_val < 1200:
            return 'morning'
        elif 1200 <= time_val < 1800:
            return 'afternoon'
        elif 1800 <= time_val <= 2100:
            return 'evening'
        return None

    def group_by_time_of_day(self):
        grouped = {'morning': [], 'afternoon': [], 'evening': []}

        for time_str, array in self.tmrt_arrays_by_time.items():
            group = self.get_time_group(time_str)
            if group:
                grouped[group].append(array)

        return grouped

    def average_time_groups(self):
        avg_by_group = {}
        for group, arrays in self.time_groups.items():
            if arrays:
                stacked = np.stack(arrays)
                avg = np.mean(stacked, axis=0)
                avg_by_group[group] = avg
            else:
                avg_by_group[group] = None
        return avg_by_group

    def calculate_stats_and_bins(self, array, pixel_size=0.5, isTmrt=True):
        if isTmrt:
            bins = [-np.inf, 15, 20, 25, 30, 35, 40, 45, 50, np.inf]
        else:
            bins = [-np.inf, 4, 8, 13, 18, 23, 29, 35, 41, np.inf]
        masked = array[~np.isnan(array)]

        stats = {
            'mean': np.nanmean(array),
            'median': np.nanmedian(array),
            'min': np.nanmin(array),
            'max': np.nanmax(array),
        }

        hist, bin_edges = np.histogram(masked, bins=bins)
        pixel_area = pixel_size ** 2
        bin_areas = hist * pixel_area
        total_area = np.sum(bin_areas)
        bin_percentages = (bin_areas / total_area) * 100

        stats['bins'] = [
            {
                'range': (bin_edges[i], bin_edges[i + 1]),
                'area_m2': bin_areas[i],
                'percentage': bin_percentages[i]
            }
            for i in range(len(hist))
        ]

        return stats

    def classify_pet(self, pet_array):
        """
        Classify PET array into bins.
        Returns an integer array where each pixel has the class index (0 to len(bins)-2).
        Pixels with NaN PET will remain NaN.
        """
        bins = [-np.inf, 4, 8, 13, 18, 23, 29, 35, 41, np.inf]
        classified = np.digitize(pet_array, bins) - 1
        classified = classified.astype(float)
        classified[np.isnan(pet_array)] = np.nan

        return classified

    def calc_pet(self, Ta, RH, va, body_type="standard_man", lookup_file="pet_lookup.h5"):
        """
        Compute PET array for the 1300 TMRT time step using precomputed PET lookup.
        Inputs:
            Ta: Air temperature in °C (scalar)
            RH: Relative humidity in % (scalar)
            va: Wind speed in m/s (scalar)
            body_type: One of "standard_man", "elderly_woman", "young_child"
            lookup_file: Path to PET HDF5 lookup file
        """
        for time_key, Tmrt in self.tmrt_arrays_by_time.items():
            if Tmrt is None:
                print(f"Skipping {time_key}: Tmrt data is None")
                continue

            try:
                pet_array = self.get_pet_raster_from_lookup(Tmrt, va, Ta, RH, body_type, lookup_file)
                self.pet_arrays_by_time[time_key] = pet_array
                self.classified_pet_by_time[time_key] = self.classify_pet(pet_array)
            except Exception as e:
                print(f"Failed to compute PET for {time_key}: {e}")

        for time_key, Tmrt in self.averaged_tmrt.items():
            if Tmrt is None:
                print(f"Skipping {time_key}: Tmrt data is None")
                continue

            try:
                pet_array = self.get_pet_raster_from_lookup(Tmrt, va, Ta, RH, body_type, lookup_file)
                self.averaged_pet[time_key] = pet_array
                self.averaged_class_pet[time_key] = self.classify_pet(pet_array)
            except Exception as e:
                print(f"Failed to compute PET for {time_key}: {e}")