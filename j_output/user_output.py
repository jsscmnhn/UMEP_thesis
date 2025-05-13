# import h5py
import numpy as np
from osgeo import gdal
from rasterio import features
import geopandas as gpd
from shapely.geometry import mapping, shape
from affine import Affine
import os
import re

class TmrtOutput:
    def __init__(self, output_folder, building_mask=None, water_mask=None):
        self.output_folder = output_folder
        self.gdal_dataset = None

        self.valid_mask = None
        self.init_mask(building_mask, water_mask)

        self.tmrt_arrays_by_time = self.calc_arrays(output_folder)
        self.time_groups = self.group_by_time_of_day()
        self.averaged_tmrt = self.average_time_groups()

        self.pet_arrays_by_time = {}
        self.classified_pet_by_time = {}
        self.averaged_pet = {}
        self.averaged_class_pet = {}

    def init_mask(self, building_mask, water_mask):
        """
        Update the valid mask by combining the building and water masks.
        Pixels marked as buildings or water are invalid (set to False).
        """
        # Start with an all-True mask (valid everywhere)
        self.valid_mask = np.ones_like(building_mask, dtype=bool)

        # Apply the building mask (mark buildings as invalid)
        if building_mask is not None:
            self.valid_mask &= (building_mask != 0)

        # Apply the water mask (mark water areas as invalid)
        if water_mask is not None:
            self.valid_mask &= (water_mask != 0)


    def get_pet_raster_from_lookup(self, tmrt_raster, wind_speed, air_temp, rh, body_type, lookup_file="pet_lookup.h5",
                               tmrt_min=0, tmrt_max=65, tmrt_step=0.5, wind_speeds=None, rhs=None, temps=None):
        """
        Get the PET raster for a given Tmrt raster and environmental conditions.

        Parameters:
            tmrt_raster (np.ndarray): TMRT raster array.
            wind_speed (float): Wind speed in m/s.
            air_temp (float): Air temperature in °C.
            rh (float): Relative humidity in %.
            body_type (str): Body type for PET lookup ('standard_man', 'elderly_woman', 'young_child').
            lookup_file (str): Path to the HDF5 lookup file.
            tmrt_min (float): Minimum TMRT value (default 0°C).
            tmrt_max (float): Maximum TMRT value (default 65°C).
            tmrt_step (float): Step size for TMRT values (default 0.5°C).
            wind_speeds (list): List of wind speed values (default [0.1, 2.0, 6.0]).
            rhs (list): List of relative humidity values (default [100, 80, 60, 40, 20, 0]).
            temps (list): List of air temperature values (default [40, 39.5, 39.0, 38.5, .... 0.5, 0.0]).

        Returns:
            np.ndarray: PET raster corresponding to the given parameters.
        """
        if wind_speeds is None:
            wind_speeds = np.array([0.1, 2.0, 6.0])  # Default wind speed values if not provided
        if rhs is None:
            rhs = np.arange(100, -1, -10)  # Default relative humidity values if not provided
        if temps is None:
            temps = np.arange(40.0, -0.1, -0.5)  # Default air temperature values if not provided

        with h5py.File(lookup_file, "r") as f:
                pet_dataset = f[body_type]

                def find_nearest_index(array, value):
                    return np.abs(array - value).argmin()

                # Find the closest indices for the environmental conditions
                ws_idx = find_nearest_index(wind_speeds, wind_speed)
                rh_idx = find_nearest_index(rhs, rh)
                ta_idx = find_nearest_index(temps, air_temp)

                # Clip and map tmrt values to lookup indices based on configurable parameters
                tmrt_clipped = np.clip(tmrt_raster, tmrt_min, tmrt_max)

                tmrt_clipped_valid = tmrt_clipped[self.valid_mask]

                # Calculate the indices only for valid TMRT values
                tmrt_indices_valid = np.round((tmrt_clipped_valid - tmrt_min) / tmrt_step).astype(int)

                # Ensure that indices stay within bounds for valid TMRT values
                max_index = int((tmrt_max - tmrt_min) / tmrt_step)
                tmrt_indices_valid = np.clip(tmrt_indices_valid, 0, max_index)
                tmrt_indices_valid = max_index - tmrt_indices_valid

                # Create an array of the same shape as the original TMRT raster and fill it with -1
                tmrt_indices = np.full_like(tmrt_raster, -1, dtype=int)

                # Place the valid indices into the correct positions
                tmrt_indices[self.valid_mask] = tmrt_indices_valid

                # Fetch the PET raster based on these indices
                pet_raster = np.take(pet_dataset[ws_idx, rh_idx, :, ta_idx], tmrt_indices)

                # Handle NaN areas where tmrt was outside the valid range
                pet_raster[~self.valid_mask] = np.nan

                return pet_raster

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

                masked_array = np.where(self.valid_mask == 0, np.nan, array)

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

        stats = {
            'mean': np.nanmean(array),
            'median': np.nanmedian(array),
            'min': np.nanmin(array),
            'max': np.nanmax(array),
        }

        hist, bin_edges = np.histogram(array[self.valid_mask], bins=bins)
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
