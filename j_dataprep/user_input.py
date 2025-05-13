import json
from rasterizer import rasterize_from_python
import numpy as np

from osgeo import gdal
from util.misc import saveraster
from rasterio.features import geometry_mask, shapes
from shapely.geometry import shape,box, mapping
import uuid
from rtree import index
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter, label
from rasterio import Affine


class Surface_input:
    def __init__(self, ncols, nrows, resolution):
        self.dictionary = json.load(open("landcover.json", "r", encoding="utf-8"))
        self.res = resolution
        self.cols = ncols
        self.rows = nrows

    def get_value(self, category, key) -> int | None:
        return self.dictionary.get(category, {}).get(key, None)

    def user_input_array(self, obj_path, type):
        array = rasterize_from_python(obj_path, self.ncols, self.nrows, self.res, [0, 0], -9999)
        typeno = self.get_value(type[0], type[1])
        return array[0], typeno


class Building3d_input:
    def __init__(self, ncols, nrows, resolution):
        self.res = resolution
        self.cols = ncols
        self.rows = nrows

    @staticmethod
    def buildings_input(arrays, num_gaps):
        if num_gaps == 0:
            return arrays[0], arrays[0]

        layers = num_gaps * 2 + 1

        if layers > len(arrays) + 1:
            f"Amount of gaps is too high for the given input. "
            return

        dsms = np.full((layers, arrays[0].shape[0], arrays[0].shape[1]), np.nan)
        arrays = [np.where(arr == -9999, np.nan, arr) for arr in arrays]
        grounded_mask = arrays[1] == 0
        direct_gaps = ~grounded_mask
        gaps = arrays[3] > 0
        both_masks = grounded_mask & gaps

        dsms[0][grounded_mask] = np.nanmin(np.stack([arrays[0][grounded_mask], arrays[2][grounded_mask]]), axis=0)

        if num_gaps == 1:
            dsms[1][direct_gaps] = arrays[1][direct_gaps]
            dsms[1][both_masks] = arrays[3][both_masks]

            dsms[2][direct_gaps] = arrays[0][direct_gaps]
            dsms[2][both_masks] = arrays[0][both_masks]

            return dsms, arrays[0]

        if num_gaps > 1:
            if 4 < len(arrays):
                # layer 1
                dsms[1][direct_gaps] = arrays[1][direct_gaps]
                dsms[1][both_masks] = arrays[3][both_masks]

                direct_gaps_valid = direct_gaps & ~np.isnan(arrays[2])
                direct_gaps_fallback = direct_gaps & np.isnan(arrays[2])

                both_masks_valid = both_masks & ~np.isnan(arrays[4])

                for i in range(1, num_gaps):
                    j = 2 * i
                    new_direct_valid = direct_gaps_valid & ~np.isnan(arrays[j])
                    new_direct_fallback = direct_gaps_valid & np.isnan(arrays[j])

                    dsms[j][new_direct_valid] = arrays[j][new_direct_valid]
                    dsms[j][new_direct_fallback] = arrays[0][new_direct_fallback]

                    if j + 1 < len(arrays):
                        dsms[j + 1][new_direct_valid] = arrays[j + 1][new_direct_valid]

                    # For both_masks
                    if j + 2 < len(arrays):
                        new_both_valid = both_masks_valid & ~np.isnan(arrays[j + 2])
                        new_both_fallback = both_masks_valid & np.isnan(arrays[j + 2])

                        dsms[j][new_both_valid] = arrays[j + 2][new_both_valid]
                        dsms[j][new_both_fallback] = arrays[0][new_both_fallback]

                        if j + 3 < len(arrays):
                            dsms[j + 1][new_both_valid] = arrays[j + 3][new_both_valid]

                    # Update masks for next iteration
                    direct_gaps_valid = new_direct_valid
                    both_masks_valid = new_both_valid if j + 2 < len(arrays) else both_masks_valid

                # final layer
                dsms[layers - 1][direct_gaps_valid] = arrays[0][direct_gaps_valid]
                dsms[layers - 1][both_masks_valid] = arrays[0][both_masks_valid]

                return dsms, arrays[0]
            elif len(arrays) == 4:
                dsms[1][direct_gaps] = arrays[1][direct_gaps]
                dsms[1][both_masks] = arrays[3][both_masks]

                direct_gaps_valid = direct_gaps & ~np.isnan(arrays[2])
                direct_gaps_fallback = direct_gaps & np.isnan(arrays[2])

                dsms[2][direct_gaps_valid] = arrays[2][direct_gaps_valid]
                dsms[2][direct_gaps_fallback] = arrays[0][direct_gaps_fallback]

                dsms[2][both_masks] = arrays[0][both_masks]

                dsms[3][direct_gaps_valid] = arrays[3][direct_gaps_valid]
                dsms[4][direct_gaps_valid] = arrays[0][direct_gaps_valid]

                return dsms, arrays[0]

    @staticmethod
    def buildings_direct_input(arrays, num_gaps):
        if num_gaps == 0:
            return arrays[0], arrays[0]

        layers = num_gaps * 2 + 1

        if layers > len(arrays) + 1:
            f"Amount of gaps is too high for the given input. "
            return

        dsms = np.full((layers, arrays[0].shape[0], arrays[0].shape[1]), np.nan)
        arrays = [np.where(arr == -9999, np.nan, arr) for arr in arrays]
        grounded_mask = arrays[1] == 0
        direct_gaps = ~grounded_mask

        dsms[0][grounded_mask] = arrays[0][grounded_mask]

        if num_gaps == 1:
            dsms[1][direct_gaps] = arrays[1][direct_gaps]
            dsms[2][direct_gaps] = arrays[0][direct_gaps]
            return dsms, arrays[0]

        if num_gaps > 1:
            # layer 1
            dsms[1][direct_gaps] = arrays[1][direct_gaps]

            # check if there is


            direct_gaps_valid = direct_gaps & ~np.isnan(arrays[2])
            direct_gaps_fallback = direct_gaps & np.isnan(arrays[2])

            for i in range(1, num_gaps):
                j = 2 * i
                new_direct_valid = direct_gaps_valid & ~np.isnan(arrays[j])
                new_direct_fallback = direct_gaps_valid & np.isnan(arrays[j])

                dsms[j][new_direct_valid] = arrays[j][new_direct_valid]
                dsms[j][new_direct_fallback] = arrays[0][new_direct_fallback]

                if j + 1 < len(arrays):
                    dsms[j + 1][new_direct_valid] = arrays[j + 1][new_direct_valid]

                # Update masks for next iteration
                direct_gaps_valid = new_direct_valid

            # final layer
            dsms[layers - 1][direct_gaps_valid] = arrays[0][direct_gaps_valid]

            return dsms, arrays[0]


    def rasterize_3dbuilding(self, obj_path, num_gaps, direct_gaps):
        input_arrays = rasterize_from_python(obj_path, self.cols, self.rows, self.res, [0, 0], -9999)

        if direct_gaps:
            dsms, highest_array = self.buildings_direct_input(input_arrays, num_gaps)
        else:
            dsms, highest_array = self.buildings_input(input_arrays, num_gaps)

        return dsms, highest_array, input_arrays


def combine_tiffs(gdalinput, datasets, output_file):
    cols = gdalinput.RasterXSize
    rows = gdalinput.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_file, cols, rows, len(datasets), gdal.GDT_Float32)

    for i, dataset in enumerate(datasets):
        if i == 0:
            dataset[np.isnan(dataset)] = 0
        band_data = dataset
        out_band = out_dataset.GetRasterBand(i + 1)
        out_band.WriteArray(band_data)

    out_dataset.SetProjection(gdalinput.GetProjection())
    out_dataset.SetGeoTransform(gdalinput.GetGeoTransform())

    print(f"Output file {output_file} created successfully!")

def insert_user_buildings(highest_array, transform, footprint_array=None):
    user_buildings_higher = []

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
        user_buildings = footprint_buildings
        user_buildings_higher = highest_buildings

        return user_buildings, user_buildings_higher

def update_dsm(dsm, dtm, user_buildings, transform, user_array=None, user_arrays=None, higher_buildings=None):
    if isinstance(dsm, np.ndarray):
        dsm = [dsm]

    template = dsm[0]
    dsm = dsm + [np.full_like(template, np.nan) for _ in range(len(dsm), len(user_arrays))]

    for building in user_buildings:
        if 'geometry' in building:
            geom = shape(building['geometry'])

            mask = geometry_mask([geom], transform=transform, invert=True, out_shape=dtm.shape)
            # Find the minimum value within the mask
            min_value = np.min(dtm[mask])

            dsm[0][mask] = user_arrays[0][mask] + min_value

            if higher_buildings:
                new_build = next(
                    (b for b in higher_buildings if b['parcel_id'] == building['parcel_id']),
                    None
                )

                if new_build and 'geometry' in new_build:
                    new_geom = shape(new_build["geometry"])
                    new_mask = geometry_mask([new_geom], transform=transform, invert=True,
                                             out_shape=dtm.shape)

                    for i in range(1, len(user_arrays)):
                        dsm[i][new_mask] = user_arrays[i][new_mask] + min_value

    return dsm

if __name__ == "__main__":
    data = Building3d_input(200, 200, 0.5)
    path = "D:/Geomatics/thesis/__newgaptesting/example/building.obj"
    tiff = "D:/Geomatics/thesis/_3drust/testing.tif"
    geodataset = gdal.Open(tiff)
    output = "D:/Geomatics/thesis/_3drust/input"

    # path = "D:/Geomatics/thesis/__newgaptesting/example/building.obj"
    # tiff =  "D:/Geomatics/thesis/oldwallvsnewwallmethod/option2/final_dsm.tif"
    # dtm_path = "D:/Geomatics/thesis/oldwallvsnewwallmethod/option2/final_dtm.tif"
    # geodataset = gdal.Open(tiff)
    # output_file = "D:/Geomatics/thesis/oldwallvsnewwallmethod/userinput/examplecomb.tif"
    #
    obj_path = "D:/Geomatics/thesis/objtryouts/3dobj.obj"
    dsm, highest, input_arrays = data.rasterize_3dbuilding(obj_path, 1, 1)

    i = 0
    for array in input_arrays:
        if i == 0:
            array[np.isnan(array)] = 0
        saveraster(geodataset, f"{output}/input_test_{i}.tiff", array)
        i += 1

    i=0
    # for array in dsms:
    #     if i == 0:
    #         array[np.isnan(array)] = 0
    #     saveraster(geodataset, f"D:/Geomatics/thesis/oldwallvsnewwallmethod/userinput/1gap_{i}.tiff", array)
    #     i += 1
    #
    # i=0
    # for array in dsm_new:
    #     if i == 0:
    #         array[np.isnan(array)] = 0
    #     saveraster(geodataset, f"D:/Geomatics/thesis/oldwallvsnewwallmethod/userinput/new1gap_{i}.tiff", array)
    #     i += 1
