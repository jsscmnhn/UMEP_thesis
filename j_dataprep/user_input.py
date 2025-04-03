import json
from rasterizer import rasterize_from_python
import numpy as np

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
            # layer 1
            dsms[1][direct_gaps] = arrays[1][direct_gaps]
            dsms[1][both_masks] = arrays[3][both_masks]

            direct_gaps_valid = direct_gaps & ~np.isnan(arrays[2])
            direct_gaps_fallback = direct_gaps & np.isnan(arrays[2])

            if 4 < len(arrays):
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
        input_arrays = rasterize_from_python(obj_path, self.ncols, self.nrows, self.res, [0, 0], -9999)

        if direct_gaps:
            dsms, highest_array = self.buildings_direct_input(input_arrays, num_gaps)
        else:
            dsms, highest_array = self.buildings_input(input_arrays, num_gaps)

        return dsms, highest_array