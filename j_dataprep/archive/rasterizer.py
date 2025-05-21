import numpy as np
from rusterizer_3d import rasterize_from_python

def create_rasters(obj_file, cols, rows, resolution, num_gaps, origin=[0,0], nodata=-9999):
    arrays = rasterize_from_python(obj_file, cols, rows, resolution, origin=origin, nodata=nodata)
    return buildings_input(arrays, num_gaps)

def buildings_input(arrays, num_gaps):
    if num_gaps == 0:
        return arrays[0]

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

    dsms[0][grounded_mask] =  np.nanmin(np.stack([arrays[0][grounded_mask], arrays[2][grounded_mask]]), axis=0)

    if num_gaps == 1:
        dsms[1][direct_gaps] = arrays[1][direct_gaps]
        dsms[1][both_masks ] = arrays[3][both_masks]

        dsms[2][direct_gaps] = arrays[0][direct_gaps]
        dsms[2][both_masks ] = arrays[0][both_masks]

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

        return dsms
