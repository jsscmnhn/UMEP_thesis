import cupy as cp
from cupyx.scipy.ndimage import maximum_filter
import cupyx.scipy.ndimage as cnd
from osgeo import gdal
from util.misc import saveraster
import matplotlib.pyplot as plt

def show_array(array, title="Array Visualization", cmap="viridis"):
    """
    Display a CuPy or NumPy array using Matplotlib.
    If the input is a CuPy array, it is converted to a NumPy array.
    """
    if isinstance(array, cp.ndarray):
        array = array.get()  # Transfer from GPU to CPU

    plt.figure(figsize=(8, 6))
    plt.imshow(array, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.show()

class WallData:
    def __init__(self, dsm, minheight):
        dsm_array = cp.array(dsm.GetRasterBand(1).ReadAsArray(), dtype=cp.float32)
        self.minheight = minheight
        self.wall_height = self.findwalls(dsm_array)
        self.wall_aspect = self.filter_aspect_sobel(dsm_array)

    def findwalls(self, dsm_array):
        # Create the domain mask
        domain = cp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        # Apply a maximum filter to get the max neighbor values (cross-shaped filter)
        walls = maximum_filter(dsm_array, footprint=domain)

        # Subtract original values
        walls = walls - dsm_array

        # Apply wall height limit
        walls[walls < self.minheight] = 0

        # Zero out edges
        walls[:, 0] = 0
        walls[:, -1] = 0
        walls[0, :] = 0
        walls[-1, :] = 0

        return walls

    def filter_aspect_sobel(self, dsm_array, sigma=0):
        """
        Compute wall aspect using a Sobel filter.
        This function computes the gradient of the DSM 'a' using Sobel,
        derives the orientation (aspect) at each pixel, and then assigns that
        orientation only to pixels where 'walls'==1.
        """
        # Ensure walls are binary
        walls = cp.where(self.wall_height > 0, 1, 0)
        dsm = dsm_array
        # Optional smoothing
        if sigma > 0:
            dsm = cnd.gaussian_filter(dsm, sigma=sigma)

        # Compute the Sobel gradients in the y and x directions
        grad_y = cnd.sobel(dsm, axis=0)
        grad_x = cnd.sobel(dsm, axis=1)

        # Compute the orientation at each pixel: arctan2 returns radians in [-π, π]
        orientation_rad = cp.arctan2(grad_y, grad_x)

        # Convert the orientation to degrees
        orientation_deg = cp.degrees(orientation_rad)

        # Adjust angles to be in the range [0, 360)
        orientation_deg = cp.where(orientation_deg < 0, orientation_deg + 360, orientation_deg)
        orientation_deg = (orientation_deg + 270) % 360

        # Create output: assign orientation only for wall pixels; background remains 0.
        dirwalls = cp.where(walls == 1, orientation_deg, 0)

        return dirwalls

if __name__ == "__main__":
    file = "E:/Geomatics/thesis/_amsterdamset/location_6/3d/dsm_0.tif"
    output_aspect = "E:/Geomatics/thesis/_amsterdamset/location_6/3d//wallaspect.tif"
    output_height = "E:/Geomatics/thesis/_amsterdamset/location_6/3d//wallheight.tif"
    gdal_dsms = gdal.Open(file)

    walldata = WallData(gdal_dsms, 2)
    wall_aspect = walldata.wall_aspect

    saveraster(gdal_dsms, output_aspect, wall_aspect.get())
    saveraster(gdal_dsms, output_height, walldata.wall_height.get())