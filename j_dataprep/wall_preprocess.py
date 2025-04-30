import cupy as cp
from cupyx.scipy.ndimage import maximum_filter
import cupyx.scipy.ndimage as cnd
from osgeo import gdal

from util.misc import saveraster
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.ndimage.interpolation as sc
import cProfile
import pstats
import io

def cart2pol(x, y, units='deg'):
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    if units in ['deg', 'degs']:
        theta = theta * 180 / np.pi
    return theta, radius


def get_ders(dsm, scale):
    # dem,_,_=read_dem_grid(dem_file)
    dx = 1 / scale
    # dx=0.5
    fy, fx = np.gradient(dsm, dx, dx)
    asp, grad = cart2pol(fy, fx, 'rad')
    grad = np.arctan(grad)
    asp = asp * -1
    asp = asp + (asp < 0) * (np.pi * 2)
    return grad, asp


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
        np_dsm_array = np.array(dsm.GetRasterBand(1).ReadAsArray(), dtype=np.float32)
        self.minheight = minheight

        profiler = cProfile.Profile()
        profiler.enable()

        self.wall_height = self.findwalls(dsm_array)
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
        ps.print_stats(20)  # top 20 functions
        profiler.dump_stats("profiling/cupy_wall_aspect_profile.prof")
        print(s.getvalue())


        self.wall_aspect = self.filter_aspect_sobel(dsm_array)
        #self.filter1Goodwin_as_aspect_v3(self.wall_height.get(), 0.5, np_dsm_array)




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


    def findwalls_np(self, a, walllimit):
        col = a.shape[0]
        row = a.shape[1]
        walls = np.zeros((col, row))
        domain = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        index = 0
        for i in np.arange(1, row - 1):
            for j in np.arange(1, col - 1):
                dom = a[j - 1:j + 2, i - 1:i + 2]
                walls[j, i] = np.max(dom[np.where(domain == 1)])  # new 20171006
                index = index + 1

        walls = np.copy(walls - a)  # new 20171006
        walls[(walls < walllimit)] = 0

        walls[0:walls.shape[0], 0] = 0
        walls[0:walls.shape[0], walls.shape[1] - 1] = 0
        walls[0, 0:walls.shape[0]] = 0
        walls[walls.shape[0] - 1, 0:walls.shape[1]] = 0
        return walls

    @staticmethod
    def filter1Goodwin_as_aspect_v3(walls, scale, a):
        """
        tThis function applies the filter processing presented in Goodwin et al (2010) but instead for removing
        linear fetures it calculates wall aspect based on a wall pixels grid, a dsm (a) and a scale factor

        Fredrik Lindberg, 2012-02-14
        fredrikl@gvc.gu.se

        Translated: 2015-09-15

        :param walls:
        :param scale:
        :param a:
        :return: dirwalls
        """

        row = a.shape[0]
        col = a.shape[1]

        filtersize = np.floor((scale + 0.0000000001) * 9)
        if filtersize <= 2:
            filtersize = 3
        else:
            if filtersize != 9:
                if filtersize % 2 == 0:
                    filtersize = filtersize + 1

        filthalveceil = int(np.ceil(filtersize / 2.))
        filthalvefloor = int(np.floor(filtersize / 2.))

        filtmatrix = np.zeros((int(filtersize), int(filtersize)))
        buildfilt = np.zeros((int(filtersize), int(filtersize)))

        filtmatrix[:, filthalveceil - 1] = 1
        n = filtmatrix.shape[0] - 1
        buildfilt[filthalveceil - 1, 0:filthalvefloor] = 1
        buildfilt[filthalveceil - 1, filthalveceil: int(filtersize)] = 2

        y = np.zeros((row, col))  # final direction
        z = np.zeros((row, col))  # temporary direction
        x = np.zeros((row, col))  # building side
        walls[walls > 0] = 1

        for h in range(0, 180):  # =0:1:180 #%increased resolution to 1 deg 20140911
            print(h)
            filtmatrix1temp = sc.rotate(filtmatrix, h, order=1, reshape=False, mode='nearest')  # bilinear
            filtmatrix1 = np.round(filtmatrix1temp)
            # filtmatrix1temp = sc.imrotate(filtmatrix, h, 'bilinear')
            # filtmatrix1 = np.round(filtmatrix1temp / 255.)
            # filtmatrixbuildtemp = sc.imrotate(buildfilt, h, 'nearest')
            filtmatrixbuildtemp = sc.rotate(buildfilt, h, order=0, reshape=False, mode='nearest')  # Nearest neighbor
            # filtmatrixbuild = np.round(filtmatrixbuildtemp / 127.)
            filtmatrixbuild = np.round(filtmatrixbuildtemp)
            index = 270 - h
            if h == 150:
                filtmatrixbuild[:, n] = 0
            if h == 30:
                filtmatrixbuild[:, n] = 0
            if index == 225:
                # n = filtmatrix.shape[0] - 1  # length(filtmatrix);
                filtmatrix1[0, 0] = 1
                filtmatrix1[n, n] = 1
            if index == 135:
                # n = filtmatrix.shape[0] - 1  # length(filtmatrix);
                filtmatrix1[0, n] = 1
                filtmatrix1[n, 0] = 1

            for i in range(int(filthalveceil) - 1, row - int(filthalveceil) - 1):  # i=filthalveceil:sizey-filthalveceil
                for j in range(int(filthalveceil) - 1,
                               col - int(filthalveceil) - 1):  # (j=filthalveceil:sizex-filthalveceil
                    if walls[i, j] == 1:
                        wallscut = walls[i - filthalvefloor:i + filthalvefloor + 1,
                                   j - filthalvefloor:j + filthalvefloor + 1] * filtmatrix1
                        dsmcut = a[i - filthalvefloor:i + filthalvefloor + 1, j - filthalvefloor:j + filthalvefloor + 1]
                        if z[i, j] < wallscut.sum():  # sum(sum(wallscut))
                            z[i, j] = wallscut.sum()  # sum(sum(wallscut));
                            if np.sum(dsmcut[filtmatrixbuild == 1]) > np.sum(dsmcut[filtmatrixbuild == 2]):
                                x[i, j] = 1
                            else:
                                x[i, j] = 2

                            y[i, j] = index

        y[(x == 1)] = y[(x == 1)] - 180
        y[(y < 0)] = y[(y < 0)] + 360

        grad, asp = get_ders(a, scale)

        y = y + ((walls == 1) * 1) * ((y == 0) * 1) * (asp / (math.pi / 180.))

        dirwalls = y

        return dirwalls


if __name__ == "__main__":
    folder_list = ['250' '500', '1000', '1500', '2000', '3000']
    D = 'D'
    for folder in folder_list:
        file = f"{D}:/Geomatics/optimization_tests/{folder}/final_dsm_over.tif"
        output_aspect = f"{D}:/Geomatics/optimization_tests/{folder}/wallaspect.tif"
        output_height = f"{D}:/Geomatics/optimization_tests/{folder}/wallheight.tif"
        gdal_dsms = gdal.Open(file)

        walldata = WallData(gdal_dsms, 2)
        wall_aspect = walldata.wall_aspect

        saveraster(gdal_dsms, output_aspect, wall_aspect.get())
        saveraster(gdal_dsms, output_height, walldata.wall_height.get())

