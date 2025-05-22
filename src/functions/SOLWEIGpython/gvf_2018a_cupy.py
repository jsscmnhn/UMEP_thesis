import numpy as np
from .sunonsurface_2018a_cupy import sunonsurface_2018a_cupy as sunonsurface_2018a
import cupy as cp


def gvf_2018a_cupy(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall,
              alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover):
    '''
    Calculates ground view factors for longwave and shortwave radiation from urban surfaces
    using GPU-accelerated computation with CuPy, based on directional azimuths.

    Parameters:
          wallsun (cp.ndarray):        Array representing sunlit portions of walls.
          walls (cp.ndarray):          Array of wall heights.
          buildings (cp.ndarray):      2D array representing building heights.
          scale (float):               Scale factor converting units to pixels.
          shadow (cp.ndarray):         2D binary shadow mask (1 = shadowed, 0 = sunlit).
          first (float):               First sensor height for radiative surface influence.
          second (float):              Second sensor height (usually first * 20) for radiative surface influence.
          dirwalls (float):            Direction of building walls in degrees.
          Tg (cp.ndarray):             2D grid of ground temperatures [°C].
          Tgwall (cp.ndarray):         2D grid of wall temperatures [°C].
          Ta (float):                  Air temperature [°C].
          emis_grid (cp.ndarray):      Emissivity grid for surfaces.
          ewall (float):               Wall emissivity.
          alb_grid (cp.ndarray):       Albedo grid for surfaces.
          SBC (float):                 Stefan-Boltzmann constant.
          albedo_b (float):            Building wall albedo.
          rows (int):                  Number of rows in grids.
          cols (int):                  Number of columns in grids.
          Twater (float):              Water temperature [°C].
          lc_grid (cp.ndarray):        Landcover classification grid.
          landcover (int):             Landcover type indicator.

    Returns:
          gvfLup (cp.ndarray):         Grid of longwave upwelling radiation view factors (averaged over azimuths).
          gvfalb (cp.ndarray):         Grid of albedo-weighted view factors including shadows (averaged over azimuths).
          gvfalbnosh (cp.ndarray):     Grid of albedo-weighted view factors excluding shadows (averaged over azimuths).
          gvfLupE (cp.ndarray):        East-facing longwave upwelling radiation view factors.
          gvfalbE (cp.ndarray):        East-facing albedo-weighted view factors including shadows.
          gvfalbnoshE (cp.ndarray):    East-facing albedo-weighted view factors excluding shadows.
          gvfLupS (cp.ndarray):        South-facing longwave upwelling radiation view factors.
          gvfalbS (cp.ndarray):        South-facing albedo-weighted view factors including shadows.
          gvfalbnoshS (cp.ndarray):    South-facing albedo-weighted view factors excluding shadows.
          gvfLupW (cp.ndarray):        West-facing longwave upwelling radiation view factors.
          gvfalbW (cp.ndarray):        West-facing albedo-weighted view factors including shadows.
          gvfalbnoshW (cp.ndarray):    West-facing albedo-weighted view factors excluding shadows.
          gvfLupN (cp.ndarray):        North-facing longwave upwelling radiation view factors.
          gvfalbN (cp.ndarray):        North-facing albedo-weighted view factors including shadows.
          gvfalbnoshN (cp.ndarray):    North-facing albedo-weighted view factors excluding shadows.
          gvfSum (cp.ndarray):         Sum of combined sun/shadow view factors across all azimuth directions.
          gvfNorm (cp.ndarray):        Normalized view factor grid where non-building areas are set to 1.
    '''
    azimuthA = np.arange(5, 359, 20)
    #### Ground View Factors ####
    gvfLup = cp.zeros((rows, cols))
    gvfalb = cp.zeros((rows, cols))
    gvfalbnosh = cp.zeros((rows, cols))
    gvfLupE = cp.zeros((rows, cols))
    gvfLupS = cp.zeros((rows, cols))
    gvfLupW = cp.zeros((rows, cols))
    gvfLupN = cp.zeros((rows, cols))
    gvfalbE = cp.zeros((rows, cols))
    gvfalbS = cp.zeros((rows, cols))
    gvfalbW = cp.zeros((rows, cols))
    gvfalbN = cp.zeros((rows, cols))
    gvfalbnoshE = cp.zeros((rows, cols))
    gvfalbnoshS = cp.zeros((rows, cols))
    gvfalbnoshW = cp.zeros((rows, cols))
    gvfalbnoshN = cp.zeros((rows, cols))
    gvfSum = cp.zeros((rows, cols))

    #  sunwall=wallinsun_2015a(buildings,azimuth(i),shadow,psi(i),dirwalls,walls);
    sunwall = (wallsun / walls * buildings) == 1  # new as from 2015a

    for j in np.arange(0, azimuthA.__len__()):
        _, gvfLupi, gvfalbi, gvfalbnoshi, gvf2 = sunonsurface_2018a(azimuthA[j], scale, buildings, shadow, sunwall,
                                                                    first,
                                                                    second, dirwalls * np.pi / 180, walls, Tg, Tgwall,
                                                                    Ta,
                                                                    emis_grid, ewall, alb_grid, SBC, albedo_b, Twater,
                                                                    lc_grid, landcover)

        gvfLup = gvfLup + gvfLupi
        gvfalb = gvfalb + gvfalbi
        gvfalbnosh = gvfalbnosh + gvfalbnoshi
        gvfSum = gvfSum + gvf2

        if (azimuthA[j] >= 0) and (azimuthA[j] < 180):
            gvfLupE = gvfLupE + gvfLupi
            gvfalbE = gvfalbE + gvfalbi
            gvfalbnoshE = gvfalbnoshE + gvfalbnoshi

        if (azimuthA[j] >= 90) and (azimuthA[j] < 270):
            gvfLupS = gvfLupS + gvfLupi
            gvfalbS = gvfalbS + gvfalbi
            gvfalbnoshS = gvfalbnoshS + gvfalbnoshi

        if (azimuthA[j] >= 180) and (azimuthA[j] < 360):
            gvfLupW = gvfLupW + gvfLupi
            gvfalbW = gvfalbW + gvfalbi
            gvfalbnoshW = gvfalbnoshW + gvfalbnoshi

        if (azimuthA[j] >= 270) or (azimuthA[j] < 90):
            gvfLupN = gvfLupN + gvfLupi
            gvfalbN = gvfalbN + gvfalbi
            gvfalbnoshN = gvfalbnoshN + gvfalbnoshi

    gvfLup = gvfLup / azimuthA.__len__() + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfalb = gvfalb / azimuthA.__len__()
    gvfalbnosh = gvfalbnosh / azimuthA.__len__()

    gvfLupE = gvfLupE / (azimuthA.__len__() / 2) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfLupS = gvfLupS / (azimuthA.__len__() / 2) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfLupW = gvfLupW / (azimuthA.__len__() / 2) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfLupN = gvfLupN / (azimuthA.__len__() / 2) + SBC * emis_grid * (Ta + 273.15) ** 4

    gvfalbE = gvfalbE / (azimuthA.__len__() / 2)
    gvfalbS = gvfalbS / (azimuthA.__len__() / 2)
    gvfalbW = gvfalbW / (azimuthA.__len__() / 2)
    gvfalbN = gvfalbN / (azimuthA.__len__() / 2)

    gvfalbnoshE = gvfalbnoshE / (azimuthA.__len__() / 2)
    gvfalbnoshS = gvfalbnoshS / (azimuthA.__len__() / 2)
    gvfalbnoshW = gvfalbnoshW / (azimuthA.__len__() / 2)
    gvfalbnoshN = gvfalbnoshN / (azimuthA.__len__() / 2)

    gvfNorm = gvfSum / (azimuthA.__len__())
    gvfNorm[buildings == 0] = 1
    
    return gvfLup, gvfalb, gvfalbnosh, gvfLupE, gvfalbE, gvfalbnoshE, gvfLupS, gvfalbS, gvfalbnoshS, gvfLupW, gvfalbW, gvfalbnoshW, gvfLupN, gvfalbN, gvfalbnoshN, gvfSum, gvfNorm