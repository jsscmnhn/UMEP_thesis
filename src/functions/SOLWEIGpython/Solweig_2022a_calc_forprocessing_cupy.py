from __future__ import absolute_import
import numpy as np
import cupy as cp
from .daylen import daylen
from src.functions.SOLWEIGpython.UTIL.clearnessindex_2013b import clearnessindex_2013b
from src.functions.SOLWEIGpython.UTIL.diffusefraction import diffusefraction
from src.functions.SOLWEIGpython.UTIL.shadowingfunction_wallheight_13_cupy import shadowingfunction_wallheight_13,  shadowingfunction_wallheight_13_3d
from src.functions.SOLWEIGpython.UTIL.shadowingfunction_wallheight_23_cupy import shadowingfunction_wallheight_23,  shadowingfunction_23_3d
from .gvf_2018a_cupy import gvf_2018a_cupy as gvf_2018a
from .cylindric_wedge_cupy import cylindric_wedge_cupy as cylindric_wedge
from .TsWaveDelay_2015a import TsWaveDelay_2015a
from .Kup_veg_2015a import Kup_veg_2015a
from .Kside_veg_v2022a_cupy import Kside_veg_v2022a
from src.functions.SOLWEIGpython.UTIL.Perez_v3 import Perez_v3
from src.functions.SOLWEIGpython.UTIL.create_patches import create_patches

# Anisotropic longwave
from .Lcyl_v2022a_cupy import Lcyl_v2022a_cupy as Lcyl_v2022a
from .Lside_veg_v2022a_cupy import Lside_veg_v2022a_cupy as Lside_veg_v2022a
from copy import deepcopy

# temp
# from osgeo import gdal
# from src.util.misc import saveraster
# gdal_dtm = gdal.Open("D:/Geomatics/thesis/_amsterdamset/location_2/original/final_dsm.tif")# gdal.Open("D:/Geomatics/thesis/_amsterdamset/3dtest/dsm.tif")

def Solweig_2022a_calc(i, dsm, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg,
                       svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdem, vegdem2, albedo_b, absK, absL,
                       ewall, Fside, Fup, Fcyl, altitude, azimuth, zen, jday, usevegdem, onlyglobal, buildings, location, psi,
                       landcover, lc_grid, dectime, altmax, dirwalls, walls, cyl, elvis, Ta, RH, radG, radD, radI, P,
                       amaxvalue, amaxvalue_dsm, bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST,
                       TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec, Tgmap1, 
                       Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, CI, TgOut1, diffsh, shmat, vegshmat, vbshvegshmat, anisotropic_sky, asvf, patch_option):

    '''
    Core computation function for SOLWEIG (SOlar and LongWave Environmental Irradiance Geometry) in the 3D-enabled version.

    Originally developed by Fredrik Lindberg (2016-08-28), Gothenburg Urban Climate Group, University of Gothenburg.
    This version (2025, Jessica Monahan) includes support for 3D geometry and GPU acceleration using CuPy.

    Parameters:
        i (int):            Index of current timestep or iteration.
        dsm (cp.ndarray):   DSM containing building and ground elevations.
        scale (float):      Height-to-pixel size ratio (e.g., 2m pixel → scale = 0.5).
        rows (int):                                                   Number of rows in the raster grid.
        cols (int):                                                   Number of columns in the raster grid.
        svf (cp.ndarray):                                             Sky View Factor (total).
        svfN, svfW, svfE, svfS (cp.ndarray):                          Directional Sky View Factors (North, West, East, South).
        svfveg (cp.ndarray):                                          SVF blocked by vegetation (total).
        svfNveg, svfEveg, svfSveg, svfWveg (cp.ndarray):              Directional vegetation-blocked SVFs.
        svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg (cp.ndarray): Vegetation SVFs blocking buildings (all directions).
        vegdem (cp.ndarray):                                          Vegetation canopy height model (chm)
        vegdem2 (cp.ndarray):                                         Vegetation trunk zone DSM.
        albedo_b (float):                                             Wall albedo for buildings.
        absK, asbL (float):                                           Human shortwave & longwave absorption coefficient.
        ewall (float):                                                Emissivity of building walls.
        Fside, Fup, Fcyl (float):       Angular view factors for a person (sides, upwards, cylindrical model).
        altitude (float):               Solar altitude angle (degrees).
        azimuth (float):                Solar azimuth angle (degrees).
        zen (float):                    Solar zenith angle (radians).
        jday (int):                     Julian day of year.
        usevegdem (bool):               Flag to enable vegetation canopy effects.
        onlyglobal (bool):              Use global shortwave input to estimate direct and diffuse (Reindl et al. 1990).
        buildings (cp.ndarray):         Boolean grid identifying building locations.
        location (tuple):               Geographic location (latitude, longitude).
        psi (float):                    Vegetation transmissivity (1 - transmissivity of shortwave).
        landcover (bool):               Enable land cover classification scheme.
        lc_grid (cp.ndarray):           Land cover classification grid.
        dectime (float):                Decimal time of day.
        altmax (float):                 Maximum solar altitude.
        dirwalls (cp.ndarray):          Wall aspect directions.
        walls (cp.ndarray):             Wall height (one pixel thick row around building footprints).
        cyl (bool):                     If True, represent person as a cylinder (otherwise box).
        elvis (int):                    Dummy parameter for compatibility.
        Ta (float):                     Air temperature (°C).
        RH (float):                     Relative humidity (%).
        radG (float):                   Global solar radiation.
        radD (float):                   Diffuse solar radiation.
        radI (float):                   Direct solar radiation.
        P (float):                      Atmospheric pressure (hPa).
        amaxvalue (float):              Maximum building height in DSM.
        amaxvalue_dsm (float):          Max height for DSM layers.
        bush (cp.ndarray):              Grid indicating bush locations.
        Twater (float):                 Daily water surface temperature (°C).
        TgK (float):                    Ground surface temperature (Kelvin).
        Tstart (float):                 Initial ground temperature (Kelvin).
        alb_grid (cp.ndarray):          Ground albedo grid.
        emis_grid (cp.ndarray):         Ground emissivity grid.
        TgK_wall (float):               Initial wall surface temperature (Kelvin).
        Tstart_wall (float):            Starting temperature for wall surface.
        TmaxLST (float):                Max land surface temperature (ground).
        TmaxLST_wall (float):           Max land surface temperature (wall).
        first, second (float):          First height (sensor height) and second (sensor* 20) for Radiative surface influence
        svfalfa (cp.ndarray):           SVF recalculated to angular format.
        svfbuveg (cp.ndarray):          Combined SVF including both buildings and vegetation.
        firstdaytime (float):                                       Time of day for initialization (decimal hours).
        timeadd (float):                                            Time step to add (hours).
        timestepdec (float):                                        Time step (decimal hours).
        Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N (cp.ndarray):    Ground temperature history maps (center + directions).
        CI (float):                                                 Clearness index.
        TgOut1 (cp.ndarray):                                        Output ground temperature map from previous model run.
        diffsh (float):                                             Shadowing fraction from vegetation or terrain.
        shmat, vegshmat, vbshvegshmat (cp.ndarray):         Shadow matrices (buildings, vegetation, and vegetation not blocked by buildings).
        anisotropic_sky (bool):                             Enable anisotropic sky model (Wallenberg et al. 2019/2022).
        asvf (cp.ndarray):                                  Anisotropic Sky View Factor.
        patch_option (int):                                 Option for amount of patches.

    Returns:
         Updated radiation and thermal comfort model outputs.
    '''

    # # # Core program start # # #
    # Instrument offset in degrees
    t = 0.

    # Stefan Bolzmans Constant
    SBC = 5.67051e-8

    # Find sunrise decimal hour - new from 2014a
    _, _, _, SNUP = daylen(jday, location['latitude'])

    # Vapor pressure
    ea = 6.107 * 10 ** ((7.5 * Ta) / (237.3 + Ta)) * (RH / 100.)

    # Determination of clear - sky emissivity from Prata (1996)
    msteg = 46.5 * (ea / (Ta + 273.15))
    esky = (1 - (1 + msteg) * np.exp(-((1.2 + 3.0 * msteg) ** 0.5))) + elvis  # -0.04 old error from Jonsson et al.2006

    if altitude > 0: # # # # # # DAYTIME # # # # # #
        # Clearness Index on Earth's surface after Crawford and Dunchon (1999) with a correction
        #  factor for low sun elevations after Lindberg et al.(2008)
        I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)
        if (CI > 1) or (CI == np.inf):
            CI = 1

        # Estimation of radD and radI if not measured after Reindl et al.(1990)
        if onlyglobal == 1:
            I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)
            if (CI > 1) or (CI == np.inf):
                CI = 1

            radI, radD = diffusefraction(radG, altitude, Kt, Ta, RH)

        # Diffuse Radiation
        # Anisotropic Diffuse Radiation after Perez et al. 1993
        if anisotropic_sky == 1:
            patchchoice = 1
            zenDeg = zen*(180/np.pi)
            # Relative luminance
            lv, pc_, pb_ = Perez_v3(zenDeg, azimuth, radD, radI, jday, patchchoice, patch_option)   
            # Total relative luminance from sky, i.e. from each patch, into each cell
            aniLum = cp.zeros((rows, cols))
            for idx in range(lv.shape[0]):
                aniLum += diffsh[:,:,idx] * lv[idx,2]     

            dRad = aniLum * radD   # Total diffuse radiation from sky into each cell
        else:
            dRad = radD * svfbuveg
            patchchoice = 1
            # zenDeg = zen*(180/np.pi)
            lv = None
            # lv, pc_, pb_ = Perez_v3(zenDeg, azimuth, radD, radI, jday, patchchoice, patch_option)   # Relative luminance

        # Shadow  images
        if usevegdem == 1:
            vegsh, sh, wallsh, wallsun, wallshve, _, facesun = shadowingfunction_wallheight_23(dsm, vegdem, vegdem2,
                                        azimuth, altitude, scale, amaxvalue, bush, walls, dirwalls * np.pi / 180.)
            shadow = sh - (1 - vegsh) * (1 - psi)

        else:
            sh, wallsh, wallsun, facesh, facesun = shadowingfunction_wallheight_13(amaxvalue_dsm, dsm, azimuth, altitude, scale,
                                                                                   walls, dirwalls * np.pi / 180.)
            shadow = sh

        # # # Surface temperature parameterisation during daytime # # # #
        # new using max sun alt.instead of  dfm
        # Tgamp = (TgK * altmax - Tstart) + Tstart # Old
        Tgamp = TgK * altmax + Tstart # Fixed 2021
        # Tgampwall = (TgK_wall * altmax - (Tstart_wall)) + (Tstart_wall) # Old
        Tgampwall = TgK_wall * altmax + Tstart_wall
        Tg = Tgamp * np.sin((((dectime - np.floor(dectime)) - SNUP / 24) / (TmaxLST / 24 - SNUP / 24)) * np.pi / 2) # 2015 a, based on max sun altitude
        Tgwall = Tgampwall * np.sin((((dectime - np.floor(dectime)) - SNUP / 24) / (TmaxLST_wall / 24 - SNUP / 24)) * np.pi / 2) # 2015a, based on max sun altitude

        if Tgwall < 0:  # temporary for removing low Tg during morning 20130205
            # Tg = 0
            Tgwall = 0

        # New estimation of Tg reduction for non - clear situation based on Reindl et al.1990
        radI0, _ = diffusefraction(I0, altitude, 1., Ta, RH)
        corr = 0.1473 * np.log(90 - (zen / np.pi * 180)) + 0.3454  # 20070329 correction of lat, Lindberg et al. 2008
        CI_Tg = (radG / radI0) + (1 - corr)
        if (CI_Tg > 1) or (CI_Tg == np.inf):
            CI_Tg = 1

        deg2rad = np.pi/180
        radG0 = radI0 * (np.sin(altitude * deg2rad)) + _
        CI_TgG = (radG / radG0) + (1 - corr)
        if (CI_TgG > 1) or (CI_TgG == np.inf):
            CI_TgG = 1
        
        # Tg = Tg * CI_Tg  # new estimation
        # Tgwall = Tgwall * CI_Tg
        Tg = Tg * CI_TgG  # new estimation
        Tgwall = Tgwall * CI_TgG
        if landcover == 1:
            Tg[Tg < 0] = 0  # temporary for removing low Tg during morning 20130205

        # # # # Ground View Factors # # # #
        gvfLup, gvfalb, gvfalbnosh, gvfLupE, gvfalbE, gvfalbnoshE, gvfLupS, gvfalbS, gvfalbnoshS, gvfLupW, gvfalbW,\
        gvfalbnoshW, gvfLupN, gvfalbN, gvfalbnoshN, gvfSum, gvfNorm = gvf_2018a(wallsun, walls, buildings, scale, shadow, first,
                second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols,
                                                                 Twater, lc_grid, landcover)


        # # # # Lup, daytime # # # #
        # Surface temperature wave delay - new as from 2014a
        Lup, timeaddnotused, Tgmap1 = TsWaveDelay_2015a(gvfLup, firstdaytime, timeadd, timestepdec, Tgmap1)
        LupE, timeaddnotused, Tgmap1E = TsWaveDelay_2015a(gvfLupE, firstdaytime, timeadd, timestepdec, Tgmap1E)
        LupS, timeaddnotused, Tgmap1S = TsWaveDelay_2015a(gvfLupS, firstdaytime, timeadd, timestepdec, Tgmap1S)
        LupW, timeaddnotused, Tgmap1W = TsWaveDelay_2015a(gvfLupW, firstdaytime, timeadd, timestepdec, Tgmap1W)
        LupN, timeaddnotused, Tgmap1N = TsWaveDelay_2015a(gvfLupN, firstdaytime, timeadd, timestepdec, Tgmap1N)
        
        # # For Tg output in POIs
        TgTemp = Tg * shadow + Ta
        TgOut, timeadd, TgOut1 = TsWaveDelay_2015a(TgTemp, firstdaytime, timeadd, timestepdec, TgOut1) #timeadd only here v2021a


        # Building height angle from svf
        F_sh = cylindric_wedge(zen, svfalfa, rows, cols)  # Fraction shadow on building walls based on sun alt and svf
        F_sh[cp.isnan(F_sh)] = 0.5


        # # # # # # # Calculation of shortwave daytime radiative fluxes # # # # # # #
        Kdown = radI * shadow * np.sin(altitude * (np.pi / 180)) + dRad + albedo_b * (1 - svfbuveg) * \
                            (radG * (1 - F_sh) + radD * F_sh) # *sin(altitude(i) * (pi / 180))


        Kup, KupE, KupS, KupW, KupN = Kup_veg_2015a(radI, radD, radG, altitude, svfbuveg, albedo_b, F_sh, gvfalb,
                    gvfalbE, gvfalbS, gvfalbW, gvfalbN, gvfalbnosh, gvfalbnoshE, gvfalbnoshS, gvfalbnoshW, gvfalbnoshN)

        Keast, Ksouth, Kwest, Knorth, KsideI, KsideD, Kside = Kside_veg_v2022a(radI, radD, radG, shadow, svfS, svfW, svfN, svfE,
                    svfEveg, svfSveg, svfWveg, svfNveg, azimuth, altitude, psi, t, albedo_b, F_sh, KupE, KupS, KupW,
                    KupN, cyl, lv, anisotropic_sky, diffsh, rows, cols, asvf, shmat, vegshmat, vbshvegshmat)

        firstdaytime = 0

    else:  # # # # # # # NIGHTTIME # # # # # # # #

        Tgwall = 0
        # CI_Tg = -999  # F_sh = []

        # Nocturnal K fluxes set to 0
        Knight = cp.zeros((rows, cols))
        Kdown = cp.zeros((rows, cols))
        Kwest = cp.zeros((rows, cols))
        Kup = cp.zeros((rows, cols))
        Keast = cp.zeros((rows, cols))
        Ksouth = cp.zeros((rows, cols))
        Knorth = cp.zeros((rows, cols))
        KsideI = cp.zeros((rows, cols))
        KsideD = cp.zeros((rows, cols))
        F_sh = cp.zeros((rows, cols))
        Tg = cp.zeros((rows, cols))
        shadow = cp.zeros((rows, cols))
        CI_Tg = deepcopy(CI)
        CI_TgG = deepcopy(CI)

        dRad = cp.zeros((rows,cols))

        Kside = cp.zeros((rows,cols))

        # # # # Lup # # # #
        Lup = SBC * emis_grid * ((Knight + Ta + Tg + 273.15) ** 4)
        if landcover == 1:
            Lup[lc_grid == 3] = SBC * 0.98 * (Twater + 273.15) ** 4  # nocturnal Water temp

        LupE = Lup
        LupS = Lup
        LupW = Lup
        LupN = Lup

        # # For Tg output in POIs
        TgOut = Ta + Tg

        I0 = 0
        timeadd = 0
        firstdaytime = 1

    # # # # Ldown # # # #
    # Anisotropic sky longwave radiation
    if anisotropic_sky == 1:
        if 'lv' not in locals():
            # Creating skyvault of patches of constant radians (Tregeneza and Sharples, 1993)
            skyvaultalt, skyvaultazi, _, _, _, _, _ = create_patches(patch_option)

            patch_emissivities = np.zeros(skyvaultalt.shape[0])

            x = np.transpose(np.atleast_2d(skyvaultalt))
            y = np.transpose(np.atleast_2d(skyvaultazi))
            z = np.transpose(np.atleast_2d(patch_emissivities))

            L_patches = np.append(np.append(x, y, axis=1), z, axis=1)

        else:
            L_patches = deepcopy(lv)

        if altitude < 0:
            CI = deepcopy(CI)

        if CI < 0.95:
            esky_c = CI * esky + (1 - CI) * 1.
            esky = esky_c

        Ldown, Lside, Least_, Lwest_, Lnorth_, Lsouth_ \
                  = Lcyl_v2022a(esky, L_patches, Ta, Tgwall, ewall, Lup, shmat, vegshmat, vbshvegshmat, 
                                altitude, azimuth, rows, cols, asvf)

    else:
        Ldown = (svf + svfveg - 1) * esky * SBC * ((Ta + 273.15) ** 4) + (2 - svfveg - svfaveg) * ewall * SBC * \
                    ((Ta + 273.15) ** 4) + (svfaveg - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) + \
                    (2 - svf - svfveg) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4)  # Jonsson et al.(2006)
        # Ldown = Ldown - 25 # Shown by Jonsson et al.(2006) and Duarte et al.(2006)

        Lside = cp.zeros((rows, cols))
        L_patches = None
 
        if CI < 0.95:  # non - clear conditions
            c = 1 - CI
            Ldown = Ldown * (1 - c) + c * ((svf + svfveg - 1) * SBC * ((Ta + 273.15) ** 4) + (2 - svfveg - svfaveg) *
                    ewall * SBC * ((Ta + 273.15) ** 4) + (svfaveg - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) +
                    (2 - svf - svfveg) * (1 - ewall) * SBC * ((Ta + 273.15) ** 4))  # NOT REALLY TESTED!!! BUT MORE CORRECT?

    # # # # Lside # # # #
    Least, Lsouth, Lwest, Lnorth = Lside_veg_v2022a(svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
                    svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth, altitude, Ta, Tgwall, SBC, ewall, Ldown,
                                                      esky, t, F_sh, CI, LupE, LupS, LupW, LupN, anisotropic_sky)

    # Box and anisotropic longwave
    if cyl == 0 and anisotropic_sky == 1:
        Least += Least_
        Lwest += Lwest_
        Lnorth += Lnorth_
        Lsouth += Lsouth_

    # # # # Calculation of radiant flux density and Tmrt # # # #
    # Human body considered as a cylinder with isotropic all-sky diffuse
    if cyl == 1 and anisotropic_sky == 0: 
        Sstr = absK * (KsideI * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
                        ((Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside)
    # Human body considered as a cylinder with Perez et al. (1993) (anisotropic sky diffuse) 
    # and Martin and Berdahl (1984) (anisotropic sky longwave)
    elif cyl == 1 and anisotropic_sky == 1:
        Sstr = absK * (Kside * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
                        ((Ldown + Lup) * Fup + Lside * Fcyl + (Lnorth + Least + Lsouth + Lwest) * Fside)
    # Knorth = nan Ksouth = nan Kwest = nan Keast = nan
    else: # Human body considered as a standing cube
        Sstr = absK * ((Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
                        ((Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside)

    Tmrt = cp.sqrt(cp.sqrt((Sstr / (absL * SBC)))) - 273.2

    # Add longwave to cardinal directions for output in POI
    if (cyl == 1) and (anisotropic_sky == 1):
        Least += Least_
        Lwest += Lwest_
        Lnorth += Lnorth_
        Lsouth += Lsouth_

    return Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, \
           timeadd, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, \
           Lsouth, Lwest, Lnorth, KsideI, TgOut1, TgOut, radI, radD, \
               Lside, L_patches, CI_Tg, CI_TgG, KsideD, dRad, Kside


def Solweig_2022a_calc_3d(i, dsms, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg,
                       svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdem, vegdem2, albedo_b, absK, absL,
                       ewall, Fside, Fup, Fcyl, altitude, azimuth, zen, jday, usevegdem, onlyglobal, buildings,
                       location, psi,
                       landcover, lc_grid, dectime, altmax, dirwalls, walls, cyl, elvis, Ta, RH, radG, radD, radI, P,
                       amaxvalue, amaxvalue_dsm, bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall,
                       TmaxLST,
                       TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec, Tgmap1,
                       Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, CI, TgOut1, diffsh, shmat, vegshmat, vbshvegshmat,
                       anisotropic_sky, asvf, patch_option, version):
    '''
    Core computation function for SOLWEIG (SOlar and LongWave Environmental Irradiance Geometry) in the 3D-enabled version.

    Originally developed by Fredrik Lindberg (2016-08-28), Gothenburg Urban Climate Group, University of Gothenburg.
    This version (2025, Jessica Monahan) includes support for 3D geometry and GPU acceleration using CuPy.

    Parameters:
        dsms (cp.ndarray):  3D-layered DSM containing building and ground elevations, and gap layers.
        other_parameters:   see Solweig_2022a_calc for the others

    Returns:
         Updated radiation and thermal comfort model outputs.
    '''

      # Instrument offset in degrees
    t = 0.

    # Stefan Bolzmans Constant
    SBC = 5.67051e-8

    # Find sunrise decimal hour - new from 2014a
    _, _, _, SNUP = daylen(jday, location['latitude'])

    # Vapor pressure
    ea = 6.107 * 10 ** ((7.5 * Ta) / (237.3 + Ta)) * (RH / 100.)

    # Determination of clear - sky emissivity from Prata (1996)
    msteg = 46.5 * (ea / (Ta + 273.15))
    esky = (1 - (1 + msteg) * np.exp(-((1.2 + 3.0 * msteg) ** 0.5))) + elvis  # -0.04 old error from Jonsson et al.2006

    for i in range(1, dsms.shape[0]):
        dsms[i] = cp.where(dsms[i] <= -500, cp.nan, dsms[i])

    if altitude > 0:  # # # # # # DAYTIME # # # # # #
        # Clearness Index on Earth's surface after Crawford and Dunchon (1999) with a correction
        #  factor for low sun elevations after Lindberg et al.(2008)
        I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)
        if (CI > 1) or (CI == np.inf):
            CI = 1

        # Estimation of radD and radI if not measured after Reindl et al.(1990)
        if onlyglobal == 1:
            I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)
            if (CI > 1) or (CI == np.inf):
                CI = 1

            radI, radD = diffusefraction(radG, altitude, Kt, Ta, RH)

        # Diffuse Radiation
        # Anisotropic Diffuse Radiation after Perez et al. 1993
        if anisotropic_sky == 1:
            patchchoice = 1
            zenDeg = zen * (180 / np.pi)
            # Relative luminance
            lv, pc_, pb_ = Perez_v3(zenDeg, azimuth, radD, radI, jday, patchchoice, patch_option)
            # Total relative luminance from sky, i.e. from each patch, into each cell
            aniLum = cp.zeros((rows, cols))
            for idx in range(lv.shape[0]):
                aniLum += diffsh[:, :, idx] * lv[idx, 2]

            dRad = aniLum * radD  # Total diffuse radiation from sky into each cell
        else:
            dRad = radD * svfbuveg
            patchchoice = 1
            # zenDeg = zen*(180/np.pi)
            lv = None
            # lv, pc_, pb_ = Perez_v3(zenDeg, azimuth, radD, radI, jday, patchchoice, patch_option)   # Relative luminance

        # Shadow  images
        if usevegdem == 1:
            vegsh, sh, wallsh, wallsun, wallshve, _, facesun = shadowingfunction_23_3d(dsms, vegdem, vegdem2,
                                                                                       azimuth, altitude, scale,
                                                                                       amaxvalue, bush, walls,
                                                                                       dirwalls * np.pi / 180.)
            shadow = sh - (1 - vegsh) * (1 - psi)
        else:
            sh, wallsh, wallsun, facesh, facesun = shadowingfunction_wallheight_13_3d(amaxvalue_dsm, dsms, azimuth,
                                                                                      altitude, scale,
                                                                                      walls, dirwalls * np.pi / 180.)
            shadow = sh

        # # # Surface temperature parameterisation during daytime # # # #
        # new using max sun alt.instead of  dfm
        # Tgamp = (TgK * altmax - Tstart) + Tstart # Old
        Tgamp = TgK * altmax + Tstart  # Fixed 2021
        Tgampwall = TgK_wall * altmax + Tstart_wall
        Tg = Tgamp * np.sin((((dectime - np.floor(dectime)) - SNUP / 24) / (
                    TmaxLST / 24 - SNUP / 24)) * np.pi / 2)  # 2015 a, based on max sun altitude

        Tgwall = Tgampwall * np.sin((((dectime - np.floor(dectime)) - SNUP / 24) / (
                    TmaxLST_wall / 24 - SNUP / 24)) * np.pi / 2)  # 2015a, based on max sun altitude

        if Tgwall < 0:  # temporary for removing low Tg during morning 20130205
            # Tg = 0
            Tgwall = 0
        dectime_str = f"{dectime:.4f}"

        # New estimation of Tg reduction for non - clear situation based on Reindl et al.1990
        radI0, _ = diffusefraction(I0, altitude, 1., Ta, RH)
        corr = 0.1473 * np.log(90 - (zen / np.pi * 180)) + 0.3454  # 20070329 correction of lat, Lindberg et al. 2008
        CI_Tg = (radG / radI0) + (1 - corr)
        if (CI_Tg > 1) or (CI_Tg == np.inf):
            CI_Tg = 1

        deg2rad = np.pi / 180
        radG0 = radI0 * (np.sin(altitude * deg2rad)) + _
        CI_TgG = (radG / radG0) + (1 - corr)
        if (CI_TgG > 1) or (CI_TgG == np.inf):
            CI_TgG = 1

        # Tg = Tg * CI_Tg  # new estimation
        # Tgwall = Tgwall * CI_Tg
        Tg = Tg * CI_TgG  # new estimation
        Tgwall = Tgwall * CI_TgG
        if landcover == 1:
            Tg[Tg < 0] = 0  # temporary for removing low Tg during morning 20130205

        # # # # Ground View Factors # # # #
        gvfLup, gvfalb, gvfalbnosh, gvfLupE, gvfalbE, gvfalbnoshE, gvfLupS, gvfalbS, gvfalbnoshS, gvfLupW, gvfalbW, \
            gvfalbnoshW, gvfLupN, gvfalbN, gvfalbnoshN, gvfSum, gvfNorm = gvf_2018a(wallsun, walls, buildings, scale,
                                                                                    shadow, first,
                                                                                    second, dirwalls, Tg, Tgwall, Ta,
                                                                                    emis_grid, ewall, alb_grid, SBC,
                                                                                    albedo_b, rows, cols,
                                                                                    Twater, lc_grid, landcover)
        # # # # Lup, daytime # # # #
        # Surface temperature wave delay - new as from 2014a
        Lup, timeaddnotused, Tgmap1 = TsWaveDelay_2015a(gvfLup, firstdaytime, timeadd, timestepdec, Tgmap1)
        LupE, timeaddnotused, Tgmap1E = TsWaveDelay_2015a(gvfLupE, firstdaytime, timeadd, timestepdec, Tgmap1E)
        LupS, timeaddnotused, Tgmap1S = TsWaveDelay_2015a(gvfLupS, firstdaytime, timeadd, timestepdec, Tgmap1S)
        LupW, timeaddnotused, Tgmap1W = TsWaveDelay_2015a(gvfLupW, firstdaytime, timeadd, timestepdec, Tgmap1W)
        LupN, timeaddnotused, Tgmap1N = TsWaveDelay_2015a(gvfLupN, firstdaytime, timeadd, timestepdec, Tgmap1N)

        # # For Tg output in POIs
        TgTemp = Tg * shadow + Ta
        TgOut, timeadd, TgOut1 = TsWaveDelay_2015a(TgTemp, firstdaytime, timeadd, timestepdec,
                                                   TgOut1)  # timeadd only here v2021a

        # Building height angle from svf
        F_sh = cylindric_wedge(zen, svfalfa, rows, cols)  # Fraction shadow on building walls based on sun alt and svf
        F_sh[cp.isnan(F_sh)] = 0.5

        # # # # # # # Calculation of shortwave daytime radiative fluxes # # # # # # #
        Kdown = radI * shadow * np.sin(altitude * (np.pi / 180)) + dRad + albedo_b * (1 - svfbuveg) * \
                (radG * (1 - F_sh) + radD * F_sh)  # *sin(altitude(i) * (pi / 180))

        Kup, KupE, KupS, KupW, KupN = Kup_veg_2015a(radI, radD, radG, altitude, svfbuveg, albedo_b, F_sh, gvfalb,
                                                    gvfalbE, gvfalbS, gvfalbW, gvfalbN, gvfalbnosh, gvfalbnoshE,
                                                    gvfalbnoshS, gvfalbnoshW, gvfalbnoshN)

        Keast, Ksouth, Kwest, Knorth, KsideI, KsideD, Kside = Kside_veg_v2022a(radI, radD, radG, shadow, svfS, svfW,
                                                                               svfN, svfE,
                                                                               svfEveg, svfSveg, svfWveg, svfNveg,
                                                                               azimuth, altitude, psi, t, albedo_b,
                                                                               F_sh, KupE, KupS, KupW,
                                                                               KupN, cyl, lv, anisotropic_sky, diffsh,
                                                                               rows, cols, asvf, shmat, vegshmat,
                                                                               vbshvegshmat)

        firstdaytime = 0

    else:  # # # # # # # NIGHTTIME # # # # # # # #

        Tgwall = 0
        # CI_Tg = -999  # F_sh = []

        # Nocturnal K fluxes set to 0
        Knight = cp.zeros((rows, cols))
        Kdown = cp.zeros((rows, cols))
        Kwest = cp.zeros((rows, cols))
        Kup = cp.zeros((rows, cols))
        Keast = cp.zeros((rows, cols))
        Ksouth = cp.zeros((rows, cols))
        Knorth = cp.zeros((rows, cols))
        KsideI = cp.zeros((rows, cols))
        KsideD = cp.zeros((rows, cols))
        F_sh = cp.zeros((rows, cols))
        Tg = cp.zeros((rows, cols))
        shadow = cp.zeros((rows, cols))
        CI_Tg = deepcopy(CI)
        CI_TgG = deepcopy(CI)

        dRad = cp.zeros((rows, cols))

        Kside = cp.zeros((rows, cols))

        # # # # Lup # # # #
        Lup = SBC * emis_grid * ((Knight + Ta + Tg + 273.15) ** 4)
        if landcover == 1:
            Lup[lc_grid == 3] = SBC * 0.98 * (Twater + 273.15) ** 4  # nocturnal Water temp

        LupE = Lup
        LupS = Lup
        LupW = Lup
        LupN = Lup

        # # For Tg output in POIs
        TgOut = Ta + Tg

        I0 = 0
        timeadd = 0
        firstdaytime = 1

    # # # # Ldown # # # #
    # Anisotropic sky longwave radiation
    if anisotropic_sky == 1:
        if 'lv' not in locals():
            # Creating skyvault of patches of constant radians (Tregeneza and Sharples, 1993)
            skyvaultalt, skyvaultazi, _, _, _, _, _ = create_patches(patch_option)

            patch_emissivities = np.zeros(skyvaultalt.shape[0])

            x = np.transpose(np.atleast_2d(skyvaultalt))
            y = np.transpose(np.atleast_2d(skyvaultazi))
            z = np.transpose(np.atleast_2d(patch_emissivities))

            L_patches = np.append(np.append(x, y, axis=1), z, axis=1)

        else:
            L_patches = deepcopy(lv)

        if altitude < 0:
            CI = deepcopy(CI)

        if CI < 0.95:
            esky_c = CI * esky + (1 - CI) * 1.
            esky = esky_c

        Ldown, Lside, Least_, Lwest_, Lnorth_, Lsouth_ \
            = Lcyl_v2022a(esky, L_patches, Ta, Tgwall, ewall, Lup, shmat, vegshmat, vbshvegshmat,
                          altitude, azimuth, rows, cols, asvf)

    else:
        Ldown = (svf + svfveg - 1) * esky * SBC * ((Ta + 273.15) ** 4) + (2 - svfveg - svfaveg) * ewall * SBC * \
                ((Ta + 273.15) ** 4) + (svfaveg - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) + \
                (2 - svf - svfveg) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4)  # Jonsson et al.(2006)
        # Ldown = Ldown - 25 # Shown by Jonsson et al.(2006) and Duarte et al.(2006)

        Lside = cp.zeros((rows, cols))
        L_patches = None

        if CI < 0.95:  # non - clear conditions
            c = 1 - CI
            Ldown = Ldown * (1 - c) + c * ((svf + svfveg - 1) * SBC * ((Ta + 273.15) ** 4) + (2 - svfveg - svfaveg) *
                                           ewall * SBC * ((Ta + 273.15) ** 4) + (svfaveg - svf) * ewall * SBC * (
                                                       (Ta + 273.15 + Tgwall) ** 4) +
                                           (2 - svf - svfveg) * (1 - ewall) * SBC * (
                                                       (Ta + 273.15) ** 4))  # NOT REALLY TESTED!!! BUT MORE CORRECT?

    # # # # Lside # # # #
    Least, Lsouth, Lwest, Lnorth = Lside_veg_v2022a(svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
                                                    svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth, altitude, Ta,
                                                    Tgwall, SBC, ewall, Ldown,
                                                    esky, t, F_sh, CI, LupE, LupS, LupW, LupN, anisotropic_sky)

    # Box and anisotropic longwave
    if cyl == 0 and anisotropic_sky == 1:
        Least += Least_
        Lwest += Lwest_
        Lnorth += Lnorth_
        Lsouth += Lsouth_

    # # # # Calculation of radiant flux density and Tmrt # # # #
    # Human body considered as a cylinder with isotropic all-sky diffuse
    if cyl == 1 and anisotropic_sky == 0:
        Sstr = absK * (KsideI * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
               ((Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside)
    # Human body considered as a cylinder with Perez et al. (1993) (anisotropic sky diffuse)
    # and Martin and Berdahl (1984) (anisotropic sky longwave)
    elif cyl == 1 and anisotropic_sky == 1:
        Sstr = absK * (Kside * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
               ((Ldown + Lup) * Fup + Lside * Fcyl + (Lnorth + Least + Lsouth + Lwest) * Fside)
    # Knorth = nan Ksouth = nan Kwest = nan Keast = nan
    else:  # Human body considered as a standing cube
        Sstr = absK * ((Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
               ((Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside)

    Tmrt = cp.sqrt(cp.sqrt((Sstr / (absL * SBC)))) - 273.2

    # Add longwave to cardinal directions for output in POI
    if (cyl == 1) and (anisotropic_sky == 1):
        Least += Least_
        Lwest += Lwest_
        Lnorth += Lnorth_
        Lsouth += Lsouth_

    return Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, \
        timeadd, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, \
        Lsouth, Lwest, Lnorth, KsideI, TgOut1, TgOut, radI, radD, \
        Lside, L_patches, CI_Tg, CI_TgG, KsideD, dRad, Kside
