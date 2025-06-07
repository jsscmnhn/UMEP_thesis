import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.util import shadowingfunctions as shadow
from src.functions.SOLWEIGpython.UTIL.create_patches import create_patches
import cupy as cp

def annulus_weight(altitude, aziinterval):
    """Unchanged function. Computes a weight for annuli at a given altitude, using sine functions based on azimuth intervals."""
    n = 90.
    steprad = (360./aziinterval) * (np.pi/180.)
    annulus = 91.-altitude
    w = (1./(2.*np.pi)) * np.sin(np.pi / (2.*n)) * np.sin((np.pi * (2. * annulus - 1.)) / (2. * n))
    weight = steprad * w

    return weight

def svf_angles_100121():
    """Unchanged function. Generates azimuth intervals and angles for calculating SVF."""
    azi1 = np.arange(1., 360., 360./16.)  #%22.5
    azi2 = np.arange(12., 360., 360./16.)  #%22.5
    azi3 = np.arange(5., 360., 360./32.)  #%11.25
    azi4 = np.arange(2., 360., 360./32.)  #%11.25
    azi5 = np.arange(4., 360., 360./40.)  #%9
    azi6 = np.arange(7., 360., 360./48.)  #%7.50
    azi7 = np.arange(6., 360., 360./48.)  #%7.50
    azi8 = np.arange(1., 360., 360./48.)  #%7.50
    azi9 = np.arange(4., 359., 360./52.)  #%6.9231
    azi10 = np.arange(5., 360., 360./52.)  #%6.9231
    azi11 = np.arange(1., 360., 360./48.)  #%7.50
    azi12 = np.arange(0., 359., 360./44.)  #%8.1818
    azi13 = np.arange(3., 360., 360./44.)  #%8.1818
    azi14 = np.arange(2., 360., 360./40.)  #%9
    azi15 = np.arange(7., 360., 360./32.)  #%10
    azi16 = np.arange(3., 360., 360./24.)  #%11.25
    azi17 = np.arange(10., 360., 360./16.)  #%15
    azi18 = np.arange(19., 360., 360./12.)  #%22.5
    azi19 = np.arange(17., 360., 360./8.)  #%45
    azi20 = 0.  #%360
    iazimuth = np.array(np.hstack((azi1, azi2, azi3, azi4, azi5, azi6, azi7, azi8, azi9, azi10, azi11, azi12, azi13,
                                    azi14, azi15, azi16, azi17, azi18, azi19, azi20)))
    aziinterval = np.array(np.hstack((16., 16., 32., 32., 40., 48., 48., 48., 52., 52., 48., 44., 44., 40., 32., 24.,
                                        16., 12., 8., 1.)))
    angleresult = {'iazimuth': iazimuth, 'aziinterval': aziinterval}

    return angleresult


def svfForProcessing153(dsm, dtm, vegdem, vegdem2, scale, usevegdem):
    """
    Calculates sky view factor (SVF) and directional SVFs using 153 sky patch divisions, an updated function.
    This function is used for the Anisotropic sky model.

    Parameters:
        dsm (cp.ndarray):      Digital Surface Model containing building + ground elevations.
        dtm (cp.ndarray):      Digital terrain model, containing only ground elevations. If None, DSM is used as base elevation.
        vegdem (cp.ndarray):   Vegetation height layer (CHM)
        vegdem2 (cp.ndarray):  Secondary vegetation layer, containing trunk heights.
        scale (float):         Pixel resolution in meters.
        usevegdem (int):       If 1, include vegetation in SVF calculation; if 0, ignore it.

    Returns
    -------
    dict :
        Contains SVF arrays for total, directional (E/S/W/N), vegetation, and shadow matrices.
    """
    dsm = cp.array(dsm, dtype=cp.float32)
    vegdem = cp.array(vegdem, dtype=cp.float32)
    vegdem2 = cp.array(vegdem2, dtype=cp.float32)

    rows = dsm.shape[0]
    cols = dsm.shape[1]
    svf = cp.zeros([rows, cols])
    svfE = cp.zeros([rows, cols])
    svfS = cp.zeros([rows, cols])
    svfW = cp.zeros([rows, cols])
    svfN = cp.zeros([rows, cols])
    svfveg = cp.zeros((rows, cols))
    svfEveg = cp.zeros((rows, cols))
    svfSveg = cp.zeros((rows, cols))
    svfWveg = cp.zeros((rows, cols))
    svfNveg = cp.zeros((rows, cols))
    svfaveg = cp.zeros((rows, cols))
    svfEaveg = cp.zeros((rows, cols))
    svfSaveg = cp.zeros((rows, cols))
    svfWaveg = cp.zeros((rows, cols))
    svfNaveg = cp.zeros((rows, cols))

    # % amaxvalue
    vegmax = vegdem.max()
    amaxvalue = dsm.max()
    amaxvalue = cp.nanmax(cp.array([amaxvalue, vegmax]))
    amaxvalueinput = amaxvalue.get()

    if dtm is not None:
        vegdem = vegdem + dtm
        vegdem[vegdem == dtm] = 0
        vegdem2 = vegdem2 + dtm
        vegdem2[vegdem2 == dtm] = 0

    else:
        # % Elevation vegdems if no DTM
        vegdem = vegdem + dsm
        vegdem[vegdem == dsm] = 0
        vegdem2 = vegdem2 + dsm
        vegdem2[vegdem2 == dsm] = 0
    # % Bush separation
    bush = cp.logical_not((vegdem2 * vegdem)) * vegdem

    index = int(0)

    # patch_option = 1 # 145 patches
    patch_option = 2 # 153 patches
    # patch_option = 3 # 306 patches
    # patch_option = 4 # 612 patches
    
    # Create patches based on patch_option
    skyvaultalt, skyvaultazi, annulino, skyvaultaltint, aziinterval, skyvaultaziint, azistart = create_patches(patch_option)

    skyvaultaziint = np.array([360/patches for patches in aziinterval])
    iazimuth = np.hstack(np.zeros((1, np.sum(aziinterval)))) # Nils

    shmat = cp.zeros((rows, cols, np.sum(aziinterval)))
    vegshmat = cp.zeros((rows, cols, np.sum(aziinterval)))
    vbshvegshmat = cp.zeros((rows, cols, np.sum(aziinterval)))

    for j in range(0, skyvaultaltint.shape[0]):
        for k in range(0, int(360 / skyvaultaziint[j])):
            iazimuth[index] = k * skyvaultaziint[j] + azistart[j]
            if iazimuth[index] > 360.:
                iazimuth[index] = iazimuth[index] - 360.
            index = index + 1
    aziintervalaniso = np.ceil(aziinterval / 2.0)
    index = int(0)

    i = 1
    for i in range(0, skyvaultaltint.shape[0]):
        for j in np.arange(0, (aziinterval[int(i)])):
              # Casting shadow
            altitude = skyvaultaltint[int(i)]
            azimuth = iazimuth[int(index)]

            if usevegdem == 1:
                if altitude == 90:
                    vegsh = cp.where(cp.logical_and(~cp.isnan(vegdem), ~cp.isnan(vegdem2)), 0.0, 1.0)
                    vbshvegsh = cp.full((rows, cols), 1.0, dtype=cp.float32)
                    sh = cp.full((rows, cols), 1.0, dtype=cp.float32)
                else:
                    shadowresult = shadow.shadowingfunction_20_cupy(dsm, vegdem, vegdem2, azimuth, altitude,
                                                               scale, amaxvalueinput,bush)
                    vegsh = shadowresult["vegsh"]
                    vbshvegsh = shadowresult["vbshvegsh"]
                    sh = shadowresult["sh"]
                vegshmat[:, :, index] = vegsh
                vbshvegshmat[:, :, index] = vbshvegsh
            else:
                if altitude == 90:
                    sh = cp.full((rows, cols), 1.0, dtype=cp.float32)
                else:
                    sh = shadow.shadowingfunctionglobalradiation_cupy(dsm, amaxvalueinput, azimuth, altitude, scale)

            shmat[:, :, index] = sh

            # Calculate svfs
            for k in np.arange(annulino[int(i)]+1, (annulino[int(i+1.)])+1):
                weight = annulus_weight(k, aziinterval[i])*sh
                svf = svf + weight
                weight = annulus_weight(k, aziintervalaniso[i]) * sh
                if (azimuth >= 0) and (azimuth < 180):
                    svfE = svfE + weight
                if (azimuth >= 90) and (azimuth < 270):
                    svfS = svfS + weight
                if (azimuth >= 180) and (azimuth < 360):
                    svfW = svfW + weight
                if (azimuth >= 270) or (azimuth < 90):
                    svfN = svfN + weight

            if usevegdem == 1:
                for k in np.arange(annulino[int(i)] + 1, (annulino[int(i + 1.)]) + 1):
                    # % changed to include 90
                    weight = annulus_weight(k, aziinterval[i])
                    svfveg = svfveg + weight * vegsh
                    svfaveg = svfaveg + weight * vbshvegsh
                    weight = annulus_weight(k, aziintervalaniso[i])
                    if (azimuth >= 0) and (azimuth < 180):
                        svfEveg = svfEveg + weight * vegsh
                        svfEaveg = svfEaveg + weight * vbshvegsh
                    if (azimuth >= 90) and (azimuth < 270):
                        svfSveg = svfSveg + weight * vegsh
                        svfSaveg = svfSaveg + weight * vbshvegsh
                    if (azimuth >= 180) and (azimuth < 360):
                        svfWveg = svfWveg + weight * vegsh
                        svfWaveg = svfWaveg + weight * vbshvegsh
                    if (azimuth >= 270) or (azimuth < 90):
                        svfNveg = svfNveg + weight * vegsh
                        svfNaveg = svfNaveg + weight * vbshvegsh

            index += 1
            print(int(index * (100. / np.sum(aziinterval))))

    svfS = svfS + 3.0459e-004
    svfW = svfW + 3.0459e-004
    # % Last azimuth is 90. Hence, manual add of last annuli for svfS and SVFW
    # %Forcing svf not be greater than 1 (some MATLAB crazyness)
    svf[(svf > 1.)] = 1.
    svfE[(svfE > 1.)] = 1.
    svfS[(svfS > 1.)] = 1.
    svfW[(svfW > 1.)] = 1.
    svfN[(svfN > 1.)] = 1.

    if usevegdem == 1:
        last = cp.zeros((rows, cols))
        last[(vegdem2 == 0.)] = 3.0459e-004
        svfSveg = svfSveg + last
        svfWveg = svfWveg + last
        svfSaveg = svfSaveg + last
        svfWaveg = svfWaveg + last
        # %Forcing svf not be greater than 1 (some MATLAB crazyness)
        svfveg[(svfveg > 1.)] = 1.
        svfEveg[(svfEveg > 1.)] = 1.
        svfSveg[(svfSveg > 1.)] = 1.
        svfWveg[(svfWveg > 1.)] = 1.
        svfNveg[(svfNveg > 1.)] = 1.
        svfaveg[(svfaveg > 1.)] = 1.
        svfEaveg[(svfEaveg > 1.)] = 1.
        svfSaveg[(svfSaveg > 1.)] = 1.
        svfWaveg[(svfWaveg > 1.)] = 1.
        svfNaveg[(svfNaveg > 1.)] = 1.

    svfresult = {'svf': svf, 'svfE': svfE, 'svfS': svfS, 'svfW': svfW, 'svfN': svfN,
                    'svfveg': svfveg, 'svfEveg': svfEveg, 'svfSveg': svfSveg, 'svfWveg': svfWveg,
                    'svfNveg': svfNveg, 'svfaveg': svfaveg, 'svfEaveg': svfEaveg, 'svfSaveg': svfSaveg,
                    'svfWaveg': svfWaveg, 'svfNaveg': svfNaveg, 'shmat': shmat, 'vegshmat': vegshmat, 'vbshvegshmat': vbshvegshmat}
    return svfresult


def svfForProcessing153_3d(dsms, dtm, vegdem, vegdem2, scale, usevegdem):
    """
    Calculates sky view factor (SVF) and directional SVFs using 153 sky patch divisions for 3D input.
    This function is used for the Anisotropic sky model.

    Parameters:
        dsms (cp.ndarray):     3D-layered Digital Surface Model containing building + ground elevations, and gap layers.
        dtm (cp.ndarray):      Digital terrain model, containing only ground elevations. If None, DSM is used as base elevation.
        vegdem (cp.ndarray):   Vegetation height layer (CHM)
        vegdem2 (cp.ndarray):  Secondary vegetation layer, containing trunk heights.
        scale (float):         Pixel resolution in meters.
        usevegdem (int):       If 1, include vegetation in SVF calculation; if 0, ignore it.

    Returns
    -------
    dict :
        Contains SVF arrays for total, directional (E/S/W/N), vegetation, and shadow matrices, all at base layer.
    """
    rows = dsms[0].shape[0]
    cols = dsms[0].shape[1]
    svf = cp.zeros([rows, cols])
    svfE = cp.zeros([rows, cols])
    svfS = cp.zeros([rows, cols])
    svfW = cp.zeros([rows, cols])
    svfN = cp.zeros([rows, cols])
    svfveg = cp.zeros((rows, cols))
    svfEveg = cp.zeros((rows, cols))
    svfSveg = cp.zeros((rows, cols))
    svfWveg = cp.zeros((rows, cols))
    svfNveg = cp.zeros((rows, cols))
    svfaveg = cp.zeros((rows, cols))
    svfEaveg = cp.zeros((rows, cols))
    svfSaveg = cp.zeros((rows, cols))
    svfWaveg = cp.zeros((rows, cols))
    svfNaveg = cp.zeros((rows, cols))

    # % amaxvalue
    vegmax = vegdem.max()
    amaxvalue = dsms[0].max()
    amaxvalue = cp.nanmax(cp.array([amaxvalue, vegmax]))

    if dtm is not None:
        vegdem = vegdem + dtm
        vegdem[vegdem == dtm] = 0
        vegdem2 = vegdem2 + dtm
        vegdem2[vegdem2 == dtm] = 0

    else:
        # % Elevation vegdems if no DTM
        vegdem = vegdem + dsms[0]
        vegdem[vegdem == dsms[0]] = 0
        vegdem2 = vegdem2 + dsms[0]
        vegdem2[vegdem2 == dsms[0]] = 0
    # % Bush separation
    bush = cp.logical_not((vegdem2 * vegdem)) * vegdem
    index = int(0)

    # patch_option = 1 # 145 patches
    patch_option = 2  # 153 patches
    # patch_option = 3 # 306 patches
    # patch_option = 4 # 612 patches

    # Create patches based on patch_option
    skyvaultalt, skyvaultazi, annulino, skyvaultaltint, aziinterval, skyvaultaziint, azistart = create_patches(
        patch_option)

    skyvaultaziint = np.array([360 / patches for patches in aziinterval])
    iazimuth = np.hstack(np.zeros((1, np.sum(aziinterval))))  # Nils

    shmat = cp.zeros((rows, cols, np.sum(aziinterval)))
    vegshmat = cp.zeros((rows, cols, np.sum(aziinterval)))
    vbshvegshmat = cp.zeros((rows, cols, np.sum(aziinterval)))

    for j in range(0, skyvaultaltint.shape[0]):
        for k in range(0, int(360 / skyvaultaziint[j])):
            iazimuth[index] = k * skyvaultaziint[j] + azistart[j]
            if iazimuth[index] > 360.:
                iazimuth[index] = iazimuth[index] - 360.
            index = index + 1
    aziintervalaniso = np.ceil(aziinterval / 2.0)
    index = int(0)

    # should always have  no data values (-9999) or lower for the not filled layers)
    for i in range(1, dsms.shape[0]):
        dsms[i] = cp.where(dsms[i] <= 0, np.nan, dsms[i])


    for i in range(0, skyvaultaltint.shape[0]):
        for j in np.arange(0, (aziinterval[int(i)])):
            altitude = skyvaultaltint[int(i)]
            azimuth = iazimuth[int(index)]
            # Casting shadow
            if usevegdem == 1:
                if altitude == 90:
                    shadowresult = shadow.shadowingfunction_20_3d_90(dsms, vegdem, vegdem2)

                else:
                    shadowresult = shadow.shadowingfunction_20_3d(dsms, vegdem, vegdem2, azimuth, altitude,
                                                            scale, amaxvalue, bush)

                vegsh = shadowresult["vegsh"]
                vbshvegsh = shadowresult["vbshvegsh"]
                sh = shadowresult["sh"]
                vegshmat[:, :, index] = vegsh
                vbshvegshmat[:, :, index] = vbshvegsh
            else:
                if altitude == 90:
                    sh = shadow.shadowingfunctionglobalradiation_3d_90(dsms)
                else:
                    sh = shadow.shadowingfunctionglobalradiation_3d(dsms, amaxvalue, azimuth, altitude, scale)

            shmat[:, :, index] = sh

            # Calculate svfs
            for k in np.arange(annulino[int(i)] + 1, (annulino[int(i + 1.)]) + 1):
                weight = annulus_weight(k, aziinterval[i]) * sh
                svf = svf + weight
                weight = annulus_weight(k, aziintervalaniso[i]) * sh
                if (azimuth >= 0) and (azimuth < 180):
                    svfE = svfE + weight
                if (azimuth >= 90) and (azimuth < 270):
                    svfS = svfS + weight
                if (azimuth >= 180) and (azimuth < 360):
                    svfW = svfW + weight
                if (azimuth >= 270) or (azimuth < 90):
                    svfN = svfN + weight

            if usevegdem == 1:
                for k in np.arange(annulino[int(i)] + 1, (annulino[int(i + 1.)]) + 1):
                    # % changed to include 90
                    weight = annulus_weight(k, aziinterval[i])
                    svfveg = svfveg + weight * vegsh
                    svfaveg = svfaveg + weight * vbshvegsh
                    weight = annulus_weight(k, aziintervalaniso[i])
                    if (azimuth >= 0) and (azimuth < 180):
                        svfEveg = svfEveg + weight * vegsh
                        svfEaveg = svfEaveg + weight * vbshvegsh
                    if (azimuth >= 90) and (azimuth < 270):
                        svfSveg = svfSveg + weight * vegsh
                        svfSaveg = svfSaveg + weight * vbshvegsh
                    if (azimuth >= 180) and (azimuth < 360):
                        svfWveg = svfWveg + weight * vegsh
                        svfWaveg = svfWaveg + weight * vbshvegsh
                    if (azimuth >= 270) or (azimuth < 90):
                        svfNveg = svfNveg + weight * vegsh
                        svfNaveg = svfNaveg + weight * vbshvegsh

            index += 1
            print(int(index * (100. / np.sum(aziinterval))))

    svfS = svfS + 3.0459e-004
    svfW = svfW + 3.0459e-004
    # % Last azimuth is 90. Hence, manual add of last annuli for svfS and SVFW
    # %Forcing svf not be greater than 1 (some MATLAB crazyness)
    svf[(svf > 1.)] = 1.
    svfE[(svfE > 1.)] = 1.
    svfS[(svfS > 1.)] = 1.
    svfW[(svfW > 1.)] = 1.
    svfN[(svfN > 1.)] = 1.

    if usevegdem == 1:
        last = cp.zeros((rows, cols))
        last[(vegdem2 == 0.)] = 3.0459e-004
        svfSveg = svfSveg + last
        svfWveg = svfWveg + last
        svfSaveg = svfSaveg + last
        svfWaveg = svfWaveg + last
        # %Forcing svf not be greater than 1 (some MATLAB crazyness)
        svfveg[(svfveg > 1.)] = 1.
        svfEveg[(svfEveg > 1.)] = 1.
        svfSveg[(svfSveg > 1.)] = 1.
        svfWveg[(svfWveg > 1.)] = 1.
        svfNveg[(svfNveg > 1.)] = 1.
        svfaveg[(svfaveg > 1.)] = 1.
        svfEaveg[(svfEaveg > 1.)] = 1.
        svfSaveg[(svfSaveg > 1.)] = 1.
        svfWaveg[(svfWaveg > 1.)] = 1.
        svfNaveg[(svfNaveg > 1.)] = 1.

    svfresult = {'svf': svf, 'svfE': svfE, 'svfS': svfS, 'svfW': svfW, 'svfN': svfN,
                 'svfveg': svfveg, 'svfEveg': svfEveg, 'svfSveg': svfSveg, 'svfWveg': svfWveg,
                 'svfNveg': svfNveg, 'svfaveg': svfaveg, 'svfEaveg': svfEaveg, 'svfSaveg': svfSaveg,
                 'svfWaveg': svfWaveg, 'svfNaveg': svfNaveg, 'shmat': shmat, 'vegshmat': vegshmat,
                 'vbshvegshmat': vbshvegshmat}
    # ,
    # 'vbshvegshmat': vbshvegshmat, 'wallshmat': wallshmat, 'wallsunmat': wallsunmat,
    # 'wallshvemat': wallshvemat, 'facesunmat': facesunmat}
    return svfresult


def svfForProcessing655(dsm, dtm, vegdem, vegdem2, scale, usevegdem):
    """
    Calculates sky view factor (SVF) and directional SVFs using 655 sky patch divisions, updated function.
    This function is used for the isotropic sky model.

    Parameters:
        dsm (cp.ndarray):      Digital Surface Model containing building + ground elevations.
        dtm (cp.ndarray):      Digital terrain model, containing only ground elevations. If None, DSM is used as base elevation.
        vegdem (cp.ndarray):   Vegetation height layer (CHM)
        vegdem2 (cp.ndarray):  Secondary vegetation layer, containing trunk heights.
        scale (float):         Pixel resolution in meters.
        usevegdem (int):       If 1, include vegetation in SVF calculation; if 0, ignore it.

    Returns
    -------
    dict :
        Contains SVF arrays for total, directional (E/S/W/N), vegetation, and shadow matrices, all at base layer.
    """
    dsm = cp.array(dsm, dtype=cp.float32)
    vegdem = cp.array(vegdem, dtype=cp.float32)
    vegdem2 = cp.array(vegdem2, dtype=cp.float32)

    rows = dsm.shape[0]
    cols = dsm.shape[1]

    svf = cp.zeros([rows, cols])
    svfE = cp.zeros([rows, cols])
    svfS = cp.zeros([rows, cols])
    svfW = cp.zeros([rows, cols])
    svfN = cp.zeros([rows, cols])
    svfveg = cp.zeros((rows, cols))
    svfEveg = cp.zeros((rows, cols))
    svfSveg = cp.zeros((rows, cols))
    svfWveg = cp.zeros((rows, cols))
    svfNveg = cp.zeros((rows, cols))
    svfaveg = cp.zeros((rows, cols))
    svfEaveg = cp.zeros((rows, cols))
    svfSaveg = cp.zeros((rows, cols))
    svfWaveg = cp.zeros((rows, cols))
    svfNaveg = cp.zeros((rows, cols))

    # % amaxvalue
    vegmax = vegdem.max()
    amaxvalue = dsm.max()
    amaxvalue = cp.nanmax(cp.array([amaxvalue, vegmax]))
    amaxvalueinput = amaxvalue.get()


    if dtm is not None:
        vegdem = vegdem + dtm
        vegdem[vegdem == dtm] = 0
        vegdem2 = vegdem2 + dtm
        vegdem2[vegdem2 == dtm] = 0

    else:
        # % Elevation vegdems if no DTM
        vegdem = vegdem + dsm
        vegdem[vegdem == dsm] = 0
        vegdem2 = vegdem2 + dsm
        vegdem2[vegdem2 == dsm] = 0
    # % Bush separation
    bush = np.logical_not((vegdem2 * vegdem)) * vegdem

    # shmat = np.zeros((rows, cols, 145))
    # vegshmat = np.zeros((rows, cols, 145))

    noa = 19.
    #% No. of anglesteps minus 1
    step = 89./noa
    iangle = np.array(np.hstack((np.arange(step/2., 89., step), 90.)))
    annulino = np.array(np.hstack((np.round(np.arange(0., 89., step)), 90.)))
    angleresult = svf_angles_100121()
    aziinterval = angleresult["aziinterval"]
    iazimuth = angleresult["iazimuth"]
    aziintervalaniso = np.ceil((aziinterval/2.))
    index = 1.

    for i in np.arange(0, iangle.shape[0]-1):
        for j in np.arange(0, (aziinterval[int(i)])):
            altitude = iangle[int(i)]
            azimuth = iazimuth[int(index)-1]

            # Casting shadow
            if usevegdem == 1:
                if altitude == 90:
                    vegsh = cp.where(cp.logical_and(~cp.isnan(vegdem), ~cp.isnan(vegdem2)), 0.0, 1.0)
                    vbshvegsh = cp.full((rows, cols), 1.0, dtype=cp.float32)
                    sh = cp.full((rows, cols), 1.0, dtype=cp.float32)
                else:
                    shadowresult = shadow.shadowingfunction_20_cupy(dsm, vegdem, vegdem2, azimuth, altitude,
                                                                    scale, amaxvalueinput, bush)
                    vegsh = shadowresult["vegsh"]
                    vbshvegsh = shadowresult["vbshvegsh"]
                    sh = shadowresult["sh"]
            else:
                if altitude == 90:
                    sh = cp.full((rows, cols), 1.0, dtype=cp.float32)
                else:
                    sh = shadow.shadowingfunctionglobalradiation_cupy(dsm, amaxvalueinput, azimuth, altitude, scale)


            # Calculate svfs
            for k in np.arange(annulino[int(i)]+1, (annulino[int(i+1.)])+1):
                weight = annulus_weight(k, aziinterval[i])*sh
                svf = svf + weight
                weight = annulus_weight(k, aziintervalaniso[i]) * sh
                if (azimuth >= 0) and (azimuth < 180):
                    svfE = svfE + weight
                if (azimuth >= 90) and (azimuth < 270):
                    svfS = svfS + weight
                if (azimuth >= 180) and (azimuth < 360):
                    svfW = svfW + weight
                if (azimuth >= 270) or (azimuth < 90):
                    svfN = svfN + weight

            if usevegdem == 1:
                for k in np.arange(annulino[int(i)] + 1, (annulino[int(i + 1.)]) + 1):
                    # % changed to include 90
                    weight = annulus_weight(k, aziinterval[i])
                    svfveg = svfveg + weight * vegsh
                    svfaveg = svfaveg + weight * vbshvegsh
                    weight = annulus_weight(k, aziintervalaniso[i])
                    if (azimuth >= 0) and (azimuth < 180):
                        svfEveg = svfEveg + weight * vegsh
                        svfEaveg = svfEaveg + weight * vbshvegsh
                    if (azimuth >= 90) and (azimuth < 270):
                        svfSveg = svfSveg + weight * vegsh
                        svfSaveg = svfSaveg + weight * vbshvegsh
                    if (azimuth >= 180) and (azimuth < 360):
                        svfWveg = svfWveg + weight * vegsh
                        svfWaveg = svfWaveg + weight * vbshvegsh
                    if (azimuth >= 270) or (azimuth < 90):
                        svfNveg = svfNveg + weight * vegsh
                        svfNaveg = svfNaveg + weight * vbshvegsh

            index += 1
            print(int(index * (100. / 655.)))

    svfS = svfS + 3.0459e-004
    svfW = svfW + 3.0459e-004
    # % Last azimuth is 90. Hence, manual add of last annuli for svfS and SVFW
    # %Forcing svf not be greater than 1 (some MATLAB crazyness)
    svf[(svf > 1.)] = 1.
    svfE[(svfE > 1.)] = 1.
    svfS[(svfS > 1.)] = 1.
    svfW[(svfW > 1.)] = 1.
    svfN[(svfN > 1.)] = 1.

    if usevegdem == 1:
        last = cp.zeros((rows, cols))
        last[(vegdem2 == 0.)] = 3.0459e-004
        svfSveg = svfSveg + last
        svfWveg = svfWveg + last
        svfSaveg = svfSaveg + last
        svfWaveg = svfWaveg + last
        # %Forcing svf not be greater than 1 (some MATLAB crazyness)
        svfveg[(svfveg > 1.)] = 1.
        svfEveg[(svfEveg > 1.)] = 1.
        svfSveg[(svfSveg > 1.)] = 1.
        svfWveg[(svfWveg > 1.)] = 1.
        svfNveg[(svfNveg > 1.)] = 1.
        svfaveg[(svfaveg > 1.)] = 1.
        svfEaveg[(svfEaveg > 1.)] = 1.
        svfSaveg[(svfSaveg > 1.)] = 1.
        svfWaveg[(svfWaveg > 1.)] = 1.
        svfNaveg[(svfNaveg > 1.)] = 1.

    svfresult = {'svf': svf, 'svfE': svfE, 'svfS': svfS, 'svfW': svfW, 'svfN': svfN,
                    'svfveg': svfveg, 'svfEveg': svfEveg, 'svfSveg': svfSveg, 'svfWveg': svfWveg,
                    'svfNveg': svfNveg, 'svfaveg': svfaveg, 'svfEaveg': svfEaveg, 'svfSaveg': svfSaveg,
                    'svfWaveg': svfWaveg, 'svfNaveg': svfNaveg}

    return svfresult


def svfForProcessing655_3d(dsms, dtm, vegdem, vegdem2, scale, usevegdem):
    """
    Calculates sky view factor (SVF) and directional SVFs using 655 sky patch divisions for 3D input.
    This function is used for the isotropic sky model.

    Parameters:
        dsms (cp.ndarray):     3D-layered Digital Surface Model containing building + ground elevations, and gap layers.
        dtm (cp.ndarray):      Digital terrain model, containing only ground elevations. If None, DSM is used as base elevation.
        vegdem (cp.ndarray):   Vegetation height layer (CHM)
        vegdem2 (cp.ndarray):  Secondary vegetation layer, containing trunk heights.
        scale (float):         Pixel resolution in meters.
        usevegdem (int):       If 1, include vegetation in SVF calculation; if 0, ignore it.

    Returns
    -------
    dict :
        Contains SVF arrays for total, directional (E/S/W/N), vegetation, and shadow matrices, all at base layer.
    """
    rows = dsms[0].shape[0]
    cols = dsms[0].shape[1]

    svf = cp.zeros([rows, cols])
    svfE = cp.zeros([rows, cols])
    svfS = cp.zeros([rows, cols])
    svfW = cp.zeros([rows, cols])
    svfN = cp.zeros([rows, cols])
    svfveg = cp.zeros((rows, cols))
    svfEveg = cp.zeros((rows, cols))
    svfSveg = cp.zeros((rows, cols))
    svfWveg = cp.zeros((rows, cols))
    svfNveg = cp.zeros((rows, cols))
    svfaveg = cp.zeros((rows, cols))
    svfEaveg = cp.zeros((rows, cols))
    svfSaveg = cp.zeros((rows, cols))
    svfWaveg = cp.zeros((rows, cols))
    svfNaveg = cp.zeros((rows, cols))

    # % amaxvalue
    vegmax = vegdem.max()
    amaxvalue = dsms[0].max()
    amaxvalue = cp.nanmax(cp.array([amaxvalue, vegmax]))

    if dtm is not None:
        vegdem = vegdem + dtm
        vegdem[vegdem == dtm] = 0
        vegdem2 = vegdem2 + dtm
        vegdem2[vegdem2 == dtm] = 0

    else:
        # % Elevation vegdems if no DTM
        vegdem = vegdem + dsms[0]
        vegdem[vegdem == dsms[0]] = 0
        vegdem2 = vegdem2 + dsms[0]
        vegdem2[vegdem2 == dsms[0]] = 0
    # % Bush separation
    bush = np.logical_not((vegdem2 * vegdem)) * vegdem

    # shmat = np.zeros((rows, cols, 145))
    # vegshmat = np.zeros((rows, cols, 145))

    noa = 19.
    # % No. of anglesteps minus 1
    step = 89. / noa
    iangle = np.array(np.hstack((np.arange(step / 2., 89., step), 90.)))
    annulino = np.array(np.hstack((np.round(np.arange(0., 89., step)), 90.)))
    angleresult = svf_angles_100121()
    aziinterval = angleresult["aziinterval"]
    iazimuth = angleresult["iazimuth"]
    aziintervalaniso = np.ceil((aziinterval / 2.))
    index = 1.

    for i in np.arange(0, iangle.shape[0] - 1):
        for j in np.arange(0, (aziinterval[int(i)])):
            altitude = iangle[int(i)]
            azimuth = iazimuth[int(index)-1]
            # Casting shadow
            if usevegdem == 1:
                if altitude == 90:
                    shadowresult = shadow.shadowingfunction_20_3d_90(dsms, vegdem, vegdem2)

                else:
                    shadowresult = shadow.shadowingfunction_20_3d(dsms, vegdem, vegdem2, azimuth, altitude,
                                                                  scale, amaxvalue, bush, 1)

                vegsh = shadowresult["vegsh"]
                vbshvegsh = shadowresult["vbshvegsh"]
                sh = shadowresult["sh"]
            else:
                if altitude == 90:
                    sh = shadow.shadowingfunctionglobalradiation_3d_90(dsms)
                else:
                    sh = shadow.shadowingfunctionglobalradiation_3d(dsms, amaxvalue, azimuth, altitude, scale)

            # Calculate svfs
            for k in np.arange(annulino[int(i)] + 1, (annulino[int(i + 1.)]) + 1):
                weight = annulus_weight(k, aziinterval[i]) * sh
                svf = svf + weight
                weight = annulus_weight(k, aziintervalaniso[i]) * sh
                if (azimuth >= 0) and (azimuth < 180):
                    svfE = svfE + weight
                if (azimuth >= 90) and (azimuth < 270):
                    svfS = svfS + weight
                if (azimuth >= 180) and (azimuth < 360):
                    svfW = svfW + weight
                if (azimuth >= 270) or (azimuth < 90):
                    svfN = svfN + weight

            if usevegdem == 1:
                for k in np.arange(annulino[int(i)] + 1, (annulino[int(i + 1.)]) + 1):
                    # % changed to include 90
                    weight = annulus_weight(k, aziinterval[i])
                    svfveg = svfveg + weight * vegsh
                    svfaveg = svfaveg + weight * vbshvegsh
                    weight = annulus_weight(k, aziintervalaniso[i])
                    if (azimuth >= 0) and (azimuth < 180):
                        svfEveg = svfEveg + weight * vegsh
                        svfEaveg = svfEaveg + weight * vbshvegsh
                    if (azimuth >= 90) and (azimuth < 270):
                        svfSveg = svfSveg + weight * vegsh
                        svfSaveg = svfSaveg + weight * vbshvegsh
                    if (azimuth >= 180) and (azimuth < 360):
                        svfWveg = svfWveg + weight * vegsh
                        svfWaveg = svfWaveg + weight * vbshvegsh
                    if (azimuth >= 270) or (azimuth < 90):
                        svfNveg = svfNveg + weight * vegsh
                        svfNaveg = svfNaveg + weight * vbshvegsh

            index += 1
            print(int(index * (100. / 655.)))

    svfS = svfS + 3.0459e-004
    svfW = svfW + 3.0459e-004
    # % Last azimuth is 90. Hence, manual add of last annuli for svfS and SVFW
    # %Forcing svf not be greater than 1 (some MATLAB crazyness)
    svf[(svf > 1.)] = 1.
    svfE[(svfE > 1.)] = 1.
    svfS[(svfS > 1.)] = 1.
    svfW[(svfW > 1.)] = 1.
    svfN[(svfN > 1.)] = 1.

    if usevegdem == 1:
        last = cp.zeros((rows, cols))
        last[(vegdem2 == 0.)] = 3.0459e-004
        svfSveg = svfSveg + last
        svfWveg = svfWveg + last
        svfSaveg = svfSaveg + last
        svfWaveg = svfWaveg + last
        # %Forcing svf not be greater than 1 (some MATLAB crazyness)
        svfveg[(svfveg > 1.)] = 1.
        svfEveg[(svfEveg > 1.)] = 1.
        svfSveg[(svfSveg > 1.)] = 1.
        svfWveg[(svfWveg > 1.)] = 1.
        svfNveg[(svfNveg > 1.)] = 1.
        svfaveg[(svfaveg > 1.)] = 1.
        svfEaveg[(svfEaveg > 1.)] = 1.
        svfSaveg[(svfSaveg > 1.)] = 1.
        svfWaveg[(svfWaveg > 1.)] = 1.
        svfNaveg[(svfNaveg > 1.)] = 1.

    svfresult = {'svf': svf, 'svfE': svfE, 'svfS': svfS, 'svfW': svfW, 'svfN': svfN,
                 'svfveg': svfveg, 'svfEveg': svfEveg, 'svfSveg': svfSveg, 'svfWveg': svfWveg,
                 'svfNveg': svfNveg, 'svfaveg': svfaveg, 'svfEaveg': svfEaveg, 'svfSaveg': svfSaveg,
                 'svfWaveg': svfWaveg, 'svfNaveg': svfNaveg}

    return svfresult
