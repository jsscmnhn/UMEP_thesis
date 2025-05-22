# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import cupy as cp

def shadowingfunctionglobalradiation_cupy(a, amaxvalue, azimuth, altitude, scale):
    '''
    Computes shadow masks for buildings using a stepped projection method, based on sun position
    (azimuth, altitude) and elevation data. This is a CuPy-accelerated version optimized for GPU use.

    The function simulates shadow casting by iteratively stepping through the DSM grid
    along the direction of the sun, lowering the sun's ray with each step, and comparing
    it to terrain heights to determine shadowed pixels.

    Parameters:
        a (cp.ndarray):            DSM.
        azimuth (float):        Sun azimuth in degrees (clockwise from north).
        altitude (float):       Sun altitude in degrees  (0° = horizon, 90° = zenith).
        scale (float):          Scale factor (pixel size in meters)
        amaxvalue (float):      Maximum vertical height to simulate in the shadow projection.

    Returns:
        sh (cp.ndarray):           Binary mask of building shadows (1 = lit, 0 = shadow),
'''
    # Conversion
    degrees = np.pi/180.
    azimuth *= degrees
    altitude *= degrees
    # Grid size
    sizex = a.shape[0]
    sizey = a.shape[1]

    # Initialize parameters
    dx = dy = dz = 0.0
    # Copy DSM to working float array
    f = a
    # Initialize shadow result masks
    temp = cp.zeros((sizex, sizey), dtype=cp.float32)
    index = 1.

    # Precompute trigonometric values
    pibyfour = np.pi/4.
    threetimespibyfour = 3.*pibyfour
    fivetimespibyfour = 5.*pibyfour
    seventimespibyfour = 7.*pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs((1./sinazimuth))
    dscos = np.abs((1./cosazimuth))
    tanaltitudebyscale = np.tan(altitude) / scale

    # Determine the stepping direction based on azimuth (sun direction)
    isVert = ((pibyfour <= azimuth) & (azimuth < threetimespibyfour)) | \
             ((fivetimespibyfour <= azimuth) & (azimuth < seventimespibyfour))

    # Vertical shadow step increment (altitude controls vertical displacement per step)
    ds = dssin * tanaltitudebyscale if isVert else dscos * tanaltitudebyscale

    # Stepwise projection loop: simulate sunlight travel across terrain
    while (amaxvalue >= dz and np.abs(dx) < sizex and np.abs(dy) < sizey):
        # Determine horizontal steps along sun vector
        if isVert:
            dy = signsinazimuth * index
            dx = -1. * signcosazimuth * np.abs(np.round(index / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1. * signcosazimuth * index

        # Vertical height offset per step
        dz = ds *index
        # Reset temporary working arrays
        temp[0:sizex, 0:sizey] = 0.
        absdx = np.abs(dx)
        absdy = np.abs(dy)

        # Compute shifted indices for stepping
        xc1 = (dx+absdx)/2.+1.
        xc2 = sizex+(dx-absdx)/2.
        yc1 = (dy+absdy)/2.+1.
        yc2 = sizey+(dy-absdy)/2.
        xp1 = -((dx-absdx)/2.)+1.
        xp2 = sizex-(dx+absdx)/2.
        yp1 = -((dy-absdy)/2.)+1.
        yp2 = sizey-(dy+absdy)/2.

        # Offset terrain height by dz for shadow test, save highest: previous step or this step.
        temp[int(xp1)-1:int(xp2), int(yp1)-1:int(yp2)] = a[int(xc1)-1:int(xc2), int(yc1)-1:int(yc2)]-dz
        f = cp.fmax(f, temp)
        index += 1.

    # Finalize shadow: Remove original DSM height from shadow volumes and invert.
    f = f-a
    f = cp.logical_not(f)
    sh = f.astype(cp.float32)

    return sh


def shadowingfunctionglobalradiation_3d(a, amaxvalue, azimuth, altitude, scale):
    '''
    Computes 3D building shadows based on sun position using stepped projection.

    This CuPy-accelerated method calculates shadow masks from a layered DSM (building & gap heights),
    simulating how shadows are cast given sun azimuth and altitude. Works on a 3D stack of DSM layers.

    Parameters:
          a (cp.ndarray):         3D Layered DSM.
          azimuth (float):        Sun azimuth in degrees (clockwise from north).
          altitude (float):       Sun altitude in degrees  (0° = horizon, 90° = zenith).
          scale (float):          Scale factor (pixel size in meters)
          amaxvalue (float):      Maximum vertical height to simulate in the shadow projection.

    Returns:
          sh (cp.ndarray):           Binary mask of building shadows on base layer (1 = lit, 0 = shadow),
    '''

    # Conversion
    degrees = np.pi / 180.0
    azimuth *= degrees
    altitude *= degrees
    # Grid size
    sizex, sizey = a[0].shape[0], a[0].shape[1]

    # Initialize parameters
    dx = dy = dz = 0.0
    num_layers = len(a)
    num_combinations = (num_layers - 1) // 2

    # Copy DSM to working float array
    dsm_ground = a[0]
    # Initialize shadow result masks
    temp = cp.zeros((sizex, sizey), dtype=cp.float32)
    temp_layers = cp.full((num_layers - 1, sizex, sizey), np.nan, dtype=cp.float32)
    sh = cp.zeros((sizex, sizey), dtype=cp.float32)  # shadows from buildings
    sh_stack = cp.full((num_combinations, sizex, sizey), np.nan, dtype=cp.float32)

    # Precompute trigonometric values
    pibyfour = np.pi / 4.0
    threetimespibyfour = 3.0 * pibyfour
    fivetimespibyfour = 5.0 * pibyfour
    seventimespibyfour = 7.0 * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs(1.0 / sinazimuth)
    dscos = np.abs(1.0 / cosazimuth)
    tanaltitudebyscale = np.tan(altitude) / scale

    # Determine the stepping direction based on azimuth (sun direction)
    isVert = ((pibyfour <= azimuth) & (azimuth < threetimespibyfour)) | \
             ((fivetimespibyfour <= azimuth) & (azimuth < seventimespibyfour))

    # Vertical shadow step increment (altitude controls vertical displacement per step)
    ds = dssin * tanaltitudebyscale if isVert else dscos * tanaltitudebyscale

    # For comparison with what the height difference would have been with the previous step
    preva = a[0] - ds
    index = 0.0

    # Stepwise projection loop: simulate sunlight travel across terrain
    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
        # Determine horizontal steps along sun vector
        if isVert:
            dy = signsinazimuth * index
            dx = -signcosazimuth * np.abs(np.round(index / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -signcosazimuth * index

        # Vertical height offset per step
        dz = ds * index

        # Reset temporary working arrays
        temp.fill(0.0)
        temp_layers[:] = np.nan
        absdx = np.abs(dx)
        absdy = np.abs(dy)

        # Compute shifted indices for stepping
        xc1 = int((dx + absdx) / 2.)
        xc2 = int(sizex + (dx - absdx) / 2.)
        yc1 = int((dy + absdy) / 2.)
        yc2 = int(sizey + (dy - absdy) / 2.)
        xp1 = int(-((dx - absdx) / 2.))
        xp2 = int(sizex - (dx + absdx) / 2.)
        yp1 = int(-((dy - absdy) / 2.))
        yp2 = int(sizey - (dy + absdy) / 2.)

        temp[xp1:xp2, yp1:yp2] = a[0][xc1:xc2, yc1:yc2] - dz
        temp_layers[:, xp1:xp2, yp1:yp2] = a[1:num_layers, xc1:xc2, yc1:yc2] - dz

        dsm_ground = cp.fmax(dsm_ground, temp)
        sh = (dsm_ground > a[0]).astype(cp.float32)

        #  Project shadows for each (gap, layer) pair
        for i in range(0, num_layers - 1, 2):
            # first gap part
            gap_layer_index = i
            layer_index = i + 1

            # Get gap and layer arrays for the current iteration
            gapabovea = temp_layers[gap_layer_index] > a[0]
            layerabovea = temp_layers[layer_index] > a[0]
            prevgapabovea = temp_layers[gap_layer_index] > preva
            prevlayerabovea = temp_layers[layer_index] > preva

            # Combine all conditions where building part casts shadow
            sh_temp = cp.add(cp.add(cp.add(layerabovea, gapabovea, dtype=float), prevgapabovea, dtype=float),
                             prevlayerabovea, dtype=float)

            # Remove cases where all four conditions are true (fully lit area)
            sh_temp = cp.where(sh_temp == 4.0, 0.0, sh_temp)
            sh_temp = cp.where(sh_temp > 0.0, 1.0, sh_temp)

            # Save highest height in stack
            sh_stack[i // 2] = cp.fmax(sh_stack[i // 2], sh_temp)

        index += 1.

    # Combine all shadow layers
    sh_combined = sh_stack[0]
    for i in range(1, num_combinations):
        sh_combined = cp.fmax(sh_combined, sh_stack[i])

    sh = cp.fmax(sh, sh_combined)
    sh = 1.0 - sh

    return sh


# @profile
def shadowingfunction_20_cupy(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush):
    '''
    Computes shadow masks for buildings and vegetation using a stepped projection method,
    based on sun position (azimuth, altitude) and elevation data. This is a CuPy-accelerated version
    optimized for GPU use. Two temporary arrays related to vegetation shading have been removed, and
    several optimizations applied to improve performance.

    The function simulates shadow casting by iteratively stepping through the DSM and CHM
    along the direction of the sun, lowering the sun's ray with each step, and comparing
    it to terrain heights to determine shadowed pixels.

    Parameters:
        a (cp.ndarray):            DSM.
        vegdem (cp.ndarray):       Vegetation height layer (CHM).
        vegdem2 (cp.ndarray):      Secondary vegetation height layer (trunk heights).
        azimuth (float):        Sun azimuth in degrees (clockwise from north).
        altitude (float):       Sun altitude in degrees  (0° = horizon, 90° = zenith).
        scale (float):          Scale factor (pixel size in meters)
        amaxvalue (float):      Maximum vertical height to simulate in the shadow projection.
        bush (cp.ndarray):         Bush indicator array (values > 1 indicate presence).

    Returns:
        dict: {
            'sh':           cp.ndarray, binary mask of building shadows (1 = lit, 0 = shadow),
            'vegsh':        cp.ndarray, binary mask of vegetation shadows (1 = lit, 0 = shadow),
            'vbshvegsh':    cp.ndarray, vegetation shadows not blocked by buildings.
        }
    '''

    # Conversion
    degrees = np.pi / 180.0
    azimuth *= degrees
    altitude *= degrees

    # Grid size
    sizex, sizey = a.shape[0], a.shape[1]

    # Initialize parameters
    dx = dy = dz = 0.0
    temp = cp.zeros((sizex, sizey), dtype=cp.float32)
    tempvegdem = cp.full((sizex, sizey), np.nan, dtype=cp.float32)
    tempvegdem2 = tempvegdem.copy()
    bushplant = bush > 1.0
    # Initialize shadow result masks
    sh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vbshvegsh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vegsh = cp.array(bushplant, dtype=cp.float32)
    # Copy DSM to working float array
    f = cp.array(a, dtype=cp.float32)

    # Precompute trigonometric values
    pibyfour = np.pi / 4.0
    threetimespibyfour = 3.0 * pibyfour
    fivetimespibyfour = 5.0 * pibyfour
    seventimespibyfour = 7.0 * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs(1.0 / sinazimuth)
    dscos = np.abs(1.0 / cosazimuth)
    tanaltitudebyscale = np.tan(altitude) /scale

    # Determine the stepping direction based on azimuth (sun direction)
    isVert = ((pibyfour <= azimuth) & (azimuth < threetimespibyfour)) | \
             ((fivetimespibyfour <= azimuth) & (azimuth < seventimespibyfour))

    # Vertical shadow step increment (altitude controls vertical displacement per step)
    ds = dssin * tanaltitudebyscale if isVert else dscos * tanaltitudebyscale

    # For comparison with what the height difference would have been with the previous step
    preva = a - ds
    i = 0.0

    # Stepwise projection loop: simulate sunlight travel across terrain
    while (amaxvalue >= dz) and (np.abs(dx)) < sizex and (np.abs(dy) < sizey):
        # Determine horizontal steps along sun vector
        if isVert:
            dy = signsinazimuth * i
            dx = -signcosazimuth * np.abs(np.round(i / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(i * tanazimuth))
            dx = -signcosazimuth * i

        # Vertical height offset per step
        dz = ds * i

        # Reset temporary working arrays
        tempvegdem.fill(np.nan)
        tempvegdem2.fill(np.nan)
        temp.fill(0.0)
        absdx = np.abs(dx)
        absdy = np.abs(dy)

        # Compute shifted indices for stepping
        xc1 = int((dx + absdx) / 2.)
        xc2 = int(sizex + (dx - absdx) / 2.)
        yc1 = int((dy + absdy) / 2.)
        yc2 = int(sizey + (dy - absdy) / 2.)
        xp1 = int(-((dx - absdx) / 2.))
        xp2 = int(sizex - (dx + absdx) / 2.)
        yp1 = int(-((dy - absdy) / 2.))
        yp2 = int(sizey - (dy + absdy) / 2.)

        # Offset terrain height by dz for shadow test, save highest: previous step or this step.
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz
        f = cp.fmax(f, temp)

        # Shadow from buildings: lit where final height is lower than DSM
        sh = cp.where(f > a, 1.0, 0.0)

        # Offset vegetation (canopy) and compare to DSM
        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2]- dz
        fabovea = tempvegdem > a
        lastfabovea = tempvegdem > preva

        # Offset vegetation (trunks) and compare to DSM
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        gabovea = tempvegdem2 > a
        lastgabovea = tempvegdem2 > preva

        # Combine all conditions where vegetation casts shadow
        vegsh2 = cp.add(cp.add(cp.add(fabovea, gabovea, dtype=cp.float32), lastfabovea, dtype=cp.float32),
                        lastgabovea, dtype=cp.float32)

        # Remove cases where all four conditions are true (fully lit area)
        vegsh2 = cp.where(vegsh2 == 4.0, 0.0, vegsh2)
        vegsh2 = cp.where(vegsh2 > 0.0, 1.0, vegsh2)

        vegsh = cp.fmax(vegsh, vegsh2)
        vegsh = cp.where(vegsh * sh > 0.0, 0.0, vegsh)
        cp.add(vbshvegsh, vegsh, out=vbshvegsh)

        i += 1.0

    # Invert shadow mask (1 = lit, 0 = shadow)
    sh = 1.0 - sh

    # Finalize vegetation mask and remove overlap with buildings
    vbshvegsh = cp.where((vbshvegsh > 0.0), 1.0, 0.0)
    vbshvegsh -= vegsh
    vegsh = 1.0 - vegsh
    vbshvegsh = 1.0 - vbshvegsh

    shadowresult = {
        'sh': sh,
        'vegsh': vegsh,
        'vbshvegsh': vbshvegsh
    }
    return shadowresult


def shadowingfunction_20_3d(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush):
    '''
    Computes 3D building shadows and Vegetation shadows based on sun position using stepped projection.

    This CuPy-accelerated method calculates shadow masks from a layered DSM (building & gap heights) & a CHM,
    simulating how shadows are cast given sun azimuth and altitude. Works on a 3D stack of DSM layers.

    Parameters:
          a (cp.ndarray):            3D Layered DSM.
          vegdem (cp.ndarray):       Vegetation height layer (CHM).
          vegdem2 (cp.ndarray):      Secondary vegetation height layer (trunk heights).
          azimuth (float):        Sun azimuth in degrees (clockwise from north).
          altitude (float):       Sun altitude in degrees  (0° = horizon, 90° = zenith).
          scale (float):          Scale factor (pixel size in meters)
          amaxvalue (float):      Maximum vertical height to simulate in the shadow projection.

    Returns:
        dict: {
            'sh':           cp.ndarray, binary mask of building shadows on base layer (1 = lit, 0 = shadow),
            'vegsh':        cp.ndarray, binary mask of vegetation shadows on base layer (1 = lit, 0 = shadow),
            'vbshvegsh':    cp.ndarray, vegetation shadows not blocked by buildings on base layer.
        }
    '''

    # Conversion
    degrees = np.pi / 180.0
    azimuth *= degrees
    altitude *= degrees
    # Grid size
    sizex, sizey = a[0].shape[0], a[0].shape[1]

    # Initialize parameters
    dx = dy = dz = 0.0
    num_layers = len(a)
    num_combinations = (num_layers - 1) // 2
    temp = cp.zeros((sizex, sizey), dtype=cp.float32)
    temp_layers = cp.full((num_layers - 1, sizex, sizey), np.nan, dtype=cp.float32)
    tempvegdem = cp.full((sizex, sizey), np.nan, dtype=cp.float32)
    tempvegdem2 = cp.full((sizex, sizey), np.nan, dtype=cp.float32)
    bushplant = bush > 1.0
    # Copy DSM to working float array
    dsm_ground = a[0]

    # Initialize shadow result masks
    sh = cp.zeros((sizex, sizey),  dtype=cp.float32) #shadows from buildings
    vbshvegsh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vegsh = cp.array(bushplant, dtype=cp.float32)
    sh_stack = cp.full((num_combinations, sizex, sizey), np.nan, dtype=cp.float32)


    # Precompute trigonometric values
    pibyfour = np.pi / 4.0
    threetimespibyfour = 3.0 * pibyfour
    fivetimespibyfour = 5.0 * pibyfour
    seventimespibyfour = 7.0 * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs(1.0 / sinazimuth)
    dscos = np.abs(1.0 / cosazimuth)
    tanaltitudebyscale = np.tan(altitude) /scale


    # Determine the stepping direction based on azimuth (sun direction)
    isVert = ((pibyfour <= azimuth) & (azimuth < threetimespibyfour)) | \
             ((fivetimespibyfour <= azimuth) & (azimuth < seventimespibyfour))

    # Vertical shadow step increment (altitude controls vertical displacement per step)
    ds = dssin * tanaltitudebyscale if isVert else dscos * tanaltitudebyscale

    # For comparison with what the height difference would have been with the previous step
    preva = a[0] - ds
    index = 0.0

    # Stepwise projection loop: simulate sunlight travel across terrain
    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
        # Determine horizontal steps along sun vector
        if isVert:
            dy = signsinazimuth * index
            dx = -signcosazimuth * np.abs(np.round(index / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -signcosazimuth * index

        # Vertical height offset per step
        dz = ds * index

        # Reset temporary working arrays
        tempvegdem.fill(np.nan)
        tempvegdem2.fill(np.nan)
        temp.fill(0.0)
        temp_layers[:] = np.nan
        absdx = np.abs(dx)
        absdy = np.abs(dy)

        # Compute shifted indices for stepping
        xc1 = int((dx + absdx) / 2.)
        xc2 = int(sizex + (dx - absdx) / 2.)
        yc1 = int((dy + absdy) / 2.)
        yc2 = int(sizey + (dy - absdy) / 2.)
        xp1 = int(-((dx - absdx) / 2.))
        xp2 = int(sizex - (dx + absdx) / 2.)
        yp1 = int(-((dy - absdy) / 2.))
        yp2 = int(sizey - (dy + absdy) / 2.)

        # ================= Building Part =================
        temp[xp1:xp2, yp1:yp2] = a[0][xc1:xc2, yc1:yc2] - dz
        temp_layers[:, xp1:xp2, yp1:yp2] = a[1:num_layers, xc1:xc2, yc1:yc2] - dz

        dsm_ground = cp.fmax(dsm_ground, temp)

        sh = (dsm_ground > a[0]).astype(cp.float32)


        #  Project shadows for each (gap, layer) pair
        for i in range(0, num_layers - 1, 2):
            # first gap part
            gap_layer_index = i
            layer_index = i + 1

            # Get gap and layer arrays for the current iteration
            gapabovea = temp_layers[gap_layer_index] > a[0]
            layerabovea = temp_layers[layer_index] > a[0]
            prevgapabovea = temp_layers[gap_layer_index] > preva
            prevlayerabovea = temp_layers[layer_index] > preva

            # Combine all conditions where building part casts shadow
            sh_temp = cp.add(cp.add(cp.add(layerabovea, gapabovea, dtype=float), prevgapabovea, dtype=float),
                              prevlayerabovea, dtype=float)

            # Remove cases where all four conditions are true (fully lit area)
            sh_temp = cp.where(sh_temp == 4.0, 0.0, sh_temp)
            sh_temp = cp.where(sh_temp > 0.0, 1.0, sh_temp)

            # Save highest height in stack
            sh_stack[i // 2] = cp.fmax(sh_stack[i // 2], sh_temp)

        # ================= Vegetation Part =================
        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2]- dz
        fabovea = tempvegdem > a[0]
        lastfabovea = tempvegdem > preva

        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        gabovea = tempvegdem2 > a[0]
        lastgabovea = tempvegdem2 > preva


        vegsh2 = cp.add(cp.add(cp.add(fabovea, gabovea, dtype=cp.float32), lastfabovea, dtype=cp.float32),
                        lastgabovea, dtype=cp.float32)

        vegsh2 = cp.where(vegsh2 == 4.0, 0.0, vegsh2)
        vegsh2 = cp.where(vegsh2 > 0.0, 1.0, vegsh2)

        vegsh = cp.fmax(vegsh, vegsh2)
        vegsh = cp.where((vegsh * sh > 0.0), 0.0, vegsh)
        vegsh = cp.where(cp.any(sh_stack * vegsh > 0.0, axis=0), 0.0, vegsh)
        cp.add(vbshvegsh, vegsh,  out=vbshvegsh)

        index += 1.0

    # Combine all shadow layers
    sh_combined = sh_stack[0]
    for i in range(1, num_combinations):
        sh_combined = cp.fmax(sh_combined, sh_stack[i])
    sh = (cp.fmax(sh, sh_combined))

    # Invert shadow mask (1 = lit, 0 = shadow)
    sh = 1.0 - sh

    # Finalize vegetation mask and remove overlap with buildings
    vbshvegsh[vbshvegsh > 0.0] = 1.0
    vbshvegsh -= vegsh
    vegsh = 1.0 - vegsh
    vbshvegsh = 1.0 - vbshvegsh

    shadowresult = {
        'sh': sh,
        'vegsh': vegsh,
        'vbshvegsh': vbshvegsh
    }
    return shadowresult


def shadowingfunction_20_3d_90(a, vegdem, vegdem2):
    '''
    Computes binary shadow masks from buildings and vegetation for 3D DSM input
    under direct overhead sunlight (solar altitude = 90°).

    Building shadows are determined by vertically projecting non-gap layers of the 3D DSM.
    Vegetation shadows are computed where both vegdem and vegdem2 are non-zero and not blocked by buildings.

    Parameters:
        a (cp.ndarray):        Layered 3D DSM.
        vegdem (cp.ndarray):   Vegetation height layer (CHM).
        vegdem2 (cp.ndarray):  Secondary vegetation height layer (trunk heights).

    Returns:
        dict: {
            'sh':           cp.ndarray, binary mask of building shadows (1 = lit, 0 = shadow),
            'vegsh':        cp.ndarray, binary mask of vegetation shadows (1 = lit, 0 = shadow),
            'vbshvegsh':    cp.ndarray, vegetation shadows not blocked by buildings
        }
    '''
    sizex, sizey = a[0].shape[0], a[0].shape[1]

    # Initialize parameters

    num_layers = len(a)
    vbshvegsh = cp.zeros((sizex, sizey), dtype=cp.float32)
    sh2 = cp.zeros((sizex, sizey),  dtype=cp.float32)

    for i in range(0, num_layers - 1, 2):
        sh_temp = cp.where(a[i + 1] > 0, 1.0, 0.0)
        sh2 = cp.fmax(sh2, sh_temp)

    # Vegetation Part
    vegsh = cp.where(cp.logical_and(vegdem > 0, vegdem2 > 0), 1.0, 0.0)
    vegsh = cp.where((vegsh * sh2 > 0.0), 0.0, vegsh)
    cp.add(vbshvegsh, vegsh,  out=vbshvegsh)

    sh = 1.0 - sh2
    vbshvegsh[vbshvegsh > 0.0] = 1.0
    vbshvegsh -= vegsh
    vegsh = 1.0 - vegsh
    vbshvegsh = 1.0 - vbshvegsh

    shadowresult = {
        'sh': sh,
        'vegsh': vegsh,
        'vbshvegsh': vbshvegsh
    }
    return shadowresult

def shadowingfunctionglobalradiation_3d_90(a):
    """
    Computes building shadow mask for a 3D DSM input assuming a solar altitude of 90 degrees.

    Shadows are determined by vertically projecting all values in the DSM layers
    (excluding gaps) that are greater than 1, simulating direct overhead sunlight.

    Parameters:
        a (cp.ndarray):    Layered 3D DSM.

    Returns:
        sh (cp.ndarray):   2D binary shadow mask (1 = illuminated, 0 = shadow).
    """

    sizex, sizey = a[0].shape[0], a[0].shape[1]
    num_layers = len(a)
    sh2 = cp.zeros((sizex, sizey),  dtype=cp.float32)

    for i in range(0, num_layers - 1, 2):
        sh_temp = cp.where(a[i + 1] > 0, 1.0, 0.0)
        sh2 = cp.fmax(sh2, sh_temp)

    return 1.0 - sh2

def shadowingfunction_20v2(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, trunkcheck, bush, forsvf):
    '''
    Computes shadow masks from buildings and vegetation based on sun position and elevation data.

    This is a first attempt at a faster version of `shadowing_20`, designed to reduce computation by terminating the
    trunk zone shade casting early once all trunk heights are below the shadow casting threshold. It steps through
    the grid in the sun's direction, checking for shadowing effects from buildings and vegetation.

    Parameters:
        a (ndarray):            DSM.
        vegdem (ndarray):       Vegetation height layer (CHM).
        vegdem2 (ndarray):      Secondary vegetation height layer (trunk heights).
        azimuth (float):        Sun azimuth in degrees.
        altitude (float):       Sun altitude in degrees.
        scale (float):          Scale factor.
        amaxvalue (float):      Maximum vertical extent to simulate shadows.
        trunkcheck (boolean):   If True, use the trunk height check to stop trunk shade casting.
        bush (ndarray):         Bush indicator array (values > 1 indicate presence).
        forsvf (int):           Flag to indicate if called from SVF plugin (0 enables progress output).

    Returns:
        dict: {
            'sh':           ndarray, binary mask of building shadows (1 = lit, 0 = shadow),
            'vegsh':        ndarray, binary mask of vegetation shadows (1 = lit, 0 = shadow),
            'vbshvegsh':    ndarray, vegetation shadows not blocked by buildings
        }
    '''

    # conversion
    degrees = np.pi/180.
    azimuth = azimuth * degrees
    altitude = altitude * degrees

    # measure the size of grid
    sizex = a.shape[0]
    sizey = a.shape[1]

    # progressbar for svf plugin
    if forsvf == 0:
        barstep = np.max([sizex, sizey])
        total = 100. / barstep

    # initialise parameters
    dx = 0.
    dy = 0.
    dz = 0.
    temp = np.zeros((sizex, sizey))
    tempvegdem = np.zeros((sizex, sizey))
    tempvegdem2 = np.zeros((sizex, sizey))
    templastfabovea = np.zeros((sizex, sizey))
    templastgabovea = np.zeros((sizex, sizey))
    bushplant = bush > 1.
    sh = np.zeros((sizex, sizey)) #shadows from buildings
    vbshvegsh = np.zeros((sizex, sizey)) #vegetation blocking buildings
    vegsh = np.add(np.zeros((sizex, sizey)), bushplant, dtype=float) #vegetation shadow
    f = a

    pibyfour = np.pi / 4.
    threetimespibyfour = 3. * pibyfour
    fivetimespibyfour = 5.* pibyfour
    seventimespibyfour = 7. * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs((1./sinazimuth))
    dscos = np.abs((1./cosazimuth))
    tanaltitudebyscale = np.tan(altitude) / scale
    # index = 1 6
    index = 0
    isVert = ((pibyfour <= azimuth) and (azimuth < threetimespibyfour) or (fivetimespibyfour <= azimuth) and (
                azimuth < seventimespibyfour))
    if isVert:
        ds = dssin
    else:
        ds = dscos

    # preva = a + ds
    dzprev = 0

    # main loop
    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
        if forsvf == 0:
            print(int(index * total))
        if isVert:
            dy = signsinazimuth * index
            dx = -1. * signcosazimuth * np.abs(np.round(index / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1. * signcosazimuth * index
        # note: dx and dy represent absolute values while ds is an incremental value
        dz = (ds * index) * tanaltitudebyscale
        tempvegdem[0:sizex, 0:sizey] = 0.
        tempvegdem2[0:sizex, 0:sizey] = 0.
        temp[0:sizex, 0:sizey] = 0.

        templastfabovea[0:sizex, 0:sizey] = 0.
        templastgabovea[0:sizex, 0:sizey] = 0.

        absdx = np.abs(dx)
        absdy = np.abs(dy)
        xc1 = int((dx + absdx) / 2.)
        xc2 = int(sizex + (dx - absdx) / 2.)
        yc1 = int((dy + absdy) / 2.)
        yc2 = int(sizey + (dy - absdy) / 2.)
        xp1 = int(-((dx - absdx) / 2.))
        xp2 = int(sizex - (dx + absdx) / 2.)
        yp1 = int(-((dy - absdy) / 2.))
        yp2 = int(sizey - (dy + absdy) / 2.)
        isTrunk = trunkcheck >= dz

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz

        f = np.fmax(f, temp)  # Moving building shadow
        sh[(f > a)] = 1.
        sh[(f <= a)] = 0.
        fabovea = tempvegdem > a  # vegdem above DEM

        templastfabovea[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dzprev
        lastfabovea = templastfabovea > a

        if isTrunk:
            tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
            gabovea = tempvegdem2 > a  # vegdem2 above DEM

            # new pergola condition
            templastgabovea[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2]- dzprev
            lastgabovea = templastgabovea > a

            vegsh2 = np.add(np.add(np.add(fabovea, gabovea, dtype=float), lastfabovea, dtype=float), lastgabovea,
                            dtype=float)

            # Apply the zeroing condition
            vegsh2[vegsh2 == 4] = 0.
            vegsh2[vegsh2 > 0] = 1.
        else:
            vegsh2 = (fabovea | lastfabovea).astype(float)
        vegsh = np.fmax(vegsh, vegsh2)

        vegsh[(vegsh * sh > 0.)] = 0.
        vbshvegsh = vegsh + vbshvegsh  # removing shadows 'behind' buildings

        dzprev = dz
        index += 1.

    sh = 1.-sh
    vbshvegsh[(vbshvegsh > 0.)] = 1.
    vbshvegsh = vbshvegsh-vegsh
    vegsh = 1.-vegsh
    vbshvegsh = 1.-vbshvegsh

    shadowresult = {'sh': sh, 'vegsh': vegsh, 'vbshvegsh': vbshvegsh}

    return shadowresult

# ============================== Original Shadow Functions  ================================
def shadowingfunctionglobalradiation(a, azimuth, altitude, scale, forsvf):
    # %This m.file calculates shadows on a DEM
    # % conversion
    degrees = np.pi / 180.
    # if azimuth == 0.0:
    # azimuth = 0.000000000001
    azimuth = np.dot(azimuth, degrees)
    altitude = np.dot(altitude, degrees)
    # % measure the size of the image
    sizex = a.shape[0]
    sizey = a.shape[1]
    if forsvf == 0:
        barstep = np.max([sizex, sizey])
        total = 100. / barstep  # dlg.progressBar.setRange(0, barstep)
    # % initialise parameters
    f = a
    dx = 0.
    dy = 0.
    dz = 0.
    temp = np.zeros((sizex, sizey))
    index = 1.
    # % other loop parameters
    amaxvalue = a.max()
    pibyfour = np.pi / 4.
    threetimespibyfour = 3. * pibyfour
    fivetimespibyfour = 5. * pibyfour
    seventimespibyfour = 7. * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs((1. / sinazimuth))
    dscos = np.abs((1. / cosazimuth))
    tanaltitudebyscale = np.tan(altitude) / scale
    # % main loop
    while (amaxvalue >= dz and np.abs(dx) < sizex and np.abs(dy) < sizey):
        if forsvf == 0:
            print(int(index * total))
            # dlg.progressBar.setValue(index)
        # while np.logical_and(np.logical_and(amaxvalue >= dz, np.abs(dx) <= sizex), np.abs(dy) <= sizey):(np.logical_and(amaxvalue >= dz, np.abs(dx) <= sizex), np.abs(dy) <= sizey):
        # if np.logical_or(np.logical_and(pibyfour <= azimuth, azimuth < threetimespibyfour), np.logical_and(fivetimespibyfour <= azimuth, azimuth < seventimespibyfour)):
        if (
                pibyfour <= azimuth and azimuth < threetimespibyfour or fivetimespibyfour <= azimuth and azimuth < seventimespibyfour):
            dy = signsinazimuth * index
            dx = -1. * signcosazimuth * np.abs(np.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1. * signcosazimuth * index
            ds = dscos

        # % note: dx and dy represent absolute values while ds is an incremental value
        dz = ds * index * tanaltitudebyscale
        temp[0:sizex, 0:sizey] = 0.
        absdx = np.abs(dx)
        absdy = np.abs(dy)
        xc1 = (dx + absdx) / 2. + 1.
        xc2 = sizex + (dx - absdx) / 2.
        yc1 = (dy + absdy) / 2. + 1.
        yc2 = sizey + (dy - absdy) / 2.
        xp1 = -((dx - absdx) / 2.) + 1.
        xp2 = sizex - (dx + absdx) / 2.
        yp1 = -((dy - absdy) / 2.) + 1.
        yp2 = sizey - (dy + absdy) / 2.
        temp[int(xp1) - 1:int(xp2), int(yp1) - 1:int(yp2)] = a[int(xc1) - 1:int(xc2), int(yc1) - 1:int(yc2)] - dz
        # f = np.maximum(f, temp)  # bad performance in python3. Replaced with fmax
        f = np.fmax(f, temp)
        index += 1.

    f = f - a
    f = np.logical_not(f)
    sh = np.double(f)

    return sh


# @jit(nopython=True)
def shadowingfunction_20(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, forsvf):
    # plt.ion()
    # fig = plt.figure(figsize=(24, 7))
    # plt.axis('image')
    # ax1 = plt.subplot(2, 3, 1)
    # ax2 = plt.subplot(2, 3, 2)
    # ax3 = plt.subplot(2, 3, 3)
    # ax4 = plt.subplot(2, 3, 4)
    # ax5 = plt.subplot(2, 3, 5)
    # ax6 = plt.subplot(2, 3, 6)
    # ax1.title.set_text('fabovea')
    # ax2.title.set_text('gabovea')
    # ax3.title.set_text('vegsh at ' + str(altitude))
    # ax4.title.set_text('lastfabovea')
    # ax5.title.set_text('lastgabovea')
    # ax6.title.set_text('vegdem')

    # This function casts shadows on buildings and vegetation units.
    # New capability to deal with pergolas 20210827

    # conversion
    degrees = np.pi / 180.
    azimuth = azimuth * degrees
    altitude = altitude * degrees

    # measure the size of grid
    sizex = a.shape[0]
    sizey = a.shape[1]

    # progressbar for svf plugin
    if forsvf == 0:
        barstep = np.max([sizex, sizey])
        total = 100. / barstep
        # dlg.progressBar.setRange(0, barstep)
        # dlg.progressBar.setValue(0)

    # initialise parameters
    dx = 0.
    dy = 0.
    dz = 0.
    temp = np.zeros((sizex, sizey))
    tempvegdem = np.zeros((sizex, sizey))
    tempvegdem2 = np.zeros((sizex, sizey))
    templastfabovea = np.zeros((sizex, sizey))
    templastgabovea = np.zeros((sizex, sizey))
    bushplant = bush > 1.
    sh = np.zeros((sizex, sizey))  # shadows from buildings
    vbshvegsh = np.zeros((sizex, sizey))  # vegetation blocking buildings
    vegsh = np.add(np.zeros((sizex, sizey)), bushplant, dtype=float)  # vegetation shadow
    f = a

    pibyfour = np.pi / 4.
    threetimespibyfour = 3. * pibyfour
    fivetimespibyfour = 5. * pibyfour
    seventimespibyfour = 7. * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs((1. / sinazimuth))
    dscos = np.abs((1. / cosazimuth))
    tanaltitudebyscale = np.tan(altitude) / scale
    # index = 1
    index = 0

    # new case with pergola (thin vertical layer of vegetation), August 2021
    dzprev = 0

    # main loop
    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
        if forsvf == 0:
            print(int(index * total))  # dlg.progressBar.setValue(index)
        if ((pibyfour <= azimuth) and (azimuth < threetimespibyfour) or (fivetimespibyfour <= azimuth) and (
                azimuth < seventimespibyfour)):
            dy = signsinazimuth * index
            dx = -1. * signcosazimuth * np.abs(np.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1. * signcosazimuth * index
            ds = dscos
        # note: dx and dy represent absolute values while ds is an incremental value
        dz = (ds * index) * tanaltitudebyscale
        tempvegdem[0:sizex, 0:sizey] = 0.
        tempvegdem2[0:sizex, 0:sizey] = 0.
        temp[0:sizex, 0:sizey] = 0.
        templastfabovea[0:sizex, 0:sizey] = 0.
        templastgabovea[0:sizex, 0:sizey] = 0.
        absdx = np.abs(dx)
        absdy = np.abs(dy)
        xc1 = int((dx + absdx) / 2.)
        xc2 = int(sizex + (dx - absdx) / 2.)
        yc1 = int((dy + absdy) / 2.)
        yc2 = int(sizey + (dy - absdy) / 2.)
        xp1 = int(-((dx - absdx) / 2.))
        xp2 = int(sizex - (dx + absdx) / 2.)
        yp1 = int(-((dy - absdy) / 2.))
        yp2 = int(sizey - (dy + absdy) / 2.)

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz

        f = np.fmax(f, temp)  # Moving building shadow
        sh[(f > a)] = 1.
        sh[(f <= a)] = 0.
        fabovea = tempvegdem > a  # vegdem above DEM
        gabovea = tempvegdem2 > a  # vegdem2 above DEM

        # new pergola condition
        templastfabovea[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dzprev
        templastgabovea[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dzprev
        lastfabovea = templastfabovea > a
        lastgabovea = templastgabovea > a
        dzprev = dz
        vegsh2 = np.add(np.add(np.add(fabovea, gabovea, dtype=float), lastfabovea, dtype=float), lastgabovea,
                        dtype=float)
        vegsh2[vegsh2 == 4] = 0.
        # vegsh2[vegsh2 == 1] = 0. # This one is the ultimate question...
        vegsh2[vegsh2 > 0] = 1.

        vegsh = np.fmax(vegsh, vegsh2)
        vegsh[(vegsh * sh > 0.)] = 0.
        vbshvegsh = vegsh + vbshvegsh  # removing shadows 'behind' buildings

        # im1 = ax1.imshow(fabovea)
        # im2 = ax2.imshow(gabovea)
        # im3 = ax3.imshow(vegsh)
        # im4 = ax4.imshow(lastfabovea)
        # im5 = ax5.imshow(lastgabovea)
        # im6 = ax6.imshow(vegshtest)
        # im1 = ax1.imshow(tempvegdem)
        # im2 = ax2.imshow(tempvegdem2)
        # im3 = ax3.imshow(vegsh)
        # im4 = ax4.imshow(templastfabovea)
        # im5 = ax5.imshow(templastgabovea)
        # im6 = ax6.imshow(vegshtest)
        # plt.show()
        # plt.pause(0.05)

        index += 1.

    sh = 1. - sh
    vbshvegsh[(vbshvegsh > 0.)] = 1.
    vbshvegsh = vbshvegsh - vegsh
    vegsh = 1. - vegsh
    vbshvegsh = 1. - vbshvegsh

    # plt.close()
    # plt.ion()
    # fig = plt.figure(figsize=(24, 7))
    # plt.axis('image')
    # ax1 = plt.subplot(1, 3, 1)
    # im1 = ax1.imshow(vegsh)
    # plt.colorbar(im1)

    # ax2 = plt.subplot(1, 3, 2)
    # im2 = ax2.imshow(vegdem2)
    # plt.colorbar(im2)
    # plt.title('TDSM')

    # ax3 = plt.subplot(1, 3, 3)
    # im3 = ax3.imshow(vegdem)
    # plt.colorbar(im3)
    # plt.tight_layout()
    # plt.title('CDSM')
    # plt.show()
    # plt.pause(0.05)

    shadowresult = {'sh': sh, 'vegsh': vegsh, 'vbshvegsh': vbshvegsh}

    return shadowresult


def shadowingfunction_20_old(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, dlg, forsvf):
    # % This function casts shadows on buildings and vegetation units
    # % conversion
    degrees = np.pi / 180.
    if azimuth == 0.0:
        azimuth = 0.000000000001
    azimuth = np.dot(azimuth, degrees)
    altitude = np.dot(altitude, degrees)
    # % measure the size of the image
    sizex = a.shape[0]
    sizey = a.shape[1]
    # % initialise parameters
    if forsvf == 0:
        barstep = np.max([sizex, sizey])
        dlg.progressBar.setRange(0, barstep)
        dlg.progressBar.setValue(0)

    dx = 0.
    dy = 0.
    dz = 0.
    temp = np.zeros((sizex, sizey))
    tempvegdem = np.zeros((sizex, sizey))
    tempvegdem2 = np.zeros((sizex, sizey))
    sh = np.zeros((sizex, sizey))
    vbshvegsh = np.zeros((sizex, sizey))
    vegsh = np.zeros((sizex, sizey))
    tempbush = np.zeros((sizex, sizey))
    f = a
    g = np.zeros((sizex, sizey))
    bushplant = bush > 1.
    pibyfour = np.pi / 4.
    threetimespibyfour = 3. * pibyfour
    fivetimespibyfour = 5. * pibyfour
    seventimespibyfour = 7. * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs((1. / sinazimuth))
    dscos = np.abs((1. / cosazimuth))
    tanaltitudebyscale = np.tan(altitude) / scale
    index = 1