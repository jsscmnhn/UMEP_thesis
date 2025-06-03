# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from math import radians
import cupy as cp

def shadowingfunction_wallheight_13(amaxvalue, a, azimuth, altitude, scale, walls, aspect):
    '''
    Computes shadow masks for buildings on a terrain, and shadow height for on building walls, using a stepped
    projection method, based on sun position (azimuth, altitude) and elevation data. This is a CuPy-accelerated
    version optimized for GPU use.

    The function simulates shadow casting by iteratively stepping through the DSM grid
    along the direction of the sun, lowering the sun's ray with each step, and comparing
    it to terrain heights to determine shadowed pixels.

    Parameters:
        a (cp.ndarray):         DSM.
        azimuth (float):        Sun azimuth in degrees (clockwise from north).
        altitude (float):       Sun altitude in degrees  (0° = horizon, 90° = zenith).
        scale (float):          Scale factor (pixel size in meters)
        amaxvalue (float):      Maximum vertical height to simulate in the shadow projection.
        walls (cp.ndarray):     DSM layer representing wall heights [m].
        aspect (cp.ndarray):    Aspect (orientation) of building walls [radians].

    Returns:
        sh (cp.ndarray):         Shadow map of ground and roof (1 = shadow, 0 = sunlit).
        wallsh (cp.ndarray):     Shadow height on walls [m].
        wallsun (cp.ndarray):    Sunlit height of walls [m].
        facesh (cp.ndarray):     Shadow mask from wall self-shadowing (1 = shadow, 0 = sunlit).
        facesun (cp.ndarray):    Sunlit mask of walls (1 = sunlit, 0 = shadow).
    '''

    # Conversion
    degrees = np.pi/180.
    azimuth *= degrees
    altitude *= degrees
    # Grid size
    sizex = a.shape[0]
    sizey = a.shape[1]

    # initialise parameters
    f = cp.copy(a)
    dx = dy = dz = 0.0
    temp = cp.zeros((sizex, sizey))
    wallbol = (walls > 0).astype(float)
    
    # Precompute trigonometric values
    pibyfour = np.pi/4
    threetimespibyfour = 3 * pibyfour
    fivetimespibyfour = 5 * pibyfour
    seventimespibyfour = 7 * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs(1/sinazimuth)
    dscos = np.abs(1/cosazimuth)
    tanaltitudebyscale = np.tan(altitude)/scale

    index = 1

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
            dx = -1 * signcosazimuth * np.abs(np.round(index / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1 * signcosazimuth * index

        #  Vertical height offset per step
        dz = ds * index

        # Reset temporary working arrays
        temp[0:sizex, 0:sizey] = 0
        absdx = np.abs(dx)
        absdy = np.abs(dy)

        # Compute shifted indices for stepping
        xc1 = int((dx+absdx)/2)
        xc2 = int(sizex+(dx-absdx)/2)
        yc1 = int((dy+absdy)/2)
        yc2 = int(sizey+(dy-absdy)/2)
        xp1 = int(-((dx-absdx)/2))
        xp2 = int(sizex-(dx+absdx)/2)
        yp1 = int(-((dy-absdy)/2))
        yp2 = int(sizey-(dy+absdy)/2)

        # Offset terrain height by dz for shadow test, save highest: previous step or this step.
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz
        f = cp.fmax(f, temp) #Moving building shadow

        index = index + 1

    # Removing walls in shadow due to selfshadowing
    azilow = azimuth-np.pi/2
    azihigh = azimuth+np.pi/2

    if azilow >= 0 and azihigh < 2*np.pi:    # 90 to 270  (SHADOW)
        facesh = (cp.logical_or(aspect < azilow, aspect >= azihigh).astype(float)-wallbol+1)
    elif azilow < 0 and azihigh <= 2*np.pi:    # 0 to 90
        azilow = azilow + 2*np.pi
        facesh = cp.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1    # (SHADOW)    # check for the -1
    elif azilow > 0 and azihigh >= 2*np.pi:    # 270 to 360
        azihigh = azihigh-2*np.pi
        facesh = cp.logical_or(aspect > azilow, aspect <= azihigh)*-1 + 1    # (SHADOW)

    sh = cp.copy(f-a)    # shadow volume
    facesun = cp.logical_and(facesh + (walls > 0).astype(float) == 1, walls > 0).astype(float)
    wallsun = cp.copy(walls-sh)
    wallsun[wallsun < 0] = 0
    wallsun[facesh == 1] = 0    # Removing walls in "self"-shadow
    wallsh = cp.copy(walls-wallsun)

    sh = cp.logical_not(np.logical_not(sh)).astype(float)
    sh = sh * -1 + 1

    return sh, wallsh, wallsun, facesh, facesun

def shadowingfunction_wallheight_13_3d(amaxvalue, a, azimuth, altitude, scale, walls, aspect):
    '''
    Computes 3D building shadows based on sun position using stepped projection.

    This CuPy-accelerated method calculates shadow masks from a layered DSM (building & gap heights), and
    calculates shadow height for on building walls. It simulates how shadows are cast given sun azimuth and
    altitude. Works on a 3D stack of DSM layers.

    Parameters:
        a (cp.ndarray):         3D Layered DSM.
        azimuth (float):        Sun azimuth in degrees (clockwise from north).
        altitude (float):       Sun altitude in degrees  (0° = horizon, 90° = zenith).
        scale (float):          Scale factor (pixel size in meters)
        amaxvalue (float):      Maximum vertical height to simulate in the shadow projection.
        walls (cp.ndarray):     DSM layer representing wall heights at base layer [m].
        aspect (cp.ndarray):    Aspect (orientation) of building walls at base layer[radians].

    Returns:
        sh (cp.ndarray):         Shadow map of ground and roof at base layer (1 = shadow, 0 = sunlit).
        wallsh (cp.ndarray):     Shadow height on walls at base layer [m].
        wallsun (cp.ndarray):    Sunlit height of walls at base layer [m].
        facesh (cp.ndarray):     Shadow mask from wall self-shadowing at base layer (1 = shadow, 0 = sunlit).
        facesun (cp.ndarray):    Sunlit mask of walls at base layer (1 = sunlit, 0 = shadow).
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

    wallbol = cp.array((walls > 0), dtype=cp.float32)

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
    # % main loop
    # Stepwise projection loop: simulate sunlight travel across terrain
    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
        # Determine horizontal steps along sun vector
        if isVert:
            dy = signsinazimuth * index
            dx = -signcosazimuth * np.abs(np.round(index / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -signcosazimuth * index

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
    if num_combinations > 0:
        sh_combined = sh_stack[0]
        for i in range(1, num_combinations):
            sh_combined = cp.fmax(sh_combined, sh_stack[i])
        sh = cp.fmax(sh, sh_combined)

    # Removing walls in shadow due to selfshadowing
    azilow = azimuth - np.pi / 2
    azihigh = azimuth + np.pi / 2

    if azilow >= 0 and azihigh < 2 * np.pi:  # 90 to 270  (SHADOW)
        facesh = cp.logical_or(aspect < azilow, aspect >= azihigh).astype(float) - wallbol + 1  # TODO check
    elif azilow < 0 and azihigh <= 2 * np.pi:  # 0 to 90
        azilow = azilow + 2 * np.pi
        facesh = cp.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1  # (SHADOW)
    elif azilow > 0 and azihigh >= 2 * np.pi:  # 270 to 360
        azihigh -= 2 * np.pi
        facesh = cp.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1  # (SHADOW)

    sh = 1 - sh

    # wall shadows
    shvo = dsm_ground - a[0]  # first layer building shadow volume
    facesun = cp.logical_and(facesh + (walls > 0).astype(float) == 1, walls > 0).astype(float)
    wallsun = cp.copy(walls - shvo)
    wallsun[wallsun < 0] = 0
    wallsun[facesh == 1] = 0  # Removing walls in "self"-shadow
    wallsh = cp.copy(walls - wallsun)

    return sh, wallsh, wallsun, facesh, facesun
