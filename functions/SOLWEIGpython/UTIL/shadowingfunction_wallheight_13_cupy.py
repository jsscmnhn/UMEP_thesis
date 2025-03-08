# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from math import radians
import cupy as cp


def shadowingfunction_wallheight_13(amaxvalue, a, azimuth, altitude, scale, walls, aspect):
    """
    This m.file calculates shadows on a DSM and shadow height on building
    walls.
    
    INPUTS:
    a = DSM
    azimuth and altitude = sun position
    scale= scale of DSM (1 meter pixels=1, 2 meter pixels=0.5)
    walls= pixel row 'outside' buildings. will be calculated if empty
    aspect = normal aspect of buildings walls
    
    OUTPUT:
    sh=ground and roof shadow
    wallsh = height of wall that is in shadow
    wallsun = hieght of wall that is in sun
    
    Fredrik Lindberg 2012-03-19
    fredrikl@gvc.gu.se
    
     Utdate 2013-03-13 - bugfix for walls alinged with sun azimuths

    :param a:
    :param azimuth:
    :param altitude:
    :param scale:
    :param walls:
    :param aspect:
    :return:
    """

    # conversion
    # degrees = np.pi/180
    azimuth = radians(azimuth)
    altitude = radians(altitude)

    # measure the size of the image
    sizex = cp.shape(a)[0]
    sizey = cp.shape(a)[1]

    # initialise parameters
    f = cp.copy(a)
    dx = 0
    dy = 0
    dz = 0
    temp = cp.zeros((sizex, sizey))
    wallbol = (walls > 0).astype(float)
    
    # other loop parameters
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

    isVert = ((pibyfour <= azimuth) & (azimuth < threetimespibyfour)) | \
             ((fivetimespibyfour <= azimuth) & (azimuth < seventimespibyfour))
    if isVert:
        ds = dssin
    else:
        ds = dscos

    # main loop
    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
        if isVert:
            dy = signsinazimuth * index
            dx = -1 * signcosazimuth * np.abs(np.round(index / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1 * signcosazimuth * index

        # note: dx and dy represent absolute values while ds is an incremental value
        dz = ds * index * tanaltitudebyscale
        temp[0:sizex, 0:sizey] = 0

        absdx = np.abs(dx)
        absdy = np.abs(dy)

        xc1 = int((dx+absdx)/2)
        xc2 = int(sizex+(dx-absdx)/2)
        yc1 = int((dy+absdy)/2)
        yc2 = int(sizey+(dy-absdy)/2)

        xp1 = int(-((dx-absdx)/2))
        xp2 = int(sizex-(dx+absdx)/2)
        yp1 = int(-((dy-absdy)/2))
        yp2 = int(sizey-(dy+absdy)/2)

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
