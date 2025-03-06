# -*- coding: utf-8 -*-
# Ready for python action!
import numpy as np
# import matplotlib.pylab as plt
# from numba import jit
import cupy as cp
import rasterio
from rasterio import CRS
from affine import Affine
import math

def write_output(output, name):
    """
    Write grid to .tiff file.
    ----
    Input:
    - dataset: Can be either a rasterio dataset (for rasters) or laspy dataset (for point clouds)
    - output (Array): the output grid, a numpy grid.
    - name (String): the name of the output file.
    - transform:
      a user defined rasterio Affine object, used for the transforming the pixel coordinates
      to spatial coordinates.
    - change_nodata (Boolean): true: use a no data value of -9999, false: use the datasets no data value
    """
    output_file = name

    output = np.squeeze(output)
    # Set the nodata value: use -9999 if nodata_value is True or dataset does not have nodata.
    crs = CRS.from_epsg(28992)
    nodata_value = -9999
    # transform = Affine(0.50, 0.00, 119300.00,
    #                    0.00, -0.50, 486500.00)
    transform = Affine(0.50, 0.00, 153100.0,
                       0.00, -0.50,  471200.0)

    # output the dataset
    with rasterio.open(output_file, 'w',
                       driver='GTiff',
                       height=output.shape[0],  # Assuming output is (rows, cols)
                       width=output.shape[1],
                       count=1,
                       dtype=np.float32,
                       crs=crs,
                       nodata=nodata_value,
                       transform=transform) as dst:
        dst.write(output, 1)
    print("File written to '%s'" % output_file)

def shadowingfunctionglobalradiation(a, amaxvalue, azimuth, altitude, scale, forsvf):
    #%This m.file calculates shadows on a DEM
    #% conversion
    degrees = np.pi/180.
    # if azimuth == 0.0:
        # azimuth = 0.000000000001
    azimuth *= degrees
    altitude *= degrees
    #% measure the size of the image
    sizex = a.shape[0]
    sizey = a.shape[1]
    if forsvf == 0:
        barstep = np.max([sizex, sizey])
        total = 100. / barstep #dlg.progressBar.setRange(0, barstep)
    #% initialise parameters
    f = a
    dx = 0.
    dy = 0.
    dz = 0.
    temp = cp.zeros((sizex, sizey), dtype=cp.float32)
    index = 1.
    #% other loop parameters
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

    isVert = ((pibyfour <= azimuth) & (azimuth < threetimespibyfour)) | \
             ((fivetimespibyfour <= azimuth) & (azimuth < seventimespibyfour))
    if isVert:
        ds = dssin
    else:
        ds = dscos

    #% main loop
    while (amaxvalue >= dz and np.abs(dx) < sizex and np.abs(dy) < sizey):
        if forsvf == 0:
            print(int(index * total))
            # dlg.progressBar.setValue(index)
        if isVert:
            dy = signsinazimuth * index
            dx = -1. * signcosazimuth * np.abs(np.round(index / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1. * signcosazimuth * index

        #% note: dx and dy represent absolute values while ds is an incremental value
        dz = ds *index * tanaltitudebyscale
        temp[0:sizex, 0:sizey] = 0.
        absdx = np.abs(dx)
        absdy = np.abs(dy)
        xc1 = (dx+absdx)/2.+1.
        xc2 = sizex+(dx-absdx)/2.
        yc1 = (dy+absdy)/2.+1.
        yc2 = sizey+(dy-absdy)/2.
        xp1 = -((dx-absdx)/2.)+1.
        xp2 = sizex-(dx+absdx)/2.
        yp1 = -((dy-absdy)/2.)+1.
        yp2 = sizey-(dy+absdy)/2.
        temp[int(xp1)-1:int(xp2), int(yp1)-1:int(yp2)] = a[int(xc1)-1:int(xc2), int(yc1)-1:int(yc2)]-dz
        # f = np.maximum(f, temp)  # bad performance in python3. Replaced with fmax
        f = cp.fmax(f, temp)
        index += 1.

    f = f-a
    f = cp.logical_not(f)
    sh = f.astype(cp.float32)

    return sh

# # @jit(nopython=True)
# # @profile
def shadowingfunction_20(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, aminvalue, trunkcheck, bush, forsvf):

    # This function casts shadows on buildings and vegetation units.
    # New capability to deal with pergolas 20210827

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
            print(int(index * total))  # dlg.progressBar.setValue(index)
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
        # print(dy, dx, dz)
        # print(f' xc1: {xc1}; xc2: {xc2}, yc1:  {yc1}, yc2: {yc2}, xp1: {xp1}, xp2: {xp2}, yp1: {yp1}, yp2: {yp2} ')
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

        # TO DO: THINK MORE ABOUT THIS LOGIC
        # vegsh2 = np.zeros_like(vegdem, dtype=float)
        # vegsh2[(fabovea) & (~gabovea)] = 1

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

    # savepath = "D:/Geomatics/thesis/shadetest/cupyoutput/"
    #
    # name = savepath + "vgog_" + str(round(azimuth, 2) )+ " " + str(round(altitude, 2)) + ".tif"
    # write_output(vegsh, name)
    #
    # name = savepath + "beog_" + str(round(azimuth, 2) )+ " " + str(round(altitude, 2)) + ".tif"
    # write_output(sh, name)

    return shadowresult

@profile
def shadowingfunction_20_cupy(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, aminvalue, trunkcheck, bush, forsvf):
    # Conversion
    degrees = np.pi / 180.0
    azimuth *= degrees
    altitude *= degrees
    # degrees = math.pi / 180.0
    # azimuth *= degrees
    # altitude *= degrees
    # factor = cp.float32(2.0)

    # Grid size
    sizex, sizey = a.shape[0], a.shape[1]

    # Initialize parameters
    dx = dy = dz = 0.0

    temp = cp.zeros((sizex, sizey), dtype=cp.float32)
    tempvegdem = cp.full((sizex, sizey), np.nan, dtype=cp.float32)
    tempvegdem2 = tempvegdem.copy()
    bushplant = bush > 1.0
    sh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vbshvegsh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vegsh = cp.array(bushplant, dtype=cp.float32)

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

    isVert = ((pibyfour <= azimuth < threetimespibyfour) or
              (fivetimespibyfour <= azimuth < seventimespibyfour))

    # Choose ds as scalar
    ds = dssin * tanaltitudebyscale if isVert else dscos * tanaltitudebyscale

    preva = a - ds
    i = 0.0
    # max_steps = int(np.floor(np.min(np.array([amaxvalue / ds, min(sizex, sizey)]))))

    while (amaxvalue >= dz) and (np.abs(dx)) <sizex and (np.abs(dy) < sizey):
        # if np.abs(dx) >= sizex:
        #     break
        # if np.abs(dy) >= sizey:
        #     break
        if isVert:
            dy = signsinazimuth * i
            dx = -signcosazimuth * np.abs(np.round(i / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(i * tanazimuth))
            dx = -signcosazimuth * i

        dz = ds * i

        tempvegdem.fill(np.nan)
        tempvegdem2.fill(np.nan)
        temp.fill(0.0)
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

        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz

        f = cp.fmax(f, temp)
        sh = cp.where(f > a, 1.0, 0.0)

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2]- dz
        fabovea = tempvegdem > a
        lastfabovea = tempvegdem > preva

        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        gabovea = tempvegdem2 > a
        lastgabovea = tempvegdem2 > preva

        vegsh2 = cp.add(cp.add(cp.add(fabovea, gabovea, dtype=cp.float32), lastfabovea, dtype=cp.float32),
                        lastgabovea, dtype=cp.float32)

        vegsh2 = cp.where(vegsh2 == 4.0, 0.0, vegsh2)
        vegsh2 = cp.where(vegsh2 > 0.0, 1.0, vegsh2)

        vegsh = cp.fmax(vegsh, vegsh2)
        vegsh = cp.where(vegsh * sh > 0.0, 0.0, vegsh)
        cp.add(vbshvegsh, vegsh, out=vbshvegsh)

        i += 1.0

    sh = 1.0 - sh
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

def shadowingfunction_20_cupy_forloop(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, aminvalue, trunkcheck, bush, forsvf):
    # Conversion
    degrees = np.pi / 180.0
    azimuth *= degrees
    altitude *= degrees
    # degrees = math.pi / 180.0
    # azimuth *= degrees
    # altitude *= degrees
    # factor = cp.float32(2.0)

    # Grid size
    sizex, sizey = a.shape[0], a.shape[1]

    # Initialize parameters
    dx = dy = dz = 0.0

    temp = cp.zeros((sizex, sizey), dtype=cp.float32)
    tempvegdem = cp.full((sizex, sizey), np.nan, dtype=cp.float32)
    tempvegdem2 = tempvegdem.copy()
    bushplant = bush > 1.0
    sh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vbshvegsh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vegsh = cp.array(bushplant, dtype=cp.float32)

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
    #
    # isVert = ((pibyfour <= azimuth) & (azimuth < threetimespibyfour)) | \
    #          ((fivetimespibyfour <= azimuth) & (azimuth < seventimespibyfour))
    # if isVert:
    #     ds = dssin
    # else:
    #     ds = dscos

    # pibyfour = math.pi / 4.0
    # threetimespibyfour = 3.0 * pibyfour
    # fivetimespibyfour = 5.0 * pibyfour
    # seventimespibyfour = 7.0 * pibyfour
    #
    # sinazimuth = math.sin(azimuth)
    # cosazimuth = math.cos(azimuth)
    # tanazimuth = math.tan(azimuth)
    # signsinazimuth = math.copysign(1.0, sinazimuth)
    # signcosazimuth = math.copysign(1.0, cosazimuth)
    # dssin = abs(1.0 / (1e-10 + sinazimuth))
    # dscos = abs(1.0 / (1e-10 + cosazimuth))
    # tanaltitudebyscale = math.tan(altitude) / scale

    isVert = ((pibyfour <= azimuth < threetimespibyfour) or
              (fivetimespibyfour <= azimuth < seventimespibyfour))

    # Choose ds as scalar
    ds = dssin * tanaltitudebyscale if isVert else dscos * tanaltitudebyscale

    preva = a - ds

    max_steps = int(np.floor(np.min(np.array([amaxvalue / ds, min(sizex, sizey)]))))


    for i in range(max_steps):
        if isVert:
            dy = signsinazimuth * i
            dx = -signcosazimuth * np.abs(np.round(i / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(i * tanazimuth))
            dx = -signcosazimuth * i

        dz = ds * i

        tempvegdem.fill(np.nan)
        tempvegdem2.fill(np.nan)
        temp.fill(0.0)
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

        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz

        f = cp.fmax(f, temp)
        sh = cp.where(f > a, 1.0, 0.0)

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2]- dz
        fabovea = tempvegdem > a
        lastfabovea = tempvegdem > preva

        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        gabovea = tempvegdem2 > a
        lastgabovea = tempvegdem2 > preva

        vegsh2 = cp.add(cp.add(cp.add(fabovea, gabovea, dtype=cp.float32), lastfabovea, dtype=cp.float32),
                        lastgabovea, dtype=cp.float32)

        vegsh2 = cp.where(vegsh2 == 4.0, 0.0, vegsh2)
        vegsh2 = cp.where(vegsh2 > 0.0, 1.0, vegsh2)

        vegsh = cp.fmax(vegsh, vegsh2)
        vegsh = cp.where(vegsh * sh > 0.0, 0.0, vegsh)
        cp.add(vbshvegsh, vegsh, out=vbshvegsh)

    sh = 1.0 - sh
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

@profile
def shadowingfunction_20_cupy_vector(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, aminvalue, trunkcheck, bush, forsvf):
    # Conversion
    degrees = np.pi / 180.0
    azimuth *= degrees
    altitude *= degrees

    # Grid size
    sizex, sizey = a.shape

    # Initialize parameters
    temp = cp.zeros((sizex, sizey), dtype=cp.float32)
    tempvegdem = cp.zeros((sizex, sizey), dtype=cp.float32)
    tempvegdem2 = cp.zeros((sizex, sizey), dtype=cp.float32)
    bushplant = bush > 1.0
    sh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vbshvegsh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vegsh = cp.array(bushplant, dtype=cp.float32)

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
    tanaltitudebyscale = np.tan(altitude) / scale

    isVert = ((pibyfour <= azimuth) & (azimuth < threetimespibyfour)) | \
             ((fivetimespibyfour <= azimuth) & (azimuth < seventimespibyfour))
    if isVert:
        ds = dssin * tanaltitudebyscale
    else:
        ds = dscos * tanaltitudebyscale

    max_steps = int(np.floor(np.min(np.array([amaxvalue / ds, min(sizex, sizey)]))))

    index = cp.arange(1, max_steps + 1)
    dx = cp.where(isVert, -signcosazimuth * cp.abs(cp.round(index / tanazimuth)), -signcosazimuth * index)
    dy = cp.where(isVert, signsinazimuth * index, signsinazimuth * cp.abs(cp.round(index * tanazimuth)))
    dz = (ds * index) * tanaltitudebyscale
    preva = a - ds * tanaltitudebyscale

    absdx = cp.abs(dx)
    absdy = cp.abs(dy)

    xc1 = ((dx + absdx) / 2.0).astype(cp.int32)
    xc2 = (sizex + (dx - absdx) / 2.0).astype(cp.int32)

    yc1 = ((dy + absdy) / 2.0).astype(cp.int32)
    yc2 = (sizey + (dy - absdy) / 2.0).astype(cp.int32)

    xp1 = (-(dx - absdx) / 2.0).astype(cp.int32)
    xp2 = (sizex - (dx + absdx) / 2.0).astype(cp.int32)

    yp1 = (-(dy - absdy) / 2.0).astype(cp.int32)
    yp2 = (sizey - (dy + absdy) / 2.0).astype(cp.int32)
    input_points = cp.stack([xc1, xc2, yc1, yc2, xp1, xp2, yp1, yp2, dz], axis=1)


    for points in input_points:
        tempvegdem.fill(np.nan)
        tempvegdem2.fill(np.nan)
        temp.fill(0.0)

        temp[points[0]:points[1], points[2]:points[3]] = a[points[4]:points[5], points[6]:points[7]] - points[8]
        f = cp.fmax(f, temp)
        sh = cp.where(f > a, 1.0, 0.0)

        tempvegdem[points[0]:points[1], points[2]:points[3]] = vegdem[points[4]:points[5], points[6]:points[7]] - points[8]
        fabovea = tempvegdem > a
        lastfabovea = tempvegdem > preva

        tempvegdem2[points[0]:points[1], points[2]:points[3]] = vegdem2[points[4]:points[5], points[6]:points[7]] - points[8]
        gabovea = tempvegdem2 > a
        lastgabovea = tempvegdem2 > preva

        vegsh2 = cp.add(cp.add(cp.add(fabovea, gabovea, dtype=cp.float32), lastfabovea, dtype=cp.float32),
                        lastgabovea, dtype=cp.float32)

        vegsh2 = cp.where(vegsh2 == 4.0, 0.0, vegsh2)
        vegsh2 = cp.where(vegsh2 > 0.0, 1.0, vegsh2)

        vegsh = cp.fmax(vegsh, vegsh2)
        vegsh = cp.where(vegsh * sh > 0.0, 0.0, vegsh)
        cp.add(vbshvegsh, vegsh, out=vbshvegsh)


    sh = 1.0 - sh
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

def shadowingfunction_20_3d(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, forsvf):
    # Conversion
    degrees = np.pi / 180.0
    azimuth *= degrees
    altitude *= degrees
    # factor = cp.float32(2.0)

    # Grid size
    sizex, sizey = a[0].shape[0], a[0].shape[1]

    # Initialize parameters
    dx = dy = dz = 0.0

    temp = cp.zeros((sizex, sizey), dtype=cp.float32)
    tempvegdem = cp.full((sizex, sizey), np.nan, dtype=cp.float32)
    tempvegdem2 = cp.full((sizex, sizey), np.nan, dtype=cp.float32)

    bushplant = bush > 1.0
    vbshvegsh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vegsh = cp.array(bushplant, dtype=cp.float32)
    dsm_ground = a[0]

    temp_firstgap = cp.full((sizex, sizey), np.nan)
    temp_secondlayer = cp.full((sizex, sizey), np.nan)
    sh = cp.zeros((sizex, sizey)) #shadows from buildings
    sh2 = cp.zeros((sizex, sizey))

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

    isVert = ((pibyfour <= azimuth) & (azimuth < threetimespibyfour)) | \
             ((fivetimespibyfour <= azimuth) & (azimuth < seventimespibyfour))
    if isVert:
        ds = dssin
    else:
        ds = dscos

    preva = a[0] - ds * tanaltitudebyscale

    index = 0.0

    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
        if isVert:
            dy = signsinazimuth * index
            dx = -signcosazimuth * np.abs(np.round(index / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -signcosazimuth * index

        dz = (ds * index) * tanaltitudebyscale

        tempvegdem.fill(np.nan)
        tempvegdem2.fill(np.nan)
        temp.fill(0.0)

        temp_firstgap[:] = np.nan
        temp_secondlayer[:] = np.nan

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

        # Building Part
        temp[xp1:xp2, yp1:yp2] = a[0][xc1:xc2, yc1:yc2] - dz
        temp_firstgap[xp1:xp2, yp1:yp2] = a[1][xc1:xc2, yc1:yc2] - dz
        temp_secondlayer[xp1:xp2, yp1:yp2] = a[2][xc1:xc2, yc1:yc2] - dz
        # Building part
        dsm_ground = cp.fmax(dsm_ground, temp)

        sh = (dsm_ground > a[0]).astype(cp.float32)

        gapabovea = temp_firstgap > a[0]
        layerabovea = temp_secondlayer > a[0]

        prevgapabovea = temp_firstgap > preva
        prevlayerabovea = temp_secondlayer > preva

        sh2_temp = cp.add(cp.add(cp.add(layerabovea, gapabovea, dtype=float), prevgapabovea, dtype=float),
                          prevlayerabovea, dtype=float)

        sh2_temp = cp.where(sh2_temp == 4.0, 0.0, sh2_temp)
        sh2_temp = cp.where(sh2_temp > 0.0, 1.0, sh2_temp)
        sh2 = cp.fmax(sh2, sh2_temp)
        # Vegetation Part
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
        vegsh = cp.where((vegsh * sh > 0.0) | (vegsh * sh2 > 0.0), 0.0, vegsh)
        cp.add(vbshvegsh, vegsh)


    sh = cp.fmax(sh, sh2)
    sh = 1.0 - sh
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


def shadowingfunction_20_3d_mult(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, forsvf):
    # Conversion
    degrees = np.pi / 180.0
    azimuth *= degrees
    altitude *= degrees
    # factor = cp.float32(2.0)

    # Grid size
    sizex, sizey = a[0].shape[0], a[0].shape[1]

    # Initialize parameters
    dx = dy = dz = 0.0

    temp = cp.zeros((sizex, sizey), dtype=cp.float32)
    tempvegdem = cp.full((sizex, sizey), np.nan, dtype=cp.float32)
    tempvegdem2 = cp.full((sizex, sizey), np.nan, dtype=cp.float32)

    bushplant = bush > 1.0
    vbshvegsh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vegsh = cp.array(bushplant, dtype=cp.float32)
    dsm_ground = a[0]

    temp_firstgap = cp.full((sizex, sizey), np.nan)
    temp_secondlayer = cp.full((sizex, sizey), np.nan)

    temp_secondgap = temp_firstgap.copy()
    temp_thirdlayer = temp_firstgap.copy()

    # temp_thirdgap = temp_firstgap.copy()
    # temp_fourthlayer = temp_firstgap.copy()
    #
    # temp_fourthgap = temp_firstgap.copy()
    # temp_fifthlayer = temp_firstgap.copy()

    sh = cp.zeros((sizex, sizey)) #shadows from buildings
    sh2 = cp.zeros((sizex, sizey))

    sh3 = sh2.copy()
    sh4 = sh2.copy()
    sh5 = sh2.copy()

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

    isVert = ((pibyfour <= azimuth) & (azimuth < threetimespibyfour)) | \
             ((fivetimespibyfour <= azimuth) & (azimuth < seventimespibyfour))
    if isVert:
        ds = dssin
    else:
        ds = dscos

    preva = a[0] - ds * tanaltitudebyscale

    index = 0.0

    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
        if isVert:
            dy = signsinazimuth * index
            dx = -signcosazimuth * np.abs(np.round(index / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -signcosazimuth * index

        dz = (ds * index) * tanaltitudebyscale

        tempvegdem.fill(np.nan)
        tempvegdem2.fill(np.nan)
        temp.fill(0.0)

        temp_firstgap[:] = np.nan
        temp_secondlayer[:] = np.nan

        temp_secondgap[:] = np.nan
        temp_thirdlayer[:] =  np.nan
        #
        # temp_thirdgap[:] = np.nan
        # temp_fourthlayer[:] =  np.nan
        #
        # temp_fourthgap[:] =  np.nan
        # temp_fifthlayer[:] =  np.nan

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

        # Building Part
        temp[xp1:xp2, yp1:yp2] = a[0][xc1:xc2, yc1:yc2] - dz
        temp_firstgap[xp1:xp2, yp1:yp2] = a[1][xc1:xc2, yc1:yc2] - dz
        temp_secondlayer[xp1:xp2, yp1:yp2] = a[2][xc1:xc2, yc1:yc2] - dz
        temp_secondgap[xp1:xp2, yp1:yp2] = a[3][xc1:xc2, yc1:yc2] - dz
        temp_thirdlayer[xp1:xp2, yp1:yp2] = a[4][xc1:xc2, yc1:yc2] - dz
        # temp_thirdgap[xp1:xp2, yp1:yp2] = a[5][xc1:xc2, yc1:yc2] - dz
        # temp_fourthlayer[xp1:xp2, yp1:yp2] = a[6][xc1:xc2, yc1:yc2] - dz
        # temp_fourthgap[xp1:xp2, yp1:yp2] = a[7][xc1:xc2, yc1:yc2] - dz
        # temp_fifthlayer[xp1:xp2, yp1:yp2] = a[8][xc1:xc2, yc1:yc2] - dz

        dsm_ground = cp.fmax(dsm_ground, temp)

        sh = (dsm_ground > a[0]).astype(cp.float32)

        # first gap part
        gapabovea = temp_firstgap > a[0]
        layerabovea = temp_secondlayer > a[0]

        prevgapabovea = temp_firstgap > preva
        prevlayerabovea = temp_secondlayer > preva

        sh2_temp = cp.add(cp.add(cp.add(layerabovea, gapabovea, dtype=float), prevgapabovea, dtype=float),
                          prevlayerabovea, dtype=float)

        sh2_temp = cp.where(sh2_temp == 4.0, 0.0, sh2_temp)
        sh2_temp = cp.where(sh2_temp > 0.0, 1.0, sh2_temp)
        sh2 = cp.fmax(sh2, sh2_temp)

        # second gap part
        gapabovea3 = temp_secondgap > a[0]
        layerabovea3 = temp_thirdlayer > a[0]

        prevgapabovea3 = temp_secondgap > preva
        prevlayerabovea3 = temp_thirdlayer > preva

        sh3_temp = cp.add(cp.add(cp.add(layerabovea3, gapabovea3, dtype=float), prevgapabovea3, dtype=float),
                          prevlayerabovea3, dtype=float)

        sh3_temp = cp.where(sh3_temp == 4.0, 0.0, sh3_temp)
        sh3_temp = cp.where(sh3_temp > 0.0, 1.0, sh3_temp)
        sh3 = cp.fmax(sh3, sh3_temp)

        # # third gap part
        # gapabovea4 = temp_thirdgap > a[0]
        # layerabovea4 = temp_fourthlayer > a[0]
        #
        # prevgapabovea4 = temp_thirdgap > preva
        # prevlayerabovea4 = temp_fourthlayer > preva
        #
        # sh4_temp = cp.add(cp.add(cp.add(layerabovea4, gapabovea4, dtype=float), prevgapabovea4, dtype=float),
        #                   prevlayerabovea4, dtype=float)
        #
        # sh4_temp = cp.where(sh4_temp == 4.0, 0.0, sh4_temp)
        # sh4_temp = cp.where(sh4_temp > 0.0, 1.0, sh4_temp)
        # sh4 = cp.fmax(sh4, sh4_temp)

        # # fourth gap part
        # gapabovea5 = temp_fourthgap > a[0]
        # layerabovea5 = temp_fifthlayer > a[0]
        #
        # prevgapabovea5 = temp_fourthgap > preva
        # prevlayerabovea5 = temp_fifthlayer > preva
        #
        # sh5_temp = cp.add(cp.add(cp.add(layerabovea5, gapabovea5, dtype=float), prevgapabovea5, dtype=float),
        #                   prevlayerabovea5, dtype=float)
        #
        # sh5_temp = cp.where(sh5_temp == 4.0, 0.0, sh5_temp)
        # sh5_temp = cp.where(sh5_temp > 0.0, 1.0, sh5_temp)
        #
        # sh5 = cp.fmax(sh5, sh5_temp)

        # Vegetation Part
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
        vegsh = cp.where((vegsh * sh > 0.0) | (vegsh * sh2 > 0.0), 0.0, vegsh)
        cp.add(vbshvegsh, vegsh)

        index += 1.0

    # sh = cp.fmax(cp.fmax(cp.fmax(cp.fmax(sh, sh2), sh3), sh4), sh5)
    sh = cp.fmax(cp.fmax(sh, sh2), sh3)
    sh = 1.0 - sh
    vbshvegsh[vbshvegsh > 0.0] = 1.0
    vbshvegsh -= vegsh
    vegsh = 1.0 - vegsh
    vbshvegsh = 1.0 - vbshvegsh


    name = "D:/Geomatics/thesis/3dthings/testcase2_output/multgap_" + str(round(azimuth, 2)) + "   " + str(round(altitude, 2)) + ".tif"
    write_output(sh.get(), name)

    shadowresult = {
        'sh': sh,
        'vegsh': vegsh,
        'vbshvegsh': vbshvegsh
    }
    return shadowresult