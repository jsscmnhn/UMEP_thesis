# -*- coding: utf-8 -*-
# Ready for python action!
import numpy as np
# import matplotlib.pylab as plt
# from numba import jit
import cupy as cp
import rasterio
from rasterio import CRS
from affine import Affine

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
    transform = Affine(0.50, 0.00, 119300.00,
                       0.00, -0.50, 486500.00)

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

# def shadowingfunction_20(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, forsvf):
#     amaxvalue = a.max()
#     pibyfour = np.pi/4.
#     threetimespibyfour = 3.*pibyfourcd thesis
#     fivetimespibyfour = 5.*pibyfour
#     seventimespibyfour = 7.*pibyfour
#     sinazimuth = np.sin(azimuth)
#     cosazimuth = np.cos(azimuth)
#     tanazimuth = np.tan(azimuth)
#     signsinazimuth = np.sign(sinazimuth)
#     signcosazimuth = np.sign(cosazimuth)
#     dssin = np.abs((1./sinazimuth))
#     dscos = np.abs((1./cosazimuth))
#     tanaltitudebyscale = np.tan(altitude) / scale
#
#     # Simplified conversion and initializations
#     degrees = np.pi / 180.
#     azimuth = azimuth * degrees
#     altitude = altitude * degrees
#
#     # Grid size
#     sizex, sizey = a.shape
#
#     # Initialize shadow arrays
#     sh = np.zeros((sizex, sizey))  # Shadows from buildings
#     vegsh = np.zeros((sizex, sizey))  # Vegetation shadows
#     vbshvegsh = np.zeros((sizex, sizey))  # Vegetation blocking building shadows
#     f = a  # Starting with the DEM
#
#     # Precompute constants
#     sinazimuth = np.sin(azimuth)
#     cosazimuth = np.cos(azimuth)
#     tanazimuth = np.tan(azimuth)
#     tanaltitudebyscale = np.tan(altitude) / scale
#     dssin = np.abs(1. / sinazimuth)
#     dscos = np.abs(1. / cosazimuth)
#
#     # Loop over shadowcasting grid    # Initialize dx, dy, dz, and index
#     dx = dy = dz = 0
#     index = 0
#
#     # Main loop for shadowcasting
#     while amaxvalue >= dz and np.abs(dx) < sizex and np.abs(dy) < sizey:
#         # Main logic
#         if forsvf == 0:
#             print(f"Progress: {index}%")
#
#         # Calculate dx, dy, dz (shadow movement)
#         if (pibyfour <= azimuth < threetimespibyfour or fivetimespibyfour <= azimuth < seventimespibyfour):
#             dy = signsinazimuth * index
#             dx = -1. * signcosazimuth * np.abs(np.round(index / tanazimuth))
#             ds = dssin
#         else:
#             dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
#             dx = -1. * signcosazimuth * index
#             ds = dscos
#
#         dz = (ds * index) * tanaltitudebyscale
#
#         # Create temporary arrays for vegetation and building shadows
#         temp = np.zeros_like(a)  # Reuse temp array for shadows
#
#         # Calculate indices for array slicing
#         absdx = np.abs(dx)
#         absdy = np.abs(dy)
#
#         xc1 = int((dx + absdx) / 2. + 1)
#         xc2 = int(sizex + (dx - absdx) / 2)
#         yc1 = int((dy + absdy) / 2. + 1)
#         yc2 = int(sizey + (dy - absdy) / 2)
#
#         xp1 = int(-((dx - absdx) / 2.) + 1)
#         xp2 = int(sizex - (dx + absdx) / 2)
#         yp1 = int(-((dy - absdy) / 2.) + 1)
#         yp2 = int(sizey - (dy + absdy) / 2)
#
#         # Update shadow and vegetation arrays
#         temp[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
#         temp[xp1:xp2, yp1:yp2] = np.maximum(temp, vegdem2[xc1:xc2, yc1:yc2] - dz)
#         temp[xp1:xp2, yp1:yp2] = np.maximum(temp, a[xc1:xc2, yc1:yc2] - dz)
#
#         # Apply maximum shadow to the f array
#         f = np.fmax(f, temp)  # Move building shadow
#
#         # Update shadow results for buildings and vegetation
#         sh[(f > a)] = 1
#         sh[(f <= a)] = 0
#
#         # Vegetation above the DEM
#         fabovea = tempvegdem > a
#         gabovea = tempvegdem2 > a
#
#         # Additional conditions for vegetation (pergola case, etc.)
#         templastfabovea[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dzprev
#         templastgabovea[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dzprev
#
#         # Check if vegetation is above the DEM
#         lastfabovea = templastfabovea > a
#         lastgabovea = templastgabovea > a
#
#         # Update dzprev for next iteration
#         dzprev = dz
#
#         # Combine all vegetation layers
#         vegsh2 = np.add(np.add(np.add(fabovea, gabovea, dtype=float), lastfabovea, dtype=float), lastgabovea, dtype=float)
#         vegsh2[vegsh2 == 4] = 0.
#         vegsh2[vegsh2 > 0] = 1
#
#         # Update vegetation shadow array
#         vegsh = np.fmax(vegsh, vegsh2)
#
#         # Remove shadows behind buildings due to vegetation
#         vegsh[(vegsh * sh > 0.)] = 0
#         vbshvegsh = vegsh + vbshvegsh
#
#         # Increment index
#         index += 1
#
#     # Finalize shadow results
#     sh = 1. - sh
#     vbshvegsh[(vbshvegsh > 0.)] = 1.
#     vbshvegsh = vbshvegsh - vegsh
#     vegsh = 1. - vegsh
#     vbshvegsh = 1. - vbshvegsh
#
#     # Return results
#     shadowresult = {'sh': sh, 'vegsh': vegsh, 'vbshvegsh': vbshvegsh}
#     return shadowresult
#
def shadowingfunctionglobalradiation(a, azimuth, altitude, scale, forsvf):
    #%This m.file calculates shadows on a DEM
    #% conversion
    degrees = np.pi/180.
    # if azimuth == 0.0:
        # azimuth = 0.000000000001
    azimuth = np.dot(azimuth, degrees)
    altitude = np.dot(altitude, degrees)
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
    temp = np.zeros((sizex, sizey))
    index = 1.
    #% other loop parameters
    amaxvalue = a.max()
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
    #% main loop
    while (amaxvalue >= dz and np.abs(dx) < sizex and np.abs(dy) < sizey):
        if forsvf == 0:
            print(int(index * total))
            # dlg.progressBar.setValue(index)
    #while np.logical_and(np.logical_and(amaxvalue >= dz, np.abs(dx) <= sizex), np.abs(dy) <= sizey):(np.logical_and(amaxvalue >= dz, np.abs(dx) <= sizex), np.abs(dy) <= sizey):
        #if np.logical_or(np.logical_and(pibyfour <= azimuth, azimuth < threetimespibyfour), np.logical_and(fivetimespibyfour <= azimuth, azimuth < seventimespibyfour)):
        if (pibyfour <= azimuth and azimuth < threetimespibyfour or fivetimespibyfour <= azimuth and azimuth < seventimespibyfour):
            dy = signsinazimuth * index
            dx = -1. * signcosazimuth * np.abs(np.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1. * signcosazimuth * index
            ds = dscos

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
        f = np.fmax(f, temp)
        index += 1.

    f = f-a
    f = np.logical_not(f)
    sh = np.double(f)

    return sh

# # @jit(nopython=True)
# # @profile
def shadowingfunction_20(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, aminvalue, trunkcheck, bush, forsvf):

    # This function casts shadows on buildings and vegetation units.
    # New capability to deal with pergolas 20210827
    # stepChange = 1. if altitude <= 10. else 2.
    stepChange = 1

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
    azimuth = azimuth * degrees
    altitude = altitude * degrees
    # factor = cp.float32(2.0)
    zero = cp.float32(0.0)

    # Grid size
    sizex, sizey = a.shape[0], a.shape[1]

    # Initialize parameters
    dx = dy = dz = 0.0

    temp = cp.zeros((sizex, sizey), dtype=cp.float32)
    tempvegdem = cp.zeros((sizex, sizey), dtype=cp.float32)
    tempvegdem2 = cp.zeros((sizex, sizey), dtype=cp.float32)
    templastfabovea = cp.zeros((sizex, sizey), dtype=cp.float32)
    templastgabovea = cp.zeros((sizex, sizey), dtype=cp.float32)
    bushplant = bush > 1.0
    sh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vbshvegsh = cp.zeros((sizex, sizey), dtype=cp.float32)
    vegsh = cp.array(bushplant, dtype=cp.float32)
    # amaxvalue = cp.float32(amaxvalue)

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

    isVert = ((pibyfour <= azimuth) & (azimuth < threetimespibyfour)) | \
             ((fivetimespibyfour <= azimuth) & (azimuth < seventimespibyfour))
    if isVert:
        ds = dssin
    else:
        ds = dscos

    # preva = a + ds

    index = 0.0
    dzprev = 0.0

    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
        if isVert:
            dy = signsinazimuth * index
            dx = -signcosazimuth * np.abs(np.round(index / tanazimuth))
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -signcosazimuth * index

        dz = (ds * index) * tanaltitudebyscale

        tempvegdem.fill(zero)
        tempvegdem2.fill(zero)
        temp.fill(zero)

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
        vegdem_slice = vegdem[xc1:xc2, yc1:yc2]
        vegdem2_slice = vegdem2[xc1:xc2, yc1:yc2]
        tempvegdem[xp1:xp2, yp1:yp2] = vegdem_slice - dz
        fabovea = tempvegdem > a
        templastfabovea[xp1:xp2, yp1:yp2] = vegdem_slice - dzprev
        lastfabovea = templastfabovea > a

        # if isTrunk:
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2_slice - dz
        gabovea = tempvegdem2 > a

        templastgabovea[xp1:xp2, yp1:yp2] = vegdem2_slice - dzprev
        lastgabovea = templastgabovea > a

        vegsh2 = cp.add(cp.add(cp.add(fabovea, gabovea, dtype=cp.float32), lastfabovea, dtype=cp.float32),
                        lastgabovea, dtype=cp.float32)

        vegsh2 = cp.where(vegsh2 == 4.0, 0.0, cp.where(vegsh2 > 0.0, 1.0, vegsh2))

        # else:
        #     vegsh2 = (fabovea | lastfabovea).astype(cp.float32)

        vegsh = cp.fmax(vegsh, vegsh2)
        vegsh = cp.where(vegsh * sh > 0.0, 0.0, vegsh)
        cp.add(vbshvegsh, vegsh, out=vbshvegsh)

        dzprev = dz
        index += 1.0

    sh = 1.0 - sh
    vbshvegsh[vbshvegsh > 0.0] = 1.0
    vbshvegsh -= vegsh
    vegsh = 1.0 - vegsh
    vbshvegsh = 1.0 - vbshvegsh

    shadowresult = {
        'sh': sh.get(),
        'vegsh': vegsh.get(),
        'vbshvegsh': vbshvegsh.get()
    }

    # savepath = "D:/Geomatics/thesis/shadetest/cupyoutput/"
    #
    # name = savepath + "vgne_" + str(round(azimuth, 2) )+ " " + str(round(altitude, 2)) + ".tif"
    # write_output(vegsh.get(), name)
    #
    # name = savepath + "bene_" + str(round(azimuth, 2) )+ " " + str(round(altitude, 2)) + ".tif"
    # write_output(sh.get(), name)

    return shadowresult

def shadowingfunction_20_old(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, dlg, forsvf):
    #% This function casts shadows on buildings and vegetation units
    #% conversion
    degrees = np.pi/180.
    if azimuth == 0.0:
        azimuth = 0.000000000001
    azimuth = np.dot(azimuth, degrees)
    altitude = np.dot(altitude, degrees)
    #% measure the size of the image
    sizex = a.shape[0]
    sizey = a.shape[1]
    #% initialise parameters
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
    index = 1

    #% main loop
    while (amaxvalue >= dz and np.abs(dx) < sizex and np.abs(dy) < sizey):
        if forsvf == 0:
            dlg.progressBar.setValue(index)
        if (pibyfour <= azimuth and azimuth < threetimespibyfour or fivetimespibyfour <= azimuth and azimuth < seventimespibyfour):
            dy = signsinazimuth * index
            dx = -1. * signcosazimuth * np.abs(np.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1. * signcosazimuth * index
            ds = dscos
        #% note: dx and dy represent absolute values while ds is an incremental value
        dz = np.dot(np.dot(ds, index), tanaltitudebyscale)
        tempvegdem[0:sizex, 0:sizey] = 0.
        tempvegdem2[0:sizex, 0:sizey] = 0.
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
        tempvegdem[int(xp1)-1:int(xp2), int(yp1)-1:int(yp2)] = vegdem[int(xc1)-1:int(xc2), int(yc1)-1:int(yc2)]-dz
        tempvegdem2[int(xp1)-1:int(xp2), int(yp1)-1:int(yp2)] = vegdem2[int(xc1)-1:int(xc2), int(yc1)-1:int(yc2)]-dz
        temp[int(xp1)-1:int(xp2), int(yp1)-1:int(yp2)] = a[int(xc1)-1:int(xc2), int(yc1)-1:int(yc2)]-dz
        # f = np.maximum(f, temp) # bad performance in python3. Replaced with fmax
        f = np.fmax(f, temp)
        sh[(f > a)] = 1.
        sh[(f <= a)] = 0.
        #%Moving building shadow
        fabovea = tempvegdem > a
        #%vegdem above DEM
        gabovea = tempvegdem2 > a
        #%vegdem2 above DEM
        # vegsh2 = np.float(fabovea)-np.float(gabovea)
        vegsh2 = np.subtract(fabovea, gabovea, dtype=float)
        # vegsh = np.maximum(vegsh, vegsh2) # bad performance in python3. Replaced with fmax
        vegsh = np.fmax(vegsh, vegsh2)
        vegsh[(vegsh*sh > 0.)] = 0.
        #% removing shadows 'behind' buildings
        vbshvegsh = vegsh+vbshvegsh
        #% vegsh at high sun altitudes
        if index == 1.:
            firstvegdem = tempvegdem-temp
            firstvegdem[(firstvegdem <= 0.)] = 1000.
            vegsh[(firstvegdem < dz)] = 1.
            vegsh = vegsh*(vegdem2 > a)
            vbshvegsh = np.zeros((sizex, sizey))

        #% Bush shadow on bush plant
        if np.logical_and(bush.max() > 0., np.max((fabovea*bush)) > 0.):
            tempbush[0:sizex, 0:sizey] = 0.
            tempbush[int(xp1)-1:int(xp2), int(yp1)-1:int(yp2)] = bush[int(xc1)-1:int(xc2),int(yc1)-1:int(yc2)]-dz
            # g = np.maximum(g, tempbush) # bad performance in python3. Replaced with fmax
            g = np.fmax(g, tempbush)
            g *= bushplant
        index += 1.

    sh = 1.-sh
    vbshvegsh[(vbshvegsh > 0.)] = 1.
    vbshvegsh = vbshvegsh-vegsh

    if bush.max() > 0.:
        g = g-bush
        g[(g > 0.)] = 1.
        g[(g < 0.)] = 0.
        vegsh = vegsh-bushplant+g
        vegsh[(vegsh<0.)] = 0.

    vegsh[(vegsh > 0.)] = 1.
    vegsh = 1.-vegsh
    vbshvegsh = 1.-vbshvegsh

    shadowresult = {'sh': sh, 'vegsh': vegsh, 'vbshvegsh': vbshvegsh}

    return shadowresult