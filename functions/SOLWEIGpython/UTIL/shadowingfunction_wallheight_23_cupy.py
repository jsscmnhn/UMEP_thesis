from __future__ import division
import numpy as np
import cupy as cp
from rasterio import CRS, Affine
import rasterio


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


def shadowingfunction_wallheight_23(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, walls, aspect):

    # conversion
    degrees = np.pi / 180.
    azimuth *= degrees
    altitude *= degrees

    # measure the size of the image
    sizex, sizey = a.shape[0], a.shape[1]

    # initialise parameters
    dx = dy = dz = 0.0

    temp = cp.full((sizex, sizey), np.nan, dtype=cp.float32)
    tempvegdem = cp.copy(temp)
    tempvegdem2 = cp.full((sizex, sizey), np.nan, dtype=cp.float32)
    bushplant = bush > 1.0
    sh = cp.zeros((sizex, sizey), dtype=cp.float32)

    vegsh = cp.array(bushplant, dtype=cp.float32)

    f = cp.array(a, dtype=cp.float32)

    shvoveg = cp.copy(vegdem)  # for vegetation shadowvolume

    wallbol = cp.array((walls > 0), dtype=cp.float32)

    # other loop parameters
    pibyfour = np.pi / 4
    threetimespibyfour = 3 * pibyfour
    fivetimespibyfour = 5 * pibyfour
    seventimespibyfour = 7 * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs(1 / sinazimuth)
    dscos = np.abs(1 / cosazimuth)
    tanaltitudebyscale = np.tan(altitude) / scale

    isVert = ((pibyfour <= azimuth) & (azimuth < threetimespibyfour)) | \
             ((fivetimespibyfour <= azimuth) & (azimuth < seventimespibyfour))
    if isVert:
        ds = dssin
    else:
        ds = dscos

    preva = a - ds * tanaltitudebyscale
    index = 0.0

    # main loop
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

        absdx = np.abs(dx)
        absdy = np.abs(dy)
        xc1 = int((dx + absdx) / 2)
        xc2 = int(sizex + (dx - absdx) / 2)
        yc1 = int((dy + absdy) / 2)
        yc2 = int(sizey + (dy - absdy) / 2)
        xp1 = -int((dx - absdx) / 2)
        xp2 = int(sizex - (dx + absdx) / 2)
        yp1 = -int((dy - absdy) / 2)
        yp2 = int(sizey - (dy + absdy) / 2)

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz

        f = cp.fmax(f, temp)
        sh = cp.where(f > a, 1.0, 0.0)
        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
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

        index += 1.0

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

    vegsh[vegsh > 0] = 1
    shvoveg = (shvoveg - a) * vegsh  # Vegetation shadow volume
    vegsh = 1 - vegsh

    # wall shadows
    shvo = f - a  # building shadow volume
    facesun = cp.logical_and(facesh + (walls > 0).astype(float) == 1, walls > 0).astype(float)
    wallsun = cp.copy(walls - shvo)
    wallsun[wallsun < 0] = 0
    wallsun[facesh == 1] = 0  # Removing walls in "self"-shadow
    wallsh = cp.copy(walls - wallsun)

    wallshve = shvoveg * wallbol
    wallshve = wallshve - wallsh
    wallshve[wallshve < 0] = 0
    id = cp.where(wallshve > walls)
    wallshve[id] = walls[id]
    wallsun = wallsun - wallshve  # problem with wallshve only
    id = cp.where(wallsun < 0)
    wallshve[id] = 0
    wallsun[id] = 0

    return vegsh, sh, wallsh, wallsun, wallshve, facesh, facesun


def shadowingfunction_23_3d(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, walls, aspect):
    # Conversion
    degrees = np.pi / 180.0
    azimuth *= degrees
    altitude *= degrees
    # factor = cp.float32(2.0)
    # Grid size
    sizex, sizey = a[0].shape[0], a[0].shape[1]

    # Initialize parameters
    dx = dy = dz = 0.0
    num_layers = len(a)
    temp = cp.zeros((sizex, sizey), dtype=cp.float32)
    temp_layers = cp.full((num_layers - 1, sizex, sizey), np.nan, dtype=cp.float32)
    tempvegdem = cp.full((sizex, sizey), np.nan, dtype=cp.float32)
    tempvegdem2 = cp.full((sizex, sizey), np.nan, dtype=cp.float32)

    bushplant = bush > 1.0
    vegsh = cp.array(bushplant, dtype=cp.float32)
    dsm_ground = a[0]


    sh = cp.zeros((sizex, sizey),  dtype=cp.float32) #shadows from buildings

    num_combinations = (num_layers - 1) // 2
    sh_stack = cp.full((num_combinations, sizex, sizey), np.nan, dtype=cp.float32)

    shvoveg = cp.copy(vegdem)

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

        temp_layers[:] = np.nan

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
        temp_layers[:, xp1:xp2, yp1:yp2] = a[1:num_layers, xc1:xc2, yc1:yc2] - dz

        dsm_ground = cp.fmax(dsm_ground, temp)

        sh = (dsm_ground > a[0]).astype(cp.float32)
        for i in range(0, num_layers - 1, 2):
            # first gap part
            gap_layer_index = i
            layer_index = i + 1

            # Get gap and layer arrays for the current iteration
            gapabovea = temp_layers[gap_layer_index] > a[0]
            layerabovea = temp_layers[layer_index] > a[0]
            prevgapabovea = temp_layers[gap_layer_index] > preva
            prevlayerabovea = temp_layers[layer_index] > preva
            sh_temp = cp.add(cp.add(cp.add(layerabovea, gapabovea, dtype=float), prevgapabovea, dtype=float),
                              prevlayerabovea, dtype=float)
            # for i in range(0, num_layers - 1):
            #     name = f"D:/Geomatics/thesis/3D_solweig/shade_debug/add_check_{i}_{index}" + str(
            #         round(azimuth, 2)) + "_" + str(round(altitude, 2)) + ".tif"
            #     write_output(sh_temp.get(), name)

            sh_temp = cp.where(sh_temp == 4.0, 0.0, sh_temp)
            sh_temp = cp.where(sh_temp > 0.0, 1.0, sh_temp)

            sh_stack[i // 2] = cp.fmax(sh_stack[i // 2], sh_temp)

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
        vegsh = cp.where((vegsh * sh > 0.0), 0.0, vegsh)
        vegsh = cp.where(cp.any(sh_stack * vegsh > 0.0, axis=0), 0.0, vegsh)

        index += 1.0

    # for i in range(0, num_combinations):
    #
    #     name = f"D:/Geomatics/thesis/3D_solweig/shade_debug/sh_multgap_stack_{i}" + str(round(azimuth, 2)) + "_" + str(round(altitude, 2)) + ".tif"
    #     write_output(sh_stack[i].get(), name)

    sh_combined = sh_stack[0]

    for i in range(1, num_combinations):
        sh_combined = cp.fmax(sh_combined, sh_stack[i])

    sh = (cp.fmax(sh, sh_combined))

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

    vegsh[vegsh > 0] = 1
    shvoveg = (shvoveg - a[0]) * vegsh  # Vegetation shadow volume
    vegsh = 1 - vegsh

    # wall shadows
    shvo = dsm_ground - a[0]  # first layer building shadow volume
    facesun = cp.logical_and(facesh + (walls > 0).astype(float) == 1, walls > 0).astype(float)
    wallsun = cp.copy(walls - shvo)
    wallsun[wallsun < 0] = 0
    wallsun[facesh == 1] = 0  # Removing walls in "self"-shadow
    wallsh = cp.copy(walls - wallsun)

    wallshve = shvoveg * wallbol
    wallshve = wallshve - wallsh
    wallshve[wallshve < 0] = 0
    id = cp.where(wallshve > walls)
    wallshve[id] = walls[id]
    wallsun = wallsun - wallshve  # problem with wallshve only
    id = cp.where(wallsun < 0)
    wallshve[id] = 0
    wallsun[id] = 0

    return vegsh, sh, wallsh, wallsun, wallshve, facesh, facesun
