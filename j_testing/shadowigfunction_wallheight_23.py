import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cupy as cp
import numpy as np
np.set_printoptions(threshold=1000)
from osgeo import gdal
from util.misc import saveraster
import matplotlib.pyplot as plt

def plot_array(array, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(array, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()

# @profile
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
    tempvegdem = cp.full((sizex, sizey), np.nan, dtype=cp.float32)
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

    # saveraster(gdal_dsm, folder + "SH.tif", sh.get())
    saveraster(gdal_dsm, folder + "vegsh.tif", vegsh.get())
    # saveraster(gdal_dsm, folder + "wallsh.tif", wallsh.get())
    # saveraster(gdal_dsm, folder + "wallsun.tif", wallsun.get())
    # saveraster(gdal_dsm, folder + "wallshve.tif", wallshve.get())
    # saveraster(gdal_dsm, folder + "facesh.tif", facesh.get())
    # saveraster(gdal_dsm, folder + "facesun.tif", facesun.get())
    return vegsh, sh, wallsh, wallsun, wallshve, facesh, facesun


INPUT_DSM = "D:/Geomatics/thesis/heattryout/preprocess/DSM_smaller.tif"
INPUT_CDSM = "D:/Geomatics/thesis/heattryout/preprocess/CHM_smaller.tif"
INPUT_HEIGHT = "D:/Geomatics/thesis/heattryout/preprocess/wallheight.tif"
INPUT_ASPECT = "D:/Geomatics/thesis/heattryout/preprocess/wallaspect.tif"

gdal_wheight = gdal.Open(INPUT_HEIGHT)
wheight = cp.array(gdal_wheight.ReadAsArray().astype(float), dtype=cp.float32)
# wheight = gdal_wheight.ReadAsArray().astype(float)
gdal_waspect = gdal.Open(INPUT_ASPECT)
waspect = cp.array(gdal_waspect.ReadAsArray().astype(float), dtype=cp.float32)
# waspect = gdal_waspect.ReadAsArray().astype(float)

gdal_dsm = gdal.Open(INPUT_DSM)
dsm = cp.array(gdal_dsm.ReadAsArray().astype(float), dtype=cp.float32)
# dsm = gdal_dsm.ReadAsArray().astype(float)
gdal_vegdsm = gdal.Open(INPUT_CDSM)
vegdsm = cp.array(gdal_dsm.ReadAsArray().astype(float), dtype=cp.float32)
# vegdsm = gdal_vegdsm.ReadAsArray().astype(float)
vegdsm2 = vegdsm * 0.25

if dsm.min() < 0:
    dsmraise = cp.abs(dsm.min())
    # dsmraise = np.abs(dsm.min())
    dsm = dsm + dsmraise
    print('Digital Surface Model (DSM) included negative values. DSM raised with ' + str(dsmraise) + 'm.')
else:
    dsmraise = 0

vegmax = vegdsm.max()
amaxvalue = dsm.max() - dsm.min()
# amaxvalue = cp.maximum(amaxvalue, vegmax)
amaxvalue = np.maximum(amaxvalue, vegmax)

# Elevation vegdsms if buildingDEM includes ground heights
vegdsm = vegdsm + dsm
vegdsm[vegdsm == dsm] = 0
vegdsm2 = vegdsm2 + dsm
vegdsm2[vegdsm2 == dsm] = 0

# % Bush separation
# bush = np.logical_not((vegdsm2 * vegdsm)) * vegdsm
bush = cp.logical_not((vegdsm2 * vegdsm)) * vegdsm
azimuth = 135
altitude = 8
scale = 2

folder = "D:/Geomatics/thesis/wallheight23/firstrun/"

shadowingfunction_wallheight_23(dsm, vegdsm, vegdsm2, azimuth, altitude, scale, amaxvalue, bush, wheight, waspect * np.pi / 180.)
