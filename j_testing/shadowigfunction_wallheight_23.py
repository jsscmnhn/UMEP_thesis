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

# def shadowingfunction_wallheight_23(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, walls, aspect):
#     """
#     This function calculates shadows on a DSM and shadow height on building
#     walls including both buildings and vegetion units.
#     New functionallity to deal with pergolas, August 2021
#
#     INPUTS:
#     a = DSM
#     vegdem = Vegetation canopy DSM (magl)
#     vegdem2 = Trunkzone DSM (magl)
#     azimuth and altitude = sun position
#     scale= scale of DSM (1 meter pixels=1, 2 meter pixels=0.5)
#     walls= pixel row 'outside' buildings. will be calculated if empty
#     aspect=normal aspect of walls
#
#     OUTPUT:
#     sh=ground and roof shadow
#
#     wallsh = height of wall that is in shadow
#     wallsun = hieght of wall that is in sun
#
#     original Matlab code:
#     Fredrik Lindberg 2013-08-14
#     fredrikl@gvc.gu.se
#
#     :param a:
#     :param vegdem:
#     :param vegdem2:
#     :param azimuth:
#     :param altitude:
#     :param scale:
#     :param amaxvalue:
#     :param bush:
#     :param walls:
#     :param aspect:
#     :return:
#     """
#
#     # conversion
#     degrees = np.pi / 180.
#     azimuth *= degrees
#     altitude *= degrees
#
#     # measure the size of the image
#     sizex = np.shape(a)[0]
#     sizey = np.shape(a)[1]
#
#     # initialise parameters
#     dx = 0
#     dy = 0
#     dz = 0
#     temp = np.zeros((sizex, sizey))
#     tempvegdem = np.zeros((sizex, sizey))
#     tempvegdem2 = np.zeros((sizex, sizey))
#     templastfabovea = np.zeros((sizex, sizey))
#     templastgabovea = np.zeros((sizex, sizey))
#     bushplant = bush > 1
#     sh = np.zeros((sizex, sizey))  # shadows from buildings
#     vbshvegsh = np.copy(sh)  # vegetation blocking buildings
#     vegsh = np.add(np.zeros((sizex, sizey)), bushplant, dtype=float)  # vegetation shadow
#     f = np.copy(a)
#     shvoveg = np.copy(vegdem)  # for vegetation shadowvolume
#     # g = np.copy(sh)
#     wallbol = (walls > 0).astype(float)
#
#     # other loop parameters
#     pibyfour = np.pi / 4
#     threetimespibyfour = 3 * pibyfour
#     fivetimespibyfour = 5 * pibyfour
#     seventimespibyfour = 7 * pibyfour
#     sinazimuth = np.sin(azimuth)
#     cosazimuth = np.cos(azimuth)
#     tanazimuth = np.tan(azimuth)
#     signsinazimuth = np.sign(sinazimuth)
#     signcosazimuth = np.sign(cosazimuth)
#     dssin = np.abs(1 / sinazimuth)
#     dscos = np.abs(1 / cosazimuth)
#     tanaltitudebyscale = np.tan(altitude) / scale
#
#     index = 0
#
#     # new case with pergola (thin vertical layer of vegetation), August 2021
#     dzprev = 0
#
#     # main loop
#     while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
#         if ((pibyfour <= azimuth) and (azimuth < threetimespibyfour)) or (
#                 (fivetimespibyfour <= azimuth) and (azimuth < seventimespibyfour)):
#             dy = signsinazimuth * index
#             dx = -1 * signcosazimuth * np.abs(np.round(index / tanazimuth))
#             ds = dssin
#         else:
#             dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
#             dx = -1 * signcosazimuth * index
#             ds = dscos
#
#         # note: dx and dy represent absolute values while ds is an incremental value
#         dz = (ds * index) * tanaltitudebyscale
#         tempvegdem[0:sizex, 0:sizey] = 0
#         tempvegdem2[0:sizex, 0:sizey] = 0
#         temp[0:sizex, 0:sizey] = 0
#         templastfabovea[0:sizex, 0:sizey] = 0.
#         templastgabovea[0:sizex, 0:sizey] = 0.
#         absdx = np.abs(dx)
#         absdy = np.abs(dy)
#         xc1 = int((dx + absdx) / 2)
#         xc2 = int(sizex + (dx - absdx) / 2)
#         yc1 = int((dy + absdy) / 2)
#         yc2 = int(sizey + (dy - absdy) / 2)
#         xp1 = -int((dx - absdx) / 2)
#         xp2 = int(sizex - (dx + absdx) / 2)
#         yp1 = -int((dy - absdy) / 2)
#         yp2 = int(sizey - (dy + absdy) / 2)
#
#         tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
#         tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
#         temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz
#
#         f = np.fmax(f, temp)  # Moving building shadow
#         shvoveg = np.fmax(shvoveg, tempvegdem)  # moving vegetation shadow volume
#         sh[f > a] = 1
#         sh[f <= a] = 0
#         fabovea = (tempvegdem > a).astype(int)  # vegdem above DEM
#         gabovea = (tempvegdem2 > a).astype(int)  # vegdem2 above DEM
#
#         # new pergola condition
#         templastfabovea[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dzprev
#         templastgabovea[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dzprev
#         lastfabovea = templastfabovea > a
#         lastgabovea = templastgabovea > a
#         dzprev = dz
#         vegsh2 = np.add(np.add(np.add(fabovea, gabovea, dtype=float), lastfabovea, dtype=float), lastgabovea,
#                         dtype=float)
#         vegsh2[vegsh2 == 4] = 0.
#         # vegsh2[vegsh2 == 1] = 0. # This one is the ultimate question...
#         vegsh2[vegsh2 > 0] = 1.
#
#         # vegsh2 = fabovea - gabovea #old without pergolas
#         # vegsh = np.max([vegsh, vegsh2], axis=0) #old without pergolas
#
#         vegsh = np.fmax(vegsh, vegsh2)
#         vegsh[vegsh * sh > 0] = 0
#         vbshvegsh = np.copy(vegsh) + vbshvegsh  # removing shadows 'behind' buildings
#
#         # # vegsh at high sun altitudes # Not needed when pergolas are included
#         # if index == 0:
#         #     firstvegdem = np.copy(tempvegdem) - np.copy(temp)
#         #     firstvegdem[firstvegdem <= 0] = 1000
#         #     vegsh[firstvegdem < dz] = 1
#         #     vegsh *= (vegdem2 > a)
#         #     vbshvegsh = np.zeros((sizex, sizey))
#
#         # # Bush shadow on bush plant # Not needed when pergolas are included
#         # if np.max(bush) > 0 and np.max(fabovea*bush) > 0:
#         #     tempbush = np.zeros((sizex, sizey))
#         #     tempbush[int(xp1):int(xp2), int(yp1):int(yp2)] = bush[int(xc1):int(xc2), int(yc1):int(yc2)] - dz
#         #     g = np.max([g, tempbush], axis=0)
#         #     g = bushplant * g
#
#         index += 1
#
#     # Removing walls in shadow due to selfshadowing
#     azilow = azimuth - np.pi / 2
#     azihigh = azimuth + np.pi / 2
#     if azilow >= 0 and azihigh < 2 * np.pi:  # 90 to 270  (SHADOW)
#         facesh = np.logical_or(aspect < azilow, aspect >= azihigh).astype(float) - wallbol + 1  # TODO check
#     elif azilow < 0 and azihigh <= 2 * np.pi:  # 0 to 90
#         azilow = azilow + 2 * np.pi
#         facesh = np.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1  # (SHADOW)
#     elif azilow > 0 and azihigh >= 2 * np.pi:  # 270 to 360
#         azihigh -= 2 * np.pi
#         facesh = np.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1  # (SHADOW)
#
#     sh = 1 - sh
#     vbshvegsh[vbshvegsh > 0] = 1
#     vbshvegsh = vbshvegsh - vegsh
#
#     # if np.max(bush) > 0: # Not needed when pergolas are included
#     #     g = g-bush
#     #     g[g > 0] = 1
#     #     g[g < 0] = 0
#     #     vegsh = vegsh-bushplant+g
#     #     vegsh[vegsh < 0] = 0
#
#     vegsh[vegsh > 0] = 1
#     shvoveg = (shvoveg - a) * vegsh  # Vegetation shadow volume
#     vegsh = 1 - vegsh
#     vbshvegsh = 1 - vbshvegsh
#
#     # wall shadows
#     shvo = f - a  # building shadow volume
#     facesun = np.logical_and(facesh + (walls > 0).astype(float) == 1, walls > 0).astype(float)
#     wallsun = np.copy(walls - shvo)
#     wallsun[wallsun < 0] = 0
#     wallsun[facesh == 1] = 0  # Removing walls in "self"-shadow
#     wallsh = np.copy(walls - wallsun)
#
#     wallshve = shvoveg * wallbol
#     wallshve = wallshve - wallsh
#     wallshve[wallshve < 0] = 0
#     id = np.where(wallshve > walls)
#     wallshve[id] = walls[id]
#     wallsun = wallsun - wallshve  # problem with wallshve only
#     id = np.where(wallsun < 0)
#     wallshve[id] = 0
#     wallsun[id] = 0
#
#     # plot_array(vegsh, "Vegetation Shadow")
#     # plot_array(sh, "Ground and Roof Shadow")
#     # plot_array(vbshvegsh, "Vegetation Blocking Shadows")
#     # plot_array(wallsh, "Wall Shadows")
#     # plot_array(wallsun, "Wall Sunlit Areas")
#     # plot_array(wallshve, "Wall Shadows from Vegetation")
#     # plot_array(facesh, "Wall Faces in Shadow")
#     # plot_array(facesun, "Wall Faces in Sun")
#
#     saveraster(gdal_dsm, folder + "SH.tif", sh)
#     saveraster(gdal_dsm, folder + "vegsh2.tif", vegsh)
#     saveraster(gdal_dsm, folder + "wallsh.tif", wallsh)
#     saveraster(gdal_dsm, folder + "wallsun.tif", wallsun)
#     saveraster(gdal_dsm, folder + "wallshve.tif", wallshve)
#     saveraster(gdal_dsm, folder + "facesh.tif", facesh)
#     saveraster(gdal_dsm, folder + "facesun.tif", facesun)
#
#     return vegsh, sh, vbshvegsh, wallsh, wallsun, wallshve, facesh, facesun

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

    saveraster(gdal_dsm, folder + f"SH_{i}.tif", sh.get())
    saveraster(gdal_dsm, folder + f"vegsh_{i}.tif", vegsh.get())

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
vegdsm = cp.array(gdal_vegdsm.ReadAsArray().astype(float), dtype=cp.float32)
# vegdsm = gdal_vegdsm.ReadAsArray().astype(float)
vegdsm2 = vegdsm * 0.25
scale = 2

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
azimuth_altitude_pairs = [
    (1.0, 50.01779819955806),
    (7.648368565989529, 61.58371657769966),
    (16.016519674811136, 72.74824421305907),
    (24.947913354435443, 83.96328707022865),
    (34.07701005698726, 95.8596089976628),
    (42.99176144519088, 109.35570605958486),
    (51.098147374599954, 125.81604248965994),
    (57.42733050284383, 146.94411459739555),
    (60.568730627457796, 173.2142247420149),
    (59.429146124409456, 200.90784863329756),
    (54.44413958470721, 224.65690050058242),
    (47.03960009151508, 243.1607533985618),
    (38.418890296336116, 257.8809553227283),
    (29.330132175084984, 270.4075796599285),
    (20.252671501603047, 281.85618205789103),
    (11.559307633362977, 292.9778828898551),
    (3.6586240011030924, 304.30261187923475),
]



folder = "D:/Geomatics/thesis/wallheight23/shadowcheck/"

for i, (altitude, azimuth) in enumerate(azimuth_altitude_pairs):
    output_filename = f"{folder}shadow_result_{i}.tif"  # Modify based on desired output format
    print(f"Processing azimuth {azimuth}, altitude {altitude}, saving to {output_filename}")

    shadowingfunction_wallheight_23(dsm, vegdsm, vegdsm2, azimuth, altitude, scale, amaxvalue, bush, wheight,
                                    (waspect) * np.pi / 180.)
