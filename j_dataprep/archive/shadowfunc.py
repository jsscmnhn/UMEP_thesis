
def shadowingfunction_20_cupy_forloop(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush):
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

# @profile
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