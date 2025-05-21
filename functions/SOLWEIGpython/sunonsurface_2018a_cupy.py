import numpy as np
import cupy as cp

def sunonsurface_2018a_cupy(azimuthA, scale, buildings, shadow, sunwall, first, second, aspect, walls, Tg, Tgwall, Ta,
                       emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover):
    '''
    Calculates surface and wall sun/shadow and radiation interactions based on building geometry and sun position.
    This CuPy-accelerated implementation simulates sunlit and shadowed surfaces on buildings and ground,
    including longwave upwelling radiation (Lup) and albedo effects modulated by landcover and shadow patterns.

    Parameters:
        azimuthA (float):            Search directions for Ground View Factors (clockwise from north).
        scale (float):               Scale factor converting units to pixels.
        buildings (cp.ndarray):      2D array representing building heights.
        shadow (cp.ndarray):         2D binary shadow mask (1 = shadowed, 0 = sunlit).
        sunwall (cp.ndarray):        2D array marking sunlit building walls.
        first (float):               First height (sensor height) for Radiative surface influence
        second (float):              Second height (sensor height * 20) for Radiative surface influence
        aspect (cp.ndarray):         2D array of building wall aspect (orientation in radians).
        walls (cp.ndarray):          2D array of wall heights.
        Tg (cp.ndarray):             2D grid of ground temperatures [°C].
        Tgwall (cp.ndarray):         2D grid of wall temperatures [°C].
        Ta (float):                  Air temperature [°C].
        emis_grid (cp.ndarray):      Emissivity grid for surfaces.
        ewall (float):               Wall emissivity.
        alb_grid (cp.ndarray):       Albedo grid for surfaces.
        SBC (float):                 Stefan-Boltzmann constant.
        albedo_b (float):            Building wall albedo.
        Twater (float):              Water temperature [°C].
        lc_grid (cp.ndarray):        Landcover classification grid.
        landcover (int):             Landcover type indicator.

    Returns:
        gvf (cp.ndarray):            Grid of combined sun/shadow view factors on surfaces.
        gvfLup (cp.ndarray):         Grid of longwave upwelling radiation view factors.
        gvfalb (cp.ndarray):         Grid of albedo-weighted view factors including shadows.
        gvfalbnosh (cp.ndarray):     Grid of albedo-weighted view factors excluding shadows.
        gvf2 (cp.ndarray):           Grid of secondary view factors combining wall and surface shadows.
    '''
    sizex = walls.shape[0]
    sizey = walls.shape[1]

    # sizex=size(buildings,1);sizey=size(buildings,2);
    wallbol = (walls > 0) * 1
    sunwall[sunwall > 0] = 1  # test 20160910

    # conversion into radians
    azimuthA *= (np.pi / 180)

    # loop parameters
    index = 0
    f = buildings
    Lup = SBC * emis_grid * (Tg * shadow + Ta + 273.15) ** 4 - SBC * emis_grid * (Ta + 273.15) ** 4  # +Ta
    if landcover == 1:
        Tg[lc_grid == 3] = Twater - Ta  # Setting water temperature

    Lwall = SBC * ewall * (Tgwall + Ta + 273.15) ** 4 - SBC * ewall * (Ta + 273.15) ** 4  # +Ta
    albshadow = alb_grid * shadow
    alb = alb_grid
    # sh(sh<=0.1)=0;
    # sh=sh-(1-vegsh)*(1-psi);
    # shadow=sh-(1-vegsh)*(1-psi);
    # dx=0;
    # dy=0;
    # ds=0; ##ok<NASGU>

    tempsh = cp.zeros((sizex, sizey))
    tempbu = cp.zeros((sizex, sizey))
    tempbub = cp.zeros((sizex, sizey))
    tempbubwall = cp.zeros((sizex, sizey))
    tempwallsun = cp.zeros((sizex, sizey))
    weightsumsh = cp.zeros((sizex, sizey))
    weightsumwall = cp.zeros((sizex, sizey))
    first = np.round(first * scale)
    if first < 1:
        first = 1
    second = np.round(second * scale)
    # tempTgsh=tempsh;
    weightsumLupsh = cp.zeros((sizex, sizey))
    weightsumLwall = cp.zeros((sizex, sizey))
    weightsumalbsh = cp.zeros((sizex, sizey))
    weightsumalbwall = cp.zeros((sizex, sizey))
    weightsumalbnosh = cp.zeros((sizex, sizey))
    weightsumalbwallnosh = cp.zeros((sizex, sizey))
    tempLupsh = cp.zeros((sizex, sizey))
    tempalbsh = cp.zeros((sizex, sizey))
    tempalbnosh = cp.zeros((sizex, sizey))

    # other loop parameters
    pibyfour = np.pi / 4
    threetimespibyfour = 3 * pibyfour
    fivetimespibyfour = 5 * pibyfour
    seventimespibyfour = 7 * pibyfour
    sinazimuth = np.sin(azimuthA)
    cosazimuth = np.cos(azimuthA)
    tanazimuth = np.tan(azimuthA)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)

    isVert = ((pibyfour <= azimuthA) & (azimuthA < threetimespibyfour)) | \
             ((fivetimespibyfour <= azimuthA) & (azimuthA < seventimespibyfour))


    ## The Shadow casting algoritm
    for n in np.arange(0, second):
        if isVert:
            dy = signsinazimuth * index
            dx = -1 * signcosazimuth * np.abs(np.round(index / tanazimuth))
        else:
            dy = signsinazimuth * abs(round(index * tanazimuth))
            dx = -1 * signcosazimuth * index

        absdx = np.abs(dx)
        absdy = np.abs(dy)

        xc1 = ((dx + absdx) / 2)
        xc2 = (sizex + (dx - absdx) / 2)
        yc1 = ((dy + absdy) / 2)
        yc2 = (sizey + (dy - absdy) / 2)

        xp1 = -((dx - absdx) / 2)
        xp2 = (sizex - (dx + absdx) / 2)
        yp1 = -((dy - absdy) / 2)
        yp2 = (sizey - (dy + absdy) / 2)

        tempbu[int(xp1):int(xp2), int(yp1):int(yp2)] = buildings[int(xc1):int(xc2),
                                                       int(yc1):int(yc2)]  # moving building
        tempsh[int(xp1):int(xp2), int(yp1):int(yp2)] = shadow[int(xc1):int(xc2), int(yc1):int(yc2)]  # moving shadow
        tempLupsh[int(xp1):int(xp2), int(yp1):int(yp2)] = Lup[int(xc1):int(xc2), int(yc1):int(yc2)]  # moving Lup/shadow
        tempalbsh[int(xp1):int(xp2), int(yp1):int(yp2)] = albshadow[int(xc1):int(xc2),
                                                          int(yc1):int(yc2)]  # moving Albedo/shadow
        tempalbnosh[int(xp1):int(xp2), int(yp1):int(yp2)] = alb[int(xc1):int(xc2), int(yc1):int(yc2)]  # moving Albedo
        f = cp.min(cp.stack([f, tempbu]), axis=0)  # utsmetning av buildings

        shadow2 = tempsh * f
        weightsumsh = weightsumsh + shadow2

        Lupsh = tempLupsh * f
        weightsumLupsh = weightsumLupsh + Lupsh

        albsh = tempalbsh * f
        weightsumalbsh = weightsumalbsh + albsh

        albnosh = tempalbnosh * f
        weightsumalbnosh = weightsumalbnosh + albnosh

        tempwallsun[int(xp1):int(xp2), int(yp1):int(yp2)] = sunwall[int(xc1):int(xc2),
                                                            int(yc1):int(yc2)]  # moving buildingwall insun image
        tempb = tempwallsun * f
        tempbwall = f * -1 + 1
        tempbub = ((tempb + tempbub) > 0) * 1
        tempbubwall = ((tempbwall + tempbubwall) > 0) * 1
        weightsumLwall += tempbub * Lwall
        weightsumalbwall += tempbub * albedo_b
        weightsumwall += tempbub
        weightsumalbwallnosh = weightsumalbwallnosh + tempbubwall * albedo_b

        ind = 1
        if (n + 1) <= first:
            weightsumwall_first = weightsumwall / ind
            weightsumsh_first = weightsumsh / ind
            wallsuninfluence_first = weightsumwall_first > 0
            weightsumLwall_first = (weightsumLwall) / ind  # *Lwall
            weightsumLupsh_first = weightsumLupsh / ind

            weightsumalbwall_first = weightsumalbwall / ind  # *albedo_b
            weightsumalbsh_first = weightsumalbsh / ind
            weightsumalbwallnosh_first = weightsumalbwallnosh / ind  # *albedo_b
            weightsumalbnosh_first = weightsumalbnosh / ind
            wallinfluence_first = weightsumalbwallnosh_first > 0
            #         gvf1=(weightsumwall+weightsumsh)/first;
            #         gvf1(gvf1>1)=1;
            ind += 1
        index += 1

    wallsuninfluence_second = weightsumwall > 0
    wallinfluence_second = weightsumalbwallnosh > 0
    # gvf2(gvf2>1)=1;

    # Removing walls in shadow due to selfshadowing
    azilow = azimuthA - np.pi / 2
    azihigh = azimuthA + np.pi / 2
    if azilow >= 0 and azihigh < 2 * np.pi:  # 90 to 270  (SHADOW)
        facesh = (cp.logical_or(aspect < azilow, aspect >= azihigh).astype(float) - wallbol + 1)
    elif azilow < 0 and azihigh <= 2 * np.pi:  # 0 to 90
        azilow = azilow + 2 * np.pi
        facesh = cp.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1  # (SHADOW)    # check for the -1
    elif azilow > 0 and azihigh >= 2 * np.pi:  # 270 to 360
        azihigh = azihigh - 2 * np.pi
        facesh = cp.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1  # (SHADOW)

    # removing walls in self shadowing
    keep = (weightsumwall == second) - facesh
    keep[keep == -1] = 0

    # gvf from shadow only
    gvf1 = ((weightsumwall_first + weightsumsh_first) / (first + 1)) * wallsuninfluence_first + \
           (weightsumsh_first) / (first) * (wallsuninfluence_first * -1 + 1)
    weightsumwall[keep == 1] = 0
    gvf2 = ((weightsumwall + weightsumsh) / (second + 1)) * wallsuninfluence_second + \
           (weightsumsh) / (second) * (wallsuninfluence_second * -1 + 1)

    gvf2[gvf2 > 1.] = 1.

    # gvf from shadow and Lup
    gvfLup1 = ((weightsumLwall_first + weightsumLupsh_first) / (first + 1)) * wallsuninfluence_first + \
              (weightsumLupsh_first) / (first) * (wallsuninfluence_first * -1 + 1)
    weightsumLwall[keep == 1] = 0
    gvfLup2 = ((weightsumLwall + weightsumLupsh) / (second + 1)) * wallsuninfluence_second + \
              (weightsumLupsh) / (second) * (wallsuninfluence_second * -1 + 1)

    # gvf from shadow and albedo
    gvfalb1 = ((weightsumalbwall_first + weightsumalbsh_first) / (first + 1)) * wallsuninfluence_first + \
              (weightsumalbsh_first) / (first) * (wallsuninfluence_first * -1 + 1)
    weightsumalbwall[keep == 1] = 0
    gvfalb2 = ((weightsumalbwall + weightsumalbsh) / (second + 1)) * wallsuninfluence_second + \
              (weightsumalbsh) / (second) * (wallsuninfluence_second * -1 + 1)

    # gvf from albedo only
    gvfalbnosh1 = ((weightsumalbwallnosh_first + weightsumalbnosh_first) / (first + 1)) * wallinfluence_first + \
                  (weightsumalbnosh_first) / (first) * (wallinfluence_first * -1 + 1)  #
    gvfalbnosh2 = ((weightsumalbwallnosh + weightsumalbnosh) / (second)) * wallinfluence_second + \
                  (weightsumalbnosh) / (second) * (wallinfluence_second * -1 + 1)

    # Weighting
    gvf = (gvf1 * 0.5 + gvf2 * 0.4) / 0.9
    gvfLup = (gvfLup1 * 0.5 + gvfLup2 * 0.4) / 0.9
    gvfLup = gvfLup + ((SBC * emis_grid * (Tg * shadow + Ta + 273.15) ** 4) - SBC * emis_grid * (Ta + 273.15) ** 4) * (
                buildings * -1 + 1)  # +Ta
    gvfalb = (gvfalb1 * 0.5 + gvfalb2 * 0.4) / 0.9
    gvfalb = gvfalb + alb_grid * (buildings * -1 + 1) * shadow
    gvfalbnosh = (gvfalbnosh1 * 0.5 + gvfalbnosh2 * 0.4) / 0.9
    gvfalbnosh = gvfalbnosh * buildings + alb_grid * (buildings * -1 + 1)

    return gvf, gvfLup, gvfalb, gvfalbnosh, gvf2