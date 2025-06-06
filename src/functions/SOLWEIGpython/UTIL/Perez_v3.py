from __future__ import division
import numpy as np
from .create_patches import create_patches

def Perez_v3(zen, azimuth, radD, radI, jday, patchchoice, patch_option):
    """
    This unchanged function calculates distribution of luminance on the skyvault based on
    Perez luminince distribution model.

    Created by:
    Fredrik Lindberg 20120527, fredrikl@gvc.gu.se
    Gothenburg University, Sweden
    Urban Climte Group

    Inputs:
         zen:     Zenith angle of the Sun (in degrees)
         azimuth: Azimuth angle of the Sun (in degrees)
         radD:    Horizontal diffuse radiation (W m-2)
         radI:    Direct radiation perpendicular to the Sun beam (W m-2)
         jday:    Day of year

    Returns
    -------
    lv: np.array
        Relative luminance map (same dimensions as theta. gamma)
    """

    m_a1 = np.array([1.3525, -1.2219, -1.1000, -0.5484, -0.6000, -1.0156, -1.0000, -1.0500])
    m_a2 = np.array([-0.2576, -0.7730, -0.2515, -0.6654, -0.3566, -0.3670, 0.0211, 0.0289])
    m_a3 = np.array([-0.2690, 1.4148, 0.8952, -0.2672, -2.5000, 1.0078, 0.5025, 0.4260])
    m_a4 = np.array([-1.4366, 1.1016, 0.0156, 0.7117, 2.3250, 1.4051, -0.5119, 0.3590])
    m_b1 = np.array([-0.7670, -0.2054, 0.2782, 0.7234, 0.2937, 0.2875, -0.3000, -0.3250])
    m_b2 = np.array([0.0007, 0.0367, -0.1812, -0.6219, 0.0496, -0.5328, 0.1922, 0.1156])
    m_b3 = np.array([1.2734, -3.9128, -4.5000, -5.6812, -5.6812, -3.8500, 0.7023, 0.7781])
    m_b4 = np.array([-0.1233, 0.9156, 1.1766, 2.6297, 1.8415, 3.3750, -1.6317, 0.0025])
    m_c1 = np.array([2.8000, 6.9750, 24.7219, 33.3389, 21.0000, 14.0000, 19.0000, 31.0625])
    m_c2 = np.array([0.6004, 0.1774, -13.0812, -18.3000, -4.7656, -0.9999, -5.0000, -14.5000])
    m_c3 = np.array([1.2375, 6.4477, -37.7000, -62.2500, -21.5906, -7.1406, 1.2438, -46.1148])
    m_c4 = np.array([1.0000, -0.1239, 34.8438, 52.0781, 7.2492, 7.5469, -1.9094, 55.3750])
    m_d1 = np.array([1.8734, -1.5798, -5.0000, -3.5000, -3.5000, -3.4000, -4.0000, -7.2312])
    m_d2 = np.array([0.6297, -0.5081, 1.5218, 0.0016, -0.1554, -0.1078, 0.0250, 0.4050])
    m_d3 = np.array([0.9738, -1.7812, 3.9229, 1.1477, 1.4062, -1.0750, 0.3844, 13.3500])
    m_d4 = np.array([0.2809, 0.1080, -2.6204, 0.1062, 0.3988, 1.5702, 0.2656, 0.6234])
    m_e1 = np.array([0.0356, 0.2624, -0.0156, 0.4659, 0.0032, -0.0672, 1.0468, 1.5000])
    m_e2 = np.array([-0.1246, 0.0672, 0.1597, -0.3296, 0.0766, 0.4016, -0.3788, -0.6426])
    m_e3 = np.array([-0.5718, -0.2190, 0.4199, -0.0876, -0.0656, 0.3017, -2.4517, 1.8564])
    m_e4 = np.array([0.9938, -0.4285, -0.5562, -0.0329, -0.1294, -0.4844, 1.4656, 0.5636])
    
    acoeff = np.transpose(np.atleast_2d([m_a1, m_a2, m_a3, m_a4]))
    bcoeff = np.transpose(np.atleast_2d([m_b1, m_b2, m_b3, m_b4]))
    ccoeff = np.transpose(np.atleast_2d([m_c1, m_c2, m_c3, m_c4]))
    dcoeff = np.transpose(np.atleast_2d([m_d1, m_d2, m_d3, m_d4]))
    ecoeff = np.transpose(np.atleast_2d([m_e1, m_e2, m_e3, m_e4]))

    deg2rad = np.pi/180
    rad2deg = 180/np.pi
    altitude = 90-zen
    zen = zen * deg2rad
    azimuth = azimuth * deg2rad
    altitude = altitude * deg2rad
    Idh = radD
    # Ibh = radI/sin(altitude)
    Ibn = radI

    # Skyclearness
    PerezClearness = ((Idh+Ibn)/(Idh+1.041*np.power(zen, 3)))/(1+1.041*np.power(zen, 3))
    # Extra terrestrial radiation
    day_angle = jday*2*np.pi/365
    #I0=1367*(1+0.033*np.cos((2*np.pi*jday)/365))
    I0 = 1367*(1.00011+0.034221*np.cos(day_angle) + 0.00128*np.sin(day_angle)+0.000719 *
               np.cos(2*day_angle)+0.000077*np.sin(2*day_angle))    # New from robinsson

    # Optical air mass
    # m=1/altitude; old
    if altitude >= 10*deg2rad:
        AirMass = 1/np.sin(altitude)
    elif altitude < 0:   # below equation becomes complex
        AirMass = 1/np.sin(altitude)+0.50572*np.power(180*complex(altitude)/np.pi+6.07995, -1.6364)
    else:
        AirMass = 1/np.sin(altitude)+0.50572*np.power(180*altitude/np.pi+6.07995, -1.6364)

    # Skybrightness
    # if altitude*rad2deg+6.07995>=0
    PerezBrightness = (AirMass*radD)/I0
    if Idh <= 10:
        # m_a=0;m_b=0;m_c=0;m_d=0;m_e=0;
        PerezBrightness = 0
    #if altitude < 0:
        #print("Airmass")
        #print(AirMass)
        #print(PerezBrightness)
    # sky clearness bins
    if PerezClearness < 1.065:
        intClearness = 0
    if PerezClearness > 1.065 and PerezClearness < 1.230:
        intClearness = 1
    if PerezClearness > 1.230 and PerezClearness < 1.500:
        intClearness = 2
    if PerezClearness > 1.500 and PerezClearness < 1.950:
        intClearness = 3
    if PerezClearness > 1.950 and PerezClearness < 2.800:
        intClearness = 4
    if PerezClearness > 2.800 and PerezClearness < 4.500:
        intClearness = 5
    if PerezClearness > 4.500 and PerezClearness < 6.200:
        intClearness = 6
    if PerezClearness > 6.200:
        intClearness = 7

    m_a = acoeff[intClearness,  0] + acoeff[intClearness,  1] * zen + PerezBrightness * (acoeff[intClearness,  2] + acoeff[intClearness,  3] * zen)
    m_b = bcoeff[intClearness,  0] + bcoeff[intClearness,  1] * zen + PerezBrightness * (bcoeff[intClearness,  2] + bcoeff[intClearness,  3] * zen)
    m_e = ecoeff[intClearness,  0] + ecoeff[intClearness,  1] * zen + PerezBrightness * (ecoeff[intClearness,  2] + ecoeff[intClearness,  3] * zen)

    if intClearness > 0:
        m_c = ccoeff[intClearness, 0] + ccoeff[intClearness, 1] * zen + PerezBrightness * (ccoeff[intClearness, 2] + ccoeff[intClearness, 3] * zen)
        m_d = dcoeff[intClearness, 0] + dcoeff[intClearness, 1] * zen + PerezBrightness * (dcoeff[intClearness, 2] + dcoeff[intClearness, 3] * zen)
    else:
        # different equations for c & d in clearness bin no. 1,  from Robinsson
        m_c = np.exp(np.power(PerezBrightness * (ccoeff[intClearness, 0] + ccoeff[intClearness, 1] * zen), ccoeff[intClearness, 2]))-1
        m_d = -np.exp(PerezBrightness * (dcoeff[intClearness, 0] + dcoeff[intClearness, 1] * zen)) + dcoeff[intClearness, 2] + \
            PerezBrightness * dcoeff[intClearness, 3] * PerezBrightness

    # print 'a = ', m_a
    # print 'b = ', m_b
    # print 'e = ', m_e
    # print 'c = ', m_c
    # print 'd = ', m_d

    if patchchoice == 2:
        skyvaultalt = np.atleast_2d([])
        skyvaultazi = np.atleast_2d([])
        # Creating skyvault at one degree intervals
        skyvaultalt = np.ones([90, 361])*90
        skyvaultazi = np.empty((90, 361))
        for j in range(90):
            skyvaultalt[j, :] = 91-j
            skyvaultazi[j, :] = range(361)
            
    elif patchchoice == 1:
        # Creating skyvault of patches of constant radians (Tregeneza and Sharples, 1993)
        skyvaultalt, skyvaultazi, _, _, _, _, _ = create_patches(patch_option)

    skyvaultzen = (90 - skyvaultalt) * deg2rad
    skyvaultalt = skyvaultalt * deg2rad
    skyvaultazi = skyvaultazi * deg2rad

    # Angular distance from the sun from Robinsson
    cosSkySunAngle = np.sin(skyvaultalt) * np.sin(altitude) + \
                     np.cos(altitude) * np.cos(skyvaultalt) * np.cos(np.abs(skyvaultazi-azimuth))

    # Main equation
    lv = (1 + m_a * np.exp(m_b / np.cos(skyvaultzen))) * ((1 + m_c * np.exp(m_d * np.arccos(cosSkySunAngle)) +
                                                           m_e * cosSkySunAngle * cosSkySunAngle))

    # Normalisation
    lv = lv / np.sum(lv)

    # plotting
    # axesm('stereo','Origin',[90 180],'MapLatLimit',[0 90],'Aspect','transverse')
    # framem off; gridm on; mlabel off; plabel off;axis on;
    # setm(gca,'MLabelParallel',-20)
    # geoshow(skyvaultalt*rad2deg,skyvaultazi*rad2deg,lv,'DisplayType','texture');
    # colorbar
    # set(gcf,'Color',[1 1 1])
    # pause(1)

    if patchchoice == 1:
        #x = np.atleast_2d([])
        #lv = np.transpose(np.append(np.append(np.append(x, skyvaultalt*rad2deg), skyvaultazi*rad2deg), lv))
        x = np.transpose(np.atleast_2d(skyvaultalt*rad2deg))
        y = np.transpose(np.atleast_2d(skyvaultazi*rad2deg))
        z = np.transpose(np.atleast_2d(lv))
        lv = np.append(np.append(x, y, axis=1), z, axis=1)
    return lv, PerezClearness, PerezBrightness