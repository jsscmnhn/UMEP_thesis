
import numpy as np

def daylen(DOY, XLAT):
    '''
    Unchanged function: Calculate solar declination, daylength, and sunrise/sunset times based on day of year and latitude.

    Parameters:
        DOY (int):             Day of year (1–365), with 1 = Jan 1.
        XLAT (float):          Latitude in degrees (positive = Northern Hemisphere, negative = Southern Hemisphere).

    Returns:
        DAYL (float):          Daylength in hours.
        DEC (float):           Solar declination angle in degrees.
        SNDN (float):          Sunset time in decimal hours (e.g., 18.0 = 6:00 PM).
        SNUP (float):          Sunrise time in decimal hours (e.g., 6.0 = 6:00 AM).

    Notes:
        - Declination is calculated using a cosine-based approximation with amplitude ±23.45°.
        - Valid for most latitudes; limited accuracy near polar circles.
    '''
    RAD=np.pi/180.0

    DEC = -23.45 * np.cos(2.0*np.pi*(DOY+10.0)/365.0)

    SOC = np.tan(RAD*DEC) * np.tan(RAD*XLAT)
    SOC = min(max(SOC,-1.0),1.0)

    DAYL = 12.0 + 24.0*np.arcsin(SOC)/np.pi
    SNUP = 12.0 - DAYL/2.0
    SNDN = 12.0 + DAYL/2.0

    return DAYL, DEC, SNDN, SNUP
