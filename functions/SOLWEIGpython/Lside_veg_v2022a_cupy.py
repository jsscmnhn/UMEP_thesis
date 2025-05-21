from __future__ import absolute_import
import numpy as np
import cupy as cp
from .Lvikt_veg import Lvikt_veg

def Lside_veg_v2022a_cupy(svfS,svfW,svfN,svfE,svfEveg,svfSveg,svfWveg,svfNveg,svfEaveg,svfSaveg,svfWaveg,svfNaveg,azimuth,altitude,Ta,Tw,SBC,ewall,Ldown,esky,t,F_sh,CI,LupE,LupS,LupW,LupN,anisotropic_longwave):
    '''
    Function updated to use CuPy. Function computes directional longwave radiation fluxes from east, south, west, and
    north walls, accounting for shadowing, vegetation, and anisotropic sky contributions using a modified
    cylindrical wedge model.

    This function estimates the longwave radiation received from each cardinal wall direction,
    considering vegetation cover, sky view factors, and diurnal sun position. The output is
    directionally separated fluxes that can be summed to get total wall irradiance.

    Parameters:
          svfX (cp.ndarray):        Directional Sky View Factors (SVFs) (X = E, S, W, N).
          svfXveg (cp.ndarray):     Vegetation-blocked SVFs.
          svfXa_veg (cp.ndarray):   Vegetation SVFs blocking buildings (all directions).
          azimuth (float):          Solar azimuth angle in degrees.
          altitude (float):         Solar altitude angle in degrees.
          Ta (float):               Air temperature in °C.
          Tw (float):               Wall surface temperature offset component.
          SBC (float):              Stefan–Boltzmann constant.
          ewall (float):            Emissivity of wall surfaces.
          Ldown (cp.ndarray):       Downwelling longwave radiation from the sky.
          esky (float):             Sky emissivity.
          t (float):                Time correction factor (in degrees) to shift solar angles.
          F_sh (cp.ndarray):        Fraction of the wall in direct sunlight (scaled 0–1).
          CI (float):               Cloud index (0 = overcast, 1 = clear).

          LupX (cp.ndarray):            Upwelling longwave radiation from ground for each direction (X = E, S, W, N).
          anisotropic_longwave (bool):  If True, anisotropic scheme is used.

    Returns:
          tuple of cp.ndarray:
              - Least: Longwave radiation received from the east-facing wall.
              - Lsouth: Longwave radiation received from the south-facing wall.
              - Lwest: Longwave radiation received from the west-facing wall.
              - Lnorth: Longwave radiation received from the north-facing wall.
    '''
    #Building height angle from svf
    svfalfaE=cp.arcsin(cp.exp((cp.log(1-svfE))/2))
    svfalfaS=cp.arcsin(cp.exp((cp.log(1-svfS))/2))
    svfalfaW=cp.arcsin(cp.exp((cp.log(1-svfW))/2))
    svfalfaN=cp.arcsin(cp.exp((cp.log(1-svfN))/2))
    
    vikttot=4.4897
    aziW=azimuth+t
    aziN=azimuth-90+t
    aziE=azimuth-180+t
    aziS=azimuth-270+t
    
    F_sh = 2*F_sh-1  #(cylindric_wedge scaled 0-1)
    
    c=1-CI
    Lsky_allsky = esky*SBC*((Ta+273.15)**4)*(1-c)+c*SBC*((Ta+273.15)**4)
    
    ## Least
    [viktveg, viktwall, viktsky, viktrefl] = Lvikt_veg(svfE, svfEveg, svfEaveg, vikttot)
    
    if altitude > 0:  # daytime
        alfaB=cp.arctan(svfalfaE)
        betaB=cp.arctan(cp.tan((svfalfaE)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaE)*(1+F_sh)) #TODO This should be considered in future versions
        if (azimuth > (180-t))  and  (azimuth <= (360-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziE*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    
    # Longwave from ground (see Lcyl_v2022a for remaining fluxes)
    if anisotropic_longwave == 1:
        Lground=LupE*0.5
        Least=Lground
    else:
        Lsky=((svfE+svfEveg-1)*Lsky_allsky)*viktsky*0.5
        Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
        Lground=LupE*0.5
        Lrefl=(Ldown+LupE)*(viktrefl)*(1-ewall)*0.5
        Least=Lsky+Lwallsun+Lwallsh+Lveg+Lground+Lrefl

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    ## Lsouth
    [viktveg,viktwall,viktsky,viktrefl]=Lvikt_veg(svfS,svfSveg,svfSaveg,vikttot)
    
    if altitude>0: # daytime
        alfaB=cp.arctan(svfalfaS)
        betaB=cp.arctan(cp.tan((svfalfaS)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaS)*(1+F_sh))
        if (azimuth <= (90-t))  or  (azimuth > (270-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziS*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    # Longwave from ground (see Lcyl_v2022a for remaining fluxes)
    if anisotropic_longwave == 1:
        Lground=LupS*0.5
        Lsouth=Lground
    else:
        Lsky=((svfS+svfSveg-1)*Lsky_allsky)*viktsky*0.5
        Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
        Lground=LupS*0.5
        Lrefl=(Ldown+LupS)*(viktrefl)*(1-ewall)*0.5
        Lsouth=Lsky+Lwallsun+Lwallsh+Lveg+Lground+Lrefl

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    ## Lwest
    [viktveg,viktwall,viktsky,viktrefl]=Lvikt_veg(svfW,svfWveg,svfWaveg,vikttot)
    
    if altitude>0: # daytime
        alfaB=cp.arctan(svfalfaW)
        betaB=cp.arctan(np.tan((svfalfaW)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaW)*(1+F_sh))
        if (azimuth > (360-t))  or  (azimuth <= (180-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziW*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    # Longwave from ground (see Lcyl_v2022a for remaining fluxes)
    if anisotropic_longwave == 1:
        Lground=LupW*0.5
        Lwest=Lground
    else:
        Lsky=((svfW+svfWveg-1)*Lsky_allsky)*viktsky*0.5
        Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
        Lground=LupW*0.5
        Lrefl=(Ldown+LupW)*(viktrefl)*(1-ewall)*0.5
        Lwest=Lsky+Lwallsun+Lwallsh+Lveg+Lground+Lrefl

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    ## Lnorth
    [viktveg,viktwall,viktsky,viktrefl]=Lvikt_veg(svfN,svfNveg,svfNaveg,vikttot)
    
    if altitude>0: # daytime
        alfaB=cp.arctan(svfalfaN)
        betaB=cp.arctan(cp.tan((svfalfaN)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaN)*(1+F_sh))
        if (azimuth > (90-t))  and  (azimuth <= (270-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziN*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*cp.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    # Longwave from ground (see Lcyl_v2022a for remaining fluxes)
    if anisotropic_longwave == 1:
        Lground=LupN*0.5
        Lnorth=Lground
    else:
        Lsky=((svfN+svfNveg-1)*Lsky_allsky)*viktsky*0.5
        Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
        Lground=LupN*0.5
        Lrefl=(Ldown+LupN)*(viktrefl)*(1-ewall)*0.5
        Lnorth=Lsky+Lwallsun+Lwallsh+Lveg+Lground+Lrefl

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    return Least, Lsouth, Lwest, Lnorth