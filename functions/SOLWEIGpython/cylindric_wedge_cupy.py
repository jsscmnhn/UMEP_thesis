import numpy as np
import cupy as cp

def cylindric_wedge_cupy(zen, svfalfa, rows, cols):
    """
    Function has been updated to use CuPy. Function Computes the fraction of sunlit wall surface based on
    the solar zenith angle and a sky-view-factor-related wall angle using the cylindric wedge model.

    This model estimates how much of a wall is sunlit based on urban canyon geometry.
    It assumes the canyon walls form a cylindrical wedge and calculates the projected
    sunlit fraction analytically.

    Parameters:
        zen (float):                    Solar zenith angle in radians (angle between the sun and the vertical).
        svfalfa (float or cp.ndarray):  Scalar or 2D CuPy array representing the SVF-weighted building wall inclination angle (in radians).
        rows (int):                     Number of rows in the output grid.
        cols (int):                     Number of columns in the output grid.

    Returns:
        F_sh (cp.ndarray):            Array representing the fraction of the wall surface that is directly sunlit for each grid cell.
    """

    np.seterr(divide='ignore', invalid='ignore')

    beta=zen
    # alfa=svfalfa
    alfa=cp.zeros((rows, cols)) + svfalfa
    # measure the size of the image
    # sizex=size(svfalfa,2)
    # sizey=size(svfalfa,1)
    
    xa=1-2./(cp.tan(alfa)*cp.tan(beta))
    ha=2./(cp.tan(alfa)*cp.tan(beta))
    ba=(1./cp.tan(alfa))
    hkil=2.*ba*ha
    
    qa=cp.zeros((rows, cols))
    # qa(length(svfalfa),length(svfalfa))=0;
    qa[xa<0]=np.tan(beta)/2
    
    Za=cp.zeros((rows, cols))
    # Za(length(svfalfa),length(svfalfa))=0;
    Za[xa<0]=((ba[xa<0]**2)-((qa[xa<0]**2)/4))**0.5
    
    phi=cp.zeros((rows, cols))
    #phi(length(svfalfa),length(svfalfa))=0;
    phi[xa<0]=cp.arctan(Za[xa<0]/qa[xa<0])
    
    A=cp.zeros((rows, cols))
    # A(length(svfalfa),length(svfalfa))=0;
    A[xa<0]=(cp.sin(phi[xa<0])-phi[xa<0]*cp.cos(phi[xa<0]))/(1-cp.cos(phi[xa<0]))
    
    ukil=cp.zeros((rows, cols))
    # ukil(length(svfalfa),length(svfalfa))=0
    ukil[xa<0]=2*ba[xa<0]*xa[xa<0]*A[xa<0]
    
    Ssurf=hkil+ukil
    
    F_sh=(2*np.pi*ba-Ssurf)/(2*np.pi*ba)#Xa
    
    return F_sh