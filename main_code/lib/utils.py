import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import quad
from scipy.special import kn
import math
from numpy import cos, sin, pi, sqrt

import lib.config as config

def cividis_r_Alpha(alpha_value):
    cmap = plt.cm.cividis_r

    # Number of color levels
    n_colors = 256

    # Sample the colormap and add alpha
    colors = cmap(np.linspace(0, 1, n_colors))
    colors[:, -1] = alpha_value  # Modify the alpha channel
    
    # Create new colormap with modified alpha
    lighter_cmap = mpl.colors.ListedColormap(colors)
    
    return lighter_cmap

###############################
### Global Parameters (MeV) ###
###############################

GF  = config.GF
me  = config.me
Mpl = config.Mpl
sw2 = config.sw2

MeV_to_invsec = config.MeV_to_invsec


# PMNS parameters
theta12 = config.PMNS_params['theta12']
theta23 = config.PMNS_params['theta23']
theta13 = config.PMNS_params['theta13']
deltaCP = config.PMNS_params['deltaCP']


def U_PMNS():
    value = (np.array([[1,0,0],
                       [0,cos(theta23),sin(theta23)],
                       [0,-sin(theta23),cos(theta23)]]) @
             np.array([[cos(theta13),0,sin(theta13)*np.exp(-1j*deltaCP)],
                       [0,1,0],
                       [-sin(theta13)*np.exp(-1j*deltaCP),0,cos(theta13)]])@
             np.array([[cos(theta12),sin(theta12),0],
                       [-sin(theta12),cos(theta12),0],
                       [0,0,1]]))
    return value


def mass_nlight(m_nl1, m_nl2, m_nl3):
    sqrtm_nl = np.diag([np.sqrt(m_nl1),np.sqrt(m_nl2),np.sqrt(m_nl3)])
    mat = sqrtm_nl @ sqrtm_nl
    return mat


def mixing_matrix(m_nl1, m_nl2, m_nl3, mN1, mN2, mN3, U_PMNS):
    
    sqrtm_nh = np.diag([np.sqrt(mN1),np.sqrt(mN2),np.sqrt(mN3)])
    sqrtm_nl = np.diag([np.sqrt(m_nl1),np.sqrt(m_nl2),np.sqrt(m_nl3)])
    inv_mN = np.diag([1/mN1,1/mN2,1/mN3])
    
    m_Dirac = U_PMNS @ sqrtm_nl @ sqrtm_nh
    mat_fin = inv_mN @ m_Dirac.T @ U_PMNS
    return mat_fin


def Geffchi(Lchi, Mass_phi):
    val = Lchi**2 / Mass_phi**2 # sqrt(5/6)* 
    return val

def Geff_to_Mphi(Lchi, Geff):
    val = sqrt( Lchi**2 / Geff) # sqrt(5/6)* 
    return val

########################################################
### Modified Bessel functions calculated numerically ###
########################################################

# Integrated f(p)
# Notation x = m/T, xi = p/T
def Jfunc(x, spin = 0):
    if spin<0 or (spin*2)%1!=0:
        raise Exception("Incorrect value of spin is selected")
    b_or_f = (-1)**(spin*2+1)
    # approximate when x = 20, val ~ 10^-6
    if x >= 30: 
        return 0.
    elif x < 30:
        val, err = quad(lambda xi: xi**2*np.sqrt(xi**2+x**2)/(np.exp(np.sqrt(xi**2+x**2))+b_or_f*1),
                        0,
                        100,
                        epsabs=10**-12,
                        epsrel=10**-12)
        return val

def Kfunc(x, spin = 0):
    if spin<0 or (spin*2)%1!=0:
        raise Exception("Incorrect value of spin is selected")
    b_or_f = (-1)**(spin*2+1)
    if x >= 30:
        # non relativistic m >> T, pressureless
        return 0.
    elif x < 30:
        val, err = quad(lambda xi: xi**4/(3*np.sqrt(xi**2+x**2))/(np.exp(np.sqrt(xi**2+x**2))+b_or_f*1),
                        0,
                        100,
                        epsabs=10**-12,
                        epsrel=10**-12)
        return val

def Lfunc(x, spin = 0):
    if spin<0 or (spin*2)%1!=0:
        raise Exception("Incorrect value of spin is selected")
    b_or_f = (-1)**(spin*2+1)
    if x >= 30:
        return 1e-10
    elif x < 30:
        val, err = quad(lambda xi: np.exp(np.sqrt(xi**2+x**2))*xi**2*(xi**2+x**2)/(np.exp(np.sqrt(xi**2+x**2))+b_or_f*1)**2,
                        0,
                        100,
                        epsabs=10**-12,
                        epsrel=10**-12)
        return val
