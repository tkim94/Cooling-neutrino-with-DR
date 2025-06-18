import numpy as np
from scipy.integrate import quad
from scipy.special import kn
import math
from numpy import cos, sin, pi, sqrt, exp

from lib.utils import *
from lib.Xsecs import *


###############################
### Global Parameters (MeV) ###
###############################

GF  = 1.1664*10**-5*10**-6 # MeV^-2
me  = 0.511
Mpl = 1.22*10**19*10**3
sw2 = 0.223

MeV_to_invsec = 1.5193*10**21


Lbound = 5
Ubound = 10


######################
### Energy density ###
######################

def rho_y(T):
    rho = (2* np.pi**2)/30*T**4
    return rho

def rho_nu(T):
    rho = 7/8*(2* np.pi**2)/30*T**4
    return rho

def rho_e(T):
    rho = 4/(2* np.pi**2)*T**4*Jfunc(me/T,spin = 1/2)
    return rho

#-------------- NP --------------#
def rho_chi(T, chi_flav):
    rho = chi_flav*7/8*(2* np.pi**2)/30*T**4
    return rho


#-- Not tracking Temperature of phi --#
def rho_phi(T, Mass_phi):
    rho = 4/(2* np.pi**2)*T**4*Jfunc(Mass_phi/T, spin = 0)
    return rho



######################
###    Pressure    ### 
######################
# pressure of radiation = 1/3 rho

def P_e(T): 
    p = 4/(2* np.pi**2)*T**4*Kfunc(me/T,spin=1/2)
    return p


#-- Not tracking Temperature of phi --#
def P_phi(T, Mass_phi): 
    p = 4/(2* np.pi**2)*T**4*Kfunc(Mass_phi/T, spin=0)
    return p




#############################
### Denominator (drho_dT) ###
#############################

def drho_y_dT(T):
    return 4*rho_y(T)/T
    
def drho_nu_dT(T):
    return 4*rho_nu(T)/T

def drho_e_dT(T):
    return 4/(2* np.pi**2)*T**3*Lfunc(me/T,spin=1/2)

#-------------- NP --------------#
def drho_chi_dT(T, chi_flav):
    return 4*rho_chi(T, chi_flav)/T


#-- Not tracking Temperature of phi --#
def drho_phi_dT(T, Mass_phi):
    return 4/(2* np.pi**2)*T**3*Lfunc(Mass_phi/T,spin=0)



##########################################
###  total energy density and pressure ###
##########################################

def rho_tot(Ty, Tnu_e, Tnu_mu, Tchi, chi_flav, Mass_phi):
    tot = rho_y(Ty) + rho_e(Ty) + rho_nu(Tnu_e) + 2*rho_nu(Tnu_mu) + rho_chi(Tchi, chi_flav)
    return tot

def P_tot(Ty, Tnu_e, Tnu_mu, Tchi, chi_flav, Mass_phi):
    tot = 1/3*rho_y(Ty) + P_e(Ty) + 1/3*(rho_nu(Tnu_e) + 2*rho_nu(Tnu_mu) + rho_chi(Tchi, chi_flav))
    return tot

def Hubble(Ty, Tnu_e, Tnu_mu, Tchi, chi_flav, Mass_phi):
    return MeV_to_invsec * np.sqrt(8*np.pi/3*rho_tot(Ty, Tnu_e, Tnu_mu, Tchi, chi_flav, Mass_phi)/Mpl**2) # sec^-1



##############################
###  Energy Transfer rate  ###
##############################

# Maxwell-Boltzmann
def FMB(T1, T2):
    fmb = 32*(T1**9-T2**9)+56*T1**4*T2**4*(T1-T2)
    return fmb

# SM neutrino sector
def delrho_nu_delt(Ty, Tnu_e, Tnu_mu, Tchi, e_or_mu = 'e'):
    if e_or_mu=='e':
        param1 = 1
        param2 = 2
        T2 = Tnu_e
    elif e_or_mu=='mu':
        param1 = -1
        param2 = -1
        T2 = Tnu_mu
    else:
        raise Exception("wrong")
    rho_transf = MeV_to_invsec * GF**2/np.pi**5*((1+param1*4*sw2+8*sw2**2)*FMB(Ty, T2)+param2*FMB(Tnu_mu,Tnu_e))
    return rho_transf



# vi vj -> chi chi
def delrho_vi_delT_chi(T1, T2, Lnu, chi_flav, Lchi, Mass_phi, flav1 = 'e', flav2 = 'e'):
    
    phi_decayW = decayWidth_phi(chi_flav, Lchi, Mass_phi)
    
    '''
    # Runge-Kutta
    val = 0
    stepsize = 0.1
    initG = -70
    finG = 70
    for seg in np.arange(initG,finG+1,stepsize):
        temp, err = quad(lambda s: -1/(4*(2*np.pi)**4)*s**2*
                         Xsec_flav(s, flav1, flav2)*
                         (T1*kn(2,np.sqrt(s)/T1) - T2*kn(2,np.sqrt(s)/T2)),
                         (Mass_phi + phi_decayW * seg)**2,
                         (Mass_phi + phi_decayW * (seg+stepsize))**2,
                         epsabs=10**-10,
                         epsrel=10**-10)
        val += temp
    #print("Runge-Kutta int : ",MeV_to_invsec*val)#
    '''
    
    val, _ = quad(lambda s: -1/(4*(2*np.pi)**4)*s**2* #local peak
                  Xsec_flav(s, Lnu, chi_flav, Lchi, Mass_phi, flav1, flav2)*
                  (T1*kn(2,np.sqrt(s)/T1) - T2*kn(2,np.sqrt(s)/T2)),
                  (Mass_phi - Lbound * phi_decayW)**2,
                  (Mass_phi + Ubound * phi_decayW)**2,
                  epsabs=1e-10,
                  epsrel=1e-10
                 )
    
    val2, _ = quad(lambda s: -1/(4*(2*np.pi)**4)*s**2* #larger scale behavior
                  Xsec_flav(s, Lnu, chi_flav, Lchi, Mass_phi, flav1, flav2)*
                  (T1*kn(2,np.sqrt(s)/T1) - T2*kn(2,np.sqrt(s)/T2)),
                  1e0,
                  1e4,
                  epsabs=1e-10,
                  epsrel=1e-10
                 )
    val = val + val2
    return MeV_to_invsec*val


# vi vj -> phi phi -> 4 chi
def delrho_vi_delT_phi(T1, T2, Lnu, Mass_phi, initmass, mN1, mN2, mN3, flav1 = 'e', flav2 = 'e'):
    
    '''
    # Runge-Kutta
    val = 0
    stepsize = 0.1
    initG = -7
    finG = 4
    for seg in np.arange(initG,finG+1,stepsize):
        temp, err = quad(lambda s: -1/(4*(2*np.pi)**4)*s**2*
                         Xsec_viphi(s, Lnu, Mass_phi, initmass, mN1, mN2, mN3, flav1, flav2)*
                         (T1*kn(2,np.sqrt(s)/T1) - T2*kn(2,np.sqrt(s)/T2)),
                         (10**seg)**2,
                         (10**(seg+stepsize))**2,
                         epsabs=10**-10,
                         epsrel=10**-10)
        val += temp
    #print("Runge-Kutta int : ",MeV_to_invsec*val)
    '''
    val = 0
    val, _ = quad(lambda s: -1/(4*(2*np.pi)**4)*s**2* # flag 2
                  Xsec_viphi(s, Lnu, Mass_phi, initmass, mN1, mN2, mN3, flav1, flav2)*
                  (T1*kn(2,np.sqrt(s)/T1) - T2*kn(2,np.sqrt(s)/T2)),
                  (2*Mass_phi)**2,
                  100**2,
                  epsabs=10**-10,
                  epsrel=10**-10
                 )
    #'''
    return MeV_to_invsec*val

# phi -> chi chi decay
def delrho_phi_delT_chi(T1, Mass_phi, phi_decayW):
    
    prefactor = -1/(2*np.pi)**2*Mass_phi*phi_decayW*T1**3
    
    x = Mass_phi/T1
    
    Phi_decay, _ = quad(lambda xi: xi**2/(np.exp(sqrt(xi**2+x**2))-1),
                        0,
                        100,
                        epsabs=10**-12,
                        epsrel=10**-12
                       )
    
    val = prefactor*Phi_decay
    if val >= 0:
        val = 0
    return MeV_to_invsec*val

