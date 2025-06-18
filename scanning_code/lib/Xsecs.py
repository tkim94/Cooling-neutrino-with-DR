import numpy as np
from scipy.integrate import quad
from scipy.special import kn
import math
from numpy import cos, sin, pi, sqrt

from lib.utils import *


import lib.config as config

###############################
### Global Parameters (MeV) ###
###############################

GF  = config.GF
me  = config.me
Mpl = config.Mpl
sw2 = config.sw2

MeV_to_invsec = config.MeV_to_invsec



##################################
###  Take Assigned Parameters  ###
##################################

m_nl1 = config.config_params['m_nl1']
m_nl2 = config.config_params['m_nl2']
m_nl3 = config.config_params['m_nl3']

mN1 = config.config_params['mN1']
mN2 = config.config_params['mN2']
mN3 = config.config_params['mN3']


# Importing necessary definitions
U_PMNS = U_PMNS()

mass_nlight = mass_nlight(m_nl1, m_nl2, m_nl3)
mixing_matrix = mixing_matrix(m_nl1, m_nl2, m_nl3, mN1, mN2, mN3, U_PMNS)


def decayWidth_phi(chi_flav, Lchi, Mass_phi):
    val = chi_flav*Lchi**2*Mass_phi/(16*pi)
    return val



##########################################
### n_l n_l phi coupling in mass basis ###
##########################################
def n_light_phi(mEigen1 = '1', mEigen2 = '1'):
    mass_list = np.array(['1','2','3'])
    mE1 = np.where(np.char.find(mass_list,mEigen1)==0)[0][0]
    mE2 = np.where(np.char.find(mass_list,mEigen2)==0)[0][0]
    factor = mixing_matrix.T[mE1,:] @ mixing_matrix[:,mE2]
    return factor.real



######################
# n_l n_l -> chi chi #
######################

def Xsec_mass(s, Lnu, chi_flav, Lchi, Mass_phi, mEigen1 = '1', mEigen2 = '1'):
    m_phi = Mass_phi
    phi_decayW = decayWidth_phi(chi_flav, Lchi, Mass_phi)
    coupl = Lnu*n_light_phi(mEigen1, mEigen2)*Lchi
    
    mass_list = np.array(['1','2','3'])
    mE1 = np.where(np.char.find(mass_list,mEigen1)==0)[0][0]
    mE2 = np.where(np.char.find(mass_list,mEigen2)==0)[0][0]
    mnl1 = mass_nlight[mE1,mE1]
    mnl2 = mass_nlight[mE2,mE2]
    numer = coupl**2*(s)*(s - 4*mnl1**2)
    denom = 16*pi*s*((s - m_phi**2)**2 + m_phi**2*phi_decayW**2)
    return numer/denom

### Change to flavor eigenstate ###
def Xsec_flav(s, Lnu, chi_flav, Lchi, Mass_phi, flav1 = 'e', flav2 = 'e'):
    flav_list = np.array(['e','mu','ta'])
    fl1 = np.where(np.char.find(flav_list,flav1)==0)[0][0]
    fl2 = np.where(np.char.find(flav_list,flav2)==0)[0][0]

    Xs11 = Xsec_mass(s, Lnu, chi_flav, Lchi, Mass_phi, '1', '1')
    Xs12 = 0
    Xs13 = 0
    Xs22 = Xsec_mass(s, Lnu, chi_flav, Lchi, Mass_phi, '2', '2')
    Xs23 = 0
    Xs33 = Xsec_mass(s, Lnu, chi_flav, Lchi, Mass_phi, '3', '3')

    totalXs = (
        Xs11*U_PMNS[fl1, 0]**2*U_PMNS[fl2, 0]**2 + Xs12*U_PMNS[fl1, 0]**2*U_PMNS[fl2, 1]**2 + Xs13*U_PMNS[fl1, 0]**2*U_PMNS[fl2, 2]**2 +
        Xs12*U_PMNS[fl1, 1]**2*U_PMNS[fl2, 0]**2 + Xs22*U_PMNS[fl1, 1]**2*U_PMNS[fl2, 1]**2 + Xs23*U_PMNS[fl1, 1]**2*U_PMNS[fl2, 2]**2 +
        Xs13*U_PMNS[fl1, 2]**2*U_PMNS[fl2, 0]**2 + Xs23*U_PMNS[fl1, 2]**2*U_PMNS[fl2, 1]**2 + Xs33*U_PMNS[fl1, 2]**2*U_PMNS[fl2, 2]**2)
    
    if totalXs.real < 0:
        totalXs = 0
    elif totalXs.real > 0:
        pass
    return totalXs.real



######################
# n_l n_l -> phi phi #
######################

def Xsec_nlphi(s, Lnu, Mass_phi, initmass, heavymass):

    if ((s-4*Mass_phi**2)/s)<=0:
        return 0
    elif ((s-4*Mass_phi**2)/s)>0:
        fs = sqrt(1-4*Mass_phi**2/s)
        gs = (heavymass**2-Mass_phi**2)/s
        coupl = Lnu**4*initmass**2/heavymass**2
    
        integval = 8/( 1+ s/heavymass**2*gs**2 ) + 16/(fs*(1+2*gs))*(6*gs**2+4*gs+1+4*heavymass**2/s)*np.arctanh(fs/(1+2*gs)) - 24
        
        Xsec = 1/(128*pi*s)*(sqrt((s-4*Mass_phi**2)/s))*coupl*integval
        return Xsec

### Change to flavor eigenstate ###
def Xsec_viphi(s, Lnu, Mass_phi, initmass, mN1, mN2, mN3, flav1 = 'e', flav2 = 'e'):
    flav_list = np.array(['e','mu','ta'])
    fl1 = np.where(np.char.find(flav_list,flav1)==0)[0][0]
    fl2 = np.where(np.char.find(flav_list,flav2)==0)[0][0]

    Xs11 = Xsec_nlphi(s, Lnu, Mass_phi, initmass, mN1)
    Xs12 = 0 
    Xs13 = 0
    Xs22 = Xsec_nlphi(s, Lnu, Mass_phi, initmass, mN2)
    Xs23 = 0
    Xs33 = Xsec_nlphi(s, Lnu, Mass_phi, initmass, mN3)

    totalXs = (
        Xs11*U_PMNS[fl1, 0]**2*U_PMNS[fl2, 0]**2 + Xs12*U_PMNS[fl1, 0]**2*U_PMNS[fl2, 1]**2 + Xs13*U_PMNS[fl1, 0]**2*U_PMNS[fl2, 2]**2 +
        Xs12*U_PMNS[fl1, 1]**2*U_PMNS[fl2, 0]**2 + Xs22*U_PMNS[fl1, 1]**2*U_PMNS[fl2, 1]**2 + Xs23*U_PMNS[fl1, 1]**2*U_PMNS[fl2, 2]**2 +
        Xs13*U_PMNS[fl1, 2]**2*U_PMNS[fl2, 0]**2 + Xs23*U_PMNS[fl1, 2]**2*U_PMNS[fl2, 1]**2 + Xs33*U_PMNS[fl1, 2]**2*U_PMNS[fl2, 2]**2)
    if totalXs.real < 0:
        totalXs = 0
    elif totalXs.real > 0:
        pass
    return totalXs.real



######################
# n_l n_l -> n_l n_l #
######################

def Xsec_nlnl(s, initmass, finmass):
    if initmass==finmass:
        Xsec_SM = 0.529112*GF**2*s
        
        '''integval, _ = quad(lambda theta: sin(theta)*(45.1949*GF*Lnu**2*initmass**2*Mass_phi**2*s**2*(3.*Mass_phi**8 
                                  +(-0.375*phi_decayW**2 - 0.375*s)*s**3 + Mass_phi**6*(6.*phi_decayW**2 + 1.5*s) + 
                                  Mass_phi**4*(3.*phi_decayW**4 - 2.25*s**2) 
                                + Mass_phi**2*s*(-1.5*phi_decayW**4 - 2.25*phi_decayW**2*s - 1.875*s**2) + 
                             (1.*Mass_phi**8 + Mass_phi**6*(2.*phi_decayW**2 - 1.5*s) + (0.25*phi_decayW**2 + 0.75*s)*s**3 + 
                             Mass_phi**4*(1.*phi_decayW**4 - 1.5*s**2) + 
                             Mass_phi**2*s*(1.5*phi_decayW**4 - 1.5*phi_decayW**2*s + 1.25*s**2))*cos(theta)**2 + 
                           s**2*(-0.25*Mass_phi**4 + Mass_phi**2*(-0.25*phi_decayW**2 + 0.625*s) + 
                           (0.125*phi_decayW**2 - 0.375*s)*s)*cos(theta)**4))/(
                            mN1**2*(Mass_phi**4 + Mass_phi**2*(phi_decayW**2 - 2.*s) + s**2)*(1.*Mass_phi**4 + 0.25*s**2 + 
                                Mass_phi**2*(1.*phi_decayW**2 + 1.*s) + (-1.*Mass_phi**2 - 0.5*s)*s*cos(theta) + 
                                0.25*s**2*cos(theta)**2)*(Mass_phi**4 + 0.25*s**2 + Mass_phi**2*(1.*phi_decayW**2 + s) +
                                (Mass_phi**2 + 0.5*s)*s*cos(theta) + 0.25*s**2*cos(theta)**2))
                           ,0,pi,epsabs=10**-10,epsrel=10**-10)'''
        Xsec_correction = 0 # 1/(64*pi**2*s)*1/4*2*pi*integval ## Almost no contribution so set to be zero. 
    else:
        Xsec_SM = 0.0529813*GF**2*s
        Xsec_correction = 0                                    ## Almost no contribution so set to be zero.
    return Xsec_SM + Xsec_correction



######################
# chi chi -> chi chi #
######################

def Xsec_chiSI(s, Lchi, Mass_phi, chi_flav=1):
    phi_decayW = decayWidth_phi(chi_flav, Lchi, Mass_phi)
    
    LOterm = Lchi**4*(s*(6*s**3 - 9*s*Mass_phi**4 + 5*Mass_phi**6)/(s + Mass_phi**2) 
                      + Mass_phi**2*(4*s**3 - 9*s*Mass_phi**4 + 5*Mass_phi**6)*(np.log((Mass_phi**2/(s+Mass_phi**2))**2))/(s +2*Mass_phi**2))
    denomi = 16*pi*s**2*((s - Mass_phi**2)**2 + Mass_phi**2*phi_decayW**2)
    Xsec = LOterm/denomi
    return Xsec
