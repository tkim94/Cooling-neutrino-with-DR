import numpy as np
from scipy.integrate import quad, odeint, solve_ivp
from scipy.special import kn
import matplotlib.pyplot as plt
import math
from numpy import cos, sin, pi

import pickle

import pandas as pd

path = '/afs/crc.nd.edu/user/t/tkim12/Work/nu_self/'
import os
os.chdir(path)

import lib.config as config

from importlib import reload

import lib.utils
import lib.Xsecs
import lib.cosmo_def

#print('Library imported')

file_no = 0

print('Processing ...')

mNval = 10

# Free parameter space
# Neutrino mass   [10^-5, 10] array
Mass_nu_arr = np.logspace(-11,-6,num=16)

# Mediator mass    [10^-5, 1] MeV array
Mass_phi_arr = np.logspace(-5,0,num=16)

for ii in range(len(Mass_nu_arr)):
    for jj in range(len(Mass_phi_arr)):
        
        ##############################
        ### Assign input variables ###
        ##############################
        
        config.config_params['Lnu'] = 1e-1
        config.config_params['Lchi'] = 3e-3
        config.config_params['chi_flav'] = 2
        
        config.config_params['m_nl1'] = Mass_nu_arr[ii]
        config.config_params['m_nl2'] = Mass_nu_arr[ii]
        config.config_params['m_nl3'] = Mass_nu_arr[ii]
        
        config.config_params['mN1'] = mNval
        config.config_params['mN2'] = mNval
        config.config_params['mN3'] = mNval
        
        config.config_params['Mass_phi'] = Mass_phi_arr[jj]
        config.config_params['Mass_chi'] = 1e-12
        
        
        Lnu = config.config_params['Lnu']
        Lchi = config.config_params['Lchi']
        
        m_nl1 = config.config_params['m_nl1']
        m_nl2 = config.config_params['m_nl2']
        m_nl3 = config.config_params['m_nl3']
        
        mN1 = config.config_params['mN1']
        mN2 = config.config_params['mN2']
        mN3 = config.config_params['mN3']
        
        Mass_phi = config.config_params['Mass_phi']
        Mass_chi = config.config_params['Mass_chi']
        
        chi_flav = config.config_params['chi_flav']
        
        
        reload(lib.utils)
        reload(lib.Xsecs)
        reload(lib.cosmo_def)
        
        from lib.utils import *
        from lib.Xsecs import *
        from lib.cosmo_def import *
        
        # Decay width of phi
        phi_decayW = decayWidth_phi(chi_flav, Lchi, Mass_phi)
        
        # G_eff of chi SI
        Geffchi = Geffchi(Lchi, Mass_phi)


        # Differential Equations of Temperature
        def dTy_dt(Ty, Tnu_e, Tnu_mu, Tchi):
            numer = -1*(Hubble(Ty, Tnu_e, Tnu_mu, Tchi, chi_flav, Mass_phi)*(4*rho_y(Ty) + 3*(rho_e(Ty)+P_e(Ty))) # + dP_int later
                        + delrho_nu_delt(Ty, Tnu_e, Tnu_mu, Tchi, e_or_mu='e') 
                        + 2*delrho_nu_delt(Ty, Tnu_e, Tnu_mu, Tchi, e_or_mu='mu')
                        )
            denomi = drho_y_dT(Ty)+drho_e_dT(Ty)
            return numer/denomi
        
        def dTnu_e_dt(Ty, Tnu_e, Tnu_mu, Tchi):
            numer = (-4*Hubble(Ty, Tnu_e, Tnu_mu, Tchi, chi_flav, Mass_phi)*rho_nu(Tnu_e)
                     + delrho_nu_delt(Ty, Tnu_e, Tnu_mu, Tchi, e_or_mu = 'e')
                     + chi_flav*4*1/2*delrho_vi_delT_chi(Tnu_e, Tchi,
                                                Lnu, chi_flav, Lchi, Mass_phi,
                                                flav1 = 'e', flav2 = 'e')       # 4(internal dof) * symmetrization factor (1/2 x 1/2 x 2)
                     + chi_flav*4*1/2*1/2*delrho_vi_delT_chi(Tnu_e, Tchi,
                                                    Lnu, chi_flav, Lchi, Mass_phi,
                                                    flav1 = 'e', flav2 = 'mu')  # 4(internal dof) * symmetrization factor (1/2 x 1/2)
                     
                     # E transfer of nu nu -> phi phi -> 4 chi 
                     # phi immediately decays, dominated by on shell phi
                     + chi_flav**2*(4*1/2*delrho_vi_delT_phi(Tnu_e, Tchi, 
                                                 Lnu, Mass_phi, m_nl1, mN1, mN2, mN3, 
                                                 flav1 = 'e', flav2 = 'e') 
                        + 4*1/2*1/2*delrho_vi_delT_phi(Tnu_e, Tchi, 
                                                       Lnu, Mass_phi, m_nl1, mN1, mN2, mN3, 
                                                       flav1 = 'e', flav2 = 'mu')
                       )
                    )
            denomi = drho_nu_dT(Tnu_e)
            return numer/denomi
        
        def dTnu_mu_dt(Ty, Tnu_e, Tnu_mu, Tchi):
            numer = (-8*Hubble(Ty, Tnu_e, Tnu_mu, Tchi, chi_flav, Mass_phi)*rho_nu(Tnu_mu)
                     + 2*delrho_nu_delt(Ty, Tnu_e, Tnu_mu, Tchi, e_or_mu = 'mu')
                     + chi_flav*2*(4*1/2*delrho_vi_delT_chi(Tnu_mu, Tchi, 
                                                   Lnu, chi_flav, Lchi, Mass_phi, 
                                                   flav1 = 'mu', flav2 = 'mu'))    # 4(internal dof) * symmetrization factor (1/2 x 1/2 x 2)
                     + chi_flav*2*(4*1/2*1/2*delrho_vi_delT_chi(Tnu_e, Tchi, 
                                                       Lnu, chi_flav, Lchi, Mass_phi, 
                                                       flav1 = 'e', flav2 = 'mu')) # 4(internal dof) * symmetrization factor (1/2 x 1/2)
        
                     # E transfer of nu nu -> phi phi -> 4 chi 
                     # phi immediately decays, dominated by on shell phi
                     + chi_flav**2*2*(4*1/2*delrho_vi_delT_phi(Tnu_e, Tchi, 
                                                   Lnu, Mass_phi, m_nl1, mN1, mN2, mN3, 
                                                   flav1 = 'mu', flav2 = 'mu') 
                          +2* 4*1/2*1/2*delrho_vi_delT_phi(Tnu_e, Tchi, 
                                                           Lnu, Mass_phi, m_nl1, mN1, mN2, mN3, 
                                                           flav1 = 'e', flav2 = 'mu')
                         )
                     )
            denomi = 2*drho_nu_dT(Tnu_mu)
            return numer/denomi
        
        def dTchi_dt(Ty, Tnu_e, Tnu_mu, Tchi):
            numer = (- 4*Hubble(Ty, Tnu_e, Tnu_mu, Tchi, chi_flav, Mass_phi)*rho_chi(Tchi, chi_flav)
                     - chi_flav*4*1/2*delrho_vi_delT_chi(Tnu_e, Tchi, 
                                                Lnu, chi_flav, Lchi, Mass_phi, flav1 = 'e', flav2 = 'e')
                     - chi_flav*2*(4*1/2*delrho_vi_delT_chi(Tnu_mu, Tchi, 
                                                   Lnu, chi_flav, Lchi, Mass_phi, flav1 = 'mu', flav2 = 'mu'))
                     - chi_flav*3*(4*1/2*1/2*delrho_vi_delT_chi(Tnu_e, Tchi, 
                                                       Lnu, chi_flav, Lchi, Mass_phi, flav1 = 'e', flav2 = 'mu'))
                    )
            denomi = drho_chi_dT(Tchi, chi_flav)
        
            # E transfer of nu nu -> phi phi -> 4 chi 
            # phi immediately decays, dominated by on shell phi
            BSM_phi_prod = chi_flav**2*(- 4*1/2*delrho_vi_delT_phi(Tnu_e, Tchi,
                                                       Lnu, Mass_phi, m_nl1, mN1, mN2, mN3, 
                                                       flav1 = 'e', flav2 = 'e') 
                            - 2*(4*1/2*delrho_vi_delT_phi(Tnu_e, Tchi, 
                                                          Lnu, Mass_phi, m_nl1, mN1, mN2, mN3, 
                                                          flav1 = 'mu', flav2 = 'mu'))
                            - 3*(4*1/2*1/2*delrho_vi_delT_phi(Tnu_e, Tchi, 
                                                              Lnu, Mass_phi, m_nl1, mN1, mN2, mN3, 
                                                              flav1 = 'e', flav2 = 'mu'))
                           )
            
            return (numer + BSM_phi_prod)/denomi
        
        
        # ODE
        def dT_dt(Tvec, t):
            Ty, Tnu_e, Tnu_mu, Tchi = Tvec
        
            #Tfloor = 1e-8
            #Ty = max(Ty, Tfloor)
            #Tnu_e = max(Tnu_e, Tfloor)
            #Tnu_mu = max(Tnu_mu, Tfloor)
            #Tchi = max(Tchi, Tfloor)
        
            dTy = dTy_dt(Ty, Tnu_e, Tnu_mu, Tchi)
            dTnu_e = dTnu_e_dt(Ty, Tnu_e, Tnu_mu, Tchi)
            dTnu_mu = dTnu_mu_dt(Ty, Tnu_e, Tnu_mu, Tchi)
            dTchi = dTchi_dt(Ty, Tnu_e, Tnu_mu, Tchi)
            
            return (dTy, dTnu_e, dTnu_mu, dTchi)
        
        
        #intial conditions Ty = Tv_i = 10 MeV , T_chi ~ small
        T_init = np.concatenate((10 * np.ones(3), np.array([1e-4])))
        t = np.logspace(np.log10(8e-3),np.log10(5e10),600)
        
        # ODE solve
        sol = odeint(dT_dt, T_init, t, rtol = 1e-6, atol= 1e-6)

        dictionary = {'phi-NN': Lnu,
                      'M_phi': Mass_phi,
                      'mtilde1': Lnu*n_light_phi('1', '1')*mN1,
                      'mtilde2': Lnu*n_light_phi('2', '2')*mN1,
                      'mtilde3': Lnu*n_light_phi('3', '3')*mN1,
                      'decay_width_phi': phi_decayW,
                      'T_photon': sol[:, 0],
                      'T_nu_e': sol[:, 1], 
                      'T_nu_mu': sol[:, 2],
                      'T_chi': sol[:, 3]}


        ###### Saving... ######

        prefix_loc = '/afs/crc.nd.edu/user/t/tkim12/Work/nu_self/data_m_scan/'
        
        with open(prefix_loc+f'mN{int(mN1)}_1e-1_flav2/Temperature_evolution_mN_{int(mN1)}_{int(file_no)}.pkl', 'wb') as f:
            pickle.dump(dictionary, f)
        
        file_no += 1

