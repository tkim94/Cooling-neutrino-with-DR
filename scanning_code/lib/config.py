import numpy as np
from numpy import cos, sin, pi

###############################
### Global Parameters (MeV) ###
###############################
GF  = 1.1664*10**-5*10**-6 # MeV^-2
me  = 0.511
Mpl = 1.22*10**19*10**3
sw2 = 0.223

MeV_to_invsec = 1.5193e21

# PMNS parameters

PMNS_params = {
    "theta12": 33.41/180*pi,
    "theta23": 49.1/180*pi,
    "theta13": 8.54/180*pi,
    "deltaCP": 0/180*pi
}


#####################
### Set of inputs ###
#####################
# Dynamic variables so define in dictionary is favorable
# All values are initialized to be 1

config_params = {
    
    "Lnu": 1,      # phi N N coupling
    "Lchi": 1,     # phi chi chi coupling
    
    "m_nl1": 1,    # light neutrino mass (MeV)
    "m_nl2": 1,
    "m_nl3": 1,
    
    "mN1": 1,      # heavy neutrino mass (MeV)
    "mN2": 1,
    "mN3": 1,
    
    "Mass_phi": 1, # NP phi mass
    "Mass_chi": 1, # NP radiation mass
    
    "chi_flav": 1  # Number of chi flavors
    
}

