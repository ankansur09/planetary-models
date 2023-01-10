import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy
from eos import chabrier, scvh, mh13_scvh
from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import math
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm
import const
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS

from astropy.constants import k_B, m_e, m_p, m_n, G, M_jup, M_earth, R_jup, R_earth
from astropy.constants import u as amu
import pdb

G=G.to('cm^3/g s^2').value # Newton's constant
R_jup = R_jup.to('cm').value
M_jup = M_jup.to('g').value
kb = k_B.to('erg/K') # ergs/K

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)

"""
This file is meant to keep the EOS funtions in one place apart from the static and evolution code. Here, RTA
adapts work from static_ys.py file so that the functions can be called in Heyney_evolution.py. These functions
will incorporate entropy and helium abundance dependence.
"""

class eos:
    def __init__(self, Y):
        self.Y = Y
        #self.eos_table = eos_table

    def rho_mix(self, y, y1, y2, rho1, rho2):
        eta1 = (y2 - y)/(y2 - y1)
        eta2 = 1-eta1
        rho_inv = eta1/rho1 + eta2/rho2
        return 1/rho_inv

    def t_mix(self, y, y1, y2, t1, t2):
        eta1 = (y2 - y)/(y2 - y1)
        eta2 = 1-eta1
        tmix = eta1*t1 + eta2*t2
        return tmix

    def get_rho_mix(self):

        if 0.22 <= self.Y < 0.25:
            Y1, Y2 = 0.22, 0.25
            s_1, lgp_1, lgt_1, lgrho_1 = np.load('inverted_eos_data/inverted_scvh_updated_22.npy')
            s_2, lgp_2, lgt_2, lgrho_2 = np.load('inverted_eos_data/inverted_scvh_updated_25.npy')

        elif 0.25 <= self.Y  <= 0.28:
            Y1, Y2 = 0.25, 0.28
            s_1, lgp_1, lgt_1, lgrho_1 = np.load('inverted_eos_data/inverted_scvh_updated_25.npy') # take out _new to get the old one
            #s_1, lgp_1, lgt_1, lgrho_1 = np.load('inverted_eos_data/inverted_adam_ST_scvh_25_new.npy') # take out _new to get the old one
            s_2, lgp_2, lgt_2, lgrho_2 = np.load('inverted_eos_data/inverted_scvh_updated_28.npy')

        elif 0.28 < self.Y  <= 0.30:
            Y1, Y2 = 0.28, 0.30
            s_1, lgp_1, lgt_1, lgrho_1 = np.load('inverted_eos_data/inverted_scvh_updated_28.npy')
            s_2, lgp_2, lgt_2, lgrho_2 = np.load('inverted_eos_data/inverted_scvh_updated_30.npy')

        else:
            # haven't yet incorporated the SCvH Helium table
            raise Exception('Helium fraction (Y) must be within 0.22 and 0.30')
        
        logp_base = []
        logrho_mix = []
        logt_mix = []
        s_base = []

        for i, s_ in enumerate(s_1[:,0]):
            logpgrid = np.linspace(3, 15, 300)
            fit = 'linear'
            logrho_interp1 = interp1d(lgp_1[i], lgrho_1[i], kind=fit, fill_value='extrapolate')
            logrho_interp2 = interp1d(lgp_2[i], lgrho_2[i], kind=fit, fill_value='extrapolate')
            
            logt_interp1 = interp1d(lgp_1[i], lgt_1[i], kind=fit, fill_value='extrapolate')
            logt_interp2 = interp1d(lgp_2[i], lgt_2[i], kind=fit, fill_value='extrapolate')
            
            
            logrho1 = logrho_interp1(logpgrid)
            logrho2 = logrho_interp2(logpgrid)
            
            logt1 = logt_interp1(logpgrid)
            logt2 = logt_interp2(logpgrid)

            rhomix = self.rho_mix(self.Y, Y1, Y2, 10**logrho1, 10**logrho2)
            tmix = self.t_mix(self.Y, Y1, Y2, 10**logt1, 10**logt2)
            logrho_mix.append(np.log10(rhomix))
            logt_mix.append(np.log10(tmix))
            s_base.append(s_1[i])
            logp_base.append(logpgrid)

        return np.array(logp_base), np.array(logt_mix), np.array(logrho_mix), np.array(s_base)

    def get_rho_p(self, s, p):
        logp_base, logt_mix, logrho_mix, s_base = self.get_rho_mix()
        
        interp = RBS(s_base[:,0], logp_base[0], logrho_mix) # logrho(S, logP)
        logrho_result = interp.ev(s, logp_base[0])
        
        logrho_interp = interp1d(logp_base[0], logrho_result, kind='linear', fill_value='extrapolate')
        
        return 10**logrho_interp(np.log10(p))

    def get_t_p(self, s, p):
        logp_base, logt_mix, logrho_mix, s_base = self.get_rho_mix()
        interp = RBS(s_base[:,0], logp_base[0], logt_mix) # logT(S, logP)
        logt_result = interp.ev(s, logp_base[0])
        
        logt_interp = interp1d(logp_base[0], logt_result, kind='linear', fill_value='extrapolate')
        
        return 10**logt_interp(np.log10(p))

    def get_grad_ad(self, s, p): 

            logp_base, logt_mix, logrho_mix, s_base = self.get_rho_mix()

            grad_ad = []

            for i, s_ in enumerate(s_base[:,0]):
                grad_ad.append(np.gradient(logt_mix[i]) / np.gradient(logp_base[i]))
            grad_ad = np.array(grad_ad)

            interp = RBS(s_base[:,0], logp_base[0], grad_ad)
            grad_ad_t = interp.ev(s, logp_base[0])

            #return logp_grid, grad_ad_t
            grad_ad_interp = interp1d(logp_base[0], grad_ad_t, kind='quadratic', fill_value='extrapolate')

            return grad_ad_interp(np.log10(p))
