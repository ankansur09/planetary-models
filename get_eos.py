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
    def __init__(self, Y, path, eos_table):
        self.Y = Y
        self.path = path
        self.eos_table = eos_table

    def rho_mix(self, y, y1, y2, rho1, rho2, s1, s2):
        if len(rho1) > len(rho2):
            rho1 = rho1[np.where(np.isin(s1[:,0],s2[:,0]))]

        elif len(rho2) > len(rho1):
            rho2 = rho2[np.where(np.isin(s2[:,0],s1[:,0]))]

        eta1 = (y2 - y)/(y2 - y1)
        eta2 = 1-eta1
        rho_inv = eta1/rho1 + eta2/rho2
        return 1/rho_inv

    def get_rho_mix(self, base=None): # this should be pressure independent because all pressures are the same
        # eos strings can only be either cms, scvh, or mh13 for now
        #path = '/Users/Helios/ongp/v2/'
        if 0.22 <= self.Y < 0.25:
            Y1, Y2 = 0.22, 0.25
            s_1, lgp_1, lgt_1, lgrho_1 = np.load(self.path+'inverted_eos_data/inverted_table_SP_{}_22.npy'.format(self.eos_table))
            s_2, lgp_2, lgt_2, lgrho_2 = np.load(self.path+'inverted_eos_data/inverted_table_SP_{}_25.npy'.format(self.eos_table))

        elif 0.25 <= self.Y  < 0.28:
            Y1, Y2 = 0.25, 0.30
            s_1, lgp_1, lgt_1, lgrho_1 = np.load(self.path+'inverted_eos_data/inverted_table_SP_{}_25.npy'.format(self.eos_table))
            s_2, lgp_2, lgt_2, lgrho_2 = np.load(self.path+'inverted_eos_data/inverted_table_SP_{}_28.npy'.format(self.eos_table))

        elif 0.28 <= self.Y  <= 0.32:
            Y1, Y2 = 0.28, 0.32
            s_1, lgp_1, lgt_1, lgrho_1 = np.load(self.path+'inverted_eos_data/inverted_table_SP_{}_28.npy'.format(self.eos_table))
            s_2, lgp_2, lgt_2, lgrho_2 = np.load(self.path+'inverted_eos_data/inverted_table_SP_{}_32.npy'.format(self.eos_table))

        else:
            # haven't yet incorporated the SCvH Helium table
            raise Exception('Helium fraction (Y) must be within 0.22 and 0.32')

        rho = self.rho_mix(self.Y, Y1, Y2, 10**np.array(lgrho_1), 10**np.array(lgrho_2), s_1, s_2)

            # making sure the bases are the same... some table inversions span across lower entropies, affecting the length of the arrays.
        if len(lgp_1) >= len(lgp_2):
            lgp_base = lgp_1[np.where(np.isin(s_1[:,0],s_2[:,0]))]
            lgt_base = lgt_1[np.where(np.isin(s_1[:,0],s_2[:,0]))]
            s_base = s_1[np.where(np.isin(s_1[:,0],s_2[:,0]))]

        elif len(lgp_2) > len(lgp_1):
            lgp_base = lgp_2[np.where(np.isin(s_2[:,0],s_1[:,0]))]
            lgt_base = lgt_2[np.where(np.isin(s_2[:,0],s_1[:,0]))]
            s_base = s_2[np.where(np.isin(s_2[:,0],s_1[:,0]))]

        # debugging
        # if base == 'SP':
        #     return lgp_base, np.log10(rho), s_base
        # elif base == 'ST':
        #     return lgt_base, np.log10(rho), s_base
        # else:
        return lgp_base, lgt_base, np.log10(rho), s_base

    # getting pressure and density for a given entropy and helium fraction
    # the pressure and density here must be 1-D interpolated

    def get_rho_p(self, s, p): # rho(S,P)
    #def get_rho_p(self, s):
        logp_base, logt_base, logrho_mix, s_base = self.get_rho_mix()
        interp = RBS(s_base[:,0], logp_base[0], logrho_mix)

        #logp_grid = np.linspace(np.min(logp_base[0,:]), np.max(logp_base[0,:]), 1000)
        logrho_result = interp.ev(s, logp_base[0])

        #return logp_grid, logrho_result
        # commenting these out for testing... we want to return the 1-d interpolation
        rho_interp = interp1d(logp_base[0], logrho_result, kind='quadratic', fill_value='extrapolate')
        return 10**rho_interp(np.log10(p))

    ### ROB: this function can't work with the current eos basis because the temps are different for each entropy;
    ### i.e., logt_base[i] != logt_base[j]

    # def get_rho_t(self, s, t): # make sure the temperature is not in log space
    # #def get_rho_t(self, s):
    #     logp_base, logt_base, logrho_mix, s_base = self.get_rho_mix()
    #     interp = RBS(s_base[:,0], logt_base[0,:], logrho_mix) # rho(S,T)
    #
    #     logt_grid = np.linspace(np.min(logt_base[0,:]), np.max(logt_base[0,:]), 1000)
    #     logrho_result = interp.ev(s, logt_grid)
    #     # commenting these out for testing... we want to return the 1-d interpolation
    #     #rho_interp = interp1d(logt_grid, logrho_result, kind='quadratic', fill_value='extrapolate')
    #     #return 10**rho_interp(np.log10(t))
    #     return logt_grid, logrho_result

    def get_t_p(self, s, p):
    #def get_t(self, s):
        logp_base, logt_base, logrho_mix, s_base = self.get_rho_mix()
        interp = RBS(s_base[:,0], logp_base[0], logt_base) # logT(S, logP)

        #logp_grid = np.linspace(np.min(logp_base[0,:]), np.max(logp_base[0,:]), 1000)
        logt_result = interp.ev(s, logp_base[0])

        t_interp = interp1d(logp_base[0], logt_result, kind='linear', fill_value='extrapolate')
        return 10**t_interp(np.log10(p))

    def get_s(self, logp, logt):
        if self.eos_table == 'cms':
            cms = chabrier.eos(path_to_data = '/Users/ankansur/Desktop/ongp-master/v2/data/')
            s = (10**cms.get(logp, logt, self.Y)['logs'])*erg_to_kbbar # S(logP, logT)

        elif self.eos_table == 'scvh':
            scvh_ = scvh.eos(path_to_data = '/Users/ankansur/Desktop/ongp-master/v2/data/')
            s = (10**scvh_.get(logp, logt, self.Y)['logs'])*erg_to_kbbar

        elif self.eos_table == 'mh13':
            mh13 = mh13_scvh.eos(path_to_data = '/Users/ankansur/Desktop/ongp-master/v2/data/')
            s = (10**mh13.get(logp, logt, self.Y)['logs'])*erg_to_kbbar

        return s

    def get_grad_ad(self, s, p): # p is meant ot be self.p in the heyney code, make sure these aren't logs

        logp_base, logt_base, logrho_mix, s_base = self.get_rho_mix()

        grad_ad = np.zeros((len(s_base),len(s_base[0])))

        for i in range(len(s_base)):
            grad_ad[i] = np.gradient(logt_base[i]) / np.gradient(logp_base[i])

        interp = RBS(s_base[:,0], logp_base[0], grad_ad)
        #logp_grid = np.linspace(np.min(logp_base[0,:]), np.max(logp_base[0,:]), 1000)
        grad_ad_t = interp.ev(s, logp_base[0])

        #return logp_grid, grad_ad_t
        grad_ad_interp = interp1d(logp_base[0], grad_ad_t, kind='quadratic', fill_value='extrapolate')

        return grad_ad_interp(np.log10(p)) # this could be set equal to self.grada as in the line below
        #self.grada = gradA_interp(np.log10(self.p))
