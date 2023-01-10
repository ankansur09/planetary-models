import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy 
import eos
from numpy.linalg import inv
from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import math
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm
import const
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.integrate import trapz, solve_ivp
from numba import jit
from astropy import units as u
from astropy.constants import u as amu
from alive_progress import alive_bar
import time
from atmospheres import atm_boundary
from scipy.interpolate import griddata
from astropy.constants import k_B, m_e, m_p, m_n, G, M_jup, M_earth, R_jup, R_earth

G=G.to('cm^3/g s^2').value # Newton's constant
R_jup = 7.15e9 #6.99e9
M_jup = 1.89914e30
kb = k_B.to('erg/K') # ergs/K
mass = M_jup

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)

import get_eos
import get_eos_scvh

temper_yix,logg_yix,entropy_yix = np.genfromtxt("atmospheres/Jupiter_isolated_1solar_table.dat",unpack=True)

def jacobian(m,x,q,y):
    J = np.zeros((2*N-1,2*N-1))
    J[0][1] = 1
    for k in range(1,N-1):
        j = 2*k-1
        i = k-1
                
        J[j,j-1]= 1 + 1.5*(m[i]-m[i+1])/4/np.pi*np.exp(-1.5*(x[i]+x[i+1])-0.5*(q[i]+q[i+1]))
        J[j,j+1]=-1 + 1.5*(m[i]-m[i+1])/4/np.pi*np.exp(-1.5*(x[i]+x[i+1])-0.5*(q[i]+q[i+1]))
        
        
        J[j+1,j-1]= -G/4./np.pi*(m[i]**2-m[i+1]**2)*np.exp(-0.5*(y[i]+y[i+1])-2.0*(x[i+1]+x[i]))
        J[j+1,j]= 1 - G/16./np.pi*(m[i]**2-m[i+1]**2)*np.exp(-0.5*(y[i]+y[i+1])-2.0*(x[i+1]+x[i]))
        J[j+1,j+1]= -G/4./np.pi*(m[i]**2-m[i+1]**2)*np.exp(-0.5*(y[i]+y[i+1])-2.0*(x[i+1]+x[i]))
        J[j+1,j+2]= -1 - G/16/np.pi*(m[i]**2-m[i+1]**2)*np.exp(-0.5*(y[i]+y[i+1])-2.0*(x[i+1]+x[i]))
                  
    J[2*N-3][-3] = 1
    J[-1][-2] = 1
    J[-1][-1] = -1-G/2.*(4*np.pi/3)**(1./3.)*(np.exp(4./3.*q[-1]-y[-1]))
    
    return J
    
def matrix_A(x,y,m,q):
    A = np.zeros(2*N-1)
    
    A[0] = y[0]-13.815
    A[1:2*N-3][::2] = x[:N-2]-x[1:N-1] - (m[:N-2]-m[1:N-1])/4.0/np.pi * np.exp(-1.5*(x[:N-2]+x[1:N-1])-0.5*(q[:N-2]+q[1:N-1]))
    A[2:2*N-2][::2] = y[:N-2]-y[1:N-1] + (m[:N-2]**2-m[1:N-1]**2)*G/8.0/np.pi*np.exp(-2*(x[:N-2]+x[1:N-1])-0.5*(y[:N-2]+y[1:N-1]))
    A[-2] = x[-2] - 1./3.*(np.log(3*m[-2])+q[0])
    A[-1] = y[-2]-y[-1] + G/2/np.pi * (4*np.pi/3.0)**(1./3.)*m[-2]**(2./3.)*np.exp(4./3.*q[-1]-y[-1])
    
    return A

rad_p,m_p,rho_p,p_p,t_p = np.genfromtxt("isentrope_ankan_s9_polytrope.txt",unpack=True)
        
N = 116
mesh_surf_amp=1e5
mesh_surf_width=1e-2
f0 = np.linspace(0, 1, N)
density_f0 = 1. / np.diff(f0)
density_f0 = np.insert(density_f0, 0, density_f0[0])
density_f0 += mesh_surf_amp * f0 * np.exp((f0 - 1.) / mesh_surf_width) * np.mean(density_f0) # boost mesh density near surface
out = np.cumsum(1. / density_f0)
out -= out[0]
out /= out[-1]

for i in range(10):
    out = np.insert(out,(i+1),out[i+1]/2.0**(10-i))
        
mesh = out
        
N = len(mesh)
    
f_p = interp1d(m_p,p_p)
f_r = interp1d(m_p,rad_p)

m = M_jup*mesh[::-1]
r_old = np.zeros(N)
r_old[:-1] =  np.log(f_r(m)[:-1])
p_old = np.log(1e12*np.ones(N))


Y_old = np.zeros(2*N-1)
error = 1
tol = 1e-5
Y = 0.25
S = 6.0
itr=1

eos = get_eos_scvh.eos(Y=Y) 
        
Y_old[:-1][::2] = r_old[:-1]
Y_old[1:-1][::2] = p_old[:-1]
Y_old[-1]=p_old[-1]

while (error>tol):
    rho = np.log(eos.get_rho_p(S,np.exp(p_old)))
    J = jacobian(m,r_old,rho,p_old)
    A = matrix_A(r_old,p_old,m,rho)
    Y_new = Y_old - np.matmul(inv(J),A)
    #print (A)
    
    error = max(abs(Y_new-Y_old)/abs(Y_old))
    
    Y_old = Y_new
    r_old[:-1] = Y_old[:-1][::2]
    p_old[:-1] = Y_old[1:-1][::2]
    p_old[-1] = Y_old[-1]
    
    print ("iteration= "+str(itr)+"  error=  "+str(error))
    
    itr=itr+1

r = np.exp(r_old)
p = np.exp(p_old)
rho = np.exp(rho)
temp = eos.get_t_p(S,p)






