### ======================================================================================
#
#
#   Princeton Planetary Evolution Code
#   (Henyey method-based)
#
#
#   Authors: Ankan Sur
#   Supervisor: Prof. Adam Burrows
#   Affiliation: Department of Astrophysical Sciences, Princeton University
#   
#   
### ======================================================================================

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
import get_eos
import get_eos_scvh


#constants

G=G.to('cm^3/g s^2').value # Newton's constant
R_jup = 7.15e9 #6.99e9
M_jup = 1.89914e30
kb = k_B.to('erg/K') # ergs/K
mass = M_jup
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)

temper_yix,logg_yix,entropy_yix = np.genfromtxt("atmospheres/Jupiter_isolated_1solar_table.dat",unpack=True)



'''
We solve the hydrostatic equilibrium with Henyey formalism
'''

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
    
def fx(x):
    return (1 + 2.0*np.arctan(0.5*(x-1)))
    
    
def get_t_y(s, grav):
    lgg = np.log10(grav)
    t_arr, logg_arr, s_arr = np.load('yixian_inverted_atm.npy')
    t_res = RBS(s_arr[:,0], logg_arr[0,:], t_arr).ev(s_arr[:,0], lgg)
    t_interp = interp1d(s_arr[:,0], t_res, kind='quadratic', fill_value='extrapolate')
    return t_interp(s)


rad_p,m_p,rho_p,p_p,t_p = np.genfromtxt("isentrope_ankan_s9_polytrope.txt",unpack=True)

        
N = 128
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

#redefine the number of mesh points      
N = len(mesh)
    
f_p = interp1d(m_p,p_p,fill_value='extrapolate')
f_r = interp1d(m_p,rad_p,fill_value='extrapolate')

# Set the grids with initial values
m = M_jup*mesh[::-1]
r_old = np.zeros(N)
r_old[:-1] =  np.log(f_r(m)[:-1])
p_old = np.log(1e12*np.ones(N))
Y_old = np.zeros(2*N-1)


#set parameters
tol = 1e-4
Y = 0.25
eos = get_eos_scvh.eos(Y=Y) 
age=0
i = 0
vt = 2.5e-3
final_age = 6*const.Gyr_to_s
bc_atm="Yixian"
S = 9.0

data_r = []
data_p = []
data_rho = []
data_temp = []
dt_arr = []
data_teff = []
data_age = []
data_grav = []
data_S = []


program_starts = time.time()

while (age<=final_age):

    print ("S=",S)
    Y_old[:-1][::2] = r_old[:-1]
    Y_old[1:-1][::2] = p_old[:-1]
    Y_old[-1]=p_old[-1]
    error = 1
    itr=1
    
    while (error>tol):
        rho = np.log(eos.get_rho_p(S,np.exp(p_old)))
        J = jacobian(m,r_old,rho,p_old)
        A = matrix_A(r_old,p_old,m,rho)
        Y_new = Y_old - np.matmul(inv(J),A)
    
        error = max(abs(Y_new-Y_old)/abs(Y_old))
    
        Y_old = Y_new
        r_old[:-1] = Y_old[:-1][::2]
        p_old[:-1] = Y_old[1:-1][::2]
        p_old[-1] = Y_old[-1]
    
        print ("iteration= "+str(itr)+"  error=  "+str(error))
    
        itr=itr+1
                
    print ("\n")
    
    r = np.exp(r_old)
    p = np.exp(p_old)
    rho = np.exp(rho)
    temp = eos.get_t_p(S,p)

    t10 = interp1d(p, temp, kind='cubic')(1e7) if itr>0 else 1e3
    g = G*m[0]/r[0]**2


    if bc_atm=="fortney2011a":
        Teff = atm_boundary.Teff(g,t10,bc_atm)
    if bc_atm=="fortney2011b":
        Teff = atm_boundary.Teff(g,t10,bc_atm)
    if bc_atm=="gT1fit":
        Teff = atm_boundary.Teff(g,t10,bc_atm)
    if bc_atm=="Burrows":
        Teff = atm_boundary.Teff(g,t10,bc_atm)   
    if bc_atm=="Yixian":
        gr = 10**(logg_yix)
        Teff = get_t_y(S,g)
    

    data_r.append(np.exp(r_old))
    data_p.append(np.exp(p_old))
    data_rho.append(np.exp(rho))
    data_temp.append(temp)
    data_teff.append(Teff)
    data_age.append(age)
    data_S.append(S)
    data_grav.append(g)
    
    if i>2:
        drho_i = np.mean(np.abs(np.array(data_rho[i])-np.array(data_rho[i-1]))/np.array(data_rho[i-1]))
        drho_i_1 = np.mean(np.abs(np.array(data_rho[i-1])-np.array(data_rho[i-2]))/np.array(data_rho[i-2]))
        dT_i = np.mean(np.abs(np.array(data_temp[i])-np.array(data_temp[i-1]))/np.array(data_temp[i-1]))
        dT_i_1 = np.mean(np.abs(np.array(data_temp[i-1])-np.array(data_temp[i-2]))/np.array(data_temp[i-2]))
        vc_i_1 = np.mean(dT_i_1 + drho_i_1)                    
        vc_i = np.mean(dT_i + drho_i)
        dtime = dtime*(fx((fx(vt/vc_i)*fx(vt/vc_i_1)/fx(np.array(dt_arr[i-1])/np.array(dt_arr[i-2])))))**0.25
        
    else:
        dtime = 290168000000.0
           
    dt = (final_age-age)
    if dt<dtime and dt>0:
        dtime = dt
    
    dt_arr.append(dtime)
    age += dtime
    t_integral = np.trapz(temp,x=m[::-1])
    l = 4*np.pi*r[0]**2*const.sigma_sb*Teff**4 
    dS = (-l*dtime/t_integral)*erg_to_kbbar
    S = S+dS
    i+=1
    now = time.time()
            
print ("Code time: " + str(np.round(now-program_starts,2))+" s")
print ("Age: " + str(np.round((age-dtime)*const.s_to_Myr/1e3,2)) + " Gyr")
print ("T_eff: "+str(np.round(Teff,2))+ " K")

np.savetxt("our_comparison_yixian_NewEOS_HR.txt",np.c_[data_age,data_teff,data_grav,data_S])



