import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import griddata
from functools import reduce
import sys

g = np.array([[3.2,3.2,3.2,3.2],[5.6,5.6,5.6,5.6,5.6],[9.1,9.1,9.1,9.1,9.1,9.1,9.1],\
            [13.5,13.5,13.5,13.5,13.5,13.5,13.5,13.5,13.5],[18.2,18.2,18.2,18.2,18.2,18.2,18.2,18.2,18.2,18.2],\
            [22.4,22.4,22.4,22.4,22.4,22.4,22.4,22.4,22.4],[25.1,25.1,25.1,25.1,25.1,25.1],\
            [28.2,28.2,28.2,28.2,28.2,28.2]],dtype=object)
              
def Teff(grav,temp,atm_bc):
    
    if atm_bc=="fortney2011a": 
        T_eff_1 = np.array([[1201.6,904.7,751.3,600.6],[1201.2,902.5,750.8,600.4,450.6],\
        [1201.8,901.5,750.5,600.4,450.6,317.4,227.3],[1201.5,901.5,750.4,600.3,450.6,317.4,227.4,175.3,146.4],\
        [750.3,600.3,450.6,317.4,227.4,175.4,146.4,131.9,125.,119.7],\
        [600.3,450.6,317.4,227.4,175.4,146.5,131.9,125,119.8],\
        [227.4,175.4,146.5,132.,125.1,119.9],[227.7,176.,146.5,132.,125.1,120.]],dtype=object)   
                             
        T_10_1 = np.array([[3471.1,3124.9,2894.2,2582.3],[3359.3,2984.3,2730.3,2391.7,1966.2],\
        [3255.1,2850.3,2571.4,2222.1,1816,1312.2,842.6],[3165.6,2731.4,2435.8,2086.1,1701.6,1233.5,791.8,558.7,451.4],\
        [2330.5,1987.6,1617.7,1174.5,754.4,533.5,430.9,375.8,348.9,322.9],\
        [1918.6,1561,1135.3,729.5,515.9,417.1,364.3,337.9,317.9],[716, 506.9,409.7,357.4,331.9,312.5],\
        [703.7,499.4,402.3,350.9,325.8,307.2]],dtype=object)                            
        
        px = np.array(reduce(lambda x, y : x + y, g))
        py = np.array(reduce(lambda x, y : x + y, T_10_1))
        f = np.array(reduce(lambda x, y : x + y, T_eff_1))
        
        X = np.zeros((8,10))
        Y = np.zeros((8,10))
        Z = np.zeros((8,10))
        for i in range(8):
            Y[i,:] = np.linspace(min(T_10_1[i]),max(T_10_1[i]),10)[::-1]
            X[i,:] = np.linspace(min(g[i]),max(g[i]),10)[::-1]
            Z[i,:] = griddata((px, py), f, (X[i,:],Y[i,:]), method='cubic')
        i_in = np.unravel_index((np.abs(X - grav/100.0)).argmin(), X.shape)
        i_in = i_in[0]
        j_in = np.unravel_index((np.abs(Y[i_in] - temp)).argmin(), Y[i_in].shape)
        j_in = j_in[0]

        teff_interp = griddata((px, py), f, (grav/100.0,temp), method='cubic')
        
        if np.isnan(teff_interp):

            return Z[i_in][j_in] + (Z[i_in][j_in]-Z[i_in-1][j_in])/(X[i_in][j_in]-X[i_in-1][j_in])*(grav/100.0-X[i_in][j_in]) +\
             (Z[i_in][j_in]-Z[i_in][j_in-1])/(Y[i_in][j_in]-Y[i_in][j_in-1])*(-Y[i_in][j_in]+temp)    
        
        
        else:
            return teff_interp
            
        
    if atm_bc=="fortney2011b":
        T_eff_07 = np.array([[1201.6,904.7,751.2,600.5],[1202.6,  902.6,  750.7,  600.4,450.4], \
        [1202.2,  901.5,  750.5,  600.3,  450.4,  317. ,  226.4], \
        [1201.8,  901. ,  750.4,  600.3,450.4,  317. ,  226.4,  173.2,  142.7], \
        [750.3,  600.2,  450.4,  317. ,  226.4,  173.2, 142.8,  126.8,  118.9,  112.9], \
        [600.2, 450.4,  317. ,  226.4,  173.3,  142.8,  126.8,  119. ,  113.],[226.4,  173.3,142.8,  126.9,  119.1,  113.1], \
        [226.6,  173.7,  142.8,  126.9,  119.1,  113.2]],dtype=object)
         
        T_10_07 = np.array([[3471.1, 3124.9, 2894.2, 2582.4],  [3360.3, 2984.3, 2730.2, 2391.6,1966.1], \
        [3255.8, 2850.2, 2571.4, 2222.1, 1815.9, 1310.2,  838.], \
        [3166. , 2731.4, 2435.8, 2086. ,1701.6, 1231.7,  787.4,  552.4,  439.9], \
        [2330.5, 1987.6, 1617.6, 1172.6,  750.4,  527.5,419.6,  360. ,  330.1,  306.9], \
        [1918.6, 1560.9, 1133.5,  725.3,  510. ,  406.5,  348.5,  320. ,  296.9], \
        [712. ,  501.1,399.2,  342.3,  314.3,  292.3],[699.5,  493.1,  391.6,  336.1,  308.7,  286.8]],dtype=object)
        
        px = np.array(reduce(lambda x, y : x + y, g))
        py = np.array(reduce(lambda x, y : x + y, T_10_07))
        f = np.array(reduce(lambda x, y : x + y, T_eff_07))
        
        X = np.zeros((8,10))
        Y = np.zeros((8,10))
        Z = np.zeros((8,10))
        for i in range(8):
            Y[i,:] = np.linspace(min(T_10_07[i]),max(T_10_07[i]),10)[::-1]
            X[i,:] = np.linspace(min(g[i]),max(g[i]),10)[::-1]
            Z[i,:] = griddata((px, py), f, (X[i,:],Y[i,:]), method='cubic')
        i_in = np.unravel_index((np.abs(X - grav/100.0)).argmin(), X.shape)
        i_in = i_in[0]
        j_in = np.unravel_index((np.abs(Y[i_in] - temp)).argmin(), Y[i_in].shape)
        j_in = j_in[0]

        teff_interp = griddata((px, py), f, (grav/100.0,temp), method='cubic')
        
        if np.isnan(teff_interp):

            return Z[i_in][j_in] + (Z[i_in][j_in]-Z[i_in-1][j_in])/(X[i_in][j_in]-X[i_in-1][j_in])*(grav/100.0-X[i_in][j_in]) +\
             (Z[i_in][j_in]-Z[i_in][j_in-1])/(Y[i_in][j_in]-Y[i_in][j_in-1])*(-Y[i_in][j_in]+temp)    
        
        
        else:
            return teff_interp
        
            
           
    if atm_bc=="gT1fit":
        K = 1.519
        expon_teff = 1.243
        return (grav**0.167*temp/K)**(1.0/expon_teff)
        
    if atm_bc=="Burrows":
        col1,col2 = np.genfromtxt("/Users/ankansur/Desktop/Planetary Code/atmospheres/Burrows_data.dat",unpack=True)
        gravity = col2[::502]
        Teffec = []
        T10 = []
        gr = []
        for i in range(31):
            Teffec.append(list(col1[i+1+i*501:502*(i+1)]))
            T10.append(list(col2[i+1+i*501:502*(i+1)]))
            gr.append(list(np.repeat(gravity[i],501)))
            
            
        X = np.array(gr)
        Y = np.array(T10)
        Z = np.array(Teffec)
        
        gr = np.array(gr).flatten()
        Teffec = np.array(Teffec).flatten()
        T10 = np.array(T10).flatten()
        
        teff_interp = griddata((gr, T10), Teffec, (grav,temp), method='cubic')
        
        '''
        if np.isnan(teff_interp):
            K = 1.519
            expon_teff = 1.243
            return (grav**0.167*temp/K)**(1.0/expon_teff)  
        else:
            return teff_interp
        '''  
        i_in = np.unravel_index((np.abs(gr - grav)).argmin(), X.shape)
        i_in = i_in[0]
        j_in = np.unravel_index((np.abs(Y[i_in] - temp)).argmin(), Y[i_in].shape)
        j_in = j_in[0]

        
        if np.isnan(teff_interp):

            return Z[i_in][j_in] + (Z[i_in][j_in]-Z[i_in-1][j_in])/(X[i_in][j_in]-X[i_in-1][j_in])*(grav-X[i_in][j_in]) +\
             (Z[i_in][j_in]-Z[i_in][j_in-1])/(Y[i_in][j_in]-Y[i_in][j_in-1])*(-Y[i_in][j_in]+temp)    
        
        else:
            return teff_interp 
         
    
        
        
        
        
        