import numpy as np
from numpy import sin, cos, exp
from pathlib import Path
from scipy.interpolate import splev, splrep

#from astropy import constants as const
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
#from scipy import optimize
from scipy import sparse
from scipy.sparse import linalg as sla
from scipy.optimize import brentq

import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
import re
from numpy import pi
from os import listdir
import time
from joblib import Parallel, delayed
import multiprocessing
from scipy.integrate import solve_bvp
start_time = time.time()
G = 6.67e-8
c_light = 3e10
h_planc = 6.626e-27
Y = 0.5
#Z = 12

m_p = 1.672621923e-24
m_n = 1.674927498e-24 
m_e = 9.1093837e-28
e_elec = 4.803e-10
aem = 1.007276466
#aem = 1.00
Q = m_n - m_p
lmbd_e = h_planc/2/pi/m_e/c_light

howdoidothat=np.array([0,1])
type_ref = type(howdoidothat)

"This is a script to tabulate different Equiations of State (EoS)"

"Phi-function in theory of degenerated gas"
def Phi(x):
    first = x * (1 + x**2)**0.5 * (2/3 * x**2 - 1)
    second = np.log(x + (1 + x**2)**0.5)
    return (first + second) / 8 / pi**2

"Chi-function in theory of degenerated gas"
def Chi(x):
    first = x * (1 + x**2)**0.5 * (2 * x**2 + 1)
    second = -np.log(x + (1 + x**2)**0.5)
    return (first + second) / 8 / pi**2

"Weizsäcker formula"
def M(a, z):
    b1, b2, b3, b4, b5 = 0.991749, 1.911e-2, 8.4e-4, 0.10175, 7.63e-4
    return m_p * c_light**2 * (b1 * a + b2 * a**(2/3) - b3 * z +
                          b4 * a * (0.5 - z / a)**2 + b5 * z**2 / a**(1/3))

"P(rho) for pne-gas"
def Pres_pne(rho):
    rho0 = m_p/3/pi**2/lmbd_e**3 * (Q**2/m_e**2-1)**1.5
    if rho < rho0:
        nn = lambda ne: 0
#        ne = rho/(m_e + m_p)
        np = lambda ne: ne
    if rho >= rho0:
        p0 = lambda ne: h_planc * (3 * ne/8/pi)**(1/3)/c_light
        n_to_p = lambda ne: ( (((m_p**2 + p0(ne)**2)**0.5 +
                  (m_e**2 + p0(ne)**2)**0.5)**2 - m_n**2)/p0(ne)**2 )**1.5
        np = lambda ne: ne
        nn = lambda ne: ne * n_to_p(ne)
#    n_to_p = lambda n_e: 100
#    eta_ = lambda n_e: 1 / n_to_p(n_e)

    pf_e = lambda n_e: h_planc/2/pi *(3*pi**2 * n_e)**(1/3)
    x = lambda n_e: pf_e(n_e) / m_e / c_light
    x_n = lambda n_e: h_planc/2/pi *(3*pi**2 * nn(n_e))**(1/3)/m_n/c_light
    x_p = lambda n_e: x(n_e) * m_e / m_p
#    x_n = lambda n_e: 0
#    x_p = lambda n_e: 0
    
    Lmbd_n = h_planc/2/pi/m_n/c_light
    P00 = m_n*c_light**2/Lmbd_n**3
    e_ = lambda n_e: P00 * (Chi(x_n(n_e)) + (m_p/m_n)**4 * Chi(x_p(n_e))+ 
     (m_e/m_n)**4 * Chi(x(n_e)) )
#    e_ = lambda n_e: 0
    if rho > 1e6:
        rho_func = lambda n_e: - rho + e_(n_e)/c_light**2 
        n = brentq(rho_func, 1e20, 1e45)
    
        P_ = P00 * (Phi(x_n(n)) + (m_p/m_n)**4 * Phi(x_p(n)) 
        + (m_e/m_n)**4 * Phi(x(n)) )
    if rho <= 1e6:
        p_f = lambda n_e: h_planc * (3 * n_e / 8 / pi)**(1/3)
        x0 = lambda n_e: p_f(n_e)/ m_e / c_light
        rho_func = lambda n_e: m_e / lmbd_e**3 * Phi(x0(n_e)) + 2*m_p*n_e-rho
        n_e = brentq(rho_func, 1e20, 1e40)

    #    print('x0 in P = ', x0)
#        Lmbd = h_planc / 2 / pi / m_e / c_light
#        Pc = -0.3 * (4*pi/3 * Z**2 * n_e**4)**(1/3)*e_e**2
        P_ =  3*m_e * c_light**2 / lmbd_e**3 * Phi(x0(n_e)) #+ Pc

    return P_

"""
Chandrasecar-like EoS: 
rho = rho(by Weitz. formula with given a, z) + E_couloumb
P = (ideal e-gas pressure calculated from rho above) + P_couloumb
"""
def Pres_ch(rho, coul):
    a, z = 62, 27.8
#    a, z = 12, 6
    p_f = lambda n_e: h_planc * (3 * n_e / 8 / pi)**(1/3)
    x0 = lambda n_e: p_f(n_e)/ m_e / c_light
    if coul == True:
        Ec = lambda n_e: -0.9 * (4*pi/3 * z**2 * n_e**4)**(1/3)*e_elec**2
    else:
        Ec = lambda a: 0

    Pc = lambda n_e: Ec(n_e)/3
    rho_func = lambda n_e: Ec(n_e)/c_light**2 - rho + M(a, z)/c_light**2*n_e/z
    
    n_e = brentq(rho_func, 1e20, 1e40)
    P_ =  m_e * c_light**2 / lmbd_e**3 * Phi(x0(n_e)) + Pc(n_e)

    return P_

"""
Harrison-Wheeler EOS according ro Shapiro & Teukolsky
returns P(a), rho(a) 
"""
def Pres_AZ(a_here, coul):
    b1, b2, b3, b4, b5 = 0.991749, 1.911e-2, 8.4e-4, 0.10175, 7.63e-4
    Lmbd_n = h_planc / 2 / pi / m_n / c_light
    P0n = m_n * c_light**2 / Lmbd_n**3
    P0e = m_e * c_light**2 / lmbd_e**3
    z = lambda a: (b2 / 2 / b5 * a)**0.5  
#    m_u = lambda a: (m_p * z(a) + m_n * (a - z(a)) ) /a# *1.007276
    m_u = lambda a: m_p / aem
    def x_n(a):
#        ans1 = (b1 + 2/3 * b2 * a**(-1/3) + b4 * (1/4 - z(a)**2 / a**2) - b5 * z(a)**2 /3 /a**(4/3))*m_u(a)/m_n
        ans1 = ( (b1 + b4/4) + 0.5*b2*a**(-1/3) - b2*b4/2/b5/a)*m_u(a)/m_n#*1.007276
        if ans1 < 1.0:
            return 0
        if ans1 >= 1.0:
            return (ans1**2 - 1)**0.5
#    n_n = lambda a: 8*pi/3 * m_n**3*c_light**3/h_planc**3 * x_n(a)**3
        
    def x(a):
        ans2 = (b3 + b4 * (1 - 2 * z(a) / a) - 2 * b5 * z(a) / a**(1/3)) * m_u(a)/m_e#*1.007276
        if ans2 >= 0:
            return (ans2**2 + 2 * ans2)**0.5
        if ans2 < 0:
#            print('ans2<0, a = ', a_here)
            return 0
    n_e = lambda a: 8 * pi/3 * (m_e * c_light / h_planc * x(a))**3
#    x_p = lambda a: x(a) * m_e / m_p
    if coul == True:
        Ec = lambda a: -0.9 * (4*pi/3 * z(a)**2 * n_e(a)**4)**(1/3)*e_elec**2
    else:
        Ec = lambda a: 0
    Pc = lambda a: Ec(a)/3
    e_e = lambda a: P0e * Chi(x(a)) - m_e * c_light**2 * n_e(a) + Ec(a)
#    e_e = lambda a: 0
    e_n = lambda a: P0n * Chi(x_n(a)) 
#    e_n = lambda a: 0
    
    rho_func = lambda a: (n_e(a) * M(a, z(a))/z(a) + e_e(a) + e_n(a) )/c_light**2
#    rho_ = rho_func(a)
#    a_here = brentq(rho_func, 1, 1000)
    P_ = (P0n * Phi(x_n(a_here))  #+ (m_p/m_n)**4 * Phi(x_p(a_here)) 
    + P0e * Phi(x(a_here)) 
    + Pc(a_here) )
    return P_, rho_func(a_here)
#    return x(a), x_n(a)
    

"""
Emperical EoS for nuclear matter: e(n)
"""
def e_emp(n):
    MN = 939.6
    n0 = 0.16 * 1e39
    E0 = 0.3/m_n*(3/16/pi*h_planc**3*n0)**(2/3) /1.602e-6
#    E0 = 22.1
    BE=-16
    K0 = 250
    sgm = (K0 + 2*E0) / (3* E0 - 9*BE)
    B = (1+sgm)/(sgm-1)*(E0/3 - BE)
    A = BE - 5/3*E0-B
    u = n / n0
    S0 = 30
    e_sym = MN + E0*u**(2/3) + A/2*u + B/(1+sgm)*u**sgm
    e_assym = (2**(2/3)-1)*(u**(2/3) - u)*E0 + S0 * u
    return (e_sym + e_assym) * 1.6*1e-6 * n
#    return (MN + 35.1 * u**(2/3) - 42.1 * u + 21 * u**2.112) * 1.6*1e-6 * n

"""
Emperical EoS for nuclear matter: P(n)
"""
def Pres_emp(n):
#    MN = 939.6
    n0 = 0.16 * 1e39
    E0 = 0.3/m_n*(3/16/pi*h_planc**3*n0)**(2/3) /1.602e-6
#    E0 = 22.1
    BE=-16
    K0 = 250
    sgm = (K0 + 2*E0) / (3* E0 - 9*BE)
    B = (1+sgm)/(sgm-1)*(E0/3 - BE)
    A = BE - 5/3*E0-B
    u = n / n0
    S0 = 30
    de_sym =  2/3*E0*u**(-1/3) + A/2 + sgm*B/(1+sgm)*u**(sgm-1)
    de_assym = (2**(2/3)-1)*(2/3*u**(-1/3) - 1)*E0 + S0
#    print(A, B, sgm)
    return n**2 * (de_sym + de_assym)/n0* 1.6*1e-6
#    return n**2 * (2/3 * 35.1*u**(-1/3) - 42.1 + 2.112*21*u**1.112)* 1.6*1e-6/n0

"""
A number of scripts which tabulate a certain EoS
Sorry for the mess
"""

#xs = np.logspace(-2.75, 1.5, 1000)
#p = []
#rho = []
#P00 = m_n * c_light**2 / (h_planc/2/pi/m_n/c_light)**3
#for x in xs:
#    p.append(Phi(x)*P00 )
#    rho.append(Chi(x)/c_light**2*P00)
#    #P = np.array(P)
#rho = np.array(rho)
#p = np.array(p)
#arr = pd.DataFrame({'rho': rho, 'p': p})
#arr.to_csv(Path(Path.cwd(), 'EoS_n'), sep= ' ')
##rho = rho 
##gamma = np.gradient(np.log10(P), np.log10(rho))
#
#plt.plot(rho/1.6/1e-6/1e36*c_light**2, p/1.6/1e-6/1e36)
##plt.plot(rho, rho/3)
##plt.plot(rho, gamma)
#
#plt.xscale('log')
#plt.yscale('log')

#rho_pne = np.logspace(np.log10(4.335e12), np.log10(2.19e14), 1000)
rho_pne = np.logspace(5, 18, 1000)
    
P_pne = []
for rho in rho_pne:
    P_pne.append(Pres_pne(rho))
P_pne = np.array(P_pne)
#gamma = np.gradient(np.log10(P), np.log10(rhos))

plt.plot(rho_pne, P_pne, color='b', label = r'$pne$-жидкость')
#plt.plot(rhos, gamma)
#arr = pd.DataFrame({'rho': rhos, 'p': P})
#arr.to_csv(Path(Path.cwd(), 'EoS_pne'), sep= ' ')
#plt.xscale('log')
#plt.yscale('log')
###    
#a1 = np.linspace(62, 70, 500)
#a2 = np.linspace(70, 183, 500)
a2 = np.linspace(62, 400, 1000)
coul = True
#a = np.concatenate((a1, a2))
a=a2
P_HW = []
rho_HW = []
for a in a:
    p_, rho_ = Pres_AZ(a, coul)
    P_HW.append(p_)
    rho_HW.append(rho_)
P_HW = np.array(P_HW)
rho_HW = np.array(rho_HW)
#rho = rho 
#gamma = np.gradient(np.log10(P), np.log10(rho))
#arr = pd.DataFrame({'rho': rho, 'p': P})
#arr.to_csv(Path(Path.cwd(), 'EoS_HW'), sep= ' ')
plt.plot(rho_HW, P_HW, color='k', ls = '-', label = 'HW-модель')
#plt.plot(rho, gamma)

plt.xscale('log')
plt.yscale('log')

#rho_ch = np.logspace(2.7, np.log10(5.81e6), 1000)
rho_ch = np.logspace(2.7, 8, 1000)

P_ch = []
for rho in rho_ch:
    P_ch.append(Pres_ch(rho, coul))
P_ch = np.array(P_ch)
plt.plot(rho_ch, P_ch, color='r', ls = '-', label = r'Ионы + $e^-$')
#
##gamma = np.gradient(np.log10(P), np.log10(rhos))
#
##
ns = np.logspace(37, 41, 1000)
#ns = np.logspace(38.15, 41, 1000)

P_emp = []
rho_emp = []
for n in ns:
    rho_ =_ = e_emp(n)/c_light**2
#    if rho_ > 1.2e10:
    P_emp.append(Pres_emp(n))
#    if rho_ <= 1.2e10:
#        p.append(Pres_pne(rho_))
    rho_emp.append(e_emp(n)/c_light**2)
    #P = np.array(P)
rho_emp = np.array(rho_emp)
P_emp = np.array(P_emp)
#arr = pd.DataFrame({'rho': rho, 'p': p})
#arr.to_csv(Path(Path.cwd(), 'EoS_emp'), sep= ' ')
#rho = rho 
#gamma = np.gradient(np.log10(P), np.log10(rho))

plt.plot(rho_emp, P_emp, color = 'g', label = 'Эмп. ядерная модель')
#plt.legend()
plt.ylabel(r'$P$')
plt.xlabel(r'$\rho$')

#Ptot, rhotot = np.concatenate((P_ch, P_HW, P_pne, P_emp)), np.concatenate((rho_ch, rho_HW, rho_pne, rho_emp)) 
#arr = pd.DataFrame({'rho': rhotot, 'p': Ptot})
#arr.to_csv(Path(Path.cwd(), 'EoS_chNoCoul_HWNoCoul_pne_emp250'), sep= ' ')

###spline = splrep(np.log10(rhotot), np.log10(Ptot))
###rho_plot =  np.logspace(1, 15, 10000)
###p_plot = np.interp(rho_plot, rhotot, Ptot)
###plt.plot(rho_plot, p_plot)
#plt.xscale('log')
#plt.yscale('log')

#
#    
#    
#    
