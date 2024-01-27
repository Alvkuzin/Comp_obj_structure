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
import scipy
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
Z = 12
m_p = 1.67262192e-24
m_n = 1.67492749e-24
m_e = 9.1093837e-28
e_e = 4.803e-10


howdoidothat=np.array([0,1])
type_ref = type(howdoidothat)


def Phi(x):
    first = x * (1 + x**2)**0.5 * (2/3 * x**2 - 1)
    second = np.log(x + (1 + x**2)**0.5)
    return (first + second) / 8 / pi**2

def Chi(x):
    first = x * (1 + x**2)**0.5 * (2 * x**2 + 1)
    second = -np.log(x + (1 + x**2)**0.5)
    return (first + second) / 8 / pi**2
  
def e_emp(n):
    n0 = 0.16 * 1e36
    u = n / n0
    return (939.6 + 35.1 * u**(2/3) - 42.1 * u + 21 * u**2.112) * 1.6*1e-6 * n

def p_emp(n):
    n0 = 0.16 * 1e36
    u = n / n0
    return n**2 * (2/3 * 35.1*u**(-1/3) - 42.1 + 2.112*21*u**1.112)
    
def P(rho, model):
    if model == 'HW':
        if rho < 1e12: model = 'HW'
        else: model = 'pne'
    data = pd.read_csv(Path(Path.cwd(), 'EoS_'+model), sep = ' ',
                       names = ['rho_t', 'p_t'], skiprows=[0])
    rho_t, p_t = np.array(data['rho_t']), np.array(data['p_t'])
    return np.interp(rho, rho_t, p_t)

def Rho(P, model):
    if model == 'HW':
        if P < 5.3e29: model = 'HW'
        else: model = 'pne'
    data = pd.read_csv(Path(Path.cwd(), 'EoS_'+model), sep = ' ',
                       names = ['rho_t', 'p_t'], skiprows=[0])
    rho_t, p_t = np.array(data['rho_t']), np.array(data['p_t'])
    return np.interp(P, p_t, rho_t)
    
def Rho_a(P, model):
    rho_ = np.zeros(P.size)
    N = P.size
    i = 0
    for i in range(N):
        rho_[i] = Rho(P[i], model)
    return rho_

def rhs(r, y, model):
    m, p = y
    rho_ = Rho(p, model)
    dm = 4 * pi * rho_ * r**2
    first = 1 + p/rho_/c_light**2
    second = 1 + 4 * pi * r**3 * p / m / c_light**2
    third = (1 - 2 * G * m / r / c_light**2)**(-1)
#    first =1
#    second =1
#    third=1
    dp = -G * m * rho_ / r**2 * first * second * third
    return np.array([dm, dp])

def WD(rho0, model):
    P0 = P(rho0, model)
#    rho0_ = Rho(P0)
#    print(rho0, rho0_)
    r0 = 1
    m0 = 4*pi/3 * rho0 * r0**3
    init_conds = np.array([m0, P0])
    def ev(r, y, model):
#        return y[1] - G*2e33 / 0.4 / (6e8)**2
        return y[1] - P0/10**5.5
    ev.direction = -1
    ev.terminal=True
    res = solve_ivp(rhs, [r0, 1e20], y0 = init_conds,
                    args = (model,), events=ev, dense_output=True)
    R = float(res.t_events[0][0])
    M = float(res.y_events[0][0][0])
#    r = res.t
#    p = res.sol(r)[1]
#    m = res.sol(r)[0]
#    rho = Rho_a(p, model)
#    n_e = Y * rho / m_p
    return M, R

rhos = np.logspace(8.6, 17, 1030)
Ms = []
Rs = []
model = 'chNoCoul_HWNoCoul_pne_emp250'
for rho00 in rhos:
    M, R = WD(rho0 = rho00, model=model)
    Ms.append(M/2e33)
#    Rs.append(100*R/7e10)
    Rs.append(R/1e5)
    print(np.log10(rho00))
#    print(rho)
Rs = np.array(Rs)
Ms = np.array(Ms)
plt.plot(Rs, Ms)
plt.xscale('log')


