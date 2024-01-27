import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from numpy import pi
import time
from joblib import Parallel, delayed
import multiprocessing

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

def rhs(r, y, model):
    m, p = y
    rho_ = Rho(p, model)
    dm = 4 * pi * rho_ * r**2
    first = 1 + p/rho_/c_light**2
    second = 1 + 4 * pi * r**3 * p / m / c_light**2
    third = (1 - 2 * G * m / r / c_light**2)**(-1)
    dp = -G * m * rho_ / r**2 * first * second * third
    return np.array([dm, dp])

def WD(rho0, model):
    P0 = P(rho0, model)
    r0 = 1
    m0 = 4*pi/3 * rho0 * r0**3
    init_conds = np.array([m0, P0])
    def ev(r, y, model):
        return y[1] - P0/10**5.5
    ev.direction = -1
    ev.terminal=True
    res = solve_ivp(rhs, [r0, 1e20], y0 = init_conds,
                    args = (model,), events=ev, dense_output=True)
    R = float(res.t_events[0][0])
    M = float(res.y_events[0][0][0])
    return M, R

Nmax = 1030
rhos = np.logspace(8.6, 17, Nmax)
model = 'chNoCoul_HWNoCoul_pne_emp250'

def func(i):
    rho00 = rhos[i]
    M, R = WD(rho0 = rho00, model=model)
    return M/2e33, R/1e5
n_cores = multiprocessing.cpu_count()
res = Parallel(n_jobs=n_cores)(delayed(func)(i) for i in range(0, Nmax))
res = np.array(res)

Ms = res[:, 0]
Rs = res[:, 1]
plt.plot(Rs, Ms)
plt.xscale('log')

print('--- took %s sec'%(time.time() - start_time))


