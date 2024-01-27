import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from numpy import pi

G = 6.67e-8
c_light = 3e10
h_planc = 6.626e-27
Y = 0.5
Z = 12
m_b = 1.673e-24
m_e = 9.11e-28
e_e = 4.8e-10


howdoidothat=np.array([0,1])
type_ref = type(howdoidothat)


def phi(x):
    first = x * (1 + x**2)**0.5 * (2/3 * x**2 - 1)
    second = np.log(x + (1 + x**2)**0.5)
    return (first + second) / 8 / pi**2

def chi(x):
    first = x * (1 + x**2)**0.5 * (2 * x**2 + 1)
    second = -np.log(x + (1 + x**2)**0.5)
    return (first + second) / 8 / pi**2

def P(rho):
    n_e = Y * rho / m_b
    p_f = h_planc * (3 * n_e / 8 / pi)**(1/3)
    x0 = p_f/ m_e / c_light
#    print('x0 in P = ', x0)
    Lmbd = h_planc / 2 / pi / m_e / c_light
    Pc = -0.3 * (4*pi/3 * Z**2 * n_e**4)**(1/3)*e_e**2
    return m_e * c_light**2 / Lmbd**3 * phi(x0) #+ Pc


#def Rho(P):
##    print(P)
#    Lmbd = h_planc / 2 / pi / m_e / c_light
#    P00 = m_e * c_light**2 / Lmbd**3
#    if P < 1e15:
#        rho_ = (P / 1e13 / Y**(5/3))**(3/5)
##    if P > 1e24:
##        rho_ = (P / 1.24e15 / Y**(4/3))**(3/4) 
#    else:
##        ne_ = lambda x: 8*pi/3 * (m_e * c_light * x / h_planc)**3
##        Pc = lambda x: -0.3 * (4*pi/3 * Z**2 * ne_(x)**4)**(1/3)*e_e**2
#        to_solve = lambda x: abs(P) / P00 - phi(x)
#        x0 = brentq(to_solve, 1e-10, 1e10, xtol=1e-10, rtol=1e-4)
#        if x0 < 0:
#            print('x0<0: p/p0 = ', P/P00)
#        p_f = x0 * m_e * c_light
#        n_e = 8*pi/3 * p_f**3/h_planc**3
#        rho_ = m_b / Y * n_e
#
#    return rho_ * np.sign(P)
def Rho(P):
    Lmbd_n = h_planc / 2 / pi / m_b / c_light
    P00 = m_b * c_light**2 / Lmbd_n**3
    if P < 1e15:
        rho_ = (P / 1e13 / Y**(5/3))**(3/5)
    else:
        n_to_p = lambda x: 1
        eta_ = lambda x: 1/n_to_p(x)
        ne = lambda x: 8*pi/3 * m_e**3*c_light**3/h_planc**3 * x**3
        np = lambda x: ne(x)
        nn = lambda x: np(x) * n_to_p(x)
        
        xn = lambda x: x / eta_(x)**(1/3) * m_e / m_b
        xp = lambda x: x * m_e / m_b
        
        P_tot = lambda x: P00 * (#Phi(xn(x)) + (m_p/m_n)**4 * Phi(xp(x)) +
                                 (m_e/m_b)**4 * phi(x) )
        func_here = lambda x: P_tot(x) - P
#        print(P)
        x0 = brentq(func_here, 1e-5, 1e7)
#        e_ = lambda x: P00 * (#Chi(xn(x)) + (m_p/m_n)**4 * Chi(xp(x))+ 
#         (m_e/m_n)**4 * Chi(x) )
        e_ = lambda x: 0
        rho_ = e_(x0)/c_light**2 + m_b * nn(x0) + np(x0) * m_b
    return rho_

def Rho_a(P):
    rho_ = np.zeros(P.size)
    N = P.size
    i = 0
    for i in range(N):
        rho_[i] = Rho(P[i])
    return rho_

def rhs(r, y):
    m, p = y
    rho_ = Rho(p)
    dm = 4 * pi * rho_ * r**2
    first = 1 + p/rho_/c_light**2
    second = 1 + 4 * pi * r**3 * p / m / c_light**2
    third = (1 - 2 * G * m / r / c_light**2)**(-1)
#    first =1
#    second =1
#    third=1
    dp = -G * m * rho_ / r**2 * first * second * third
    return np.array([dm, dp])

def WD(rho0):
    P0 = P(rho0)
#    rho0_ = Rho(P0)
#    print(rho0, rho0_)
    r0 = 1
    m0 = 4*pi/3 * rho0 * r0**3
    init_conds = np.array([m0, P0])
    def ev(r, y):
#        return y[1] - G*2e33 / 0.4 / (6e8)**2
        return y[1] - P0/1e8
    ev.direction = -1
    ev.terminal=True
    res = solve_ivp(rhs, [r0, 1e20], y0 = init_conds, events=ev, dense_output=True)
    R = float(res.t_events[0][0])
    M = float(res.y_events[0][0][0])
    r = res.t
    p = res.sol(r)[1]
    m = res.sol(r)[0]
    rho = Rho_a(p)
    n_e = Y * rho / m_b
#    plt.plot(r, n_e, lw=1, ls = '-', color='k')
#    plt.plot(r, m/max(m), lw=1, ls = '--', color='k')
    
#    plt.xscale('linear')
#    plt.yscale('log')
    return M, R

rhos = np.logspace(9, 11.3, 50)
Ms = []
Rs = []
for rho00 in rhos:
    M, R = WD(rho00)
    Ms.append(M/2e33)
    Rs.append(100*R/7e10)
#    print(rho)
Rs = np.array(Rs)
Ms = np.array(Ms)

plt.scatter(Ms, Rs, s=1)    