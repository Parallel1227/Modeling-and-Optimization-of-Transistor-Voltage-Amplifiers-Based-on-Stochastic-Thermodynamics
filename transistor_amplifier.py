import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import scipy
from scipy.linalg import null_space
from scipy import integrate
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def Fermi(x):
    return 1.0 / (exp(x) + 1)


def Bose(x):
    if (x > 1e-4):
        return 1.0 / (exp(x) - 1)
    else:
        return 1e5


def NMOS(V_DD, Vin, Gamma):
    A = zeros((2, 2))
    Gamma_l = 0.2
    Gamma_r = 0.2
    Gamma_g = 0.2

    mu_l = 0.0
    kBT = 1.0

    E_N = - Vin
    Vout = V_DD

    for V_D in range(0, 1500, 1):
        V_D = V_D / 100.0
        mu_r = mu_l - V_D
        mu_DD = mu_l - V_DD

        J_d1 = 0.5 * Gamma * (Fermi((mu_r - mu_r) / kBT) - Fermi((mu_r - mu_DD) / kBT))

        k_Nl = Gamma_l * Fermi((E_N - mu_l) / kBT)
        k_lN = Gamma_l * (1.0 - Fermi((E_N - mu_l) / kBT))
        k_rN = Gamma_r * (1.0 - Fermi((E_N - mu_r) / kBT))
        k_Nr = Gamma_r * Fermi((E_N - mu_r) / kBT)

        A[1][0] = k_Nr + k_Nl
        A[0][0] = -A[1][0]
        A[0][1] = k_rN + k_lN
        A[1][1] = -A[0][1]

        p = null_space(A)
        sum = p[0][0] + p[1][0]
        p[0][0] = p[0][0] / sum
        p[1][0] = p[1][0] / sum
        p_N = p[1][0]
        J_d2 = -(k_Nr * (1 - p_N) - k_rN * p_N)

        if abs(J_d1 - J_d2) < 1e-3:
            Vout = V_D
            break

    return J_d1, Vout


T = 20
tint = 1
Ntot = int(T / tint)
adj = 2  # 增加以减少周期数
A = 0.1  # 幅值
V_DD = 15.0
Gamma1 = 0.25
I_DSS = 0.05

time = np.zeros(Ntot*20)
Vin = np.zeros(Ntot*20)
Vout1 = np.zeros(Ntot*20)
Vout1_d = np.zeros(Ntot*20)
# diss_rate1 = 0.0
# diss_rate2 = 0.0
# diss1 = np.zeros(Ntot*20)
# diss2 = np.zeros(Ntot*20)

I_DSS1, VDSQ1 = NMOS(V_DD, 0.0, Gamma1)

i = 0
for t in np.arange(0,Ntot,0.05):
    time[i] = t * tint
    Vin[i] = sin(t / adj) * A
    J_d1, Vout1[i] = NMOS(V_DD, Vin[i], Gamma1)
    Vout1_d[i] = Vout1[i]
    Vout1[i] = Vout1[i] - VDSQ1
    i = i + 1

with open('/results/transistor_amplifier.txt', 'w') as f:
    f.write("time\tVin\tVout\n")
    for i in range(len(time)):
        f.write(f"{time[i]:.6f}\t{Vin[i]:.6f}\t{Vout1[i]:.6f}\n")