from pylab import *
import numpy as np
import scipy
from scipy.linalg import null_space
from scipy import integrate
from matplotlib import rcParams
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

def Fermi(x):
		return 1.0/(exp(x)+1)

def Bose(x):
		return 1.0/(exp(x)-1)

def NMOS(V_DD,Vin,E0):
	A=zeros((2,2))
	Gamma_l=0.2
	Gamma_r=0.2
	Gamma=0.2
	Gamma_g=0.2

	J_d = 0.0
	mu_l=0.0
	kBT=1.0

	tint=10
	T=1000
	Ntot=int(T/tint)


	for i in range(Ntot):
		V_D = V_DD
		E_N = E0 - Vin
		mu_r = mu_l - V_D

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
		J_d = -(k_Nr * (1 - p_N) - k_rN * p_N)

	return J_d, V_D

V_DD_max = 11
# V_DD = np.zeros(V_DD_max)
Vin = np.zeros(V_DD_max*10)
J_d1 = np.zeros(V_DD_max*10)
J_d2 = np.zeros(V_DD_max*10)
J_d3 = np.zeros(V_DD_max*10)
J_d4 = np.zeros(V_DD_max*10)
J_d5 = np.zeros(V_DD_max*10)
V_D1 = np.zeros(V_DD_max*10)
V_D2 = np.zeros(V_DD_max*10)
V_D3 = np.zeros(V_DD_max*10)
V_D4 = np.zeros(V_DD_max*10)
V_D5 = np.zeros(V_DD_max*10)

i = 0
for V in np.arange(0,V_DD_max,0.1):
	Vin[i] = V-10.0
	J_d1[i], V_D1[i] = NMOS(15.0, Vin[i], -2.0)
	J_d2[i], V_D2[i] = NMOS(15.0, Vin[i], -1.0)
	J_d3[i], V_D3[i] = NMOS(15.0, Vin[i], 0.0)
	J_d4[i], V_D4[i] = NMOS(15.0, Vin[i], 1.0)
	J_d5[i], V_D5[i] = NMOS(15.0, Vin[i], 2.0)
	i = i + 1

with open('/results/NMOS_transfer.txt', 'w') as f:
    f.write("J_d1\tJ_d2\tJ_d3\tJ_d4\tJ_d5\n")
    for i in range(len(Vin)):
        f.write(f"{J_d1[i]:.6f}\t{J_d2[i]:.6f}\t{J_d3[i]:.6f}\t{J_d4[i]:.6f}\t{J_d5[i]:.6f}\n")
