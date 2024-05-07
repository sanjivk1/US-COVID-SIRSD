import traceback
import numpy as np
import pandas as pd
import time
import math
import concurrent.futures
import multiprocessing
from scipy.optimize import minimize
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sys
from sklearn.metrics import r2_score, mean_squared_error
from SIRfunctions import SEIARG, SEIARG_fixed, weighted_deviation, weighted_relative_deviation, computeBeta_combined
import datetime
from numpy.random import uniform as uni
import os
import warnings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

np.set_printoptions(threshold=sys.maxsize)
# Geo = 0.98
Geo = 0.98
num_para = 14

num_threads = 10
# num_threads = 1 #SK-12-24
num_CI = 1000
# num_CI = 5
start_dev = 0

num_threads_dist = 0

# weight of G in initial fitting
theta = 0.7
# weight of G in release fitting
theta2 = 0.8

I_0 = 5
Om_Intital = 50
beta_range = (1, 50)
gammaE_range = (0.2, 0.3)
alpha_range = (0.1, 0.9)
gamma_range = (0.04, 0.2)
gamma2_range = (0.04, 0.2)
gamma3_range = (0.04, 0.2)
# sigma_range = (0.001, 1)
a1_range = (0.01, 0.5)
a2_range = (0.001, 0.2)
a3_range = (0.01, 0.2)
eta_range = (0.001, 0.95)
c1_fixed = (0.9, 0.9)
c1_range = (0.8, 1)
h1_range = (1 / 30, 1 / 14)
Hiding_init1_range = (0, 10)
h2_range = (1 / 30, 1 / 14)
Hiding_init2_range = (0, 10)
k_range = (0.1, 2)
k2_range = (0.1, 2)
E_initial_range = (0, 1)
I_initial_range = (0, 1)
start_date = '2021-02-01'
reopen_date1 = '2021-03-15'
reopen_date2 = '2021-06-10'
reopen_date3 = '2021-12-16'
fitting_enddate = '2021-12-15'

Om_start_date = '2021-12-15'

vac_date1 = '2021-02-01'
daily_vspeed1 = 0.0015
vac_date2 = '2021-06-15'
daily_vspeed2 = 0.00225
daily_vspeed3 = 0.0036
daily_vspeed4 = 0.004
vac_date3 = '2021-08-16'
vac3_On = True
# daily_vspeed = 0mohfwOct2.json
v_period1 = 14
v_period2 = 7 * 12
v_period3 = 120
v_eff1 = 0.65
v_eff2 = 0.8
v_eff3 = 0.4
# release_duration = 30
# k_drop = 14
# p_m = 1
# Hiding = 0.33
# delay = 7
# change_eta2 = False
size_ext = 150

fig_row = 5
fig_col = 3

ANTIBODY_RATIO = 0.40
HIDING_DELAY = 25
HIDING_FRACTION = 0.20

states = ['kl', 'dl', 'tg', 'rj', 'hr', 'jk', 'ka', 'la', 'mh', 'pb', 'tn', 'up', 'ap', 'ut', 'or', 'wb', 'py', 'ch',
          'ct', 'gj', 'hp', 'mp', 'br', 'mn', 'mz', 'ga', 'an', 'as', 'jh', 'ar', 'tr', 'nl', 'ml', 'sk', 'dn_dd', 'ld']

state_dict = {'up': 'Uttar Pradesh',
              'mh': 'Maharastra',
              'br': 'Bihar',
              'wb': 'West Bengal',
              'mp': 'Madhya Pradesh',
              'tn': 'Tamil Nadu',
              'rj': 'Rajesthan',
              'ka': 'Karnataka',
              'gj': 'Gujarat',
              'ap': 'Andhra Pradesh',
              'or': 'Odisha',
              'tg': 'Telangana',
              'kl': 'Kerala',
              'jh': 'Jharkhand',
              'as': 'Assam',
              'pb': 'Punjab',
              'ct': 'Chhattisgarh',
              'hr': 'Haryana',
              'dl': 'Delhi',
              'jk': 'Jammu and Kashmir',
              'ut': 'Uttarakhand',
              'hp': 'Himachal Pradesh',
              'tr': 'Tripura',
              'ml': 'Meghalaya',
              'mn': 'Manipur',
              'nl': 'Nagaland',
              'ga': 'Goa',
              'ar': 'Arunachal Pradesh',
              'py': 'Puducherry',
              'mz': 'Mizoram',
              'ch': 'Chandigarh',
              'sk': 'Sikkim',
              'dn_dd': 'Daman and Diu',
              'an': 'Andaman and Nicobar',
              'ld': 'Ladakh',
              'la': 'Lakshdweep',
              'India': 'India'
              }


def simulate_vac(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, c1,
                 n_0, vaccine_speeds, vac_period1, vac_period2, vac_eff1, vac_eff2, releases):
	result = True

	# vaccine_speeds = [vac_speed] * size

	# no dose
	S0 = S.copy()
	E0 = E.copy()
	I0 = I.copy()
	A0 = A.copy()
	IH0 = IH.copy()
	IN0 = IN.copy()
	D0 = D.copy()
	R0 = R.copy()
	G0 = G.copy()
	H0 = H.copy()
	betas = [beta]

	# 1st dose
	S1 = [0]
	E1 = [0]
	I1 = [0]
	A1 = [0]
	IH1 = [0]
	IN1 = [0]
	D1 = [0]
	R1 = [0]
	G1 = [0]
	H1 = [0]

	# 2nd dose (or 2 weeks after 1st dose for it to take effect)
	S2 = [0]
	E2 = [0]
	I2 = [0]
	A2 = [0]
	IH2 = [0]
	IN2 = [0]
	D2 = [0]
	R2 = [0]
	G2 = [0]
	H2 = [0]

	# fully vaccinated
	S3 = [0]
	E3 = [0]
	I3 = [0]
	A3 = [0]
	IH3 = [0]
	IN3 = [0]
	D3 = [0]
	R3 = [0]
	G3 = [0]
	H3 = [0]

	Hiding0 = H0[0] + H1[0] + H2[0] + H3[0]
	# HH = [release_size * n_0]
	# daily_release = release_speed * release_size * n_0
	for i in range(1, size):

		vaccine_speed = vaccine_speeds[i]

		beta_t = computeBeta_combined(beta, eta, n_0,
		                              S0[-1] + S1[-1] + S2[-1] + S3[-1],
		                              0,
		                              H0[-1] + H1[-1] + H2[-1] + H3[-1],
		                              c1, Hiding0)
		dS0 = -beta_t * S0[-1] * (I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0
		dE0 = beta_t * S0[-1] * (I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - gammaE * \
		      E0[-1]
		dI0 = (1 - alpha) * gammaE * E0[-1] - (gamma + gamma2) * I0[-1]
		dA0 = alpha * gammaE * E0[-1] - gamma3 * A0[-1]
		dIH0 = gamma * I0[-1] - (a1 + a2) * IH0[-1]
		dIN0 = gamma2 * I0[-1] - a3 * IN0[-1]
		dD0 = a2 * IH0[-1]
		dR0 = a1 * IH0[-1] + a3 * IN0[-1] + gamma3 * A0[-1]
		dG0 = (1 - alpha) * gammaE * E0[-1]

		dS1 = -beta_t * S1[-1] * (I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0
		dE1 = beta_t * S1[-1] * (I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - gammaE * \
		      E1[-1]
		dI1 = (1 - alpha) * gammaE * E1[-1] - (gamma + gamma2) * I1[-1]
		dA1 = alpha * gammaE * E1[-1] - gamma3 * A1[-1]
		dIH1 = gamma * I1[-1] - (a1 + a2) * IH1[-1]
		dIN1 = gamma2 * I1[-1] - a3 * IN1[-1]
		dD1 = a2 * IH1[-1]
		dR1 = a1 * IH1[-1] + a3 * IN1[-1] + gamma3 * A1[-1]
		dG1 = (1 - alpha) * gammaE * E1[-1]

		dS2 = -beta_t * (1 - vac_eff1) * S2[-1] * (
				I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0
		dE2 = beta_t * (1 - vac_eff1) * S2[-1] * (
				I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - gammaE * E2[-1]
		dI2 = (1 - alpha) * gammaE * E2[-1] - (gamma + gamma2) * I2[-1]
		dA2 = alpha * gammaE * E2[-1] - gamma3 * A2[-1]
		dIH2 = gamma * I2[-1] - (a1 + a2) * IH2[-1]
		dIN2 = gamma2 * I2[-1] - a3 * IN2[-1]
		dD2 = a2 * IH2[-1]
		dR2 = a1 * IH2[-1] + a3 * IN2[-1] + gamma3 * A2[-1]
		dG2 = (1 - alpha) * gammaE * E2[-1]

		dS3 = -beta_t * (1 - vac_eff2) * S3[-1] * (
				I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0
		dE3 = beta_t * (1 - vac_eff2) * S3[-1] * (
				I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - gammaE * E3[-1]
		dI3 = (1 - alpha) * gammaE * E3[-1] - (gamma + gamma2) * I3[-1]
		dA3 = alpha * gammaE * E3[-1] - gamma3 * A3[-1]
		dIH3 = gamma * I3[-1] - (a1 + a2) * IH3[-1]
		dIN3 = gamma2 * I3[-1] - a3 * IN3[-1]
		dD3 = a2 * IH3[-1]
		dR3 = a1 * IH3[-1] + a3 * IN3[-1] + gamma3 * A3[-1]
		dG3 = (1 - alpha) * gammaE * E3[-1]

		S0.append(S0[-1] + dS0)
		E0.append(E0[-1] + dE0)
		I0.append(I0[-1] + dI0)
		A0.append(A0[-1] + dA0)
		IH0.append(IH0[-1] + dIH0)
		IN0.append(IN0[-1] + dIN0)
		D0.append(D0[-1] + dD0)
		R0.append(R0[-1] + dR0)
		G0.append(G0[-1] + dG0)

		S1.append(S1[-1] + dS1)
		E1.append(E1[-1] + dE1)
		I1.append(I1[-1] + dI1)
		A1.append(A1[-1] + dA1)
		IH1.append(IH1[-1] + dIH1)
		IN1.append(IN1[-1] + dIN1)
		D1.append(D1[-1] + dD1)
		R1.append(R1[-1] + dR1)
		G1.append(G1[-1] + dG1)

		S2.append(S2[-1] + dS2)
		E2.append(E2[-1] + dE2)
		I2.append(I2[-1] + dI2)
		A2.append(A2[-1] + dA2)
		IH2.append(IH2[-1] + dIH2)
		IN2.append(IN2[-1] + dIN2)
		D2.append(D2[-1] + dD2)
		R2.append(R2[-1] + dR2)
		G2.append(G2[-1] + dG2)

		S3.append(S3[-1] + dS3)
		E3.append(E3[-1] + dE3)
		I3.append(I3[-1] + dI3)
		A3.append(A3[-1] + dA3)
		IH3.append(IH3[-1] + dIH3)
		IN3.append(IN3[-1] + dIN3)
		D3.append(D3[-1] + dD3)
		R3.append(R3[-1] + dR3)
		G3.append(G3[-1] + dG3)

		H0.append(H0[-1])
		H1.append(H1[-1])
		H2.append(H2[-1])
		H3.append(H3[-1])
		# HH0.append(HH0[-1])
		# HH1.append(HH1[-1])
		# HH2.append(HH2[-1])
		# HH3.append(HH3[-1])

		betas.append(beta_t)

		dS12 = S1[i] / vac_period1
		dS23 = S2[i] / vac_period2
		S1[i] -= dS12
		S2[i] = S2[i] - dS23 + dS12
		S3[i] += dS23

		dH12 = H1[i] / vac_period1
		dH23 = H2[i] / vac_period2
		H1[i] -= dH12
		H2[i] = H2[i] - dH23 + dH12
		H3[i] += dH23

		# if i >= reopen_day:
		# 	release = min(H0[-1], r)
		# 	S0[-1] += release
		# 	H0[-1] -= release

		total_H = H0[-1] + H1[-1] + H2[-1] + H3[-1]
		if total_H > 0:
			release = min(releases[i], total_H)
			frac0 = H0[-1] / total_H
			frac1 = H1[-1] / total_H
			frac2 = H2[-1] / total_H
			frac3 = H3[-1] / total_H
			S0[-1] += release * frac0
			S1[-1] += release * frac1
			S2[-1] += release * frac2
			S3[-1] += release * frac3
			H0[-1] -= release * frac0
			H1[-1] -= release * frac1
			H2[-1] -= release * frac2
			H3[-1] -= release * frac3
		# Hiding0 += release

		S1[-1] += S0[-1] * vaccine_speed
		S0[-1] -= S0[-1] * vaccine_speed
		H1[-1] += H0[-1] * vaccine_speed
		H0[-1] -= H0[-1] * vaccine_speed

	# if S0[-1] < 0:
	# 	result = False
	# 	break

	if result:
		S = [S0[i] + S1[i] + S2[i] + S3[i] for i in range(size)]
		E = [E0[i] + E1[i] + E2[i] + E3[i] for i in range(size)]
		I = [I0[i] + I1[i] + I2[i] + I3[i] for i in range(size)]
		A = [A0[i] + A1[i] + A2[i] + A3[i] for i in range(size)]
		IH = [IH0[i] + IH1[i] + IH2[i] + IH3[i] for i in range(size)]
		IN = [IN0[i] + IN1[i] + IN2[i] + IN3[i] for i in range(size)]
		D = [D0[i] + D1[i] + D2[i] + D3[i] for i in range(size)]
		R = [R0[i] + R1[i] + R2[i] + R3[i] for i in range(size)]
		G = [G0[i] + G1[i] + G2[i] + G3[i] for i in range(size)]
		H = [H0[i] + H1[i] + H2[i] + H3[i] for i in range(size)]

	return result, [S, E, I, A, IH, IN, D, R, G, H,
	                S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0,
	                S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
	                S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2,
	                S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
	                betas]


# Om_factor = 4/3  #2.0
beta_om_fac = 1.7


def simulate_vac_Omicron(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3,
                         eta, c1, n_0, vaccine_speeds, vac_period1, vac_period2, vac_period3, vac_eff1, vac_eff2,
                         vac_eff3, releases, OmStart_day, hiding_day, hiding_frac):
	result = True

	# no dose
	S0 = S.copy()
	E0 = E.copy()
	I0 = I.copy()
	A0 = A.copy()
	IH0 = IH.copy()
	IN0 = IN.copy()
	D0 = D.copy()
	R0 = R.copy()
	G0 = G.copy()
	H0 = H.copy()
	# beta = beta * beta_om_fac
	betas = [beta]

	E0_Om = [0]
	I0_Om = [0]
	A0_Om = [0]
	R0_Om = [0]
	G0_Om = [0]

	# 1st dose
	S1 = [0]
	E1 = [0]
	I1 = [0]
	A1 = [0]
	IH1 = [0]
	IN1 = [0]
	D1 = [0]
	R1 = [0]
	G1 = [0]
	H1 = [0]

	E1_Om = [0]
	I1_Om = [0]
	R1_Om = [0]
	G1_Om = [0]
	A1_Om = [0]

	# 2nd dose (or 2 weeks after 1st dose for it to take effect)
	S2 = [0]
	E2 = [0]
	I2 = [0]
	A2 = [0]
	IH2 = [0]
	IN2 = [0]
	D2 = [0]
	R2 = [0]
	G2 = [0]
	H2 = [0]

	E2_Om = [0]
	I2_Om = [0]
	A2_Om = [0]
	R2_Om = [0]
	G2_Om = [0]

	# fully vaccinated
	S3 = [0]
	E3 = [0]
	I3 = [0]
	A3 = [0]
	IH3 = [0]
	IN3 = [0]
	D3 = [0]
	R3 = [0]
	G3 = [0]
	H3 = [0]

	E3_Om = [0]
	I3_Om = [0]
	A3_Om = [0]
	R3_Om = [0]
	G3_Om = [0]

	# Vaccine compromised
	S4 = [0]
	E4 = [0]
	I4 = [0]
	A4 = [0]
	IH4 = [0]
	IN4 = [0]
	D4 = [0]
	R4 = [0]
	G4 = [0]
	H4 = [0]

	E4_Om = [0]
	I4_Om = [0]
	A4_Om = [0]
	R4_Om = [0]
	G4_Om = [0]

	Hiding0 = H0[0] + H1[0] + H2[0] + H3[0] + H4[0]
	# HH = [release_size * n_0]
	# daily_release = release_speed * release_size * n_0
	for i in range(1, size):

		vaccine_speed = vaccine_speeds[i]

		beta_t = computeBeta_combined(beta, eta, n_0,
		                              S0[-1] + S1[-1] + S2[-1] + S3[-1] + S4[-1],
		                              0,
		                              H0[-1] + H1[-1] + H2[-1] + H3[-1] + H4[-1],
		                              c1, Hiding0)
		# SK-OM
		I_Om = I0_Om[-1] + I1_Om[-1] + I2_Om[-1] + I3_Om[-1] + I4_Om[-1]
		A_Om = A0_Om[-1] + A1_Om[-1] + A2_Om[-1] + A3_Om[-1] + A4_Om[-1]
		# Increase_R1 = 15043/560
		# Increase_R2 = 3061/1234
		# beta_Om =   (np.log(Increase_R1/Increase_R2)/(eta*18)) + beta_t  # * Om_factor #SK-12/7
		# beta_Om =  (np.log(15043/560)/18)
		beta_Om = beta_t * beta_om_fac
		dS0_Om = beta_Om * S0[-1] * (I_Om + A_Om) / n_0  # SK-12/7

		I = I0[-1] + I1[-1] + I2[-1] + I3[-1] + I4[-1]
		A = A0[-1] + A1[-1] + A2[-1] + A3[-1] + A4[-1]
		# dS0 = -beta_t * S0[-1] * (I0[-1] + I1[-1] + I2[-1] + I3[-1] + I4[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1] + A4[-1]) / n_0 - dS0_Om
		# dE0 = beta_t * S0[-1] * (I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - gammaE * \
		#       E0[-1]
		dS0 = - beta_t * S0[-1] * (I + A) / n_0 - dS0_Om
		dE0 = beta_t * S0[-1] * (I + A) / n_0 - gammaE * E0[-1]

		dI0 = (1 - alpha) * gammaE * E0[-1] - (gamma + gamma2) * I0[-1]
		dA0 = alpha * gammaE * E0[-1] - gamma3 * A0[-1]
		dIH0 = gamma * I0[-1] - (a1 + a2) * IH0[-1]
		dIN0 = gamma2 * I0[-1] - a3 * IN0[-1]
		dD0 = a2 * IH0[-1]
		dR0 = a1 * IH0[-1] + a3 * IN0[-1] + gamma3 * A0[-1]
		dG0 = (1 - alpha) * gammaE * E0[-1]
		# SK
		dE0_Om = dS0_Om - gammaE * E0_Om[-1]
		dI0_Om = (1 - alpha) * gammaE * E0_Om[-1] - (gamma + gamma2) * I0_Om[-1]
		dA0_Om = alpha * gammaE * E0_Om[-1] - gamma3 * A0_Om[-1]
		dR0_Om = (gamma + gamma2) * I0_Om[-1]
		dG0_Om = (1 - alpha) * gammaE * E0_Om[-1]

		# OM
		# SK-12/7
		dS1_Om = beta_Om * S1[-1] * (I_Om + A_Om) / n_0  # SK-12/7

		# dS1 = -beta_t * S1[-1] * (I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - dS1_Om
		# dE1 = beta_t * S1[-1] * (I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - gammaE * \
		#       E1[-1]
		dS1 = - beta_t * S1[-1] * (I + A) / n_0 - dS1_Om
		dE1 = beta_t * S1[-1] * (I + A) / n_0 - gammaE * E1[-1]
		dI1 = (1 - alpha) * gammaE * E1[-1] - (gamma + gamma2) * I1[-1]
		dA1 = alpha * gammaE * E1[-1] - gamma3 * A1[-1]
		dIH1 = gamma * I1[-1] - (a1 + a2) * IH1[-1]
		dIN1 = gamma2 * I1[-1] - a3 * IN1[-1]
		dD1 = a2 * IH1[-1]
		dR1 = a1 * IH1[-1] + a3 * IN1[-1] + gamma3 * A1[-1]
		dG1 = (1 - alpha) * gammaE * E1[-1]
		# SK
		dE1_Om = dS1_Om - gammaE * E1_Om[-1]
		dI1_Om = (1 - alpha) * gammaE * E1_Om[-1] - (gamma + gamma2) * I1_Om[-1]
		dA1_Om = alpha * gammaE * E1_Om[-1] - gamma3 * A1_Om[-1]
		dR1_Om = (gamma + gamma2) * I1_Om[-1]
		dG1_Om = (1 - alpha) * gammaE * E1_Om[-1]

		vac_om_factor = 2
		# SK
		# SK-12/7
		dS2_Om = beta_Om * (1 - vac_eff1 / vac_om_factor) * S2[-1] * (I_Om + A_Om) / n_0  # SK-12/7
		dE2_Om = dS2_Om - gammaE * E2_Om[-1]
		dI2_Om = (1 - alpha) * gammaE * E2_Om[-1] - (gamma + gamma2) * I2_Om[-1]
		dA2_Om = alpha * gammaE * E2_Om[-1] - gamma3 * A2_Om[-1]
		dR2_Om = (gamma + gamma2) * I2_Om[-1]
		dG2_Om = (1 - alpha) * gammaE * E2_Om[-1]

		# dS2 = -beta_t * (1 - vac_eff1) * S2[-1] * (
		# 		I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - dS2_Om
		# dE2 = beta_t * (1 - vac_eff1) * S2[-1] * (
		# 		I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - gammaE * E2[-1]
		dS2 = - beta_t * (1 - vac_eff1) * S2[-1] * (I + A) / n_0 - dS2_Om
		dE2 = beta_t * (1 - vac_eff1) * S2[-1] * (I + A) / n_0 - gammaE * E2[-1]
		dI2 = (1 - alpha) * gammaE * E2[-1] - (gamma + gamma2) * I2[-1]
		dA2 = alpha * gammaE * E2[-1] - gamma3 * A2[-1]
		dIH2 = gamma * I2[-1] - (a1 + a2) * IH2[-1]
		dIN2 = gamma2 * I2[-1] - a3 * IN2[-1]
		dD2 = a2 * IH2[-1]
		dR2 = a1 * IH2[-1] + a3 * IN2[-1] + gamma3 * A2[-1]
		dG2 = (1 - alpha) * gammaE * E2[-1]

		# SK-12/7
		dS3_Om = beta_Om * (1 - vac_eff2 / vac_om_factor) * S3[-1] * (I_Om + A_Om) / n_0  # SK-12/7
		dE3_Om = dS3_Om - gammaE * E3_Om[-1]
		dI3_Om = (1 - alpha) * gammaE * E3_Om[-1] - (gamma + gamma2) * I3_Om[-1]
		dA3_Om = alpha * gammaE * E3_Om[-1] - gamma3 * A3_Om[-1]
		dR3_Om = (gamma + gamma2) * I3_Om[-1]
		dG3_Om = (1 - alpha) * gammaE * E3_Om[-1]

		# dS3 = -beta_t * (1 - vac_eff2) * S3[-1] * (
		# 		I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - dS3_Om
		# dE3 = beta_t * (1 - vac_eff2) * S3[-1] * (
		# 		I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - gammaE * E3[-1]
		dS3 = - beta_t * (1 - vac_eff2) * S3[-1] * (I + A) / n_0 - dS3_Om
		dE3 = beta_t * (1 - vac_eff2) * S3[-1] * (I + A) / n_0 - gammaE * E3[-1]
		dI3 = (1 - alpha) * gammaE * E3[-1] - (gamma + gamma2) * I3[-1]
		dA3 = alpha * gammaE * E3[-1] - gamma3 * A3[-1]
		dIH3 = gamma * I3[-1] - (a1 + a2) * IH3[-1]
		dIN3 = gamma2 * I3[-1] - a3 * IN3[-1]
		dD3 = a2 * IH3[-1]
		dR3 = a1 * IH3[-1] + a3 * IN3[-1] + gamma3 * A3[-1]
		dG3 = (1 - alpha) * gammaE * E3[-1]

		# New added

		dS4_Om = beta_Om * (1 - vac_eff3 / vac_om_factor) * S4[-1] * (I_Om + A_Om) / n_0  # SK-12/7
		dE4_Om = dS3_Om - gammaE * E4_Om[-1]
		dI4_Om = (1 - alpha) * gammaE * E4_Om[-1] - (gamma + gamma2) * I4_Om[-1]
		dA4_Om = alpha * gammaE * E4_Om[-1] - gamma3 * A4_Om[-1]
		dR4_Om = (gamma + gamma2) * I4_Om[-1]
		dG4_Om = (1 - alpha) * gammaE * E4_Om[-1]

		dS4 = - beta_t * (1 - vac_eff3) * S4[-1] * (I + A) / n_0 - dS4_Om
		dE4 = beta_t * (1 - vac_eff3) * S4[-1] * (I + A) / n_0 - gammaE * E4[-1]
		dI4 = (1 - alpha) * gammaE * E4[-1] - (gamma + gamma2) * I4[-1]
		dA4 = alpha * gammaE * E4[-1] - gamma3 * A4[-1]
		dIH4 = gamma * I4[-1] - (a1 + a2) * IH4[-1]
		dIN4 = gamma2 * I4[-1] - a3 * IN4[-1]
		dD4 = a2 * IH4[-1]
		dR4 = a1 * IH4[-1] + a3 * IN4[-1] + gamma3 * A4[-1]
		dG4 = (1 - alpha) * gammaE * E4[-1]

		S0.append(S0[-1] + dS0)
		E0.append(E0[-1] + dE0)
		I0.append(I0[-1] + dI0)
		A0.append(A0[-1] + dA0)
		IH0.append(IH0[-1] + dIH0)
		IN0.append(IN0[-1] + dIN0)
		D0.append(D0[-1] + dD0)
		R0.append(R0[-1] + dR0)
		G0.append(G0[-1] + dG0)
		# Om
		E0_Om.append(E0_Om[-1] + dE0_Om)
		I0_Om.append(I0_Om[-1] + dI0_Om)
		A0_Om.append(A0_Om[-1] + dA0_Om)
		R0_Om.append(R0_Om[-1] + dR0_Om)
		G0_Om.append(G0_Om[-1] + dG0_Om)

		S1.append(S1[-1] + dS1)
		E1.append(E1[-1] + dE1)
		I1.append(I1[-1] + dI1)
		A1.append(A1[-1] + dA1)
		IH1.append(IH1[-1] + dIH1)
		IN1.append(IN1[-1] + dIN1)
		D1.append(D1[-1] + dD1)
		R1.append(R1[-1] + dR1)
		G1.append(G1[-1] + dG1)
		# Om

		E1_Om.append(E1_Om[-1] + dE1_Om)
		I1_Om.append(I1_Om[-1] + dI1_Om)
		A1_Om.append(A1_Om[-1] + dA1_Om)
		R1_Om.append(R1_Om[-1] + dR1_Om)
		G1_Om.append(G1_Om[-1] + dG1_Om)

		S2.append(S2[-1] + dS2)
		E2.append(E2[-1] + dE2)
		I2.append(I2[-1] + dI2)
		A2.append(A2[-1] + dA2)
		IH2.append(IH2[-1] + dIH2)
		IN2.append(IN2[-1] + dIN2)
		D2.append(D2[-1] + dD2)
		R2.append(R2[-1] + dR2)
		G2.append(G2[-1] + dG2)
		# Om

		E2_Om.append(E2_Om[-1] + dE2_Om)
		I2_Om.append(I2_Om[-1] + dI2_Om)
		A2_Om.append(A2_Om[-1] + dA2_Om)
		R2_Om.append(R2_Om[-1] + dR2_Om)
		G2_Om.append(G2_Om[-1] + dG2_Om)

		S3.append(S3[-1] + dS3)
		E3.append(E3[-1] + dE3)
		I3.append(I3[-1] + dI3)
		A3.append(A3[-1] + dA3)
		IH3.append(IH3[-1] + dIH3)
		IN3.append(IN3[-1] + dIN3)
		D3.append(D3[-1] + dD3)
		R3.append(R3[-1] + dR3)
		G3.append(G3[-1] + dG3)

		# Om

		E3_Om.append(E3_Om[-1] + dE3_Om)
		I3_Om.append(I3_Om[-1] + dI3_Om)
		A3_Om.append(A3_Om[-1] + dA3_Om)
		R3_Om.append(R3_Om[-1] + dR3_Om)
		G3_Om.append(G3_Om[-1] + dG3_Om)

		S4.append(S4[-1] + dS4)
		E4.append(E4[-1] + dE4)
		I4.append(I4[-1] + dI4)
		A4.append(A4[-1] + dA4)
		IH4.append(IH4[-1] + dIH4)
		IN4.append(IN4[-1] + dIN4)
		D4.append(D4[-1] + dD4)
		R4.append(R4[-1] + dR4)
		G4.append(G4[-1] + dG4)

		# Om

		E4_Om.append(E4_Om[-1] + dE4_Om)
		I4_Om.append(I4_Om[-1] + dI4_Om)
		A4_Om.append(A4_Om[-1] + dA4_Om)
		R4_Om.append(R4_Om[-1] + dR4_Om)
		G4_Om.append(G4_Om[-1] + dG4_Om)

		H0.append(H0[-1])
		H1.append(H1[-1])
		H2.append(H2[-1])
		H3.append(H3[-1])
		H4.append(H4[-1])
		# HH0.append(HH0[-1])
		# HH1.append(HH1[-1])
		# HH2.append(HH2[-1])
		# HH3.append(HH3[-1])

		betas.append(beta_t)

		dS12 = S1[i] / vac_period1
		dS23 = S2[i] / vac_period2
		dS34 = S3[i] / vac_period3
		S1[i] -= dS12
		S2[i] = S2[i] - dS23 + dS12
		S3[i] = S3[i] + dS23 - dS34
		S4[i] = S4[i] + dS34

		dH12 = H1[i] / vac_period1
		dH23 = H2[i] / vac_period2
		dH34 = H3[i] / vac_period3
		H1[i] -= dH12
		H2[i] = H2[i] - dH23 + dH12
		H3[i] = H3[i] + dH23 - dH34
		H4[i] = H4[i] + dH34

		# if i >= reopen_day:
		# 	release = min(H0[-1], r)
		# 	S0[-1] += release
		# 	H0[-1] -= release
		if i < hiding_day:
			total_H = H0[-1] + H1[-1] + H2[-1] + H3[-1] + H4[-1]
			if total_H > 0:
				release = min(releases[i], total_H)
				frac0 = H0[-1] / total_H
				frac1 = H1[-1] / total_H
				frac2 = H2[-1] / total_H
				frac3 = H3[-1] / total_H
				frac4 = H4[-1] / total_H
				S0[-1] += release * frac0
				S1[-1] += release * frac1
				S2[-1] += release * frac2
				S3[-1] += release * frac3
				S4[-1] += release * frac4
				H0[-1] -= release * frac0
				H1[-1] -= release * frac1
				H2[-1] -= release * frac2
				H4[-1] -= release * frac4
		# Hiding0 += release
		elif i == hiding_day:
			H0[-1] += S0[-1] * hiding_frac
			H1[-1] += S1[-1] * hiding_frac
			H2[-1] += S2[-1] * hiding_frac
			H3[-1] += S3[-1] * hiding_frac
			H4[-1] += S4[-1] * hiding_frac
			S0[-1] -= S0[-1] * hiding_frac
			S1[-1] -= S1[-1] * hiding_frac
			S2[-1] -= S2[-1] * hiding_frac
			S3[-1] -= S3[-1] * hiding_frac
			S4[-1] -= S4[-1] * hiding_frac

		S1[-1] += S0[-1] * vaccine_speed
		S0[-1] -= S0[-1] * vaccine_speed
		H1[-1] += H0[-1] * vaccine_speed
		H0[-1] -= H0[-1] * vaccine_speed

		# if S0[-1] < 0:
		# 	result = False
		# 	break
		if i == OmStart_day:
			I0_Om[i] = I0_Om[i] + Om_Intital

	if result:
		S = [S0[i] + S1[i] + S2[i] + S3[i] + S4[i] for i in range(size)]
		E = [E0[i] + E1[i] + E2[i] + E3[i] + E4[i] for i in range(size)]
		I = [I0[i] + I1[i] + I2[i] + I3[i] + I4[i] for i in range(size)]
		A = [A0[i] + A1[i] + A2[i] + A3[i] + A4[i] for i in range(size)]
		IH = [IH0[i] + IH1[i] + IH2[i] + IH3[i] + IH4[i] for i in range(size)]
		IN = [IN0[i] + IN1[i] + IN2[i] + IN3[i] + IN4[i] for i in range(size)]
		D = [D0[i] + D1[i] + D2[i] + D3[i] + D4[i] for i in range(size)]
		R = [R0[i] + R1[i] + R2[i] + R3[i] + R4[i] for i in range(size)]
		G = [G0[i] + G1[i] + G2[i] + G3[i] + G4[i] for i in range(size)]
		H = [H0[i] + H1[i] + H2[i] + H3[i] + H4[i] for i in range(size)]
		E_Om = [E0_Om[i] + E1_Om[i] + E2_Om[i] + E3_Om[i] + E4_Om[i] for i in range(size)]
		I_Om = [I0_Om[i] + I1_Om[i] + I2_Om[i] + I3_Om[i] + I4_Om[i] for i in range(size)]
		A_Om = [A0_Om[i] + A1_Om[i] + A2_Om[i] + A3_Om[i] + A4_Om[i] for i in range(size)]
		R_Om = [R0_Om[i] + R1_Om[i] + R2_Om[i] + R3_Om[i] + R4_Om[i] for i in range(size)]
		G_Om = [G0_Om[i] + G1_Om[i] + G2_Om[i] + G3_Om[i] + G4_Om[i] for i in range(size)]

	return result, [S, E, I, A, IH, IN, D, R, G, H,
	                S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0,
	                S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
	                S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2,
	                S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
	                S4, E4, I4, A4, IH4, IN4, D4, R4, G4, H4,
	                E_Om, I_Om, A_Om, R_Om, G_Om,
	                betas]


def loss_init(point, c1, confirmed, death, n_0, vaccine_speeds, reopen_day1):
	size = len(confirmed)
	beta = point[0]
	gammaE = point[1]
	alpha = point[2]
	gamma = point[3]
	gamma2 = point[4]
	gamma3 = point[5]
	a1 = point[6]
	a2 = point[7]
	a3 = point[8]
	eta = point[9]
	E_initial = point[10]
	I_initial = point[11]
	h1 = point[12]
	Hiding_init1 = point[13]
	S = [n_0 * eta]
	E = [n_0 * eta * E_initial]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [n_0 * eta * Hiding_init1]
	releases = [0] * size
	H1_left = n_0 * eta * Hiding_init1
	r = h1 * H1_left
	release_days1 = math.ceil(1 / h1)
	release1 = []
	for i in range(release_days1):
		release1.append(min(r, H1_left))
		H1_left -= min(r, H1_left)
	for i in range(len(release1)):
		releases[reopen_day1 + i] = release1[i]
	# H = [0]
	result, [S, E, I, A, IH, IN, D, R, G, H,
	         S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
	         S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2, S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
	         betas] = simulate_vac(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1,
	                               a2, a3, eta, c1, n_0, vaccine_speeds, v_period1, v_period2, v_eff1,
	                               v_eff2, releases)

	if not result:
		return 1000

	size1 = reopen_day1
	size2 = size - size1
	weights1 = [Geo ** n for n in range(size1)]
	weights1.reverse()
	weights2 = [Geo ** n for n in range(size2)]
	weights2.reverse()
	weights = weights1
	weights.extend(weights2)

	dG = [G[i] - G[i - 1] for i in range(1, len(G))]
	dG.insert(0, 0)
	dD = [D[i] - D[i - 1] for i in range(1, len(D))]
	dD.insert(0, 0)
	d_confirmed = [confirmed[i] - confirmed[i - 1] for i in range(1, len(confirmed))]
	d_confirmed.insert(0, 0)
	d_death = [death[i] - death[i - 1] for i in range(1, len(death))]
	d_death.insert(0, 0)

	weighted_d_confirmed = [d_confirmed[i] * weights[i] for i in range(size)]
	weighted_d_G = [dG[i] * weights[i] for i in range(size)]
	weighted_d_death = [d_death[i] * weights[i] for i in range(size)]
	weighted_d_D = [dD[i] * weights[i] for i in range(size)]

	weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
	weighted_G = [G[i] * weights[i] for i in range(size)]
	weighted_death = [death[i] * weights[i] for i in range(size)]
	weighted_D = [D[i] * weights[i] for i in range(size)]

	metric0 = r2_score(weighted_confirmed, weighted_G)
	metric1 = r2_score(weighted_death, weighted_D)

	metricd0 = r2_score(weighted_d_confirmed, weighted_d_G)
	metricd1 = r2_score(weighted_d_death, weighted_d_D)

	metricf0 = 0.50 * metric0 + 0.50 * metricd0
	metricf1 = 0.50 * metric1 + 0.50 * metricd1
	sim_antibody_ratio = (R[-1] + S2[-1] + S3[-1] + H2[-1] + H3[-1]) / (S[0] + H[0])
	metric_antibody = 1 - (sim_antibody_ratio / ANTIBODY_RATIO - 1) ** 2
	return -(0.88 * metricf0 + 0.02 * metricf1 + 0.10 * metric_antibody)


# return -(0.90 * metricf0 + 0.10 * metricf1)


def loss_reopen(point, confirmed, death, n_0, vaccine_speeds, reopen_day2, reopen_day2_gov, para_init):
	size = len(confirmed)
	h2 = point[0]
	Hiding_init2 = point[1]
	[beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, c1, E_initial, I_initial, h1, Hiding_init1,
	 reopen_day1, metric_antibody] = para_init
	S = [n_0 * eta]
	E = [n_0 * eta * E_initial]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [n_0 * eta * (Hiding_init1 + Hiding_init2)]
	releases = [0] * size
	H1_left = n_0 * eta * Hiding_init1
	r = h1 * H1_left
	release_days1 = math.ceil(1 / h1)
	release1 = []
	for i in range(release_days1):
		release1.append(min(r, H1_left))
		H1_left -= min(r, H1_left)
	for i in range(len(release1)):
		releases[reopen_day1 + i] = release1[i]
	H2_left = n_0 * eta * Hiding_init2
	r = h2 * H2_left
	release_days2 = math.ceil(1 / h2)
	release2 = []
	for i in range(release_days2):
		release2.append(min(r, H2_left))
		H2_left -= min(r, H2_left)
	for i in range(len(release2)):
		if reopen_day2 + i < size:
			releases[reopen_day2 + i] = release2[i]
	# H = [0]
	result, [S, E, I, A, IH, IN, D, R, G, H,
	         S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
	         S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2, S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3, betas] = \
		simulate_vac(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta,
		             c1, n_0, vaccine_speeds, v_period1, v_period2, v_eff1, v_eff2, releases)

	if not result:
		return 1000

	size1 = reopen_day2_gov  # reopen_day2
	size2 = size - size1
	weights1 = [Geo ** n for n in range(size1)]
	weights1.reverse()
	weights2 = [Geo ** n for n in range(size2)]
	weights2.reverse()
	weights = weights1
	weights.extend(weights2)
	if len(weights) != size:
		print('wrong weights!')
	dG = [G[i] - G[i - 1] for i in range(1, len(G))]
	dG.insert(0, 0)
	dD = [D[i] - D[i - 1] for i in range(1, len(D))]
	dD.insert(0, 0)
	d_confirmed = [confirmed[i] - confirmed[i - 1] for i in range(1, len(confirmed))]
	d_confirmed.insert(0, 0)
	d_death = [death[i] - death[i - 1] for i in range(1, len(death))]
	d_death.insert(0, 0)

	weighted_d_confirmed = [d_confirmed[i] * weights[i] for i in range(size)]
	weighted_d_G = [dG[i] * weights[i] for i in range(size)]
	weighted_d_death = [d_death[i] * weights[i] for i in range(size)]
	weighted_d_D = [dD[i] * weights[i] for i in range(size)]

	weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
	weighted_G = [G[i] * weights[i] for i in range(size)]
	weighted_death = [death[i] * weights[i] for i in range(size)]
	weighted_D = [D[i] * weights[i] for i in range(size)]

	weighted_confirmed = weighted_confirmed[-size2:]
	weighted_G = weighted_G[-size2:]
	weighted_death = weighted_death[-size2:]
	weighted_D = weighted_D[-size2:]

	weighted_d_confirmed = weighted_d_confirmed[-size2:]
	weighted_d_G = weighted_d_G[-size2:]
	weighted_d_death = weighted_d_death[-size2:]
	weighted_d_D = weighted_d_D[-size2:]

	metric0 = r2_score(weighted_confirmed, weighted_G)
	metric1 = r2_score(weighted_death, weighted_D)

	metricd0 = r2_score(weighted_d_confirmed, weighted_d_G)
	metricd1 = r2_score(weighted_d_death, weighted_d_D)
	# print("metricd0, metricd1", metricd0,  metricd1)
	metricf0 = 0.50 * metric0 + 0.50 * metricd0
	metricf1 = 0.50 * metric1 + 0.50 * metricd1
	return -(0.97 * metricf0 + 0.03 * metricf1)


def fit_init(state, thread, confirmed0, death0, vaccine_speeds, reopen_day1_gov, n_0):
	print('fit init #', thread, state)
	np.random.seed()
	confirmed = confirmed0.copy()
	death = death0.copy()
	size = len(confirmed)
	# if metric2 != 0 or metric1 != 0:
	# 	scale1 = pd.Series(np.random.normal(1, metric1, size))
	# 	confirmed = [max(confirmed[i] * scale1[i], 1) for i in range(size)]
	# 	scale2 = pd.Series(np.random.normal(1, metric2, size))
	# 	death = [max(death[i] * scale2[i], 1) for i in range(size)]
	c_max = 0
	min_loss = 10000
	for reopen_day1 in range(reopen_day1_gov, reopen_day1_gov + 8):
		for c1 in np.arange(c1_range[0], c1_range[1], 0.01):  # Sk-12-24 changed from .01 reverted
			# optimal = minimize(loss, [10, 0.05, 0.01, 0.1, 0.1, 0.1, 0.02], args=(c1, confirmed, death, n_0, SIDRG_sd),
			optimal = minimize(loss_init, [uni(beta_range[0], beta_range[1]),
			                               uni(gammaE_range[0], gammaE_range[1]),
			                               uni(alpha_range[0], alpha_range[1]),
			                               uni(gamma_range[0], gamma_range[1]),
			                               uni(gamma2_range[0], gamma2_range[1]),
			                               uni(gamma3_range[0], gamma3_range[1]),
			                               uni(a1_range[0], a1_range[1]),
			                               uni(a2_range[0], a2_range[1]),
			                               uni(a3_range[0], a3_range[1]),
			                               uni(eta_range[0], eta_range[1]),
			                               uni(E_initial_range[0], E_initial_range[1]),
			                               uni(I_initial_range[0], I_initial_range[1]),
			                               uni(h1_range[0], h1_range[1]),
			                               uni(Hiding_init1_range[0], Hiding_init1_range[1])],
			                   args=(c1, confirmed, death, n_0, vaccine_speeds, reopen_day1),
			                   method='L-BFGS-B',
			                   bounds=[beta_range,
			                           gammaE_range,
			                           alpha_range,
			                           gamma_range,
			                           gamma2_range,
			                           gamma3_range,
			                           a1_range,
			                           a2_range,
			                           a3_range,
			                           eta_range,
			                           E_initial_range,
			                           I_initial_range,
			                           h1_range,
			                           Hiding_init1_range])
			current_loss = loss_init(optimal.x, c1, confirmed, death, n_0, vaccine_speeds, reopen_day1)
			if current_loss < min_loss:
				# print(f'updating loss={current_loss} with c1={c1} reopen={reopen_day1}')
				min_loss = current_loss
				c_max = c1
				reopen_day1_best = reopen_day1
				beta = optimal.x[0]
				gammaE = optimal.x[1]
				alpha = optimal.x[2]
				gamma = optimal.x[3]
				gamma2 = optimal.x[4]
				gamma3 = optimal.x[5]
				a1 = optimal.x[6]
				a2 = optimal.x[7]
				a3 = optimal.x[8]
				eta = optimal.x[9]
				E_initial = optimal.x[10]
				I_initial = optimal.x[11]
				h1 = optimal.x[12]
				Hiding_init1 = optimal.x[13]

	c1 = c_max
	reopen_day1 = reopen_day1_best
	S = [n_0 * eta]
	E = [n_0 * eta * E_initial]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [n_0 * eta * Hiding_init1]
	releases = [0] * size
	H1_left = n_0 * eta * Hiding_init1
	r = h1 * H1_left
	release_days1 = math.ceil(1 / h1)
	release1 = []
	for i in range(release_days1):
		release1.append(min(r, H1_left))
		H1_left -= min(r, H1_left)
	for i in range(len(release1)):
		releases[reopen_day1 + i] = release1[i]

	result, [S, E, I, A, IH, IN, D, R, G, H,
	         S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
	         S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2, S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3, betas] = \
		simulate_vac(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta,
		             c1, n_0, vaccine_speeds, v_period1, v_period2, v_eff1, v_eff2, releases)

	sim_antibody_ratio = (R[-1] + S2[-1] + S3[-1] + H2[-1] + H3[-1]) / (S[0] + H[0])
	metric_antibody = 1 - (sim_antibody_ratio / ANTIBODY_RATIO - 1) ** 2

	# weights = [Geo ** n for n in range(size)]
	# weights.reverse()

	# metric1 = weighted_relative_deviation(weights, confirmed, G, start_dev, num_para)
	# metric2 = weighted_relative_deviation(weights, death, D, start_dev, num_para)

	# r1 = r2_score(confirmed, G)
	# r2 = r2_score(death, D)

	return [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, c1, E_initial, I_initial, h1, Hiding_init1,
	        reopen_day1, sim_antibody_ratio], min_loss


def fit_reopen(state, thread, confirmed0, death0, vaccine_speeds, reopen_day2_gov, n_0, para_init):
	print('fit reopen #', thread, state)
	np.random.seed()
	confirmed = confirmed0.copy()
	death = death0.copy()
	size = len(confirmed)
	[beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, c1, E_initial, I_initial, h1, Hiding_init1,
	 reopen_day1, metric_antibody] = para_init
	# if metric2 != 0 or metric1 != 0:
	# 	scale1 = pd.Series(np.random.normal(1, metric1, size))
	# 	confirmed = [max(confirmed[i] * scale1[i], 1) for i in range(size)
	# 	scale2 = pd.Series(np.random.normal(1, metric2, size))
	# 	death = [max(death[i] * scale2[i], 1) for i in range(size)]
	c_max = 0
	min_loss = 10000
	# print(f'inside fit_reopen size={size} with reopen_day2_gov={reopen_day2_gov}') #SK-12-24-added
	# 8/5/2021 REQ: change reopen end to be any range upto current date. WHAT is 22???
	for reopen_day2 in range(reopen_day2_gov + 8, size - 14):  # min(reopen_day2_gov + 22, size + 1)):
		# for reopen_day2 in range(reopen_day2_gov+1,  size - 1 ):
		optimal = minimize(loss_reopen, [uni(h2_range[0], h2_range[1]),
		                                 # uni(Hiding_init_range[0], Hiding_init_range[1])],
		                                 Hiding_init2_range[0]],
		                   args=(confirmed, death, n_0, vaccine_speeds, reopen_day2, reopen_day2_gov, para_init),
		                   method='L-BFGS-B',
		                   bounds=[h2_range, Hiding_init2_range])
		# print(f'inside fit_reopen before current_loss={reopen_day2_gov}')  # SK-12-24-added
		current_loss = loss_reopen(optimal.x, confirmed, death, n_0, vaccine_speeds, reopen_day2, reopen_day2_gov,
		                           para_init)
		# print(f'inside fit_reopen  current_loss={current_loss}')  # SK-12-24-added
		if current_loss < min_loss:
			# print(f'updating loss={current_loss} with c1={c1}') #SK-12-24-uncommented
			min_loss = current_loss
			h2 = optimal.x[0]
			Hiding_init2 = optimal.x[1]
			reopen_day2_best = reopen_day2

	reopen_day2 = reopen_day2_best
	S = [n_0 * eta]
	E = [n_0 * eta * E_initial]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [n_0 * eta * (Hiding_init1 + Hiding_init2)]
	releases = [0] * size
	H1_left = n_0 * eta * Hiding_init1
	r = h1 * H1_left
	release_days1 = math.ceil(1 / h1)
	release1 = []
	for i in range(release_days1):
		release1.append(min(r, H1_left))
		H1_left -= min(r, H1_left)
	for i in range(len(release1)):
		releases[reopen_day1 + i] = release1[i]
	H2_left = n_0 * eta * Hiding_init2
	r = h2 * H2_left
	release_days2 = math.ceil(1 / h2)
	release2 = []
	for i in range(release_days2):
		release2.append(min(r, H2_left))
		H2_left -= min(r, H2_left)
	for i in range(len(release2)):
		if reopen_day2 + i < size:
			releases[reopen_day2 + i] = release2[i]

	result, [S, E, I, A, IH, IN, D, R, G, H,
	         S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
	         S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2, S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3, betas] = \
		simulate_vac(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta,
		             c1, n_0, vaccine_speeds, v_period1, v_period2, v_eff1, v_eff2, releases)

	size1 = reopen_day1
	size2 = reopen_day2 - size1
	size3 = size - size1 - size2
	weights1 = [Geo ** n for n in range(size1)]
	weights1.reverse()
	weights2 = [Geo ** n for n in range(size2)]
	weights2.reverse()
	weights3 = [Geo ** n for n in range(size3)]
	weights3.reverse()
	weights = weights1
	weights.extend(weights2)
	weights.extend(weights3)
	# if len(weights) != size:
	# 	print('wrong weights!')

	metric1 = weighted_relative_deviation(weights, confirmed, G, start_dev, num_para)
	metric2 = weighted_relative_deviation(weights, death, D, start_dev, num_para)

	r1 = r2_score(confirmed, G)
	r2 = r2_score(death, D)

	return [h2, Hiding_init2, reopen_day2, metric1, metric2, r1, r2], min_loss


def MT_fitting_init(state, confirmed, death, n_0, vaccine_speeds, reopen_day1_gov):
	print('MT fitting init', state)
	para_best = []
	min_loss = 10000
	with concurrent.futures.ProcessPoolExecutor() as executor:
		# t1 = time.perf_counter()
		results = [executor.submit(fit_init, state, i, confirmed, death, vaccine_speeds, reopen_day1_gov, n_0) for i in
		           range(num_threads)]

		threads = 0
		try:
			for f in concurrent.futures.as_completed(results):
				para, current_loss = f.result()
				threads += 1
				# print(f'thread {threads} returned')
				if current_loss < min_loss:
					min_loss = current_loss
					para_best = para
					print(f'{state} initial #{threads} loss={min_loss}')
				else:
					print(f'{state} initial #{threads}')
		except:
			print('init crash in', state)
			traceback.print_exception(*sys.exc_info())
	# if threads % 10 == 0:
	# 	print(f'{threads}/{num_threads} thread(s) completed')

	# t2 = time.perf_counter()
	# print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_threads, 3)} seconds per job')

	# print('initial best fitting completed\n')
	return para_best


def MT_fitting_reopen(state, confirmed, death, n_0, vaccine_speeds, reopen_day_gov, para_init):
	print('MT fitting reopen', state)
	para_best = []
	min_loss = 10000
	# print('reopen entry\n') #SK-12-24
	with concurrent.futures.ProcessPoolExecutor() as executor:
		# t1 = time.perf_counter()
		results = [
			executor.submit(fit_reopen, state, i, confirmed, death, vaccine_speeds, reopen_day_gov, n_0, para_init)
			for i in range(num_threads)]

		threads = 0
		try:
			for f in concurrent.futures.as_completed(results):
				para, current_loss = f.result()
				threads += 1
				# print(f'thread {threads} returned')
				if current_loss < min_loss:
					min_loss = current_loss
					para_best = para
					print(f'{state} reopen #{threads} loss={min_loss}')
				else:
					print(f'{state} reopen #{threads}')
		except:
			print('reopen crash in', state)
			traceback.print_exception(*sys.exc_info())
	# if threads % 10 == 0:
	# 	print(f'{threads}/{num_threads} thread(s) completed')

	# t2 = time.perf_counter()
	# print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_threads, 3)} seconds per job')

	# print('initial best fitting completed\n')
	return para_best


def fit_state_split(state, ConfirmFile, DeathFile, PopFile, end_date, path):
	t1 = time.perf_counter()
	state_path = f'{path}/{state}'
	if not os.path.exists(state_path):
		os.makedirs(state_path)

	print(state)
	print()

	# read population
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

	# select confirmed and death data
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]

	days = list(confirmed.columns)
	days_full = days[days.index(start_date):days.index(end_date) + 1]
	days_init = days[days.index(start_date):days.index(reopen_date2) + 1]
	days_full = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days_full]
	days_init = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days_init]
	confirmed_full = confirmed.iloc[0].loc[start_date: end_date]
	death_full = death.iloc[0].loc[start_date: end_date]
	for i in range(len(death)):
		if death_full.iloc[i] == 0:
			death_full.iloc[i] = 0.01
	death_full = death_full.tolist()
	confirmed_init = confirmed.iloc[0].loc[start_date: reopen_date2]
	death_init = death.iloc[0].loc[start_date: reopen_date2]
	for i in range(len(death)):
		if death_init.iloc[i] == 0:
			death_init.iloc[i] = 0.01
	death_init = death_init.tolist()

	reopen_day1_gov = days_full.index(datetime.datetime.strptime(reopen_date1, '%Y-%m-%d'))

	reopen_day2_gov = days_full.index(datetime.datetime.strptime(reopen_date2, '%Y-%m-%d'))
	vaccine_day1 = days_full.index(datetime.datetime.strptime(vac_date1, '%Y-%m-%d'))
	vaccine_day2 = days_full.index(datetime.datetime.strptime(vac_date2, '%Y-%m-%d'))
	vaccine_speeds = [0] * len(days_full)
	for i in range(vaccine_day1, vaccine_day2):
		vaccine_speeds[i] = daily_vspeed1
	if vac3_On:
		vaccine_day3 = days_full.index(datetime.datetime.strptime(vac_date3, '%Y-%m-%d'))
		for i in range(vaccine_day2, vaccine_day3):
			vaccine_speeds[i] = daily_vspeed2
		for i in range(vaccine_day3, len(days_full)):
			vaccine_speeds[i] = daily_vspeed3
	else:
		for i in range(vaccine_day2, len(days_full)):
			vaccine_speeds[i] = daily_vspeed2

	# initial fitting
	para_init = MT_fitting_init(state, confirmed_init, death_init, n_0, vaccine_speeds, reopen_day1_gov)
	print("DONE WITH FIRST FIT", state)

	# reopen fitting
	para_reopen = MT_fitting_reopen(state, confirmed_full, death_full, n_0, vaccine_speeds, reopen_day2_gov, para_init)
	print("DONE WITH REOPEN FIT", state)  # Sk-12-24
	para = para_init + para_reopen
	#
	# plotting
	# print("DONE WITH para add") #Sk-12-24
	[S, E, I, A, IH, IN, D, R, G, H,
	 S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0,
	 S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
	 S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2,
	 S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
	 betas] = plot_full(state, confirmed_full, death_full, days_full, n_0, para, state_path, vaccine_speeds)

	save_sim_vac([S, E, I, A, IH, IN, D, R, G, H,
	              S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0,
	              S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
	              S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2,
	              S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
	              betas], days_full, state_path)

	para[-5] = days_full[para[-5]]
	para[-9] = days_full[para[-9]]
	save_para_vac([para], state_path)
	t2 = time.perf_counter()
	print(f'{round((t2 - t1) / 60, 1)} minutes in total for {state}\n')

	return


def save_para_vac(paras, state_path):
	para_label = ['beta', 'gammaE', 'alpha', 'gamma', 'gamma2', 'gamma3', 'a1', 'a2', 'a3', 'eta', 'c1', 'E_initial',
	              'I_initial', 'h1', 'Hiding_init1', 'reopen1', 'antibody ratio', 'h2', 'Hiding_init2', 'reopen2',
	              'metric1', 'metric2', 'r1', 'r2']
	df = pd.DataFrame(paras, columns=para_label)
	df.to_csv(f'{state_path}/para.csv', index=False, header=True)

	# print('parameters saved\n')

	return


def save_sim_vac(data, days, state_path):
	days = [day.strftime('%Y-%m-%d') for day in days]
	c0 = ['S', 'E', 'I', 'A', 'IH', 'IN', 'D', 'R', 'G', 'H',
	      'S0', 'E0', 'I0', 'A0', 'IH0', 'IN0', 'D0', 'R0', 'G0', 'H0',
	      'S1', 'E1', 'I1', 'A1', 'IH1', 'IN1', 'D1', 'R1', 'G1', 'H1',
	      'S2', 'E2', 'I2', 'A2', 'IH2', 'IN2', 'D2', 'R2', 'G2', 'H2',
	      'S3', 'E3', 'I3', 'A3', 'IH3', 'IN3', 'D3', 'R3', 'G3', 'H3',
	      'beta']
	df = pd.DataFrame(data, columns=days)
	df.insert(0, 'series', c0)
	df.to_csv(f'{state_path}/sim.csv', index=False)
	# print('simulation saved\n')

	return


def plot_full(state, confirmed, death, days, n_0, para, state_path, vaccine_speeds):
	# print('inside plot1\n')  # SK-12-24
	[beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, c1, E_initial, I_initial, h1, Hiding_init1,
	 reopen_day1, metric_antibody, h2, Hiding_init2, reopen_day2, metric1, metric2, r1, r2] = para
	print(f'inside plot state = {state}\n')  # SK-12-24
	size = len(days)
	S = [n_0 * eta]
	E = [n_0 * eta * E_initial]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [n_0 * eta * Hiding_init1]
	H = [n_0 * eta * (Hiding_init1 + Hiding_init2)]
	releases = [0] * size
	H1_left = n_0 * eta * Hiding_init1
	r = h1 * H1_left
	release_days1 = math.ceil(1 / h1)
	release1 = []
	for i in range(release_days1):
		release1.append(min(r, H1_left))
		H1_left -= min(r, H1_left)
	for i in range(len(release1)):
		releases[reopen_day1 + i] = release1[i]
	H2_left = n_0 * eta * Hiding_init2
	r = h2 * H2_left
	release_days2 = math.ceil(1 / h2)
	release2 = []
	for i in range(release_days2):
		release2.append(min(r, H2_left))
		H2_left -= min(r, H2_left)
	for i in range(len(release2)):
		if reopen_day2 + i < size:
			releases[reopen_day2 + i] = release2[i]
	# H = [0]
	result, [S, E, I, A, IH, IN, D, R, G, H,
	         S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0,
	         S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
	         S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2,
	         S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
	         betas] = simulate_vac(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1,
	                               a2, a3, eta, c1, n_0, vaccine_speeds, v_period1, v_period2, v_eff1, v_eff2, releases)

	fig = plt.figure(figsize=(20, 16))
	fig.suptitle(state)
	ax = fig.add_subplot(421)
	# ax.set_title(state)
	ax2 = fig.add_subplot(422)
	ax3 = fig.add_subplot(423)
	ax4 = fig.add_subplot(424)
	ax5 = fig.add_subplot(425)
	ax6 = fig.add_subplot(426)
	ax7 = fig.add_subplot(427)
	ax.axvline(days[reopen_day1], linestyle='dashed', color='tab:grey')
	ax2.axvline(days[reopen_day1], linestyle='dashed', color='tab:grey')
	ax3.axvline(days[reopen_day1], linestyle='dashed', color='tab:grey')
	ax4.axvline(days[reopen_day1], linestyle='dashed', color='tab:grey')
	ax5.axvline(days[reopen_day1], linestyle='dashed', color='tab:grey', label=days[reopen_day1].strftime('%Y-%m-%d'))
	ax6.axvline(days[reopen_day1], linestyle='dashed', color='tab:grey')
	ax7.axvline(days[reopen_day1], linestyle='dashed', color='tab:grey')

	ax.axvline(days[reopen_day2], linestyle='dashed', color='tab:grey')
	ax2.axvline(days[reopen_day2], linestyle='dashed', color='tab:grey')
	ax3.axvline(days[reopen_day2], linestyle='dashed', color='tab:grey')
	ax4.axvline(days[reopen_day2], linestyle='dashed', color='tab:grey')
	ax5.axvline(days[reopen_day2], linestyle='dashed', color='tab:grey', label=days[reopen_day2].strftime('%Y-%m-%d'))
	ax6.axvline(days[reopen_day2], linestyle='dashed', color='tab:grey')
	ax7.axvline(days[reopen_day2], linestyle='dashed', color='tab:grey')

	ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
	ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
	ax.plot(days, [i / 1000 for i in G], label='G')
	ax2.plot(days, [i / 1000 for i in D], label='D')

	ax3.plot(days, [i / 1000 for i in S0], label='S0')
	ax3.plot(days, [i / 1000 for i in S1], label='S1')
	ax3.plot(days, [i / 1000 for i in S2], label='S2')
	ax3.plot(days, [i / 1000 for i in S3], label='S3')

	ax4.plot(days, [i / 1000 for i in H0], label='H0')
	ax4.plot(days, [i / 1000 for i in H1], label='H1')
	ax4.plot(days, [i / 1000 for i in H2], label='H2')
	ax4.plot(days, [i / 1000 for i in H3], label='H3')

	ax5.plot(days, betas, label='beta')

	ax7.plot(days, [i / 1000 for i in S], label='S')
	ax7.plot(days, [i / 1000 for i in H], label='H')

	diff_G = pd.Series(np.diff(G))
	diff_confirmed = pd.Series(np.diff(confirmed))
	ax6.plot(days[-len(diff_confirmed):], [i / 1000 for i in diff_confirmed], label='daily new cases')
	ax6.plot(days[-len(diff_G):], [i / 1000 for i in diff_G], label='dG')

	ax.legend()
	ax2.legend()
	ax3.legend()
	ax4.legend()
	ax5.legend()
	ax6.legend()
	ax7.legend()
	fig.autofmt_xdate()
	fig.savefig(f'{state_path}/sim.png', bbox_inches="tight")
	# fig.savefig(f'init_only_{end_date}/{state}/sim.png', bbox_inches="tight")
	plt.close(fig)
	return [S, E, I, A, IH, IN, D, R, G, H,
	        S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0,
	        S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
	        S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2,
	        S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
	        betas]


def fit_all_split(end_date):
	path = f'india/T{num_threads}_fitting_split2C_{end_date}_{reopen_date1}_{reopen_date2}'
	t1 = time.perf_counter()
	matplotlib.use('Agg')

	with concurrent.futures.ProcessPoolExecutor() as executor:
		[executor.submit(fit_state_split, state, 'india/indian_cases_confirmed_cases.csv',
		                 'india/indian_cases_confirmed_deaths.csv', 'india/state_population.csv', end_date, path) for
		 state in states]

	# for state in states:
	# 	fit_state_split(state, 'india/indian_cases_confirmed_cases.csv',
	# 	                'india/indian_cases_confirmed_deaths.csv', 'india/state_population.csv', end_date)

	t2 = time.perf_counter()
	print(f'{round((t2 - t1) / 60, 3)} minutes for all states ending on {end_date}')
	save_para_all_vac(path)
	return


def tmp():
	l = [1, 3, 5, 7, 9, 11]
	l2 = moving_avg(l, 3)
	print(l2)

	return


def extend_state_split(state, ConfirmFile, DeathFile, PopFile, ParaFile, state_path, sim_enddate, data_enddate,
                       ext_days, h3, Hiding_init3, reopen_date3, boostVac, hiding_delay, hiding_frac):
	print('extending', state)
	# state_path = f'india/extended_split2C_{sim_enddate}_{reopen_date1}_{reopen_date2}/{state}'
	# if not os.path.exists(state_path):
	# 	os.makedirs(state_path)
	df = pd.read_csv(ParaFile)
	beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, c1, E_initial, I_initial, h1, Hiding_init1, \
	reopen_day1, antibody_ratio, h2, Hiding_init2, reopen_day2, metric1, metric2, r1, r2 = df.iloc[0]

	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]

	dates = list(confirmed.columns)
	dates = dates[dates.index(start_date):dates.index(data_enddate) + 1]

	confirmed = confirmed.iloc[0].loc[start_date: data_enddate]
	death = death.iloc[0].loc[start_date: data_enddate]
	size = len(confirmed)
	days_full = [datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=i) for i in
	             range(size + ext_days)]

	d_confirmed = [confirmed[i] - confirmed[i - 1] for i in range(1, len(confirmed))]
	d_confirmed.insert(0, 0)
	d_death = [death[i] - death[i - 1] for i in range(1, len(death))]
	d_death.insert(0, 0)

	reopen_day1 = dates.index(reopen_day1)
	reopen_day2 = dates.index(reopen_day2)
	reopen_day3 = days_full.index(datetime.datetime.strptime(reopen_date3, '%Y-%m-%d'))
	hiding_day = reopen_day3 + hiding_delay
	# print(reopen_day1, reopen_day2, reopen_day3)
	sim_endday = days_full.index(datetime.datetime.strptime(sim_enddate, '%Y-%m-%d'))
	data_endday = days_full.index(datetime.datetime.strptime(data_enddate, '%Y-%m-%d'))
	vaccine_day = days_full.index(datetime.datetime.strptime(vac_date1, '%Y-%m-%d'))

	Om_start_ix = days_full.index(datetime.datetime.strptime(Om_start_date, '%Y-%m-%d'))

	S = [n_0 * eta]
	E = [n_0 * eta * E_initial]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [n_0 * eta * (Hiding_init1 + Hiding_init2 + 1 + Hiding_init1 - Hiding_init2 * Hiding_init3)]

	releases = [0] * (size + ext_days)
	H1_left = n_0 * eta * Hiding_init1
	r = h1 * H1_left
	release_days1 = math.ceil(1 / h1)
	release1 = []
	for i in range(release_days1):
		release1.append(min(r, H1_left))
		H1_left -= min(r, H1_left)
	for i in range(len(release1)):
		releases[reopen_day1 + i] = release1[i]

	H2_left = n_0 * eta * Hiding_init2
	r = h2 * H2_left
	release_days2 = math.ceil(1 / h2)
	release2 = []
	for i in range(release_days2):
		release2.append(min(r, H2_left))
		H2_left -= min(r, H2_left)
	for i in range(len(release2)):
		if reopen_day2 + i < size + ext_days:
			releases[reopen_day2 + i] = release2[i]

	H3_left = n_0 * eta * (1 + Hiding_init1 - Hiding_init2 * Hiding_init3)
	r = h3 * H3_left
	release_days3 = math.ceil(1 / h3)
	release3 = []
	for i in range(release_days3):
		release3.append(min(r, H3_left))
		H3_left -= min(r, H3_left)
	for i in range(len(release3)):
		if reopen_day3 + i < size + ext_days:
			releases[reopen_day3 + i] += release3[i]

	vaccine_day1 = days_full.index(datetime.datetime.strptime(vac_date1, '%Y-%m-%d'))
	vaccine_day2 = days_full.index(datetime.datetime.strptime(vac_date2, '%Y-%m-%d'))
	vaccine_speeds = [0] * len(days_full)
	for i in range(vaccine_day1, vaccine_day2):
		vaccine_speeds[i] = daily_vspeed1
	# for i in range(vaccine_day2, len(days_full)):
	# 	vaccine_speeds[i] = daily_vspeed2
	if vac3_On:
		vaccine_day3 = days_full.index(datetime.datetime.strptime(vac_date3, '%Y-%m-%d'))
		for i in range(vaccine_day2, vaccine_day3):
			vaccine_speeds[i] = daily_vspeed2
		for i in range(vaccine_day3, len(days_full)):
			vaccine_speeds[i] = daily_vspeed3
	else:
		for i in range(vaccine_day2, len(days_full)):
			vaccine_speeds[i] = daily_vspeed2
	if boostVac:
		for i in range(reopen_day3, len(days_full)):
			vaccine_speeds[i] = daily_vspeed4

	result, [S, E, I, A, IH, IN, D, R, G, H,
	         S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0,
	         S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
	         S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2,
	         S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
	         S4, E4, I4, A4, IH4, IN4, D4, R4, G4, H4,
	         E_Om, I_Om, A_Om, R_Om, G_Om, betas] = \
		simulate_vac_Omicron(size + ext_days, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2,
		                     gamma3, a1, a2, a3, eta, c1, n_0, vaccine_speeds, v_period1, v_period2, v_period3, v_eff1,
		                     v_eff2, v_eff3, releases, Om_start_ix, hiding_day, hiding_frac)

	dG = [G[i] - G[i - 1] for i in range(1, len(G))]
	dG.insert(0, 0)
	dD = [D[i] - D[i - 1] for i in range(1, len(D))]
	dD.insert(0, 0)

	# Om
	dG_Om = [G_Om[i] - G_Om[i - 1] for i in range(1, len(G))]
	dG_Om.insert(0, 0)

	return G0, G1, G2, G3, G4, D0, D1, D2, D3, D4, G, D, dG, dD, confirmed, death, d_confirmed, d_death, days_full, dG_Om


def extend_release_allV2(sim_enddate, sim_folder, ext_days, h3, Hiding_init3_range, reopen_date3, hiding_delay,
                         hiding_frac):
	# postfix = '_1_9'
	matplotlib.use('Agg')

	data_enddate = fitting_enddate

	fig_India = plt.figure(figsize=(12, 4.5))
	fig_India.suptitle(f'India {round(1 / h3)} Days')
	ax = fig_India.add_subplot(132)
	ax2 = fig_India.add_subplot(133)
	ax3 = fig_India.add_subplot(131)
	ax.set_title('Daily current variant Cases (Thousand)')
	ax2.set_title('Daily Total Cases (Thousand)')
	ax3.set_title('Daily Omicron Cases (Thousand)')

	state_results = {}
	for state in states:
		state_results[state] = []
	India_results = []

	# Hiding_init3_range = np.arange(0.5, 1.01, 0.1)
	for Hiding_init3 in Hiding_init3_range:
		boostVac = True

		India_G0 = []
		India_G1 = []
		India_G2 = []
		India_G3 = []
		India_G4 = []
		India_D0 = []
		India_D1 = []
		India_D2 = []
		India_D3 = []
		India_D4 = []
		India_G = []
		India_D = []
		India_dG = []
		India_dD = []
		India_confirmed = []
		India_death = []
		India_d_confirmed = []
		India_d_death = []
		India_dG_Om = []
		# states = ['dl', 'mh', 'kl', 'ar']
		for state in states:
			state_path = f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}/{state}'
			G0, G1, G2, G3, G4, D0, D1, D2, D3, D4, G, D, dG, dD, confirmed, death, d_confirmed, d_death, days, dG_Om = \
				extend_state_split(state, 'india/indian_cases_confirmed_cases.csv',
				                   'india/indian_cases_confirmed_deaths.csv', 'india/state_population.csv',
				                   f'{sim_folder}/{state}/para.csv', state_path, sim_enddate, data_enddate, ext_days,
				                   h3, Hiding_init3, reopen_date3, boostVac, hiding_delay, hiding_frac)
			# WHY-SK
			if not state_results[state]:
				state_results[state].extend(
					[[datetime.datetime.strftime(day, '%Y-%m-%d') for day in days], G0, G1, G2, G3, G4, D0, D1, D2, D3,
					 D4, G, D, dG, dD, dG_Om])
			else:
				state_results[state].extend([G0, G1, G2, G3, G4, D0, D1, D2, D3, D4, G, D, dG, dD, dG_Om])

			if len(India_G) == 0:
				India_G0 = G0.copy()
				India_G1 = G1.copy()
				India_G2 = G2.copy()
				India_G3 = G3.copy()
				India_G4 = G4.copy()
				India_D0 = D0.copy()
				India_D1 = D1.copy()
				India_D2 = D2.copy()
				India_D3 = D3.copy()
				India_D4 = D4.copy()
				India_G = G.copy()
				India_dG = dG.copy()
				India_D = D.copy()
				India_dD = dD.copy()
				India_confirmed = confirmed.copy()
				India_d_confirmed = d_confirmed.copy()
				India_death = death.copy()
				India_d_death = d_death.copy()

				India_dG_Om = dG_Om.copy()
			else:
				India_G0 = [India_G0[i] + G0[i] for i in range(len(G0))]
				India_G1 = [India_G1[i] + G1[i] for i in range(len(G1))]
				India_G2 = [India_G2[i] + G2[i] for i in range(len(G2))]
				India_G3 = [India_G3[i] + G3[i] for i in range(len(G3))]
				India_G4 = [India_G4[i] + G4[i] for i in range(len(G4))]
				India_D0 = [India_D0[i] + D0[i] for i in range(len(D0))]
				India_D1 = [India_D1[i] + D1[i] for i in range(len(D1))]
				India_D2 = [India_D2[i] + D2[i] for i in range(len(D2))]
				India_D3 = [India_D3[i] + D3[i] for i in range(len(D3))]
				India_D4 = [India_D4[i] + D4[i] for i in range(len(D4))]
				India_G = [India_G[i] + G[i] for i in range(len(G))]
				India_dG = [India_dG[i] + dG[i] for i in range(len(G))]
				India_D = [India_D[i] + D[i] for i in range(len(G))]
				India_dD = [India_dD[i] + dD[i] for i in range(len(G))]
				India_confirmed = [India_confirmed[i] + confirmed[i] for i in range(len(confirmed))]
				India_d_confirmed = [India_d_confirmed[i] + d_confirmed[i] for i in range(len(confirmed))]
				India_death = [India_death[i] + death[i] for i in range(len(confirmed))]
				India_d_death = [India_d_death[i] + d_death[i] for i in range(len(confirmed))]

				India_dG_Om = [India_dG_Om[i] + dG_Om[i] for i in range(len(dG_Om))]

		sim_endday = days.index(datetime.datetime.strptime(sim_enddate, '%Y-%m-%d'))
		data_endday = days.index(datetime.datetime.strptime(data_enddate, '%Y-%m-%d'))

		if sim_endday + 7 <= data_endday:
			error_ratio = 0
			MA_India_dG = moving_avg(India_dG, 7)
			MA_India_d_confirmed = moving_avg(India_d_confirmed, 7)
			for i in range(sim_endday + 1, sim_endday + 8):
				error_ratio += (abs(MA_India_dG[i] - MA_India_d_confirmed[i]) / MA_India_d_confirmed[i])
			error_ratio /= 7
			print(f'India  7-day new case error ratio from {sim_enddate} = {round(error_ratio, 6)}')

			error_ratio = 0
			MA_India_G = moving_avg(India_G, 7)
			MA_India_confirmed = moving_avg(India_confirmed, 7)
			for i in range(sim_endday + 1, sim_endday + 8):
				error_ratio += (abs(MA_India_G[i] - MA_India_confirmed[i]) / MA_India_confirmed[i])
			error_ratio /= 7
			print(f'India  7-day cumulative error ratio from {sim_enddate} = {round(error_ratio, 6)}')

		if sim_endday + 14 <= data_endday:
			error_ratio = 0
			MA_India_dG = moving_avg(India_dG, 7)
			MA_India_d_confirmed = moving_avg(India_d_confirmed, 7)
			for i in range(sim_endday + 1, sim_endday + 15):
				error_ratio += (abs(MA_India_dG[i] - MA_India_d_confirmed[i]) / MA_India_d_confirmed[i])
			error_ratio /= 14
			print(f'India 14-day new case error ratio from {sim_enddate} = {round(error_ratio, 6)}')

			error_ratio = 0
			MA_India_G = moving_avg(India_G, 7)
			MA_India_confirmed = moving_avg(India_confirmed, 7)
			for i in range(sim_endday + 1, sim_endday + 15):
				error_ratio += (abs(MA_India_G[i] - MA_India_confirmed[i]) / MA_India_confirmed[i])
			error_ratio /= 14
			print(f'India 14-day cumulative error ratio from {sim_enddate} = {round(error_ratio, 6)}')
		# PLOT HERE 8/7/2021
		# India_dG_comb = [(India_dG[i] + India_dG_Om[i]) / 1000 for i in range(1, len(India_dG))]
		lendG = len(India_dG)
		lendO = len(India_dG_Om)
		fig_color = 'r'
		if Hiding_init3 == 1:
			fig_color = 'orange'
		ax.plot(days[1:len(India_dG)], [i / 1000 for i in India_dG[1:]], label=f'{round(Hiding_init3 * 100)}%',
		        color=fig_color)
		# ax2.plot(days[1:len(India_dD)], [i / 1000 for i in India_dG[1:] + India_dG_Om[1:]], label=f'{round(Hiding_init3 * 100)}%',
		#		 color=fig_color)
		ax2.plot(days[1:len(India_dG)], [(India_dG[i] + India_dG_Om[i]) / 1000 for i in range(1, len(India_dG))],
		         label=f'{round(Hiding_init3 * 100)}%', color=fig_color)
		ax3.plot(days[1:len(India_dG_Om)], [i / 1000 for i in India_dG_Om[1:]], label=f'{round(Hiding_init3 * 100)}%',
		         color=fig_color)

		India_results.extend(
			[India_G0, India_G1, India_G2, India_G3, India_G4, India_D0, India_D1, India_D2, India_D3, India_D4,
			 India_G, India_D, India_dG, India_dD, India_dG_Om])

	# ax.fill_between(days[1:len(India_dG)], [i / 1000 for i in India_results[12][1:]],
	#                 [i / 1000 for i in India_results[-3][1:]], alpha=0.3, color='orange')
	# ax2.fill_between(days[1:len(India_dD)], [i / 1000 for i in India_results[13][1:]],
	#                  [i / 1000 for i in India_results[-2][1:]], alpha=0.3, color='orange')
	#	ax3.fill_between(days[1:len(India_dG_Om)], [i / 1000 for i in India_results[12][1:]],
	#					 [i / 1000 for i in India_results[-1][1:]], alpha=0.3, color='red')

	# ax.scatter(days[1:len(India_d_confirmed)], [i / 1000 for i in India_d_confirmed[1:]], linewidth=5, linestyle=':')
	# ax2.scatter(days[1:len(India_d_death)], [i / 1000 for i in India_d_death[1:]], linewidth=5, linestyle=':')
	ax.scatter(days[1:len(India_d_confirmed)], [i / 1000 if i > 0 else 0 for i in India_d_confirmed[1:]], s=1)
	ax2.scatter(days[1:len(India_d_confirmed)], [i / 1000 if i > 0 else 0 for i in India_d_confirmed[1:]], s=1)
	#	ax2.scatter(days[1:len(India_d_death)], [i / 1000 for i in India_d_death[1:]], s=1)
	# ax.legend()
	# ax2.legend()
	if not os.path.exists(f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}'):
		os.makedirs(f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}')

	fig_India.autofmt_xdate()
	fig_India.savefig(
		f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}/India_{round(1 / h3)}.png',
		bbox_inches="tight")
	plt.close(fig_India)
	# print(state_results)

	India_df = pd.DataFrame(India_results, columns=[datetime.datetime.strftime(day, '%Y-%m-%d') for day in days])
	India_df.insert(0, 'series',
	                ['G0', 'G1', 'G2', 'G3', 'G4', 'D0', 'D1', 'D2', 'D3', 'D4', 'G', 'D', 'dG', 'dD', 'dG_Om'] * len(
		                Hiding_init3_range))
	India_df.to_csv(
		f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}/india{round(1 / h3)}.csv',
		index=False)

	for state in states:
		state_path = f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}/{state}'
		if not os.path.exists(state_path):
			os.makedirs(state_path)
		df = pd.DataFrame(state_results[state])
		out_df = pd.DataFrame(df.values[1:], columns=df.iloc[0])
		out_df.insert(0, 'series',
		              ['G0', 'G1', 'G2', 'G3', 'G4', 'D0', 'D1', 'D2', 'D3', 'D4', 'G', 'D', 'dG', 'dD', 'dG_Om'] * len(
			              Hiding_init3_range))
		out_df.to_csv(f'{state_path}/{round(1 / h3)}.csv', index=False)

	return


def save_para_all_vac(fitting_folder):
	# fitting_folder = f'india/ab_fitting_split2C_{end_date}_{reopen_date1}_{reopen_date2}'
	out_table = []
	for state in states:
		df = pd.read_csv(f'{fitting_folder}/{state}/para.csv')
		cols = df.columns
		row = list(df.iloc[0])
		row.insert(0, state)
		row.insert(1, state_dict[state])
		out_table.append(row)
	cols = list(cols)
	cols.insert(0, 'state')
	cols.insert(1, 'state full')
	out_df = pd.DataFrame(out_table, columns=cols)
	out_df.to_csv(f'{fitting_folder}/paras.csv', index=False)
	return


def moving_avg(original_list, days):
	MA_list = pd.Series(original_list)
	MA_list = MA_list.rolling(days).mean()
	return list(MA_list)


def compareVacRatesExtended():
	sim_date = fitting_enddate
	sim_folder = f'india/T10_fitting_split2C_{sim_date}_{reopen_date1}_{reopen_date2}'
	# Hiding_init3_range = np.arange(0, 1.01, 1)
	Hiding_init3_range = [0, 1]

	extend_release_allV2(sim_date, sim_folder, size_ext, 1 / 30, Hiding_init3_range, reopen_date3, HIDING_DELAY,
	                     HIDING_FRACTION)
	# extend_release_allV2(sim_date, sim_folder, size_ext, 1 / 90, Hiding_init3_range, reopen_date3, HIDING_DELAY,
	#                      HIDING_FRACTION)
	extend_release_allV2(sim_date, sim_folder, size_ext, 1 / 60, Hiding_init3_range, reopen_date3, HIDING_DELAY,
	                     HIDING_FRACTION)
	extend_release_allV2(sim_date,  sim_folder, size_ext, 1 / 90, Hiding_init3_range, reopen_date3, HIDING_DELAY, HIDING_FRACTION)

	gridFiguresV2(sim_date, 30, Hiding_init3_range)
	# gridFiguresV2(sim_date, 90, Hiding_init3_range)
	gridFiguresV2(sim_date, 60, Hiding_init3_range)
	gridFiguresV2(sim_date, 90, Hiding_init3_range)

	plotExtensionComposition(sim_date, 30, reopen_date3)
	# plotExtensionComposition(sim_date, 90, reopen_date3)
	plotExtensionComposition(sim_date, 60, reopen_date3)
	plotExtensionComposition(sim_date, 90, reopen_date3)
	return


def gridFiguresV2(sim_enddate, releaseDays, Hiding_init3_range):
	data_enddate = fitting_enddate
	for state in states:
		plotExtV2(state, sim_enddate, data_enddate, releaseDays, Hiding_init3_range)
	return


def plotExtV2(state, sim_enddate, data_enddate, releaseDays, Hiding_init3_range):
	print('plotting', state)
	ConfirmFile = 'india/indian_cases_confirmed_cases.csv'
	DeathFile = 'india/indian_cases_confirmed_deaths.csv'
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[start_date: data_enddate]
	death = death.iloc[0].loc[start_date: data_enddate]
	d_confirmed = [confirmed[i] - confirmed[i - 1] for i in range(1, len(confirmed))]
	d_confirmed.insert(0, 0)
	d_death = [death[i] - death[i - 1] for i in range(1, len(death))]
	d_death.insert(0, 0)
	state_path = f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}/{state}'
	df = pd.read_csv(f'{state_path}/{releaseDays}.csv')
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in df.columns[1:]]
	fig = plt.figure(figsize=(12, 4.5))
	fig.suptitle(f'{state_dict[state]} {releaseDays} Days')
	ax = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax.set_title('Daily Cases (Thousand)')
	ax2.set_title('Daily Deaths (Thousand)')
	for i in range(len(Hiding_init3_range)):
		Hiding_init3 = Hiding_init3_range[i]
		# G = df.iloc[i * 15 + 10][1:]
		# D = df.iloc[i * 15 + 11][1:]
		dG = df.iloc[i * 15 + 12][1:]
		dD = df.iloc[i * 15 + 13][1:]
		dG_Om = df.iloc[i * 15 + 14][1:]
		dG = [dG[i] + dG_Om[i] for i in range(len(dG))]
		fig_color = 'r'
		if Hiding_init3 == 1:
			fig_color = 'orange'
		ax.plot(days[1:len(dG)], [i / 1000 for i in dG[1:]], label=f'{round(Hiding_init3 * 100)}%', color=fig_color)
		ax2.plot(days[1:len(dD)], [i / 1000 for i in dD[1:]], label=f'{round(Hiding_init3 * 100)}%', color=fig_color)

	# ax.fill_between(days[1:len(dG)], [i / 1000 for i in df.iloc[12][2:]], [i / 1000 for i in df.iloc[-3][2:]],
	#                 alpha=0.5, color='orange')
	# ax2.fill_between(days[1:len(dG)], [i / 1000 for i in df.iloc[13][2:]], [i / 1000 for i in df.iloc[-2][2:]],
	#                  alpha=0.5, color='orange')

	# ax.legend()
	# ax2.legend()
	ax.scatter(days[1:len(d_confirmed)], [i / 1000 for i in d_confirmed[1:]], s=1)
	ax2.scatter(days[1:len(d_death)], [i / 1000 for i in d_death[1:]], s=1)
	fig.autofmt_xdate()
	fig.savefig(f'{state_path}/{state}_{releaseDays}.png', bbox_inches='tight')
	plt.close(fig)
	return


def plotExtensionComposition(sim_enddate, releaseDays, reopen_date3):
	plotStateComposition('India', sim_enddate, releaseDays, reopen_date3)
	for state in states:
		plotStateComposition(state, sim_enddate, releaseDays, reopen_date3)
	return


def plotStateComposition(state, sim_enddate, releaseDays, reopen_date3):
	print(f'plotting composition {state_dict[state]} {releaseDays} days')
	if state == 'India':
		state_path = f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}'
		df = pd.read_csv(f'{state_path}/india{releaseDays}.csv')
	else:
		state_path = f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}/{state}'
		df = pd.read_csv(f'{state_path}/{releaseDays}.csv')
	dates = list(df.columns[1:])
	# dates = dates[dates.index(reopen_date3):]
	reopen_day3 = dates.index(reopen_date3)
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates]
	fig = plt.figure(figsize=(12, 4.5))
	fig.suptitle(f'{state_dict[state]} {releaseDays} Days')
	ax = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax.set_title('Daily Cases (Thousand)')
	ax2.set_title('Daily Deaths (Thousand)')

	i = 0
	G0 = df.iloc[0 + i * 12][1:]
	G1 = df.iloc[1 + i * 12][1:]
	G2 = df.iloc[2 + i * 12][1:]
	G3 = df.iloc[3 + i * 12][1:]
	G4 = df.iloc[4 + i * 12][1:]
	D0 = df.iloc[5 + i * 12][1:]
	D1 = df.iloc[6 + i * 12][1:]
	D2 = df.iloc[7 + i * 12][1:]
	D3 = df.iloc[8 + i * 12][1:]
	D3 = df.iloc[9 + i * 12][1:]
	# G = df.iloc[i * 12 + 10][1:]
	# D = df.iloc[i * 12 + 11][1:]
	dG = df.iloc[i * 12 + 12][1:]
	dD = df.iloc[i * 12 + 13][1:]

	dG0 = [G0[i] - G0[i - 1] for i in range(1, len(G0))]
	dG1 = [G1[i] - G1[i - 1] for i in range(1, len(G1))]
	dG2 = [G2[i] - G2[i - 1] for i in range(1, len(G2))]
	dG3 = [G3[i] - G3[i - 1] for i in range(1, len(G3))]
	dG0.insert(0, 0)
	dG1.insert(0, 0)
	dG2.insert(0, 0)
	dG3.insert(0, 0)

	dD0 = [D0[i] - D0[i - 1] for i in range(1, len(D0))]
	dD1 = [D1[i] - D1[i - 1] for i in range(1, len(D1))]
	dD2 = [D2[i] - D2[i - 1] for i in range(1, len(D2))]
	dD3 = [D3[i] - D3[i - 1] for i in range(1, len(D3))]
	dD0.insert(0, 0)
	dD1.insert(0, 0)
	dD2.insert(0, 0)
	dD3.insert(0, 0)

	# plotStackedBar(ax, dG0, dG1, dG2, dG3, dG, days)
	# plotStackedBar(ax2, dD0, dD1, dD2, dD3, dD, days)
	plotCompositionCurves(ax, dG0, dG1, dG2, dG3, dG, days, reopen_day3)
	plotCompositionCurves(ax2, dD0, dD1, dD2, dD3, dD, days, reopen_day3)

	ax2.legend(loc='lower left', bbox_to_anchor=(1, 0))
	# ax2.yaxis.ticks.set_color('w')
	fig.autofmt_xdate()
	# plt.show()
	fig.savefig(f'{state_path}/{state}_comp_{releaseDays}.png', bbox_inches="tight")
	plt.close(fig)
	return


def plotCompositionCurves(ax, G0, G1, G2, G3, G, days, reopen_day3):
	comp0 = [k / 1000 for k in G0]
	comp1 = [(G0[i] + G1[i] + G2[i]) / 1000 for i in range(len(G0))]
	G = [k / 1000 for k in G]
	ax.plot(days[1:], G[1:], label='Total')
	ax.fill_between(days[reopen_day3:], comp0[reopen_day3:], label='Unvaccinated', color='orange', alpha=0.7)
	ax.fill_between(days[reopen_day3:], comp0[reopen_day3:], comp1[reopen_day3:], label='First Shot', color='orange',
	                alpha=0.3)
	ax.fill_between(days[reopen_day3:], comp1[reopen_day3:], G[reopen_day3:], label='Fully vaccinated', color='green',
	                alpha=0.3)
	return


def plotStackedBar(ax, G0, G1, G2, G3, G, days):
	# G = [G0[i] + G1[i] + G2[i] + G3[i] for i in range(len(G0))]
	comp0 = [G0[i] / G[i] if G[i] != 0 else 0 for i in range(len(G))]
	comp1 = [G1[i] / G[i] if G[i] != 0 else 0 for i in range(len(G))]
	comp2 = [G2[i] / G[i] if G[i] != 0 else 0 for i in range(len(G))]
	comp3 = [G3[i] / G[i] if G[i] != 0 else 0 for i in range(len(G))]

	ax.bar(days, comp0, label='Unvaccinated')
	ax.bar(days, comp1, bottom=comp0, label='First Shot')
	ax.bar(days, comp2, bottom=[i + j for i, j in zip(comp0, comp1)], label='First shot effective')
	ax.bar(days, comp3, bottom=[i + j + k for i, j, k in zip(comp0, comp1, comp2)], label='Fully vaccinated')
	ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
	# ax.set_xticks(days)

	return


def comparisonTable(sim_enddate, reopen_date3):
	PopFile = 'india/state_population.csv'
	PopDF = pd.read_csv(PopFile)
	cols = ['State', 'Ratio', 'H1', 'H2', 'G30', 'G45', 'G60', 'D30', 'D45', 'D60', 'G_impr_45', 'G_impr_60',
	        'D_impr_45', 'D_impr_60']
	table = []
	India_H1 = 0
	India_H2 = 0
	India_G30 = 0
	India_G45 = 0
	India_G60 = 0
	India_D30 = 0
	India_D45 = 0
	India_D60 = 0
	for state in states:
		row = [state_dict[state]]
		pop = PopDF[PopDF['state'] == state].iloc[0, 2]
		ParaFile = f'india/fitting_split2C_{sim_enddate}_{reopen_date1}_{reopen_date2}/{state}/para.csv'
		ParaDF = pd.read_csv(ParaFile)
		paras = ParaDF.iloc[0]
		eta = paras['eta']
		Hiding_init1 = paras['Hiding_init1']
		Hiding_init2 = paras['Hiding_init2']
		row.append(Hiding_init2 / (1 + Hiding_init1))
		row.append(pop * eta * (1 + Hiding_init1))
		row.append(pop * eta * Hiding_init2)
		India_H1 += pop * eta * (1 + Hiding_init1)
		India_H2 += pop * eta * Hiding_init2

		Ext30DF = pd.read_csv(
			f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}/{state}/30.csv')
		Ext45DF = pd.read_csv(
			f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}/{state}/45.csv')
		Ext60DF = pd.read_csv(
			f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}/{state}/60.csv')

		G30 = Ext30DF[Ext30DF['series'] == 'G'].iloc[0, -1] - Ext30DF[Ext30DF['series'] == 'G'].iloc[0][reopen_date3]
		G45 = Ext45DF[Ext45DF['series'] == 'G'].iloc[0, -1] - Ext45DF[Ext45DF['series'] == 'G'].iloc[0][reopen_date3]
		G60 = Ext60DF[Ext60DF['series'] == 'G'].iloc[0, -1] - Ext60DF[Ext60DF['series'] == 'G'].iloc[0][reopen_date3]

		D30 = Ext30DF[Ext30DF['series'] == 'D'].iloc[0, -1] - Ext30DF[Ext30DF['series'] == 'D'].iloc[0][reopen_date3]
		D45 = Ext45DF[Ext45DF['series'] == 'D'].iloc[0, -1] - Ext45DF[Ext45DF['series'] == 'D'].iloc[0][reopen_date3]
		D60 = Ext60DF[Ext60DF['series'] == 'D'].iloc[0, -1] - Ext60DF[Ext60DF['series'] == 'D'].iloc[0][reopen_date3]

		row.extend([G30, G45, G60, D30, D45, D60, 1 - G45 / G30, 1 - G60 / G30, 1 - D45 / D30, 1 - D60 / D30])
		India_G30 += G30
		India_G45 += G45
		India_G60 += G60
		India_D30 += D30
		India_D45 += D45
		India_D60 += D60
		table.append(row)

	table.append(
		['India', India_H2 / India_H1, India_H1, India_H2, India_G30, India_G45, India_G60, India_D30, India_D45,
		 India_D60, 1 - India_G45 / India_G30, 1 - India_G60 / India_G30, 1 - India_D45 / India_D30,
		 1 - India_D60 / India_D30])

	out_df = pd.DataFrame(table, columns=cols)
	out_df.to_csv(f'india/extended_split2V2_{sim_enddate}_{reopen_date1}_{reopen_date2}/comparison.csv',
	              index=False)

	return


postfix = '_0_1'


def checkHidingSum():
	#	end_date = '2021-09-08'
	for state in states:
		df = pd.read_csv(f'india/fitting_split2C_{end_date}_{reopen_date1}_{reopen_date2}/{state}/para.csv')
		beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, c1, E_initial, I_initial, h1, Hiding_init1, reopen1, h2, Hiding_init2, reopen2, metric1, metric2, r1, r2 = \
			df.iloc[0]
		if eta * (1 + Hiding_init1 + Hiding_init2) > 0.4:
			print(state_dict[state], eta * (1 + Hiding_init1 + Hiding_init2))
	return


def IndiaCumulativeByVacGroup():
	releasingDays = [30, 45, 60]
	table = []
	for releasing in releasingDays:
		df = pd.read_csv(
			f'india/extended_split2V2_{fitting_enddate}_{reopen_date1}_{reopen_date2}/india{releasing}.csv')
		G = df[df['series'] == 'G'].iloc[0]
		G0 = df[df['series'] == 'G0'].iloc[0]
		G1 = df[df['series'] == 'G1'].iloc[0]
		G2 = df[df['series'] == 'G2'].iloc[0]
		G3 = df[df['series'] == 'G3'].iloc[0]
		print('releasing in', releasing, 'days')
		newG = G.iloc[-1] - G.iloc[-size_ext]
		newG0 = G0.iloc[-1] - G0.iloc[-size_ext]
		newG1 = G1.iloc[-1] - G1.iloc[-size_ext]
		newG2 = G2.iloc[-1] - G2.iloc[-size_ext]
		newG3 = G3.iloc[-1] - G3.iloc[-size_ext]
		# print(newG - newG0 - newG1 - newG2 - newG3)
		table.append([releasing, newG0 / newG, (newG1 + newG2) / newG, newG3 / newG])
	out_df = pd.DataFrame(table, columns=['Releasing Days', 'Unvaccinated', 'Partially Vaccinated', 'Fully Vaccinated'])
	out_df.to_csv(f'india/extended_split2V2_{fitting_enddate}_{reopen_date1}_{reopen_date2}/new case comp.csv',
	              index=False)
	return


def main():
	# fit_all_split(fitting_enddate)
	compareVacRatesExtended()

	# comparisonTable(fitting_enddate, reopen_date3)
	# IndiaCumulativeByVacGroup()
	# checkHidingSum()
	# tmp()
	return


if __name__ == '__main__':
	main()
