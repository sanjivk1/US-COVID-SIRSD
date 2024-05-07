import numpy as np
import pandas as pd
import time
import concurrent.futures
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score, mean_squared_error
from SIRfunctions import SEIARG, computeBeta_combined
import datetime
from numpy.random import uniform as uni
import os

np.set_printoptions(threshold=sys.maxsize)

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
              'la': 'Lakshdweep'
              }

start_date = '2021-02-01'
# reopen_date = '2021-03-15'
end_date = '2021-06-01'

size_ext = 400

HH_frac1 = 0.13
HH_frac2 = 0
HH_frac3 = 0
daily_vspeed1 = 0.0015
daily_vspeed2 = 0.003
r_frac = 1
release_duration = 90
daily_rspeed = 1 / release_duration
v_period1 = 14
v_period2 = 30
v_eff1 = 0.65
v_eff2 = 0.8

parasFile = f'india/fittingV2_{end_date}/paras.csv'
popFile = 'india/state_population.csv'
confirmFile = 'india/indian_cases_confirmed_cases.csv'
deathFile = 'india/indian_cases_confirmed_deaths.csv'
paras_df = pd.read_csv(parasFile)
pop_df = pd.read_csv(popFile)
confirm_df = pd.read_csv(confirmFile)
death_df = pd.read_csv(deathFile)

v_date = '2021-06-15'
r_date = '2021-06-30'
state_objs = {}
G0s = {}
D0s = {}
G1s = {}
D1s = {}
G2s = {}
D2s = {}
days = []
dates = []
india_G0 = []
india_D0 = []
india_G1 = []
india_D1 = []
india_G2 = []
india_D2 = []


def simulate_release(size, S, E, I, A, IH, IN, D, R, G, H, HH, betas, beta, gammaE, alpha, gamma, gamma2, gamma3, a1,
                     a2, a3, h, Hiding_init, eta, c1, n_0, reopen_day, release_day, release_size, release_speed):
	result = True
	eta2 = eta * (1 - Hiding_init)
	Hiding0 = H[0]
	r = h * H[0]
	# HH = [release_size * n_0]
	daily_release = release_speed * release_size * n_0
	betas.append(beta)
	for i in range(1, size):

		# if i > reopen_day:
		# 	release = min(H[-1], r)
		# 	S[-1] += release
		# 	H[-1] -= release

		delta = SEIARG(i,
		               [S[i - 1], E[i - 1], I[i - 1], A[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1],
		                beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta2, n_0, c1, H[-1], Hiding0])
		S.append(S[-1] + delta[0])
		E.append(E[-1] + delta[1])
		I.append(I[-1] + delta[2])
		A.append(A[-1] + delta[3])
		IH.append(IH[-1] + delta[4])
		IN.append(IN[-1] + delta[5])
		D.append(D[-1] + delta[6])
		R.append(R[-1] + delta[7])
		G.append(G[-1] + delta[8])
		H.append(H[-1])
		HH.append(HH[-1])
		betas.append(delta[9])

		if i >= reopen_day:
			release = min(H[-1], r)
			S[-1] += release
			H[-1] -= release

		if i >= release_day and HH[-1] > 0:
			release = min(daily_release, HH[-1])
			S[-1] += release
			HH[-1] -= release
			Hiding0 += release

		if S[-1] < 0:
			result = False
			break
	return result


def simulate_vac(start_index, size,
                 S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, HH0,
                 S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1, HH1,
                 S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2, HH2,
                 S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3, HH3,
                 betas,
                 beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, eta, c1, n_0, reopen_day,
                 release_day, release_size, release_speed, vaccine_day, vaccine_speeds, vac_period1, vac_period2,
                 vac_eff1, vac_eff2):
	result = True
	eta2 = eta * (1 - Hiding_init)
	Hiding0 = H0[0] + HH0[0] - HH0[-1] - HH1[-1] - HH2[-1] - HH3[-1]
	r = h * H0[0]
	# HH = [release_size * n_0]
	daily_release = release_speed * release_size * n_0
	for i in range(start_index, size):

		vaccine_speed = vaccine_speeds[i]

		beta_t = computeBeta_combined(beta, eta2, n_0,
		                              S0[-1] + S1[-1] + S2[-1] + S3[-1],
		                              0, H0[-1], c1, Hiding0)
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
		HH0.append(HH0[-1])
		HH1.append(HH1[-1])
		HH2.append(HH2[-1])
		HH3.append(HH3[-1])

		betas.append(beta_t)

		dS12 = S1[i] / vac_period1
		dS23 = S2[i] / vac_period2
		S1[i] -= dS12
		S2[i] = S2[i] - dS23 + dS12
		S3[i] += dS23

		dHH12 = HH1[i] / vac_period1
		dHH23 = HH2[i] / vac_period2
		HH1[i] -= dHH12
		HH2[i] = HH2[i] - dHH23 + dHH12
		HH3[i] += dHH23

		if i >= reopen_day:
			release = min(H0[-1], r)
			S0[-1] += release
			H0[-1] -= release

		total_HH = HH0[-1] + HH1[-1] + HH2[-1] + HH3[-1]
		if i >= release_day and total_HH > 0:
			release = min(daily_release, total_HH)
			frac0 = HH0[-1] / total_HH
			frac1 = HH1[-1] / total_HH
			frac2 = HH2[-1] / total_HH
			frac3 = HH3[-1] / total_HH
			S0[-1] += release * frac0
			S1[-1] += release * frac1
			S2[-1] += release * frac2
			S3[-1] += release * frac3
			HH0[-1] -= release * frac0
			HH1[-1] -= release * frac1
			HH2[-1] -= release * frac2
			HH3[-1] -= release * frac3
			Hiding0 += release

		# if i >= vaccine_day:
		S1[-1] += S0[-1] * vaccine_speed
		S0[-1] -= S0[-1] * vaccine_speed
		HH1[-1] += HH0[-1] * vaccine_speed
		HH0[-1] -= HH0[-1] * vaccine_speed

		if S0[-1] < 0:
			result = False
			break

	return result


class State:
	def __init__(self, state, paras_df, pop_df, confirm_df, death_df):
		self.state = state

		row = list(paras_df[paras_df['state'] == state].iloc[0])
		# self.state = row[0]
		self.beta = row[1]
		self.gammaE = row[2]
		self.alpha = row[3]
		self.gamma = row[4]
		self.gamma2 = row[5]
		self.gamma3 = row[6]
		self.a1 = row[7]
		self.a2 = row[8]
		self.a3 = row[9]
		self.eta = row[10]
		self.h = row[11]
		self.Hiding_init = row[12]
		self.c1 = row[13]
		self.I_initial = row[14]
		self.vday = 0
		self.rday = 0

		self.n_0 = pop_df[pop_df['state'] == state].iloc[0]['POP']

		self.confirmed = confirm_df[confirm_df.iloc[:, 0] == state]
		self.death = death_df[death_df.iloc[:, 0] == state]
		dates = list(self.confirmed.columns)
		self.dates = dates[dates.index(start_date):dates.index(end_date) + 1]
		self.days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in self.dates]
		self.confirmed = self.confirmed.iloc[0].loc[start_date: end_date]
		self.death = self.death.iloc[0].loc[start_date: end_date]

		self.reopen_date = row[19]
		self.reopen_day = self.days.index(datetime.datetime.strptime(self.reopen_date, '%Y-%m-%d'))

		self.size = len(self.days)
		self.days_ext = [self.days[0] + datetime.timedelta(days=i) for i in range(self.size + size_ext)]  # datetime
		self.dates_ext = [d.strftime('%Y-%m-%d') for d in self.days_ext]  # string

		self.S = [self.n_0 * self.eta * (1 - self.Hiding_init)]
		self.E = [0]
		self.I = [self.n_0 * self.eta * self.I_initial * (1 - self.alpha)]
		self.A = [self.n_0 * self.eta * self.I_initial * self.alpha]
		self.IH = [0]
		self.IN = [self.I[-1] * self.gamma2]
		self.D = [self.death[0]]
		self.R = [0]
		self.G = [self.confirmed[0]]
		self.H = [self.n_0 * self.eta * self.Hiding_init]
		self.HH = []

		self.S0 = []
		self.E0 = []
		self.I0 = []
		self.A0 = []
		self.IH0 = []
		self.IN0 = []
		self.D0 = []
		self.R0 = []
		self.G0 = []
		self.H0 = []
		self.HH0 = []

		self.S1 = []
		self.E1 = []
		self.I1 = []
		self.A1 = []
		self.IH1 = []
		self.IN1 = []
		self.D1 = []
		self.R1 = []
		self.G1 = []
		self.H1 = []
		self.HH1 = []

		self.S2 = []
		self.E2 = []
		self.I2 = []
		self.A2 = []
		self.IH2 = []
		self.IN2 = []
		self.D2 = []
		self.R2 = []
		self.G2 = []
		self.H2 = []
		self.HH2 = []

		self.S3 = []
		self.E3 = []
		self.I3 = []
		self.A3 = []
		self.IH3 = []
		self.IN3 = []
		self.D3 = []
		self.R3 = []
		self.G3 = []
		self.H3 = []
		self.HH3 = []

		self.betas = []
		self.betas2 = []

		# self.vaccine_date = end_date
		# self.vaccine_day = self.days.index(datetime.datetime.strptime(self.vaccine_date, '%Y-%m-%d'))
		return

	def sim(self, release_date, release_frac, release_speed):
		release_day = self.dates_ext.index(release_date)
		self.rday = release_day
		release_size = min(1 - self.eta, release_frac * self.eta)
		self.HH = [release_size * self.n_0]
		result = simulate_release(self.size + size_ext,
		                          self.S, self.E, self.I, self.A, self.IH, self.IN, self.D, self.R, self.G, self.H,
		                          self.HH,
		                          self.betas, self.beta, self.gammaE, self.alpha, self.gamma, self.gamma2, self.gamma3,
		                          self.a1, self.a2, self.a3, self.h, self.Hiding_init, self.eta, self.c1, self.n_0,
		                          self.reopen_day, release_day, release_size, release_speed)

		G = self.G.copy()
		D = self.D.copy()

		return G, D

	def vac(self, release_date, release_frac, release_speed, vaccine_date, vaccine_speeds, vac_period1, vac_period2,
	        vac_eff1, vac_eff2):
		release_day = self.dates_ext.index(release_date)
		vaccine_day = self.dates_ext.index(vaccine_date)
		self.vday = vaccine_day
		release_size = min(1 - self.eta, release_frac * self.eta)
		extend_start_day = self.dates_ext.index(end_date)

		# no dose
		self.S0 = self.S[:extend_start_day].copy()
		self.E0 = self.E[:extend_start_day].copy()
		self.I0 = self.I[:extend_start_day].copy()
		self.A0 = self.A[:extend_start_day].copy()
		self.IH0 = self.IH[:extend_start_day].copy()
		self.IN0 = self.IN[:extend_start_day].copy()
		self.D0 = self.D[:extend_start_day].copy()
		self.R0 = self.R[:extend_start_day].copy()
		self.G0 = self.G[:extend_start_day].copy()
		self.H0 = self.H[:extend_start_day].copy()
		self.betas2 = self.betas[:extend_start_day].copy()

		# 1st dose
		self.S1 = [0] * extend_start_day
		self.E1 = [0] * extend_start_day
		self.I1 = [0] * extend_start_day
		self.A1 = [0] * extend_start_day
		self.IH1 = [0] * extend_start_day
		self.IN1 = [0] * extend_start_day
		self.D1 = [0] * extend_start_day
		self.R1 = [0] * extend_start_day
		self.G1 = [0] * extend_start_day
		self.H1 = [0] * extend_start_day

		# 2nd dose (or 2 weeks after 1st dose for it to take effect)
		self.S2 = [0] * extend_start_day
		self.E2 = [0] * extend_start_day
		self.I2 = [0] * extend_start_day
		self.A2 = [0] * extend_start_day
		self.IH2 = [0] * extend_start_day
		self.IN2 = [0] * extend_start_day
		self.D2 = [0] * extend_start_day
		self.R2 = [0] * extend_start_day
		self.G2 = [0] * extend_start_day
		self.H2 = [0] * extend_start_day

		# fully vaccinated
		self.S3 = [0] * extend_start_day
		self.E3 = [0] * extend_start_day
		self.I3 = [0] * extend_start_day
		self.A3 = [0] * extend_start_day
		self.IH3 = [0] * extend_start_day
		self.IN3 = [0] * extend_start_day
		self.D3 = [0] * extend_start_day
		self.R3 = [0] * extend_start_day
		self.G3 = [0] * extend_start_day
		self.H3 = [0] * extend_start_day

		# new releases
		self.HH0 = self.HH[:extend_start_day].copy()
		self.HH1 = [0] * extend_start_day
		self.HH2 = [0] * extend_start_day
		self.HH3 = [0] * extend_start_day
		self.HH1[-1] = self.HH0[-1] * HH_frac1
		self.HH2[-1] = self.HH0[-1] * HH_frac2
		self.HH3[-1] = self.HH0[-1] * HH_frac3
		self.HH0[-1] = self.HH0[-1] * (1 - HH_frac1 - HH_frac2 - HH_frac3)

		simulate_vac(extend_start_day, self.size + size_ext,
		             self.S0, self.E0, self.I0, self.A0, self.IH0, self.IN0, self.D0, self.R0, self.G0, self.H0,
		             self.HH0,
		             self.S1, self.E1, self.I1, self.A1, self.IH1, self.IN1, self.D1, self.R1, self.G1, self.H1,
		             self.HH1,
		             self.S2, self.E2, self.I2, self.A2, self.IH2, self.IN2, self.D2, self.R2, self.G2, self.H2,
		             self.HH2,
		             self.S3, self.E3, self.I3, self.A3, self.IH3, self.IN3, self.D3, self.R3, self.G3, self.H3,
		             self.HH3,
		             self.betas2, self.beta, self.gammaE, self.alpha, self.gamma, self.gamma2, self.gamma3, self.a1,
		             self.a2, self.a3, self.h, self.Hiding_init, self.eta, self.c1, self.n_0, self.reopen_day,
		             release_day, release_size, release_speed, vaccine_day, vaccine_speeds, vac_period1, vac_period2,
		             vac_eff1, vac_eff2)

		G = [self.G0[i] + self.G1[i] + self.G2[i] + self.G3[i] for i in range(len(self.G0))]
		D = [self.D0[i] + self.D1[i] + self.D2[i] + self.D3[i] for i in range(len(self.D0))]

		return G, D

	def plot(self):
		dG = [self.G[i] - self.G[i - 1] for i in range(1, len(self.G))]
		dG.insert(0, 0)
		dD = [self.D[i] - self.D[i - 1] for i in range(1, len(self.D))]
		dD.insert(0, 0)

		GG = [self.G0[i] + self.G1[i] + self.G2[i] + self.G3[i] for i in range(len(self.G0))]
		dGG = [GG[i] - GG[i - 1] for i in range(1, len(GG))]
		dGG.insert(0, 0)
		DD = [self.D0[i] + self.D1[i] + self.D2[i] + self.D3[i] for i in range(len(self.G0))]
		dDD = [DD[i] - DD[i - 1] for i in range(1, len(DD))]
		dDD.insert(0, 0)

		fig = plt.figure()
		ax = fig.add_subplot(221)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(223)
		ax4 = fig.add_subplot(224)
		fig.suptitle(self.state)
		ax.plot(self.days_ext, dG, label='dG')
		ax.plot(self.days_ext, dGG, label='dG2')
		ax2.plot(self.days_ext, dD, label='dD')
		ax2.plot(self.days_ext, dDD, label='dD2')
		ax.axvline(self.days_ext[self.rday], linestyle='dashed', color='tab:grey', label='release')
		ax.axvline(self.days_ext[self.vday], linestyle='dashed', color='tab:green', label='vaccine')
		ax2.axvline(self.days_ext[self.rday], linestyle='dashed', color='tab:grey', label='release')
		ax2.axvline(self.days_ext[self.vday], linestyle='dashed', color='tab:green', label='vaccine')

		# ax3.plot(self.days_ext, self.HH0, label='HH0')
		ax3.plot(self.days_ext, self.HH1, label='HH1')
		ax3.plot(self.days_ext, self.HH2, label='HH2')
		ax3.plot(self.days_ext, self.HH3, label='HH3')
		ax4.plot(self.days_ext, self.betas, label='beta')
		ax4.plot(self.days_ext, self.betas2, label='beta2')
		# ax.plot(self.days_ext, [i / 1000 for i in self.G], label='G')
		# ax.plot(self.days_ext,
		#         [(self.G0[i] + self.G1[i] + self.G2[i] + self.G3[i]) / 1000 for i in range(len(self.G0))], label='G2')
		# ax2.plot(self.days_ext, [i / 1000 for i in self.D], label='D')
		# ax2.plot(self.days_ext,
		#          [(self.D0[i] + self.D1[i] + self.D2[i] + self.D3[i]) / 1000 for i in range(len(self.D0))], label='D2')
		# ax.plot(self.days, [i / 1000 for i in self.confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
		# ax2.plot(self.days, [i / 1000 for i in self.death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
		ax.legend()
		ax2.legend()
		ax3.legend()
		ax4.legend()
		fig.autofmt_xdate()
		plt.show()
		plt.close(fig)
		return


def vac_all():
	state_path = f'india/vaccine/{round(1 / daily_rspeed)} day'
	if not os.path.exists(state_path):
		os.makedirs(state_path)
	global india_G0
	global india_D0
	global india_G1
	global india_D1
	global india_G2
	global india_D2
	global days
	global dates

	for state in states:

		state_obj = State(state, paras_df, pop_df, confirm_df, death_df)
		state_objs[state] = state_obj
		G0, D0 = state_obj.sim(release_date=r_date, release_frac=r_frac, release_speed=daily_rspeed)

		vac_speeds = [daily_vspeed1] * (state_obj.size + size_ext)
		G1, D1 = state_obj.vac(release_date=r_date, release_frac=r_frac, release_speed=daily_rspeed,
		                       vaccine_date=v_date, vaccine_speeds=vac_speeds, vac_period1=v_period1,
		                       vac_period2=v_period2, vac_eff1=v_eff1, vac_eff2=v_eff2)

		vac_speeds = [daily_vspeed1] * state_obj.vday + [daily_vspeed2] * (state_obj.size + size_ext - state_obj.vday)
		G2, D2 = state_obj.vac(release_date=r_date, release_frac=r_frac, release_speed=daily_rspeed,
		                       vaccine_date=v_date, vaccine_speeds=vac_speeds, vac_period1=v_period1,
		                       vac_period2=v_period2, vac_eff1=v_eff1, vac_eff2=v_eff2)

		G0s[state] = G0
		G1s[state] = G1
		G2s[state] = G2
		D0s[state] = D0
		D1s[state] = D1
		D2s[state] = D2
		days = state_obj.days_ext
		dates = state_obj.dates_ext
		if len(india_G1) == 0:
			india_G0 = G0.copy()
			india_D0 = D0.copy()
			india_G1 = G1.copy()
			india_D1 = D1.copy()
			india_G2 = G2.copy()
			india_D2 = D2.copy()
		else:
			india_G0 = [india_G0[i] + G0[i] for i in range(len(india_G0))]
			india_D0 = [india_D0[i] + D0[i] for i in range(len(india_D0))]
			india_G1 = [india_G1[i] + G1[i] for i in range(len(india_G1))]
			india_D1 = [india_D1[i] + D1[i] for i in range(len(india_D1))]
			india_G2 = [india_G2[i] + G2[i] for i in range(len(india_G2))]
			india_D2 = [india_D2[i] + D2[i] for i in range(len(india_D2))]

	# plot_comparison_india(india_G0, india_D0, india_G1, india_D1, india_G2, india_D2, r_date, v_date, dates, days)
	# for state in states:
	# 	plot_comparison_state(state)

	# improvements()
	print(india_G1[-1] - india_G1[-size_ext])
	print(india_G2[-1] - india_G2[-size_ext])
	return


def tmp():
	a = [1, 2, 3]
	b = [4, 5]
	print(a + b)

	return


def plot_comparison_india(G0, D0, G1, D1, G2, D2, release_date, vaccine_date, dates, days):
	v_day = dates.index(vaccine_date)
	r_day = dates.index(release_date)

	dG0 = [G0[i] - G0[i - 1] for i in range(1, len(G0))]
	dG0.insert(0, 0)
	dD0 = [D0[i] - D0[i - 1] for i in range(1, len(D0))]
	dD0.insert(0, 0)

	dG1 = [G1[i] - G1[i - 1] for i in range(1, len(G1))]
	dG1.insert(0, 0)
	dD1 = [D1[i] - D1[i - 1] for i in range(1, len(D1))]
	dD1.insert(0, 0)

	dG2 = [G2[i] - G2[i - 1] for i in range(1, len(G2))]
	dG2.insert(0, 0)
	dD2 = [D2[i] - D2[i - 1] for i in range(1, len(D2))]
	dD2.insert(0, 0)

	fig = plt.figure(figsize=(12, 4.5))
	fig.suptitle("India")
	ax = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax.set_title('Daily Cases')
	ax2.set_title('Daily Deaths')

	ax.plot(days, dG0, label='No vaccine')
	ax.plot(days, dG1, label='Current rate')
	ax.plot(days, dG2, label='Projected rate')

	ax2.plot(days, dD0, label='No vaccine')
	ax2.plot(days, dD1, label='Current rate')
	ax2.plot(days, dD2, label='Projected rate')

	# ax.axvline(days[r_day], linestyle='dashed', color='tab:grey', label='release')
	# ax.axvline(days[v_day], linestyle='dashed', color='tab:green', label='vaccine')
	# ax2.axvline(days[r_day], linestyle='dashed', color='tab:grey', label='release')
	# ax2.axvline(days[v_day], linestyle='dashed', color='tab:green', label='vaccine')

	ax.legend()
	ax2.legend()
	fig.autofmt_xdate()
	# plt.show()
	fig.savefig(f'india/vaccine/{round(1 / daily_rspeed)} day/vac_india_{round(1 / daily_rspeed)}.png',
	            bbox_inches='tight')
	plt.close(fig)

	return


def plot_comparison_state(state):
	state_obj = state_objs[state]
	G0 = G0s[state]
	D0 = D0s[state]
	G1 = G1s[state]
	D1 = D1s[state]
	G2 = G2s[state]
	D2 = D2s[state]

	dG0 = [G0[i] - G0[i - 1] for i in range(1, len(G0))]
	dG0.insert(0, 0)
	dD0 = [D0[i] - D0[i - 1] for i in range(1, len(D0))]
	dD0.insert(0, 0)

	dG1 = [G1[i] - G1[i - 1] for i in range(1, len(G1))]
	dG1.insert(0, 0)
	dD1 = [D1[i] - D1[i - 1] for i in range(1, len(D1))]
	dD1.insert(0, 0)

	dG2 = [G2[i] - G2[i - 1] for i in range(1, len(G2))]
	dG2.insert(0, 0)
	dD2 = [D2[i] - D2[i - 1] for i in range(1, len(D2))]
	dD2.insert(0, 0)

	fig = plt.figure(figsize=(12, 4.5))
	fig.suptitle(state_dict[state_obj.state])
	ax = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax.set_title('Daily Cases')
	ax2.set_title('Daily Deaths')

	ax.plot(state_obj.days_ext, dG0, label='No vaccine')
	ax.plot(state_obj.days_ext, dG1, label='Current rate')
	ax.plot(state_obj.days_ext, dG2, label='Projected rate')

	ax2.plot(state_obj.days_ext, dD0, label='No vaccine')
	ax2.plot(state_obj.days_ext, dD1, label='Current rate')
	ax2.plot(state_obj.days_ext, dD2, label='Projected rate')

	# ax.axvline(state_obj.days_ext[state_obj.rday], linestyle='dashed', color='tab:grey', label='release')
	# ax.axvline(state_obj.days_ext[state_obj.vday], linestyle='dashed', color='tab:green', label='vaccine')
	# ax2.axvline(state_obj.days_ext[state_obj.rday], linestyle='dashed', color='tab:grey', label='release')
	# ax2.axvline(state_obj.days_ext[state_obj.vday], linestyle='dashed', color='tab:green', label='vaccine')

	ax.legend()
	ax2.legend()
	fig.autofmt_xdate()
	# plt.show()
	fig.savefig(f'india/vaccine/{round(1 / daily_rspeed)} day/vac_{state_obj.state}_{round(1 / daily_rspeed)}.png',
	            bbox_inches='tight')
	plt.close(fig)

	return


def improvements():
	table = []
	col = ['State', 'Reduction in Cases', 'Improvement in Cases', 'Reduction in Deaths', 'Improvement in Deaths', ]
	for state in states:
		G1 = G1s[state]
		D1 = D1s[state]
		G2 = G2s[state]
		D2 = D2s[state]
		G_reduction = round((G1[-1] - G1[- size_ext]) - (G2[-1] - G2[- size_ext]))
		G_improvement = 1 - (G2[-1] - G2[- size_ext]) / (G1[-1] - G1[- size_ext])
		D_reduction = round((D1[-1] - D1[- size_ext]) - (D2[-1] - D2[- size_ext]))
		D_improvement = 1 - (D2[-1] - D2[- size_ext]) / (D1[-1] - D1[- size_ext])
		table.append([state_dict[state], G_reduction, G_improvement, D_reduction, D_improvement])
	G_reduction = round((india_G1[-1] - india_G1[- size_ext]) - (india_G2[-1] - india_G2[- size_ext]))
	G_improvement = 1 - (india_G2[-1] - india_G2[- size_ext]) / (india_G1[-1] - india_G1[- size_ext])
	D_reduction = round((india_D1[-1] - india_D1[- size_ext]) - (india_D2[-1] - india_D2[- size_ext]))
	D_improvement = 1 - (india_D2[-1] - india_D2[- size_ext]) / (india_D1[-1] - india_D1[- size_ext])
	table.insert(0, ['India', G_reduction, G_improvement, D_reduction, D_improvement])
	df = pd.DataFrame(table, columns=col)
	df.to_csv(f'india/vaccine/{round(1 / daily_rspeed)} day/improvement_{round(1 / daily_rspeed)}.csv', index=False)

	return


def main():
	# tmp()
	vac_all()
	return


if __name__ == '__main__':
	main()
