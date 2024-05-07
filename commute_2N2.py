import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from SIRfunctions import SEIARG, SEIARG_fixed, computeBeta_combined
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter
import os

# city_vaccine_rate = 0.0015
city_vaccine_rate = 0.002
city_test_rate = 0.02
school_beta_multiplier = 1
city_beta_multiplier = 1
vaccination_period = 1 / 14
log_scale = False
days = 150
fix_school_beta = False
daytime_fraction = 1 / 2

test_cost = 25
hosp_cost = [5500, 8500, 11600]


class Node:

	def pop(self, day):
		return self.S[day] + self.SV[day] + self.GV[day] + self.E[day] + self.I[day] + self.A[day] + self.IH[day] + \
		       self.IN[day] + self.D[day] + self.R[day] + self.Q[day]

	def read_para(self, SimFolder):
		ParaFile = f'{SimFolder}/{self.state}/para.csv'
		df = pd.read_csv(ParaFile)
		beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3 = \
			df.iloc[0]
		return [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1]

	def read_pop(self, PopFile):
		df = pd.read_csv(PopFile)
		n_0 = df[df.iloc[:, 0] == self.state].iloc[0]['POP']
		return n_0

	def __init__(self, state, SimFolder, PopFile, initial_infection, test_rate, vaccine_rate):
		self.state = state
		self.t_rate = test_rate
		self.v_rate = vaccine_rate
		self.gammaV = vaccination_period
		self.n_0 = self.read_pop(PopFile)

		[self.beta,
		 self.gammaE,
		 self.alpha,
		 self.gamma,
		 self.gamma2,
		 self.gamma3,
		 self.a1,
		 self.a2,
		 self.a3,
		 self.h,
		 self.Hiding_init,
		 self.k,
		 self.k2,
		 self.eta,
		 self.c1] = self.read_para(SimFolder)

		self.S = [self.n_0 * self.eta]
		self.SV = [0]
		self.E = [0]
		self.I = [initial_infection]
		self.A = [0]
		self.IH = [0]
		self.IN = [0]
		self.Q = [0]
		self.D = [0]
		self.R = [0]
		self.G = [initial_infection]
		self.H = [self.Hiding_init * self.n_0 * self.eta]
		self.Betas = []
		self.GH = [0]
		self.GT = [0]
		self.GV = [0]
		self.GI = [initial_infection]

	def vac_with_com(self, day, com):
		self.SV[day] += self.S[day] * self.v_rate
		com.SV[day] += com.S[day] * self.v_rate

		self.S[day] -= self.S[day] * self.v_rate
		com.S[day] -= com.S[day] * self.v_rate

	def vac(self, day):
		self.SV[day] += self.S[day] * self.v_rate
		self.S[day] -= self.S[day] * self.v_rate

	def sim_with_com(self, day, com, fixed_beta, frac):
		if fixed_beta:
			beta = self.beta
		else:
			beta = computeBeta_combined(self.beta, self.eta, self.n_0 + com.n_0,
			                            self.S[-1] + com.S[-1] + self.SV[-1] + com.SV[-1], self.I[-1] + com.I[-1], 0,
			                            self.c1, 0)
		# print(f'city beta{beta}, {self.beta}')
		dS = - beta * self.S[-1] * (self.I[-1] + self.A[-1] + com.I[-1] + com.A[-1]) / self.n_0 * frac
		dS_c = - beta * com.S[-1] * (self.I[-1] + self.A[-1] + com.I[-1] + com.A[-1]) / self.n_0 * frac

		dSV = - self.gammaV * self.SV[-1] * frac \
		      - beta * self.SV[-1] * (self.I[-1] + self.A[-1] + com.I[-1] + com.A[-1]) / self.n_0 * frac
		dSV_c = - self.gammaV * com.SV[-1] * frac \
		        - beta * com.SV[-1] * (self.I[-1] + self.A[-1] + com.I[-1] + com.A[-1]) / self.n_0 * frac

		dGV = self.gammaV * self.SV[-1] * frac
		dGV_c = self.gammaV * com.SV[-1] * frac

		dE = (beta * (self.S[-1] + self.SV[-1]) * (self.I[-1] + self.A[-1] + com.I[-1] + com.A[-1]) / self.n_0
		      - self.gammaE * self.E[-1]) * frac
		dE_c = (beta * (com.S[-1] + com.SV[-1]) * (self.I[-1] + self.A[-1] + com.I[-1] + com.A[-1]) / self.n_0
		        - self.gammaE * com.E[-1]) * frac

		dI = ((1 - self.alpha) * self.gammaE * self.E[-1] - (self.gamma + self.gamma2) * self.I[-1]) * frac
		dI_c = ((1 - self.alpha) * self.gammaE * com.E[-1] - (self.gamma + self.gamma2) * com.I[-1]) * frac

		dA = (self.alpha * self.gammaE * self.E[-1] - self.gamma3 * self.A[-1]) * frac
		dA_c = (self.alpha * self.gammaE * com.E[-1] - self.gamma3 * com.A[-1]) * frac

		dQ = - (self.gamma + self.gamma2) * self.Q[-1] * frac
		dQ_c = - (self.gamma + self.gamma2) * com.Q[-1] * frac

		dIH = (self.gamma * (self.I[-1] + self.Q[-1]) - (self.a1 + self.a2) * self.IH[-1]) * frac
		dIH_c = (self.gamma * (com.I[-1] + com.Q[-1]) - (self.a1 + self.a2) * com.IH[-1]) * frac

		dIN = (self.gamma2 * (self.I[-1] + self.Q[-1]) - self.a3 * self.IN[-1]) * frac
		dIN_c = (self.gamma2 * (com.I[-1] + com.Q[-1]) - self.a3 * com.IN[-1]) * frac

		dD = self.a2 * self.IH[-1] * frac
		dD_c = self.a2 * com.IH[-1] * frac

		dR = (self.a1 * self.IH[-1] + self.a3 * self.IN[-1] + self.gamma3 * self.A[-1]) * frac
		dR_c = (self.a1 * com.IH[-1] + self.a3 * com.IN[-1] + self.gamma3 * com.A[-1]) * frac

		dG = (1 - self.alpha) * self.gammaE * self.E[-1] * frac
		dG_c = (1 - self.alpha) * self.gammaE * com.E[-1] * frac

		dGH = self.gamma * (self.I[-1] + self.Q[-1]) * frac
		dGH_c = self.gamma * (com.I[-1] + com.Q[-1]) * frac

		dGI = (1 - self.alpha) * self.gammaE * self.E[-1] * frac
		dGI_c = (1 - self.alpha) * self.gammaE * com.E[-1] * frac

		self.S.append(self.S[-1] + dS)
		com.S.append(com.S[-1] + dS_c)

		self.SV.append(self.SV[-1] + dSV)
		com.SV.append(com.SV[-1] + dSV_c)

		self.GV.append(self.GV[-1] + dGV)
		com.GV.append(com.GV[-1] + dGV_c)

		self.E.append(self.E[-1] + dE)
		com.E.append(com.E[-1] + dE_c)

		self.I.append(self.I[-1] + dI)
		com.I.append(com.I[-1] + dI_c)

		self.A.append(self.A[-1] + dA)
		com.A.append(com.A[-1] + dA_c)

		self.Q.append(self.Q[-1] + dQ)
		com.Q.append(com.Q[-1] + dQ_c)

		self.IH.append(self.IH[-1] + dIH)
		com.IH.append(com.IH[-1] + dIH_c)

		self.IN.append(self.IN[-1] + dIN)
		com.IN.append(com.IN[-1] + dIN_c)

		self.D.append(self.D[-1] + dD)
		com.D.append(com.D[-1] + dD_c)

		self.R.append(self.R[-1] + dR)
		com.R.append(com.R[-1] + dR_c)

		self.G.append(self.G[-1] + dG)
		com.G.append(com.G[-1] + dG_c)

		self.GH.append(self.GH[-1] + dGH)
		com.GH.append(com.GH[-1] + dGH_c)

		self.GT.append(self.GT[-1])
		com.GT.append(com.GT[-1])

		self.GI.append(self.GI[-1] + dGI)
		com.GI.append(com.GI[-1] + dGI_c)

	def sim(self, day, fixed_beta, frac):
		if fixed_beta:
			beta = self.beta
		else:
			beta = computeBeta_combined(self.beta, self.eta, self.n_0, self.S[-1] + self.SV[-1], self.I[-1], 0, self.c1,
			                            0)

		dS = - beta * self.S[-1] * (self.I[-1] + self.A[-1]) / self.n_0 * frac

		dSV = - self.gammaV * self.SV[-1] * frac \
		      - beta * self.SV[-1] * (self.I[-1] + self.A[-1]) / self.n_0 * frac

		dGV = self.gammaV * self.SV[-1] * frac

		dE = (beta * (self.S[-1] + self.SV[-1])
		      * (self.I[-1] + self.A[-1]) / self.n_0 - self.gammaE * self.E[-1]) * frac

		dI = ((1 - self.alpha) * self.gammaE * self.E[-1] - (self.gamma + self.gamma2) * self.I[-1]) * frac

		dA = (self.alpha * self.gammaE * self.E[-1] - self.gamma3 * self.A[-1]) * frac

		dQ = - (self.gamma + self.gamma2) * self.Q[-1] * frac

		dIH = (self.gamma * (self.I[-1] + self.Q[-1]) - (self.a1 + self.a2) * self.IH[-1]) * frac

		dIN = (self.gamma2 * (self.I[-1] + self.Q[-1]) - self.a3 * self.IN[-1]) * frac

		dD = self.a2 * self.IH[-1] * frac

		dR = (self.a1 * self.IH[-1] + self.a3 * self.IN[-1] + self.gamma3 * self.A[-1]) * frac

		dG = (1 - self.alpha) * self.gammaE * self.E[-1] * frac

		dGH = self.gamma * (self.I[-1] + self.Q[-1]) * frac

		dGI = (1 - self.alpha) * self.gammaE * self.E[-1] * frac

		self.S.append(self.S[-1] + dS)

		self.SV.append(self.SV[-1] + dSV)

		self.GV.append(self.GV[-1] + dGV)

		self.E.append(self.E[-1] + dE)

		self.I.append(self.I[-1] + dI)

		self.A.append(self.A[-1] + dA)

		self.Q.append(self.Q[-1] + dQ)

		self.IH.append(self.IH[-1] + dIH)

		self.IN.append(self.IN[-1] + dIN)

		self.D.append(self.D[-1] + dD)

		self.R.append(self.R[-1] + dR)

		self.G.append(self.G[-1] + dG)

		self.GH.append(self.GH[-1] + dGH)

		self.GT.append(self.GT[-1])

		self.GI.append(self.GI[-1] + dGI)

	def sim2(self, day, fixed_beta, frac):
		if fixed_beta:
			delta = SEIARG_fixed(day,
			                     [self.S[-1], self.E[- 1], self.I[- 1], self.A[- 1], self.IH[- 1], self.IN[- 1],
			                      self.D[- 1], self.R[- 1], self.G[- 1], self.beta, self.gammaE, self.alpha, self.gamma,
			                      self.gamma2, self.gamma3, self.a1, self.a2, self.a3, self.eta, self.n_0, self.c1,
			                      self.H[-1], self.H[0]])
		else:
			delta = SEIARG(day,
			               [self.S[-1], self.E[- 1], self.I[- 1], self.A[- 1], self.IH[- 1], self.IN[- 1], self.D[- 1],
			                self.R[- 1], self.G[- 1], self.beta, self.gammaE, self.alpha, self.gamma, self.gamma2,
			                self.gamma3, self.a1, self.a2, self.a3, self.eta, self.n_0, self.c1, self.H[-1], self.H[0]])

		self.GI.append(self.GI[-1] + (1 - self.alpha) * self.gammaE * self.E[-1] * frac)
		self.S.append(self.S[-1] + delta[0] * frac)
		self.E.append(self.E[-1] + delta[1] * frac)
		self.I.append(self.I[-1] + delta[2] * frac)
		self.A.append(self.A[-1] + delta[3] * frac)
		self.IH.append(self.IH[-1] + delta[4] * frac)
		self.IN.append(self.IN[-1] + delta[5] * frac)
		self.D.append(self.D[-1] + delta[6] * frac)
		self.R.append(self.R[-1] + delta[7] * frac)
		self.G.append(self.G[-1] + delta[8] * frac)
		self.H.append(self.H[-1])
		self.Betas.append(delta[9])
		self.GH.append(self.GH[-1] + self.I[-1] * self.gamma * frac)

		# Q
		self.IH[-1] += self.Q[-1] * self.gamma * frac
		self.GH[-1] += self.Q[-1] * self.gamma * frac
		self.IN[-1] += self.Q[-1] * self.gamma2 * frac
		self.Q.append(self.Q[-1] * (1 - (self.gamma + self.gamma2) * frac))

		self.GT.append(self.GT[-1])
		self.GV.append(self.GV[-1])

	def test_with_com(self, day, com):
		self.GT[day] += (self.S[day] + self.E[day] + self.I[day] + self.A[day]) * self.t_rate
		com.GT[day] += (com.S[day] + com.E[day] + com.I[day] + com.A[day]) * self.t_rate

		self.Q[day] += self.I[day] * self.t_rate
		com.Q[day] += com.I[day] * self.t_rate

		self.I[day] -= self.I[day] * self.t_rate
		com.I[day] -= com.I[day] * self.t_rate

		self.IN[day] += self.A[day] * self.t_rate
		com.IN[day] += com.A[day] * self.t_rate

		self.G[day] += self.A[day] * self.t_rate
		com.G[day] += com.A[day] * self.t_rate

		self.A[day] -= self.A[day] * self.t_rate
		com.A[day] -= com.A[day] * self.t_rate

	def test(self, day):
		self.GT[day] += (self.S[day] + self.E[day] + self.I[day] + self.A[day]) * self.t_rate

		self.Q[day] += self.I[day] * self.t_rate

		self.I[day] -= self.I[day] * self.t_rate

		self.IN[day] += self.A[day] * self.t_rate

		self.G[day] += self.A[day] * self.t_rate

		self.A[day] -= self.A[day] * self.t_rate


class School(Node):
	pass

	def commute_out(self, commuter):
		commuter.S[-1] = self.S[-1] * 0.8
		commuter.n_0 = commuter.S[-1] / self.eta
		self.S[-1] -= commuter.S[-1]
		self.n_0 -= commuter.n_0
		self.E[-1] -= commuter.E[-1]
		self.I[-1] -= commuter.I[-1]
		self.A[-1] -= commuter.A[-1]
		# print(f'commuter n_0:{commuter.n_0}\n school n_0:{self.n_0}')
		return commuter


class Commuter:
	def __init__(self):
		self.S = [0]
		self.SV = [0]
		self.E = [0]
		self.I = [0]
		self.A = [0]
		self.IH = [0]
		self.IN = [0]
		self.D = [0]
		self.R = [0]
		self.Q = [0]
		self.G = [0]
		self.GH = [0]
		self.GT = [0]
		self.GV = [0]
		self.GI = [0]
		self.n_0 = 0

	def pop(self, day):
		return self.S[day] + self.SV[day] + self.GV[day] + self.E[day] + self.I[day] + self.A[day] + self.IH[day] + \
		       self.IN[day] + self.R[day] + self.D[day] + self.Q[day]

	def sim(self, day):
		self.S.append(self.S[-1])
		self.E.append(self.E[-1])
		self.I.append(self.I[-1])
		self.A.append(self.A[-1])


def simulate(testing, vaccinating, ct_rate, cv_rate, st_rate, sv_rate, fix_s_beta, day_frac):
	print()
	print(f'city test={ct_rate} city vaccine ={cv_rate}')
	print(f'school test={st_rate} school vaccine ={sv_rate}')
	school = School('IL-Cook', '2N', '2N/Population.csv', 0, st_rate, sv_rate)
	city = Node('IL-Cook', '2N', '2N/Population.csv', 5, ct_rate, cv_rate)
	# city.S[0] *= 1.1
	# city.eta *= 1.1
	school.S[-1] = 10000
	school.n_0 = school.S[-1] / school.eta
	commuter = Commuter()
	school.commute_out(commuter)
	if fix_s_beta:
		school.beta = 6 * (school.gamma + school.gamma2 + school.gamma3)
		print(f'fixed school beta={school.beta}')

	school.beta *= school_beta_multiplier
	city.beta *= city_beta_multiplier

	for i in range(days * 2):
		if i % 2 == 0:  # day time
			city.sim(i + 1, False, day_frac)
			school.sim_with_com(i + 1, commuter, fix_s_beta, day_frac)
			if testing:
				city.test(i + 1)
				school.test_with_com(i + 1, commuter)
			if vaccinating:
				city.vac(i + 1)
				school.vac_with_com(i + 1, commuter)

		else:  # night time
			city.sim_with_com(i + 1, commuter, False, 1 - day_frac)
			school.sim(i + 1, fix_s_beta, 1 - day_frac)

		# print(school.pop(i), city.pop(i), commuter.pop(i))

	return city, school, commuter


def comparison_2D_I():
	t_low, t_up, t_step = 0, 0.50, 0.02
	v_low, v_up, v_step = 0, 0.010, 0.0005
	t_rates = np.arange(t_low, t_up + t_step, t_step)
	v_rates = np.arange(v_low, v_up + v_step, v_step)
	# print(t_rates, v_rates)
	cities = []
	schools = []
	commuters = []
	for t_rate in t_rates:
		cities.append([])
		schools.append([])
		commuters.append([])
		for v_rate in v_rates:
			# print(f'testing rate={round(t_rate, 3)} vac rate={round(v_rate, 3)}')
			city, school, commuter = simulate(testing=True, vaccinating=True, ct_rate=city_test_rate,
			                                  cv_rate=city_vaccine_rate, st_rate=t_rate, sv_rate=v_rate,
			                                  fix_s_beta=fix_school_beta, day_frac=daytime_fraction)
			cities[-1].append(city)
			schools[-1].append(school)
			commuters[-1].append(commuter)

	fig = plt.figure()
	ax = fig.add_subplot()
	peakI_city, peakI_school, peakI_commuter, peakI_school2 = select_peak_I(cities, schools, commuters)
	x, y = np.mgrid[slice(t_low, t_up + t_step, t_step), slice(v_low, v_up + v_step, v_step)]
	v_min = min(min([peakI_school2[i] for i in range(len(peakI_commuter))]))
	v_max = max(max([peakI_school2[i] for i in range(len(peakI_commuter))]))
	# print(v_min, v_max)
	if log_scale:
		c = ax.pcolor(x, y, peakI_school2, norm=LogNorm(vmin=v_min, vmax=v_max), cmap='RdBu_r')
	else:
		c = ax.pcolor(x, y, peakI_school2, vmin=v_min, vmax=v_max, cmap='RdBu_r')
	ax.set_xlabel('testing rate')
	ax.set_ylabel('vaccinating rate')
	fig.colorbar(c, ax=ax)

	cols = ['test\\vac']
	cols.extend(v_rates)
	rows = list(t_rates)
	for i in range(len(rows)):
		rows[i] = [rows[i]]
		rows[i].extend(peakI_school2[i])
	out_df = pd.DataFrame(rows, columns=cols)

	if fix_school_beta:
		fig.suptitle(
			f'School peak infection with fixed beta and D/N={round(daytime_fraction / (1 - daytime_fraction), 1)}')
		fig.savefig(f'2N/grid_I_fixed_{round(daytime_fraction, 2)}.png', bbox_inches="tight")
		out_df.to_csv(f'2N/PeakI_fixed_{round(daytime_fraction, 2)}.csv', index=False)
	else:
		fig.suptitle(
			f'School peak infection with dynamic beta and D/N={round(daytime_fraction / (1 - daytime_fraction), 1)}')
		fig.savefig(f'2N/grid_I_dynamic_{round(daytime_fraction, 2)}.png', bbox_inches="tight")
		out_df.to_csv(f'2N/PeakI_dynamic_{round(daytime_fraction, 2)}.csv', index=False)

	# plt.show()
	return


def select_peak_I(cities, schools, commuters):
	peakI_city = []
	peakI_school = []
	peakI_commuter = []
	peakI_school2 = []
	for i in range(len(cities)):
		peakI_city.append([])
		peakI_school.append([])
		peakI_commuter.append([])
		peakI_school2.append([])
		for j in range(len(cities[0])):
			peakI_city[-1].append(max(cities[i][j].I))
			peakI_school[-1].append(max(schools[i][j].I))
			peakI_commuter[-1].append(max(commuters[i][j].I))
			school_commuter = [schools[i][j].I[t] + commuters[i][j].I[t] for t in range(len(schools[i][j].I))]
			peakI_school2[-1].append(max(school_commuter))
	# print(peakI_city)
	return peakI_city, peakI_school, peakI_commuter, peakI_school2


def comparison_2D_IH():
	t_low, t_up, t_step = 0, 0.50, 0.02
	v_low, v_up, v_step = 0, 0.010, 0.0005
	t_rates = np.arange(t_low, t_up + t_step, t_step)
	v_rates = np.arange(v_low, v_up + v_step, v_step)
	# print(t_rates, v_rates)
	cities = []
	schools = []
	commuters = []
	for t_rate in t_rates:
		cities.append([])
		schools.append([])
		commuters.append([])
		for v_rate in v_rates:
			# print(f'testing rate={round(t_rate, 3)} vac rate={round(v_rate, 3)}')
			city, school, commuter = simulate(testing=True, vaccinating=True, ct_rate=city_test_rate,
			                                  cv_rate=city_vaccine_rate, st_rate=t_rate, sv_rate=v_rate,
			                                  fix_s_beta=fix_school_beta, day_frac=daytime_fraction)
			cities[-1].append(city)
			schools[-1].append(school)
			commuters[-1].append(commuter)

	fig = plt.figure()
	ax = fig.add_subplot()
	peakIH_city, peakIH_school, peakIH_commuter, peakIH_school2 = select_peak_IH(cities, schools, commuters)
	x, y = np.mgrid[slice(t_low, t_up + t_step, t_step), slice(v_low, v_up + v_step, v_step)]
	v_min = min(min([peakIH_school2[i] for i in range(len(peakIH_commuter))]))
	v_max = max(max([peakIH_school2[i] for i in range(len(peakIH_commuter))]))
	# print(v_min, v_max)
	if log_scale:
		c = ax.pcolor(x, y, peakIH_school2, norm=LogNorm(vmin=v_min, vmax=v_max), cmap='RdBu_r')
	else:
		c = ax.pcolor(x, y, peakIH_school2, vmin=v_min, vmax=v_max, cmap='RdBu_r')
	ax.set_xlabel('testing rate')
	ax.set_ylabel('vaccinating rate')
	fig.colorbar(c, ax=ax)

	cols = ['test\\vac']
	cols.extend(v_rates)
	rows = list(t_rates)
	for i in range(len(rows)):
		rows[i] = [rows[i]]
		rows[i].extend(peakIH_school2[i])
	out_df = pd.DataFrame(rows, columns=cols)

	if fix_school_beta:
		fig.suptitle(
			f'School peak hospitalization with fixed beta and D/N={round(daytime_fraction / (1 - daytime_fraction), 1)}')
		fig.savefig(f'2N/grid_IH_fixed_{round(daytime_fraction, 2)}.png', bbox_inches="tight")
		out_df.to_csv(f'2N/PeakIH_fixed_{round(daytime_fraction, 2)}.csv', index=False)
	else:
		fig.suptitle(
			f'School peak hospitalization with dynamic beta and D/N={round(daytime_fraction / (1 - daytime_fraction), 1)}')
		fig.savefig(f'2N/grid_IH_dynamic_{round(daytime_fraction, 2)}.png', bbox_inches="tight")
		out_df.to_csv(f'2N/PeakIH_dynamic_{round(daytime_fraction, 2)}.csv', index=False)

	# plt.show()
	return


def select_peak_IH(cities, schools, commuters):
	peakIH_city = []
	peakIH_school = []
	peakIH_commuter = []
	peakIH_school2 = []
	for i in range(len(cities)):
		peakIH_city.append([])
		peakIH_school.append([])
		peakIH_commuter.append([])
		peakIH_school2.append([])
		for j in range(len(cities[0])):
			peakIH_city[-1].append(max(cities[i][j].IH))
			peakIH_school[-1].append(max(schools[i][j].IH))
			peakIH_commuter[-1].append(max(commuters[i][j].IH))
			school_commuter = [schools[i][j].IH[t] + commuters[i][j].IH[t] for t in range(len(schools[i][j].IH))]
			peakIH_school2[-1].append(max(school_commuter))
	# print(peakI_city)
	return peakIH_city, peakIH_school, peakIH_commuter, peakIH_school2


def tmp():
	city, school, commuter = simulate(testing=True, vaccinating=True, ct_rate=city_test_rate,
	                                  cv_rate=city_vaccine_rate, st_rate=0.475, sv_rate=city_vaccine_rate,
	                                  fix_s_beta=False, day_frac=daytime_fraction)
	return


def comparison_school_testing():
	t_low, t_up, t_step = 0, 1, 0.2
	t_rates = np.arange(t_low, t_up + t_step, t_step)
	cities = []
	schools = []
	commuters = []
	for t_rate in t_rates:
		city, school, commuter = simulate(testing=True, vaccinating=True, ct_rate=city_test_rate,
		                                  cv_rate=city_vaccine_rate, st_rate=t_rate, sv_rate=city_vaccine_rate,
		                                  fix_s_beta=fix_school_beta, day_frac=daytime_fraction)
		cities.append(city)
		schools.append(school)
		commuters.append(commuter)

	fig, axes = plt.subplots(1, 2)
	fig.set_size_inches(14, 6)

	days = np.arange(0, 150.5, 0.5)

	for i in range(len(t_rates)):
		I_combined = [schools[i].I[t] + commuters[i].I[t] for t in range(len(schools[i].I))]
		I_combined = I_combined[::2]
		IH_combined = [schools[i].IH[t] + commuters[i].IH[t] for t in range(len(schools[i].IH))]
		IH_combined = IH_combined[::2]
		axes[0].plot(I_combined, label=f'Testing rate={round(t_rates[i], 3)}')
		axes[1].plot(IH_combined, label=f'Testing rate={round(t_rates[i], 3)}')

	axes[0].set_xlabel('time')
	# axes[0].set_ylabel('I')

	axes[1].set_xlabel('time')
	# axes[1].set_ylabel('IH')

	axes[0].set_title('Active Infection')
	axes[1].set_title('Hospitalization')

	axes[0].legend()
	axes[1].legend()
	# axes[1][0].legend()
	# axes[1][1].legend()
	if fix_school_beta:
		fig.suptitle(f'Fixed beta in school with D/N={round(daytime_fraction / (1 - daytime_fraction), 2)}')
		fig.savefig(f'2N/I_IH_fixed_{round(daytime_fraction, 2)}.png', bbox_inches="tight")
	else:
		fig.suptitle(f'Dynamic beta in school with D/N={round(daytime_fraction / (1 - daytime_fraction), 2)}')
		fig.savefig(f'2N/I_IH_dynamic_{round(daytime_fraction, 2)}.png', bbox_inches="tight")
	# plt.show()
	plt.close(fig)

	fig, ax = plt.subplots()
	ax.plot(t_rates, [schools[i].GH[-1] + commuters[i].GH[-1] for i in range(len(t_rates))], label='cum IH')
	ax.plot(t_rates, [schools[i].GT[-1] + commuters[i].GT[-1] for i in range(len(t_rates))], label='cum Tests')

	# for i in range(len(t_rates)):
	# 	ax.plot(t_rates[i], schools[i].GH[-1] + commuters[i].GH[-1], label=f't_rate={round(t_rates[i], 2)}')
	# ax.legend()
	plt.show()
	return


def school_testing_cost():
	t_low, t_up, t_step = 0, 1, 0.025
	t_rates = np.arange(t_low, t_up + t_step, t_step)
	days = np.arange(0, 150.5, 1)

	cols = ['save \\ test rate']
	cols.extend(t_rates)

	# normal vaccination rate
	cities = []
	schools = []
	commuters = []
	for t_rate in t_rates:
		city, school, commuter = simulate(testing=True, vaccinating=True, ct_rate=city_test_rate,
		                                  cv_rate=city_vaccine_rate, st_rate=t_rate, sv_rate=city_vaccine_rate,
		                                  fix_s_beta=fix_school_beta, day_frac=daytime_fraction)
		cities.append(city)
		schools.append(school)
		commuters.append(commuter)
	# fig2 = plt.figure()
	# ax = fig2.add_subplot()
	# print(school.S)
	# ax.plot([school.S[i] + commuter.S[i] for i in range(len(school.S))], label='S')
	# ax.plot([school.E[i] + commuter.E[i] for i in range(len(school.E))], label='E')
	# ax.plot([school.I[i] + commuter.I[i] for i in range(len(school.I))], label='I')
	# ax.plot([school.A[i] + commuter.A[i] for i in range(len(school.A))], label='A')
	# ax.plot(commuter.S, label='commuter S')
	# ax.plot(commuter.E, label='commuter E')
	# ax.plot(commuter.I, label='commuter I')
	# ax.plot(commuter.A, label='commuter A')

	# ax.plot(city.S, label='city S')
	# ax.plot(city.E, label='city E')
	# ax.plot(city.I, label='city I')
	# ax.plot(city.A, label='city A')
	# ax.legend()
	# plt.show()
	# plt.close(fig2)

	fig, axes = plt.subplots(1, 2)
	fig.set_size_inches(14, 6)

	base_GI = schools[0].GI[-1] + commuters[0].GI[-1]
	base_GT = schools[0].GT[-1] + commuters[0].GT[-1]

	l0 = axes[0].axhline(y=0, color='grey', linestyle=':')
	y1 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[0] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y1 = np.asarray(y1)
	l1 = axes[0].plot(t_rates, y1, label=f'saving w/ {hosp_cost[0]} / infection')
	y2 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[1] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y2 = np.asarray(y2)
	l2 = axes[0].plot(t_rates, y2, label=f'saving w/ {hosp_cost[1]} / infection')
	y3 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[2] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y3 = np.asarray(y3)
	l3 = axes[0].plot(t_rates, y3, label=f'saving w/ {hosp_cost[2]} / infection')

	axes[0].plot(t_rates, [(schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost / 1000000
	                       for i in range(len(t_rates))], label=f'testing cost')

	# save to csv
	table = []
	table.append([hosp_cost[0]])
	table[-1].extend(y1)
	table.append([hosp_cost[1]])
	table[-1].extend(y2)
	table.append([hosp_cost[2]])
	table[-1].extend(y3)
	out_df = pd.DataFrame(table, columns=cols)
	out_df.to_csv(f'2N/saving_{round(city_vaccine_rate, 4)}.csv', index=False)

	# fill between curves
	z = np.zeros(len(t_rates))
	axes[0].fill_between(t_rates, z, y1, where=y1 > z, interpolate=True, color=l1[0].get_color(), alpha=0.5)
	y1 = [max(y1[i], z[i]) for i in range(len(z))]
	axes[0].fill_between(t_rates, y1, y2, where=y2 > y1, interpolate=True, color=l2[0].get_color(), alpha=0.5)
	y2 = [max(y2[i], z[i]) for i in range(len(z))]
	axes[0].fill_between(t_rates, y3, y2, where=y3 > y2, interpolate=True, color=l3[0].get_color(), alpha=0.5)

	axes[0].set_xlabel('testing rate')
	axes[0].set_ylabel('million')
	axes[0].set_title(f'{round(city_vaccine_rate * 100, 2)}% vaccination rate')
	axes[0].legend()

	# double vaccination rate
	cities = []
	schools = []
	commuters = []
	for t_rate in t_rates:
		city, school, commuter = simulate(testing=True, vaccinating=True, ct_rate=city_test_rate,
		                                  cv_rate=2 * city_vaccine_rate, st_rate=t_rate, sv_rate=2 * city_vaccine_rate,
		                                  fix_s_beta=fix_school_beta, day_frac=daytime_fraction)
		cities.append(city)
		schools.append(school)
		commuters.append(commuter)

	base_GI = schools[0].GI[-1] + commuters[0].GI[-1]
	base_GT = schools[0].GT[-1] + commuters[0].GT[-1]
	print(base_GI, base_GT)

	l0 = axes[1].axhline(y=0, color='grey', linestyle=':')
	y1 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[0] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y1 = np.asarray(y1)
	l1 = axes[1].plot(t_rates, y1, label=f'saving w/ {hosp_cost[0]} / infection')
	y2 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[1] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y2 = np.asarray(y2)
	l2 = axes[1].plot(t_rates, y2, label=f'saving w/ {hosp_cost[1]} / infection')
	y3 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[2] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y3 = np.asarray(y3)
	l3 = axes[1].plot(t_rates, y3, label=f'saving w/ {hosp_cost[2]} / infection')

	axes[1].plot(t_rates, [(schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost / 1000000
	                       for i in range(len(t_rates))], label=f'testing cost')

	# save to csv
	table = []
	table.append([hosp_cost[0]])
	table[-1].extend(y1)
	table.append([hosp_cost[1]])
	table[-1].extend(y2)
	table.append([hosp_cost[2]])
	table[-1].extend(y3)
	out_df = pd.DataFrame(table, columns=cols)
	out_df.to_csv(f'2N/saving_{round(2 * city_vaccine_rate, 4)}.csv', index=False)

	# fill between curves
	z = np.zeros(len(t_rates))
	axes[1].fill_between(t_rates, z, y1, where=y1 >= z, interpolate=True, color=l1[0].get_color(), alpha=0.5)
	y1 = [max(y1[i], z[i]) for i in range(len(z))]
	axes[1].fill_between(t_rates, y1, y2, where=y2 >= y1, interpolate=True, color=l2[0].get_color(), alpha=0.5)
	y2 = [max(y2[i], z[i]) for i in range(len(z))]
	axes[1].fill_between(t_rates, y3, y2, where=y3 >= y2, interpolate=True, color=l3[0].get_color(), alpha=0.5)

	axes[1].set_xlabel('testing rate')
	axes[1].set_ylabel('million')
	axes[1].set_title(f'{round(2 * city_vaccine_rate * 100, 2)}% vaccination rate')
	axes[1].legend()

	axes[0].set_xlim(t_low, t_up)
	axes[1].set_xlim(t_low, t_up)
	# fig.savefig(f'2N/testing saving.png', bbox_inches="tight")
	plt.show()
	return


def school_testing_cost2():
	t_low, t_up, t_step = 0, 1, 0.005
	t_rates = [0, 1 / 14, 1 / 7, 1 / 5, 1 / 3, 1 / 2, 1]
	# fig, axes = plt.subplots(1, 2)
	# fig.set_size_inches(14, 6)
	days = np.arange(0, 150.5, 1)

	cols = ['save \\ test rate']
	cols.extend(t_rates)

	# normal vaccination rate
	cities = []
	schools = []
	commuters = []
	for t_rate in t_rates:
		city, school, commuter = simulate(testing=True, vaccinating=True, ct_rate=city_test_rate,
		                                  cv_rate=city_vaccine_rate, st_rate=t_rate, sv_rate=city_vaccine_rate,
		                                  fix_s_beta=fix_school_beta, day_frac=daytime_fraction)
		cities.append(city)
		schools.append(school)
		commuters.append(commuter)

	base_GI = schools[0].GI[-1] + commuters[0].GI[-1]
	base_GT = schools[0].GT[-1] + commuters[0].GT[-1]

	y1 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[0] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y1 = np.asarray(y1)
	y2 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[1] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y2 = np.asarray(y2)
	y3 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[2] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y3 = np.asarray(y3)

	# save to csv
	table = []
	table.append([hosp_cost[0]])
	table[-1].extend(y1)
	table.append([hosp_cost[1]])
	table[-1].extend(y2)
	table.append([hosp_cost[2]])
	table[-1].extend(y3)
	out_df = pd.DataFrame(table, columns=cols)
	out_df.to_csv(f'2N/saving2_{round(city_vaccine_rate, 4)}.csv', index=False)

	# double vaccination rate
	cities = []
	schools = []
	commuters = []
	for t_rate in t_rates:
		city, school, commuter = simulate(testing=True, vaccinating=True, ct_rate=city_test_rate,
		                                  cv_rate=2 * city_vaccine_rate, st_rate=t_rate, sv_rate=2 * city_vaccine_rate,
		                                  fix_s_beta=fix_school_beta, day_frac=daytime_fraction)
		cities.append(city)
		schools.append(school)
		commuters.append(commuter)

	base_GI = schools[0].GI[-1] + commuters[0].GI[-1]
	base_GT = schools[0].GT[-1] + commuters[0].GT[-1]
	print(base_GI, base_GT)

	y1 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[0] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y1 = np.asarray(y1)
	y2 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[1] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y2 = np.asarray(y2)
	y3 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[2] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y3 = np.asarray(y3)

	# save to csv
	table = []
	table.append([hosp_cost[0]])
	table[-1].extend(y1)
	table.append([hosp_cost[1]])
	table[-1].extend(y2)
	table.append([hosp_cost[2]])
	table[-1].extend(y3)
	out_df = pd.DataFrame(table, columns=cols)
	out_df.to_csv(f'2N/saving2_{round(2 * city_vaccine_rate, 4)}.csv', index=False)

	return


def school_testing_cost3():
	t_low, t_up, t_step = 0.005, 1, 0.005
	t_rates = np.arange(t_low, t_up + t_step, t_step)
	# fig, axes = plt.subplots(1, 2)
	# fig.set_size_inches(14, 6)
	days = np.arange(0, 150.5, 1)

	cols = ['save \\ test rate']
	cols.extend(t_rates)

	# normal vaccination rate
	cities = []
	schools = []
	commuters = []
	for t_rate in t_rates:
		city, school, commuter = simulate(testing=True, vaccinating=True, ct_rate=city_test_rate,
		                                  cv_rate=city_vaccine_rate, st_rate=t_rate, sv_rate=city_vaccine_rate,
		                                  fix_s_beta=fix_school_beta, day_frac=daytime_fraction)
		cities.append(city)
		schools.append(school)
		commuters.append(commuter)

	base_GI = schools[0].GI[-1] + commuters[0].GI[-1]
	base_GT = schools[0].GT[-1] + commuters[0].GT[-1]

	y1 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[0] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y1 = np.asarray(y1)
	y2 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[1] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y2 = np.asarray(y2)
	y3 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[2] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y3 = np.asarray(y3)

	# save to csv
	table = []
	table.append([hosp_cost[0]])
	table[-1].extend(y1)
	table.append([hosp_cost[1]])
	table[-1].extend(y2)
	table.append([hosp_cost[2]])
	table[-1].extend(y3)
	out_df = pd.DataFrame(table, columns=cols)
	out_df.to_csv(f'2N/saving3_{round(city_vaccine_rate, 4)}.csv', index=False)

	# double vaccination rate
	cities = []
	schools = []
	commuters = []
	for t_rate in t_rates:
		city, school, commuter = simulate(testing=True, vaccinating=True, ct_rate=city_test_rate,
		                                  cv_rate=2 * city_vaccine_rate, st_rate=t_rate, sv_rate=2 * city_vaccine_rate,
		                                  fix_s_beta=fix_school_beta, day_frac=daytime_fraction)
		cities.append(city)
		schools.append(school)
		commuters.append(commuter)

	base_GI = schools[0].GI[-1] + commuters[0].GI[-1]
	base_GT = schools[0].GT[-1] + commuters[0].GT[-1]
	print(base_GI, base_GT)

	y1 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[0] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y1 = np.asarray(y1)
	y2 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[1] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y2 = np.asarray(y2)
	y3 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[2] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
	y3 = np.asarray(y3)

	# save to csv
	table = []
	table.append([hosp_cost[0]])
	table[-1].extend(y1)
	table.append([hosp_cost[1]])
	table[-1].extend(y2)
	table.append([hosp_cost[2]])
	table[-1].extend(y3)
	out_df = pd.DataFrame(table, columns=cols)
	out_df.to_csv(f'2N/saving3_{round(2 * city_vaccine_rate, 4)}.csv', index=False)

	return


def main():
	# comparison_2D_I()
	# comparison_2D_IH()
	# comparison_school_testing()
	school_testing_cost()
	# school_testing_cost2()
	# school_testing_cost3()

	# tmp()

	return


if __name__ == "__main__":
	main()
