import numpy as np
import pandas as pd
import time
import math
import concurrent.futures
import multiprocessing
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score, mean_squared_error
from SIRfunctions import SEIARG
import datetime
from numpy.random import uniform as uni
import os
import warnings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# matplotlib.use('Agg')

np.set_printoptions(threshold=sys.maxsize)
Geo = 0.98

# num_threads = 200
num_threads = 50
num_CI = 1000
# num_CI = 5

num_threads_dist = 0

# weight of G in initial fitting
theta = 0.7
# weight of G in release fitting
theta2 = 0.8

I_0 = 5
beta_range = (0.1, 100)
gammaE_range = (0.2, 0.3)
alpha_range = (0.1, 0.9)
gamma_range = (0.04, 0.2)
gamma2_range = (0.04, 0.2)
gamma3_range = (0.04, 0.2)
# sigma_range = (0.001, 1)
a1_range = (0.01, 0.5)
a2_range = (0.01, 0.2)
a3_range = (0.01, 0.2)
eta_range = (0.001, 0.05)
c1_fixed = (0.9, 0.9)
c1_range = (0.9, 1)
h_range = (0, 10)
k_range = (0.1, 2)
k2_range = (0.1, 2)
# end_date = '2020-06-10'
# end_date = '2020-08-16'
# end_date = '2020-09-23'
# end_date = '2020-09-22'
end_date = '2020-08-31'
release_duration = 30
k_drop = 14
p_m = 1
# Hiding = 0.33
delay = 7
change_eta2 = False

fig_row = 5
fig_col = 3


# save simulation of SIRG fitting to csv
def save_sim_combined(csv_filename, data, days):
	days = [day.strftime('%Y-%m-%d') for day in days]
	c0 = ['S', 'E', 'I', 'A', 'IH', 'IN', 'D', 'R', 'G', 'H', 'beta']
	df = pd.DataFrame(data, columns=days)
	df.insert(0, 'series', c0)
	df.to_csv(csv_filename, index=False)
	print('simulation saved\n')


# save simulation of SIRG fitting to csv for initial phase only
def save_sim_init(csv_filename, data, days):
	days = [day.strftime('%Y-%m-%d') for day in days]
	c0 = ['S', 'I', 'IH', 'IN', 'D', 'R', 'G', 'H', 'beta']
	df = pd.DataFrame(data, columns=days)
	df.insert(0, 'series', c0)
	df.to_csv(csv_filename, index=False)
	print('simulation saved\n')


# save the parameters distribution to CSV
def save_para_combined(para_file, paras):
	para_label = ['beta', 'gammaE', 'alpha', 'gamma', 'gamma2', 'gamma3', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k',
	              'k2', 'eta', 'c1', 'metric1', 'metric2', 'metric3', 'r1', 'r2', 'r3']
	df = pd.DataFrame(paras, columns=para_label)
	df.to_csv(para_file, index=False, header=True)
	# df.to_csv(f'init_only_{end_date}/{state}/para.csv', index=False, header=True)
	print('parameters saved\n')


# save the parameters distribution to CSV for initial phase only
def save_para_init(state, paras):
	para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1', 'metric1',
	              'metric2', 'metric3', 'r1', 'r2', 'r3']
	df = pd.DataFrame(paras, columns=para_label)
	df.to_csv(f'ASYM/init_only_{end_date}/{state}/para.csv', index=False, header=True)
	# df.to_csv(f'init_only_{end_date}/{state}/para.csv', index=False, header=True)
	print('parameters saved\n')


# simulate combined phase
def simulate_combined(size, SIRG, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
                      a3, h, k, k2, eta, c1, n_0, reopen_day):
	result = True
	H0 = H[0]
	eta2 = eta
	kk = 1
	kk2 = 1
	r = h * H0
	betas = [beta]
	for i in range(1, size):

		if i > reopen_day:
			kk = k
			kk2 = k2
			release = min(H[-1], r * funcmod(i))
			S[-1] += release
			H[-1] -= release
		delta = SIRG(i,
		             [S[i - 1], E[i - 1], I[i - 1], A[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1], beta,
		              gammaE, alpha, kk * gamma, gamma2, gamma3, a1, kk2 * a2, a3, eta2, n_0, c1, H[-1], H0])
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
		betas.append(delta[9])
		if S[-1] < 0:
			result = False
			break
	return result, [S, E, I, A, IH, IN, D, R, G, H, betas]


# combined fitting
def loss_combined(point, c1, confirmed, death, hosp, n_0, SIRG, reopen_day):
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
	h = point[10]
	Hiding_init = point[11]
	k = point[12]
	k2 = point[13]
	S = [n_0 * eta]
	E = [0]
	I = [confirmed[0]]
	A = [0]
	IH = [hosp[0]]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [Hiding_init * eta * n_0]
	result, [S, E, I, A, IH, IN, D, R, G, H, betas] \
		= simulate_combined(size, SIRG, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1,
		                    a2, a3, h, k, k2, eta, c1, n_0, reopen_day)

	if not result:
		return 1000000

	size1 = reopen_day
	size2 = size - size1
	weights1 = [Geo ** n for n in range(size1)]
	weights1.reverse()
	weights2 = [Geo ** n for n in range(size2)]
	weights2.reverse()
	weights = weights1
	weights.extend(weights2)
	if len(weights) != size:
		print('wrong weights!')

	# weights = [Geo ** n for n in range(size)]
	# weights.reverse()
	weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
	weighted_G = [G[i] * weights[i] for i in range(size)]
	weighted_death = [death[i] * weights[i] for i in range(size)]
	weighted_D = [D[i] * weights[i] for i in range(size)]
	# weighted_hosp = [hosp[i] * weights[i] for i in range(size)]
	# weighted_IH = [IH[i] * weights[i] for i in range(size)]

	# confirmed_derivative = np.diff(confirmed)
	# G_derivative = np.diff(G)
	# confirmed_derivative = [confirmed_derivative[i] * weights[i] for i in range(size - 1)]
	# G_derivative = [G_derivative[i] * weights[i] for i in range(size - 1)]
	# alpha = 0.5
	# metric00 = r2_score(weighted_confirmed, weighted_G)
	# metric01 = r2_score(confirmed_derivative, G_derivative)
	# metric0 = (alpha * metric00 + (1 - alpha) * metric01)

	weighted_hosp = hosp
	weighted_IH = IH

	metric0 = r2_score(weighted_confirmed, weighted_G)
	metric1 = r2_score(weighted_death, weighted_D)
	metric2 = r2_score(weighted_hosp, weighted_IH)
	# return -(theta * metric0 + 1 * (1 - theta) / 2 * metric1 + 1 * (1 - theta) / 2 * metric2)
	return -(0.4 * metric0 + 0.4 * metric1 + 0.2 * metric2)


# initial phase fitting
def loss_init(point, c1, confirmed, death, hosp, n_0, SIRG, reopen_day):
	size = len(confirmed)
	beta = point[0]
	gamma = point[1]
	gamma2 = point[2]
	a1 = point[3]
	a2 = point[4]
	a3 = point[5]
	eta = point[6]
	h = point[7]
	Hiding_init = point[8]
	k = point[9]
	k2 = point[10]
	S = [n_0 * eta]
	I = [confirmed[0]]
	IH = [hosp[0]]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [Hiding_init * eta * n_0]
	result, [S, I, IH, IN, D, R, G, H, betas] \
		= simulate_combined(size, SIRG, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2, eta, c1,
		                    n_0, reopen_day)

	if not result:
		return 1000000

	size1 = reopen_day
	size2 = size - size1
	weights1 = [Geo ** n for n in range(size1)]
	weights1.reverse()
	weights2 = [Geo ** n for n in range(size2)]
	weights2.reverse()
	weights = weights1
	weights.extend(weights2)
	if len(weights) != size:
		print('wrong weights!')

	# weights = [Geo ** n for n in range(size)]
	# weights.reverse()
	weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
	weighted_G = [G[i] * weights[i] for i in range(size)]
	weighted_death = [death[i] * weights[i] for i in range(size)]
	weighted_D = [D[i] * weights[i] for i in range(size)]
	# weighted_hosp = [hosp[i] * weights[i] for i in range(size)]
	# weighted_IH = [IH[i] * weights[i] for i in range(size)]

	# confirmed_derivative = np.diff(confirmed)
	# G_derivative = np.diff(G)
	# confirmed_derivative = [confirmed_derivative[i] * weights[i] for i in range(size - 1)]
	# G_derivative = [G_derivative[i] * weights[i] for i in range(size - 1)]
	# alpha = 0.5
	# metric00 = r2_score(weighted_confirmed, weighted_G)
	# metric01 = r2_score(confirmed_derivative, G_derivative)
	# metric0 = (alpha * metric00 + (1 - alpha) * metric01)

	weighted_hosp = hosp
	weighted_IH = IH

	metric0 = r2_score(weighted_confirmed, weighted_G)
	metric1 = r2_score(weighted_death, weighted_D)
	metric2 = r2_score(weighted_hosp, weighted_IH)
	# return -(theta * metric0 + 1 * (1 - theta) / 2 * metric1 + 1 * (1 - theta) / 2 * metric2)
	return -(0.99 * metric0 + 0.005 * metric1 + 0.005 * metric2)


def fit_combined(confirmed0, death0, hosp0, days, reopen_day, n_0, metric1, metric2, metric3):
	np.random.seed()
	confirmed = confirmed0.copy()
	death = death0.copy()
	hosp = hosp0.copy()
	size = len(confirmed)
	if metric2 != 0 or metric1 != 0:
		scale1 = pd.Series(np.random.normal(1, metric1, size))
		confirmed = [max(confirmed[i] * scale1[i], 1) for i in range(size)]
		scale2 = pd.Series(np.random.normal(1, metric2, size))
		death = [max(death[i] * scale2[i], 1) for i in range(size)]
		scale3 = pd.Series(np.random.normal(1, metric3, size))
		hosp = [max(hosp[i] * scale3[i], 1) for i in range(size)]
	c_max = 0
	min_loss = 10000
	for c1 in np.arange(c1_range[0], c1_range[1], 0.01):
		# optimal = minimize(loss, [10, 0.05, 0.01, 0.1, 0.1, 0.1, 0.02], args=(c1, confirmed, death, n_0, SIDRG_sd),
		optimal = minimize(loss_combined, [uni(beta_range[0], beta_range[1]),
		                                   uni(gammaE_range[0], gammaE_range[1]),
		                                   uni(alpha_range[0], alpha_range[1]),
		                                   uni(gamma_range[0], gamma_range[1]),
		                                   uni(gamma2_range[0], gamma2_range[1]),
		                                   uni(gamma3_range[0], gamma3_range[1]),
		                                   uni(a1_range[0], a1_range[1]),
		                                   uni(a2_range[0], a2_range[1]),
		                                   uni(a3_range[0], a3_range[1]),
		                                   uni(eta_range[0], eta_range[1]),
		                                   uni(0, 1 / 14),
		                                   0.5,
		                                   uni(k_range[0], k_range[1]),
		                                   uni(k2_range[0], k2_range[1])],
		                   args=(c1, confirmed, death, hosp, n_0, SEIARG, reopen_day), method='L-BFGS-B',
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
		                           (0, 1 / 14),
		                           (0, 5),
		                           k_range,
		                           k2_range])
		current_loss = loss_combined(optimal.x, c1, confirmed, death, hosp, n_0, SEIARG, reopen_day)
		if current_loss < min_loss:
			# print(f'updating loss={current_loss} with c1={c1}')
			min_loss = current_loss
			c_max = c1
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
			h = optimal.x[10]
			Hiding_init = optimal.x[11]
			k = optimal.x[12]
			k2 = optimal.x[13]

	c1 = c_max
	S = [n_0 * eta]
	E = [0]
	I = [confirmed[0]]
	A = [0]
	IH = [hosp[0]]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [Hiding_init * n_0 * eta]
	# Betas = [beta]

	result, [S, E, I, A, IH, IN, D, R, G, H, betas] \
		= simulate_combined(size, SEIARG, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3,
		                    a1, a2, a3, h, k, k2, eta, c1, n_0, reopen_day)

	data1 = [(confirmed[i] - G[i]) / confirmed[i] for i in range(size)]
	data2 = [(death[i] - D[i]) / death[i] for i in range(size)]
	data3 = [(hosp[i] - IH[i]) / hosp[i] for i in range(size)]

	# metric1 = math.sqrt(sum([i ** 2 for i in data1]) / (len(data1) - 8))
	# metric2 = math.sqrt(sum([i ** 2 for i in data2]) / (len(data2) - 8))

	size1 = reopen_day
	size2 = size - size1
	weights1 = [Geo ** n for n in range(size1)]
	weights1.reverse()
	weights2 = [Geo ** n for n in range(size2)]
	weights2.reverse()
	weights = weights1
	weights.extend(weights2)

	# weights = [Geo ** n for n in range(size)]
	# weights.reverse()

	sum_wt = sum(weights)
	metric1 = math.sqrt(sum([data1[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 12) * sum_wt / size)
	                    )
	metric2 = math.sqrt(sum([data2[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 12) * sum_wt / size)
	                    )
	metric3 = math.sqrt(sum([data3[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 12) * sum_wt / size)
	                    )

	r1 = r2_score(confirmed, G)
	r2 = r2_score(hosp, IH)
	r3 = r2_score(death, D)

	return [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2,
	        metric3, r1, r2, r3], min_loss


def fit_init(confirmed0, death0, hosp0, days, reopen_day, n_0, metric1, metric2, metric3):
	np.random.seed()
	confirmed = confirmed0.copy()
	death = death0.copy()
	hosp = hosp0.copy()
	size = len(confirmed)
	if metric2 != 0 or metric1 != 0:
		scale1 = pd.Series(np.random.normal(1, metric1, size))
		confirmed = [max(confirmed[i] * scale1[i], 1) for i in range(size)]
		scale2 = pd.Series(np.random.normal(1, metric2, size))
		death = [max(death[i] * scale2[i], 1) for i in range(size)]
		scale3 = pd.Series(np.random.normal(1, metric3, size))
		hosp = [max(hosp[i] * scale3[i], 1) for i in range(size)]
	c_max = 0
	min_loss = 10000
	for c1 in np.arange(c1_range[0], c1_range[1], 0.01):
		# optimal = minimize(loss, [10, 0.05, 0.01, 0.1, 0.1, 0.1, 0.02], args=(c1, confirmed, death, n_0, SIDRG_sd),
		optimal = minimize(loss_init, [uni(beta_range[0], beta_range[1]),
		                               uni(gamma_range[0], gamma_range[1]),
		                               uni(gamma2_range[0], gamma2_range[1]),
		                               uni(a1_range[0], a1_range[1]),
		                               uni(a2_range[0], a2_range[1]),
		                               uni(a3_range[0], a3_range[1]),
		                               uni(eta_range[0], eta_range[1]),
		                               uni(0, 1 / 14),
		                               0.5,
		                               uni(k_range[0], k_range[1]),
		                               uni(k2_range[0], k2_range[1])],
		                   args=(c1, confirmed, death, hosp, n_0, SIARG, reopen_day), method='L-BFGS-B',
		                   bounds=[beta_range,
		                           gamma_range,
		                           gamma2_range,
		                           a1_range,
		                           a2_range,
		                           a3_range,
		                           eta_range,
		                           (0, 1 / 14),
		                           (0, 5),
		                           k_range,
		                           k2_range])
		current_loss = loss_init(optimal.x, c1, confirmed, death, hosp, n_0, SIARG, reopen_day)
		if current_loss < min_loss:
			# print(f'updating loss={current_loss} with c1={c1}')
			min_loss = current_loss
			c_max = c1
			beta = optimal.x[0]
			gamma = optimal.x[1]
			gamma2 = optimal.x[2]
			a1 = optimal.x[3]
			a2 = optimal.x[4]
			a3 = optimal.x[5]
			eta = optimal.x[6]
			h = optimal.x[7]
			Hiding_init = optimal.x[8]
			k = optimal.x[9]
			k2 = optimal.x[10]

	c1 = c_max
	S = [n_0 * eta]
	I = [confirmed[0]]
	IH = [hosp[0]]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [Hiding_init * n_0 * eta]
	# Betas = [beta]

	result, [S, I, IH, IN, D, R, G, H, betas] \
		= simulate_combined(size, SIARG, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2,
		                    eta, c1, n_0, reopen_day)

	data1 = [(confirmed[i] - G[i]) / confirmed[i] for i in range(size)]
	data2 = [(death[i] - D[i]) / death[i] for i in range(size)]
	data3 = [(hosp[i] - IH[i]) / hosp[i] for i in range(size)]

	# metric1 = math.sqrt(sum([i ** 2 for i in data1]) / (len(data1) - 8))
	# metric2 = math.sqrt(sum([i ** 2 for i in data2]) / (len(data2) - 8))

	size1 = reopen_day
	size2 = size - size1
	weights1 = [Geo ** n for n in range(size1)]
	weights1.reverse()
	weights2 = [Geo ** n for n in range(size2)]
	weights2.reverse()
	weights = weights1
	weights.extend(weights2)

	# weights = [Geo ** n for n in range(size)]
	# weights.reverse()

	sum_wt = sum(weights)
	metric1 = math.sqrt(sum([data1[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 12) * sum_wt / size)
	                    )
	metric2 = math.sqrt(sum([data2[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 12) * sum_wt / size)
	                    )
	metric3 = math.sqrt(sum([data3[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 12) * sum_wt / size)
	                    )

	r1 = r2_score(confirmed, G)
	r2 = r2_score(hosp, IH)
	r3 = r2_score(death, D)

	return [beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2,
	        r3], min_loss


def funcmod(i):
	# return 0.5 * np.log(1 + i)
	# return 1.00 * np.power(i, -0.4)
	return 1


def fit_state_combined(state, ConfirmFile, DeathFile, PopFile, dates):
	t1 = time.perf_counter()

	if not os.path.exists(f'ASYM/combined2W_{end_date}/{state}'):
		os.makedirs(f'ASYM/combined2W_{end_date}/{state}')
	# if not os.path.exists(f'init_only_{end_date}/{state}'):
	# 	os.makedirs(f'init_only_{end_date}/{state}')

	# add the delay in dates
	for i in range(len(dates) - 1):
		date = datetime.datetime.strptime(dates[i], '%Y-%m-%d')
		date += datetime.timedelta(days=delay)
		dates[i] = date.strftime('%Y-%m-%d')
	print(state)
	print(dates)
	print()

	# read population
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
	dens = df[df.iloc[:, 0] == state].iloc[0]['DENS']

	# select confirmed and death data
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	for start_date in confirmed.columns[1:]:
		# if confirmed.iloc[0].loc[start_date] >= I_0 and death.iloc[0].loc[start_date] > 0:
		if confirmed.iloc[0].loc[start_date] >= I_0:
			break
	days = list(confirmed.columns)
	days = days[days.index(start_date):days.index(end_date) + 1]
	# days = days[days.index(start_date):days.index(dates[0]) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[start_date: end_date]
	# confirmed = confirmed.iloc[0].loc[start_date: dates[0]]
	death = death.iloc[0].loc[start_date: end_date]
	# death = death.iloc[0].loc[start_date: dates[0]]
	for i in range(len(death)):
		if death.iloc[i] == 0:
			death.iloc[i] = 0.01
	death = death.tolist()

	# select hospitalization data
	hosp = readHosp(state)
	hosp = hosp[:dates[1]].tolist()
	diff_len = len(days) - len(hosp)
	if diff_len > 0:

		hosp = [0] * diff_len.extend(hosp)
	else:
		hosp = hosp[-len(days):]
	hosp = [0.01 if h == 0 else h for h in hosp]
	reopen_day = days.index(datetime.datetime.strptime(dates[0], '%Y-%m-%d'))

	# fitting
	para = MT_combined(confirmed, death, hosp, n_0, days, reopen_day)
	[S, E, I, A, IH, IN, D, R, G, H, betas] = plot_combined(state, confirmed, death, hosp, days, n_0, reopen_day, para)
	csv_file = f'ASYM/combined2W_{end_date}/{state}/sim.csv'
	# csv_file = f'init_only_{end_date}/{state}/sim.csv'
	save_sim_combined(csv_file, [S, E, I, A, IH, IN, D, R, G, H, betas], days)
	para_file = f'ASYM/combined2W_{end_date}/{state}/para.csv'
	save_para_combined(para_file, [para])
	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds in total for {state}\n')

	return


# fit with SD for initial phase only
def fit_state_init(state, ConfirmFile, DeathFile, PopFile, dates):
	t1 = time.perf_counter()

	# if not os.path.exists(f'JHU/combined2W_{end_date}/{state}'):
	# 	os.makedirs(f'JHU/combined2W_{end_date}/{state}')
	if not os.path.exists(f'ASYM/init_only_{end_date}/{state}'):
		os.makedirs(f'ASYM/init_only_{end_date}/{state}')

	# add the delay in dates
	for i in range(len(dates) - 1):
		date = datetime.datetime.strptime(dates[i], '%Y-%m-%d')
		date += datetime.timedelta(days=delay)
		dates[i] = date.strftime('%Y-%m-%d')
	print(state)
	print(dates)
	print()

	# read population
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
	dens = df[df.iloc[:, 0] == state].iloc[0]['DENS']

	# select confirmed and death data
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	for start_date in confirmed.columns[1:]:
		# if confirmed.iloc[0].loc[start_date] >= I_0 and death.iloc[0].loc[start_date] > 0:
		if confirmed.iloc[0].loc[start_date] >= I_0:
			break
	days = list(confirmed.columns)
	# days = days[days.index(start_date):days.index(end_date) + 1]
	days = days[days.index(start_date):days.index(dates[0]) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	# confirmed = confirmed.iloc[0].loc[start_date: end_date]
	confirmed = confirmed.iloc[0].loc[start_date: dates[0]]
	# death = death.iloc[0].loc[start_date: end_date]
	death = death.iloc[0].loc[start_date: dates[0]]
	for i in range(len(death)):
		if death.iloc[i] == 0:
			death.iloc[i] = 0.01
	death = death.tolist()

	# select hospitalization data
	hosp = readHosp(state)
	hosp = hosp[:dates[1]].tolist()
	diff_len = len(days) - len(hosp)
	if diff_len > 0:

		hosp = [0] * diff_len.extend(hosp)
	else:
		hosp = hosp[-len(days):]
	hosp = [0.01 if h == 0 else h for h in hosp]
	reopen_day = days.index(datetime.datetime.strptime(dates[0], '%Y-%m-%d'))

	# fitting
	para = MT_init(confirmed, death, hosp, n_0, days, reopen_day)
	[S, I, IH, IN, D, R, G, H, betas] = plot_init(state, confirmed, death, hosp, days, n_0, reopen_day, para)
	# csv_file = f'JHU/combined2W_{end_date}/{state}/sim.csv'
	csv_file = f'ASYM/init_only_{end_date}/{state}/sim.csv'
	save_sim_init(csv_file, [S, I, IH, IN, D, R, G, H, betas], days)
	save_para_init(state, [para])
	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds in total for {state}\n')

	return


# plot result
def plot_combined(state, confirmed, death, hosp, days, n_0, reopen_day, para):
	[beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3,
	 r1, r2, r3] = para
	para_label = ['beta', 'gammaE', 'alpha', 'gamma', 'gamma2', 'gamma3', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k',
	              'k2', 'eta', 'c1', 'metric1', 'metric2', 'metric3', 'r1', 'r2', 'r3']
	for i in range(len(para)):
		print(f'{para_label[i]}={para[i]} ', end=' ')
		if i % 4 == 1:
			print()

	S = [n_0 * eta]
	E = [0]
	I = [confirmed[0]]
	A = [0]
	IH = [hosp[0]]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [Hiding_init * n_0 * eta]
	size = len(days)
	result, [S, E, I, A, IH, IN, D, R, G, H, betas] \
		= simulate_combined(size, SEIARG, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3,
		                    a1, a2, a3, h, k, k2, eta, c1, n_0, reopen_day)

	# weights = [Geo ** n for n in range(size)]
	# weights.reverse()
	# weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
	# weighted_G = [G[i] * weights[i] for i in range(size)]
	# weighted_death = [death[i] * weights[i] for i in range(size)]
	# weighted_D = [D[i] * weights[i] for i in range(size)]
	# # weighted_hosp = [hosp[i] * weights[i] for i in range(size)]
	# # weighted_IH = [IH[i] * weights[i] for i in range(size)]
	#
	# confirmed_derivative = np.diff(confirmed)
	# G_derivative = np.diff(G)
	# confirmed_derivative = [confirmed_derivative[i] * weights[i] for i in range(size - 1)]
	# G_derivative = [G_derivative[i] * weights[i] for i in range(size - 1)]
	# alpha = 0.5
	# metric00 = r2_score(weighted_confirmed, weighted_G)
	# metric01 = r2_score(confirmed_derivative, G_derivative)
	# # metric0 = (alpha * metric00 + (1 - alpha) * metric01)
	#
	# weighted_hosp = hosp
	# weighted_IH = IH
	#
	# metric0 = r2_score(weighted_confirmed, weighted_G)
	# metric1 = r2_score(weighted_death, weighted_D)
	# metric2 = r2_score(weighted_hosp, weighted_IH)
	# print(f'\nloss={-(theta * metric0 + 1 * (1 - theta) / 2 * metric1 + 1 * (1 - theta) / 2 * metric2)}')

	fig = plt.figure(figsize=(6, 12))
	# fig.suptitle(state)
	ax = fig.add_subplot(411)
	ax.set_title(state)
	ax2 = fig.add_subplot(412)
	ax3 = fig.add_subplot(413)
	ax4 = fig.add_subplot(414)
	ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax2.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax3.axvline(days[reopen_day], linestyle='dashed', color='tab:grey', label=days[reopen_day].strftime('%Y-%m-%d'))
	ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
	ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
	ax2.plot(days, [i / 1000 for i in hosp], linewidth=5, linestyle=':', label="Hospitalization")
	ax.plot(days, [i / 1000 for i in G], label='G')
	ax2.plot(days, [i / 1000 for i in D], label='D')
	ax2.plot(days, [i / 1000 for i in IH], label='IH')
	ax3.plot(days, betas, label='beta')
	ax.plot(days, [i / 1000 for i in H], label='H')
	diff_G = pd.Series(np.diff(G))
	diff_confirmed = pd.Series(np.diff(confirmed))
	ax4.plot(days[-len(diff_confirmed):], [i / 1000 for i in diff_confirmed], label='daily new cases')
	ax4.plot(days[-len(diff_G):], [i / 1000 for i in diff_G], label='dG')
	ax.legend()
	ax2.legend()
	ax3.legend()
	ax4.legend()
	fig.autofmt_xdate()
	fig.savefig(f'ASYM/combined2W_{end_date}/{state}/sim.png', bbox_inches="tight")
	# fig.savefig(f'init_only_{end_date}/{state}/sim.png', bbox_inches="tight")
	plt.close(fig)
	return [S, E, I, A, IH, IN, D, R, G, H, betas]


# plot result for initial phase only
def plot_init(state, confirmed, death, hosp, days, n_0, reopen_day, para):
	[beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3] = para
	para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1', 'metric1',
	              'metric2', 'metric3', 'r1', 'r2', 'r3']
	for i in range(len(para)):
		print(f'{para_label[i]}={para[i]} ', end=' ')
		if i % 4 == 1:
			print()

	S = [n_0 * eta]
	I = [confirmed[0]]
	IH = [hosp[0]]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [Hiding_init * n_0 * eta]
	size = len(days)
	result, [S, I, IH, IN, D, R, G, H, betas] \
		= simulate_combined(size, SIARG, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2,
		                    eta, c1, n_0, reopen_day)

	# weights = [Geo ** n for n in range(size)]
	# weights.reverse()
	# weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
	# weighted_G = [G[i] * weights[i] for i in range(size)]
	# weighted_death = [death[i] * weights[i] for i in range(size)]
	# weighted_D = [D[i] * weights[i] for i in range(size)]
	# # weighted_hosp = [hosp[i] * weights[i] for i in range(size)]
	# # weighted_IH = [IH[i] * weights[i] for i in range(size)]
	#
	# confirmed_derivative = np.diff(confirmed)
	# G_derivative = np.diff(G)
	# confirmed_derivative = [confirmed_derivative[i] * weights[i] for i in range(size - 1)]
	# G_derivative = [G_derivative[i] * weights[i] for i in range(size - 1)]
	# alpha = 0.5
	# metric00 = r2_score(weighted_confirmed, weighted_G)
	# metric01 = r2_score(confirmed_derivative, G_derivative)
	# # metric0 = (alpha * metric00 + (1 - alpha) * metric01)
	#
	# weighted_hosp = hosp
	# weighted_IH = IH
	#
	# metric0 = r2_score(weighted_confirmed, weighted_G)
	# metric1 = r2_score(weighted_death, weighted_D)
	# metric2 = r2_score(weighted_hosp, weighted_IH)
	# print(f'\nloss={-(theta * metric0 + 1 * (1 - theta) / 2 * metric1 + 1 * (1 - theta) / 2 * metric2)}')

	fig = plt.figure(figsize=(6, 10))
	# fig.suptitle(state)
	ax = fig.add_subplot(311)
	ax.set_title(state)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)
	ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax2.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax3.axvline(days[reopen_day], linestyle='dashed', color='tab:grey', label=days[reopen_day].strftime('%Y-%m-%d'))
	ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
	ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
	ax2.plot(days, [i / 1000 for i in hosp], linewidth=5, linestyle=':', label="Hospitalization")
	ax.plot(days, [i / 1000 for i in G], label='G')
	ax2.plot(days, [i / 1000 for i in D], label='D')
	ax2.plot(days, [i / 1000 for i in IH], label='IH')
	ax3.plot(days, betas, label='beta')
	ax.plot(days, [i / 1000 for i in H], label='H')
	ax.legend()
	ax2.legend()
	ax3.legend()
	fig.autofmt_xdate()
	fig.savefig(f'ASYM/init_only_{end_date}/{state}/sim.png', bbox_inches="tight")
	# fig.savefig(f'init_only_{end_date}/{state}/sim.png', bbox_inches="tight")
	plt.close(fig)
	return [S, I, IH, IN, D, R, G, H, betas]


# read hospitalization data
def readHosp(state):
	df = pd.read_csv(f'JHU/hosp/{state}/Top_Counties_by_Cases_Full_Data_data.csv', usecols=['Date', 'Hospitalization'])
	df = df[df['Hospitalization'].notna()]

	dates = [datetime.datetime.strptime(date, '%m/%d/%Y') for date in df['Date']]
	dates = [date.strftime('%Y-%m-%d') for date in dates]

	df['Date'] = dates
	df = df.sort_values('Date')
	df = df.drop_duplicates()
	df = df.T
	df.columns = df.iloc[0]
	df = df.iloc[1]
	return df


# combined fitting
def MT_combined(confirmed, death, hosp, n_0, days, reopen_day):
	para_best = []
	min_loss = 10000
	with concurrent.futures.ProcessPoolExecutor() as executor:
		t1 = time.perf_counter()
		results = [executor.submit(fit_combined, confirmed, death, hosp, days, reopen_day, n_0, 0, 0, 0) for _ in
		           range(num_threads)]

		threads = 0
		for f in concurrent.futures.as_completed(results):
			para, current_loss = f.result()
			threads += 1
			print(f'thread {threads} returned')
			if current_loss < min_loss:
				min_loss = current_loss
				para_best = para
				print(f'best paras updated at {threads} with loss={min_loss}')
		# if threads % 10 == 0:
		# 	print(f'{threads}/{num_threads} thread(s) completed')

		t2 = time.perf_counter()
		print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_threads, 3)} seconds per job')

	print('initial best fitting completed\n')
	return para_best


# combined fitting for initial phase only
def MT_init(confirmed, death, hosp, n_0, days, reopen_day):
	para_best = []
	min_loss = 10000
	with concurrent.futures.ProcessPoolExecutor() as executor:
		t1 = time.perf_counter()
		results = [executor.submit(fit_init, confirmed, death, hosp, days, reopen_day, n_0, 0, 0, 0) for _ in
		           range(num_threads)]

		threads = 0
		for f in concurrent.futures.as_completed(results):
			para, current_loss = f.result()
			threads += 1
			print(f'thread {threads} returned')
			if current_loss < min_loss:
				min_loss = current_loss
				para_best = para
				print(f'best paras updated at {threads} with loss={min_loss}')
		# if threads % 10 == 0:
		# 	print(f'{threads}/{num_threads} thread(s) completed')

		t2 = time.perf_counter()
		print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_threads, 3)} seconds per job')

	print('initial best fitting completed\n')
	return para_best


# multi thread fit with deviations in input series
def CI_combined(confirmed, death, hosp, n_0, days, reopen_day, metric1, metric2, metric3):
	paras = []
	with concurrent.futures.ProcessPoolExecutor() as executor:
		t1 = time.perf_counter()
		results = [
			executor.submit(fit_combined, confirmed, death, hosp, days, reopen_day, n_0, metric1, metric2, metric3) for
			_ in range(num_CI)]

		threads = 0
		for f in concurrent.futures.as_completed(results):
			para, current_loss = f.result()
			threads += 1
			print(f'thread {threads} returned')
			paras.append(para)

		t2 = time.perf_counter()
		print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_CI, 3)} seconds per job')

	print('initial best fitting completed\n')
	return paras


def fit_all_combined():
	t1 = time.perf_counter()

	state = 'NY-New York'
	dates = ['2020-06-22', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'CA-Los Angeles'
	dates = ['2020-06-12', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'FL-Miami-Dade'
	dates = ['2020-06-03', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'IL-Cook'
	dates = ['2020-06-03', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'TX-Dallas'
	dates = ['2020-05-22', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'TX-Harris'
	dates = ['2020-06-03', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'AZ-Maricopa'
	dates = ['2020-05-28', end_date]
	# dates = ['2020-06-22', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'GA-Fulton'
	dates = ['2020-06-12', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'NJ-Bergen'
	dates = ['2020-06-22', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'PA-Philadelphia'
	dates = ['2020-06-05', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'MD-Prince George\'s'
	dates = ['2020-06-29', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'NV-Clark'
	dates = ['2020-05-29', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'NC-Mecklenburg'
	dates = ['2020-05-22', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'LA-Jefferson'
	dates = ['2020-06-05', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'CA-Riverside'
	dates = ['2020-06-12', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'FL-Broward'
	dates = ['2020-06-12', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'NJ-Hudson'
	dates = ['2020-06-22', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'MA-Middlesex'
	dates = ['2020-06-22', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'OH-Franklin'
	dates = ['2020-05-21', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'VA-Fairfax'
	dates = ['2020-06-12', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'TN-Shelby'
	dates = ['2020-06-15', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'WI-Milwaukee'
	dates = ['2020-07-01', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'UT-Salt Lake'
	dates = ['2020-05-15', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	state = 'MN-Hennepin'
	dates = ['2020-06-04', end_date]
	fit_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                   'JHU/CountyPopulation.csv', dates)

	t2 = time.perf_counter()
	print(f'{round((t2 - t1) / 60, 3)} minutes for all counties')

	return


# fit all state with SD for initial phase only
def fit_all_init():
	t1 = time.perf_counter()

	state = 'NY-New York'
	dates = ['2020-06-22', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'CA-Los Angeles'
	dates = ['2020-06-12', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'FL-Miami-Dade'
	dates = ['2020-06-03', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'IL-Cook'
	dates = ['2020-06-03', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'TX-Dallas'
	dates = ['2020-05-22', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'TX-Harris'
	dates = ['2020-06-03', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'AZ-Maricopa'
	dates = ['2020-05-28', end_date]
	# dates = ['2020-06-22', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'GA-Fulton'
	dates = ['2020-06-12', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'NJ-Bergen'
	dates = ['2020-06-22', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'PA-Philadelphia'
	dates = ['2020-06-05', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'MD-Prince George\'s'
	dates = ['2020-06-29', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'NV-Clark'
	dates = ['2020-05-29', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'NC-Mecklenburg'
	dates = ['2020-05-22', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'LA-Jefferson'
	dates = ['2020-06-05', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'CA-Riverside'
	dates = ['2020-06-12', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'FL-Broward'
	dates = ['2020-06-12', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'NJ-Hudson'
	dates = ['2020-06-22', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'MA-Middlesex'
	dates = ['2020-06-22', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'OH-Franklin'
	dates = ['2020-05-21', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'VA-Fairfax'
	dates = ['2020-06-12', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'TN-Shelby'
	dates = ['2020-06-15', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'WI-Milwaukee'
	dates = ['2020-07-01', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'UT-Salt Lake'
	dates = ['2020-05-15', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	state = 'MN-Hennepin'
	dates = ['2020-06-04', end_date]
	fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	               dates)

	t2 = time.perf_counter()
	print(f'{round((t2 - t1) / 60, 3)} minutes for all counties')

	return


# add deviations to input series and fit to generate new estimates for all parameters
def CI_state_combined(state, ConfirmFile, DeathFile, PopFile, dates):
	SimFile = f'ASYM/combined2W_{end_date}/{state}/sim.csv'
	ParaFile = f'ASYM/combined2W_{end_date}/{state}/para.csv'
	t1 = time.perf_counter()

	# if not os.path.exists(f'combined2W_{end_date}/{state}'):
	# 	os.makedirs(f'combined2W_{end_date}/{state}')
	if not os.path.exists(f'ASYM/CI_{end_date}/{state}'):
		os.makedirs(f'ASYM/CI_{end_date}/{state}')

	# add the delay in dates
	for i in range(len(dates) - 1):
		date = datetime.datetime.strptime(dates[i], '%Y-%m-%d')
		date += datetime.timedelta(days=delay)
		dates[i] = date.strftime('%Y-%m-%d')
	print(state)
	print(dates)
	print()

	# read population
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
	dens = df[df.iloc[:, 0] == state].iloc[0]['DENS']

	# select useful portion of simulation
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	for start_date in confirmed.columns[1:]:
		if confirmed.iloc[0].loc[start_date] >= I_0 and death.iloc[0].loc[start_date] > 0:
			break
	confirmed = confirmed.iloc[0].loc[start_date:end_date]
	death = death.iloc[0].loc[start_date:end_date]
	hosp = readHosp(state)
	hosp = hosp[:end_date]
	for start_date2 in hosp.index:
		if hosp[start_date2] > 0:
			break
	if start_date2 > start_date:
		start_date = start_date2
	confirmed = confirmed[start_date:]
	death = death[start_date:]
	hosp = hosp[start_date:]
	days = confirmed.index.tolist()

	df_sim = pd.read_csv(SimFile)
	G = df_sim[df_sim['series'] == 'G'].iloc[0].loc[start_date:]
	D = df_sim[df_sim['series'] == 'D'].iloc[0].loc[start_date:]
	IH = df_sim[df_sim['series'] == 'IH'].iloc[0].loc[start_date:]

	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	reopen_day = days.index(datetime.datetime.strptime(dates[0], '%Y-%m-%d'))

	# calculate weighted deviations
	size = len(G)
	size1 = reopen_day
	size2 = size - size1
	weights1 = [Geo ** n for n in range(size1)]
	weights1.reverse()
	weights2 = [Geo ** n for n in range(size2)]
	weights2.reverse()
	weights = weights1
	weights.extend(weights2)
	data1 = [(confirmed[i] - G[i]) / confirmed[i] for i in range(size)]
	data2 = [(death[i] - D[i]) / death[i] for i in range(size)]
	data3 = [(hosp[i] - IH[i]) / hosp[i] for i in range(size)]
	sum_wt = sum(weights)
	metric1 = math.sqrt(sum([data1[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 12) * sum_wt / size)
	                    )
	metric2 = math.sqrt(sum([data2[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 12) * sum_wt / size)
	                    )
	metric3 = math.sqrt(sum([data3[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 12) * sum_wt / size)
	                    )

	# fitting
	paras = CI_combined(G, D, IH, n_0, days, reopen_day, metric1, metric2, metric3)
	para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1', 'metric1',
	              'metric2', 'metric3', 'r1', 'r2', 'r3']
	out_df = pd.DataFrame(paras, columns=para_label)
	out_df.to_csv(f'ASYM/CI_{end_date}/{state}/paraCI.csv', index=False)
	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds in total for {state}\n')

	return 0


# add deviations to input series and fit to generate new estimates for all parameters for all states
def CI_all_combined():
	t1 = time.perf_counter()

	state = 'NY-New York'
	dates = ['2020-06-22', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'CA-Los Angeles'
	dates = ['2020-06-12', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'FL-Miami-Dade'
	dates = ['2020-06-03', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'IL-Cook'
	dates = ['2020-06-03', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'TX-Dallas'
	dates = ['2020-05-22', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'TX-Harris'
	dates = ['2020-06-03', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'AZ-Maricopa'
	dates = ['2020-05-28', end_date]
	# dates = ['2020-06-22', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'GA-Fulton'
	dates = ['2020-06-12', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'NJ-Bergen'
	dates = ['2020-06-22', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'PA-Philadelphia'
	dates = ['2020-06-05', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'MD-Prince George\'s'
	dates = ['2020-06-29', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'NV-Clark'
	dates = ['2020-05-29', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'NC-Mecklenburg'
	dates = ['2020-05-22', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'LA-Jefferson'
	dates = ['2020-06-05', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'CA-Riverside'
	dates = ['2020-06-12', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'FL-Broward'
	dates = ['2020-06-12', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'NJ-Hudson'
	dates = ['2020-06-22', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'MA-Middlesex'
	dates = ['2020-06-22', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'OH-Franklin'
	dates = ['2020-05-21', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'VA-Fairfax'
	dates = ['2020-06-12', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'TN-Shelby'
	dates = ['2020-06-15', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'WI-Milwaukee'
	dates = ['2020-07-01', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'UT-Salt Lake'
	dates = ['2020-05-15', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	state = 'MN-Hennepin'
	dates = ['2020-06-04', end_date]
	CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv',
	                  dates)

	t2 = time.perf_counter()
	print(f'{round((t2 - t1) / 60, 3)} minutes for all counties')


def test():
	r1 = [1, 2, 3]
	r2 = [4, 5, 6]
	col = ['c2', 'c3', 'c4']
	c1 = [0, 5]
	df = pd.DataFrame([r1, r2], columns=col)
	print(df)
	df = df.iloc[0].loc[:'c3']
	# df.insert(0, 'c1', c1)
	print(df)
	df.to_csv('index.csv', index=True)
	df.to_csv('no_index.csv', index=False)


def hist(state):
	fig = plt.figure(figsize=(10, 8))
	para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1']
	df = pd.read_csv(f'ASYM/CI_{end_date}/{state}/paraCI.csv')

	for i in range(len(para_label)):
		ax = fig.add_subplot(3, 4, i + 1)
		data = df[para_label[i]]
		ax.hist(data, bins='auto', alpha=0.3, histtype='stepfilled')
		ax.set_title(para_label[i])
	fig.subplots_adjust(hspace=0.3, wspace=0.3)
	fig.suptitle(state)
	fig.savefig(f'ASYM/CI_{end_date}/{state}/hist.png', bbox_inches='tight')
	# plt.show()
	plt.close(fig)


# parameter histogram for all states
def hist_all():
	state = 'NY-New York'
	hist(state)

	state = 'CA-Los Angeles'
	hist(state)

	state = 'FL-Miami-Dade'
	hist(state)

	state = 'IL-Cook'
	hist(state)

	state = 'TX-Dallas'
	hist(state)

	state = 'TX-Harris'
	hist(state)

	state = 'AZ-Maricopa'
	hist(state)

	state = 'GA-Fulton'
	hist(state)

	state = 'NJ-Bergen'
	hist(state)

	state = 'PA-Philadelphia'
	hist(state)

	state = 'MD-Prince George\'s'
	hist(state)

	state = 'NV-Clark'
	hist(state)

	state = 'NC-Mecklenburg'
	hist(state)

	state = 'LA-Jefferson'
	hist(state)

	state = 'CA-Riverside'
	hist(state)

	state = 'FL-Broward'
	hist(state)

	state = 'NJ-Hudson'
	hist(state)

	state = 'MA-Middlesex'
	hist(state)

	state = 'OH-Franklin'
	hist(state)

	state = 'VA-Fairfax'
	hist(state)

	state = 'TN-Shelby'
	hist(state)

	state = 'WI-Milwaukee'
	hist(state)

	state = 'UT-Salt Lake'
	hist(state)

	state = 'MN-Hennepin'
	hist(state)


# generate confidence interval of G,D
def interval(state, reopen_date, ax):
	print('interval', state)
	PopFile = 'JHU/CountyPopulation.csv'
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

	SimFile = f'ASYM/combined2W_{end_date}/{state}/sim.csv'
	df = pd.read_csv(SimFile)
	S0 = df[df['series'] == 'S'].iloc[0].iloc[1:]
	I0 = df[df['series'] == 'I'].iloc[0].iloc[1:]
	IH0 = df[df['series'] == 'IH'].iloc[0].iloc[1:]
	IN0 = df[df['series'] == 'IN'].iloc[0].iloc[1:]
	R0 = df[df['series'] == 'R'].iloc[0].iloc[1:]
	D0 = df[df['series'] == 'D'].iloc[0].iloc[1:]
	G0 = df[df['series'] == 'G'].iloc[0].iloc[1:]
	H0 = df[df['series'] == 'H'].iloc[0].iloc[1:]
	days = S0.index.to_list()
	size = len(S0)

	ParaFile = f'ASYM/combined2W_{end_date}/{state}/para.csv'
	df = pd.read_csv(ParaFile)
	beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3 = df.iloc[0]
	# print(beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3)

	date = datetime.datetime.strptime(reopen_date, '%Y-%m-%d')
	date += datetime.timedelta(days=delay)
	reopen_date = date.strftime('%Y-%m-%d')

	reopen_day = days.index(reopen_date)

	CIFile = f'ASYM/CI_{end_date}/{state}/paraCI.csv'
	df = pd.read_csv(CIFile)
	Gs = [G0]
	Ds = [D0]
	df_Gs = pd.DataFrame(columns=days)
	df_Ds = pd.DataFrame(columns=days)

	for i in range(len(df)):
		beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3 = \
			df.iloc[i]
		S2 = [n_0 * eta]
		I2 = [I0[0]]
		IH2 = [IH0[0]]
		IN2 = [IN0[0]]
		D2 = [D0[0]]
		R2 = [R0[0]]
		G2 = [G0[0]]
		H2 = [Hiding_init * eta * n_0]
		result, [S, I, IH, IN, D, R, G, H, betas] \
			= simulate_combined(size, SIARG, S2, I2, IH2, IN2, D2, R2, G2, H2, beta, gamma, gamma2, a1, a2, a3,
			                    h, k, k2, eta, c1, n_0, reopen_day)
		Gs.append(G)
		Ds.append(D)

	G_max = []
	G_min = []
	D_max = []
	D_min = []

	for i in range(len(days)):
		G_current = [G[i] for G in Gs]
		G_current.sort()
		D_current = [D[i] for D in Ds]
		D_current.sort()
		size = len(G_current)
		G_max.append(G_current[max(round(size * 0.975) - 1, 0)])
		G_min.append(G_current[max(round(size * 0.025) - 1, 0)])
		D_max.append(D_current[max(round(size * 0.975) - 1, 0)])
		D_min.append(D_current[max(round(size * 0.025) - 1, 0)])
	# G_max.append(max([G[i] for G in Gs]))
	# G_min.append(min([G[i] for G in Gs]))
	# D_max.append(max([D[i] for D in Ds]))
	# D_min.append(min([D[i] for D in Ds]))

	col = ['series']
	col.extend(days)
	df = pd.DataFrame(columns=col)

	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	# ax.plot(days, [i / 1000 for i in G0], label='G')
	# ax.plot(days, [i / 1000 for i in G_max], label='G_max')
	# ax.plot(days, [i / 1000 for i in G_min], label='G_min')
	ax.fill_between(days, [i / 1000 for i in G_min], [i / 1000 for i in G_max], alpha=0.3, label='G_CI')
	ax.legend()
	ax.set_title(state)

	G_max.insert(0, 'G_high')
	G_min.insert(0, 'G_low')
	D_max.insert(0, 'D_high')
	D_min.insert(0, 'D_low')
	df.loc[len(df)] = G_max
	df.loc[len(df)] = G_min
	df.loc[len(df)] = D_max
	df.loc[len(df)] = D_min
	df.to_csv(f'ASYM/CI_{end_date}/{state}/GD_high_low.csv', index=False)
	return


# generate confidence interval of G,D for all states
def interval_all():
	fig = plt.figure(figsize=(10, 18))
	i = 0

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'NY-New York'
	reopen_date = '2020-06-22'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'CA-Los Angeles'
	reopen_date = '2020-06-12'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'FL-Miami-Dade'
	reopen_date = '2020-06-03'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'IL-Cook'
	reopen_date = '2020-06-03'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'TX-Dallas'
	reopen_date = '2020-05-22'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'TX-Harris'
	reopen_date = '2020-06-03'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'AZ-Maricopa'
	reopen_date = '2020-05-28'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'GA-Fulton'
	reopen_date = '2020-06-12'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'NJ-Bergen'
	reopen_date = '2020-06-22'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'PA-Philadelphia'
	reopen_date = '2020-06-05'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'MD-Prince George\'s'
	reopen_date = '2020-06-29'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'NV-Clark'
	reopen_date = '2020-05-29'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'NC-Mecklenburg'
	reopen_date = '2020-05-22'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'LA-Jefferson'
	reopen_date = '2020-06-05'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'CA-Riverside'
	reopen_date = '2020-06-12'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'FL-Broward'
	reopen_date = '2020-06-12'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'NJ-Hudson'
	reopen_date = '2020-06-22'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'MA-Middlesex'
	reopen_date = '2020-06-22'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'OH-Franklin'
	reopen_date = '2020-05-21'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'VA-Fairfax'
	reopen_date = '2020-06-12'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'TN-Shelby'
	reopen_date = '2020-06-15'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'WI-Milwaukee'
	reopen_date = '2020-07-01'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'UT-Salt Lake'
	reopen_date = '2020-05-15'
	interval(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(6, 4, i)
	state = 'MN-Hennepin'
	reopen_date = '2020-06-04'
	interval(state, reopen_date, ax)

	fig.autofmt_xdate()
	# plt.show()
	plt.close(fig)


def CI_para(state):
	df = pd.read_csv(f'ASYM/combined2W_{end_date}/{state}/para.csv')
	beta = df['beta'].iloc[0]
	gamma = df['gamma'].iloc[0]
	gamma2 = df['gamma2'].iloc[0]
	a2 = df['a2'].iloc[0]

	df2 = pd.read_csv(f'ASYM/CI_{end_date}/{state}/paraCI.csv')
	betas = df2['beta'].tolist()
	gammas = df2['gamma'].tolist()
	gamma2s = df2['gamma2'].tolist()
	a2s = df2['a2'].tolist()
	betas.sort()
	gammas.sort()
	gamma2s.sort()
	a2s.sort()

	size = len(betas)
	row = [state]
	row.append(
		f'{round(beta, 3)}, ({round(betas[max(round(size * 0.025) - 1, 0)], 3)}, {round(betas[max(round(size * 0.975) - 1, 1)], 3)})')
	row.append(
		f'{round(gamma, 3)}, ({round(gammas[max(round(size * 0.025) - 1, 0)], 3)}, {round(gammas[max(round(size * 0.975) - 1, 1)], 3)})')
	row.append(
		f'{round(gamma2, 3)}, ({round(gamma2s[max(round(size * 0.025) - 1, 0)], 3)}, {round(gamma2s[max(round(size * 0.975) - 1, 1)], 3)})')
	row.append(
		f'{round(a2, 3)}, ({round(a2s[max(round(size * 0.025) - 1, 0)], 3)}, {round(a2s[max(round(size * 0.975) - 1, 1)], 3)})')
	return row


# confidence interval for all parameters for all states
def CI_para_all():
	states = ['AZ-Maricopa', 'CA-Los Angeles', 'CA-Riverside', 'FL-Broward', 'FL-Miami-Dade',
	          'GA-Fulton', 'IL-Cook', 'LA-Jefferson', 'MA-Middlesex', 'MD-Prince George\'s', 'MN-Hennepin',
	          'NC-Mecklenburg', 'NJ-Bergen', 'NJ-Hudson', 'NV-Clark', 'NY-New York', 'OH-Franklin', 'PA-Philadelphia',
	          'TN-Shelby', 'TX-Dallas', 'TX-Harris', 'UT-Salt Lake', 'VA-Fairfax', 'WI-Milwaukee']

	# states = ['TX-Harris']

	columns = ['State - County', 'beta', 'gamma', 'gamma2', 'a2']
	table = []
	for state in states:
		table.append(CI_para(state))

	out_df = pd.DataFrame(table, columns=columns)
	out_df.to_csv(f'ASYM/CI_{end_date}/para_table.csv', index=False)
	# print(table)
	return


def main():
	# fit_all_init()

	fit_all_combined()
	# CI_all_combined()

	# hist_all()

	# interval_all()
	# CI_para_all()

	# test()
	return 0


if __name__ == '__main__':
	main()
