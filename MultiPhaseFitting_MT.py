import numpy as np
import pandas as pd
import time
import math
import concurrent.futures
import multiprocessing
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score, mean_squared_error
from SIRfunctions import SIRG_sd, SIRG, computeBeta, SIDRG_sd
import datetime
from numpy.random import uniform as uni
import os
import warnings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

np.set_printoptions(threshold=sys.maxsize)
Geo = 0.98

num_threads = 5
num_threads_dist = 5
num_threads_reopen = 100
num_threads_dist_reopen = 5

# weight of G in initial fitting
theta = 0.7
# weight of G in release fitting
theta2 = 0.7

I_0 = 50
beta_range = (0.1, 100)
gamma_range = (0.04, 0.08)
sigma_range = (0.001, 1)
a1_range = (0.01, 0.5)
a2_range = (0.01, 0.5)
a3_range = (0.01, 0.5)
eta_range = (0.001, 0.05)
c1_fixed = (0.9, 0.9)
c1_range = (0, 0.98)
h_range = (0, 10)
k_range = (0.1, 1)
# end_date = '2020-08-01'
# end_date = '2020-09-23'
end_date = '2020-09-22'
p_m = 1
# Hiding = 0.33
delay = 7
change_eta2 = False

fig_row = 5
fig_col = 3


# loss function for traditional SIRG
def loss_SIRG(point, confirmed, n_0, SIRG):
	size = len(confirmed)
	beta = point[0]
	gamma = point[1]
	eta = point[2]
	# c1 = point[3]
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
		if S[-1] < 0:
			return 1000000

	weights = [Geo ** n for n in range(size)]
	weights.reverse()
	weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
	weighted_G = [G[i] * weights[i] for i in range(size)]

	metric0 = r2_score(weighted_confirmed, weighted_G)
	return - metric0


# initial fitting
def loss1(point, c1, confirmed, death, hosp, n_0, SIRG):
	size = len(confirmed)
	beta = point[0]
	gamma = point[1]
	sigma = point[2]
	a1 = point[3]
	a2 = point[4]
	a3 = point[5]
	eta = point[6]
	S = [n_0 * eta]
	I = [confirmed[0]]
	IH = [I[-1] * sigma]
	IN = [I[-1] * (1 - sigma)]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SIRG(i, [S[i - 1], I[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1], beta, gamma, sigma, a1,
		                 a2, a3, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		IH.append(IH[-1] + delta[2])
		IN.append(IN[-1] + delta[3])
		D.append(D[-1] + delta[4])
		R.append(R[-1] + delta[5])
		G.append(G[-1] + delta[6])
		if S[-1] < 0:
			return 1000000

	weights = [Geo ** n for n in range(size)]
	weights.reverse()
	weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
	weighted_G = [G[i] * weights[i] for i in range(size)]
	weighted_death = [death[i] * weights[i] for i in range(size)]
	weighted_D = [D[i] * weights[i] for i in range(size)]
	weighted_hosp = [hosp[i] * weights[i] for i in range(size)]
	weighted_IH = [IH[i] * weights[i] for i in range(size)]

	metric0 = r2_score(weighted_confirmed, weighted_G)
	metric1 = r2_score(weighted_death, weighted_D)
	metric2 = r2_score(weighted_hosp, weighted_IH)
	return -(theta * metric0 + (1 - theta) / 2 * metric1 + (1 - theta) / 2 * metric2)
	# return -(theta * metric0 + (1 - theta) * metric1)


# one release phase with h and Hiding and drop in hospitalization
def loss2(point, para_init, confirmed, death, hosp, start_point, n_0, SIRG):
	h = point[0]
	Hiding_init = point[1]
	k = point[2]
	[beta, gamma, sigma, a1, a2, a3, eta, c1, metric1, metric2, metric3] = para_init
	size = len(confirmed)
	[S0, I0, IH0, IN0, D0, R0, G0] = start_point
	S = [S0]
	I = [I0]
	IH = [IH0]
	IN = [IN0]
	D = [D0]
	R = [R0]
	G = [G0]
	H = [Hiding_init * eta * n_0]
	eta2 = eta
	for i in range(1, size):
		# if i < 40:
		release = min(H[-1], h * funcmod(i))
		S[-1] += release
		H[-1] -= release
		if change_eta2:
			eta2 += release / n_0
		delta = SIRG(i, [S[i - 1], I[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1], beta, gamma, k * sigma,
		                 a1, a2, a3, eta2, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		IH.append(IH[-1] + delta[2])
		IN.append(IN[-1] + delta[3])
		D.append(D[-1] + delta[4])
		R.append(R[-1] + delta[5])
		G.append(G[-1] + delta[6])
		H.append(H[-1])
		if S[-1] < 0:
			return 1000000

	weights = [Geo ** n for n in range(size)]
	weights.reverse()
	weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
	weighted_G = [G[i] * weights[i] for i in range(size)]
	weighted_death = [death[i] * weights[i] for i in range(size)]
	weighted_D = [D[i] * weights[i] for i in range(size)]
	weighted_hosp = [hosp[i] * weights[i] for i in range(size)]
	weighted_IH = [IH[i] * weights[i] for i in range(size)]

	metric0 = r2_score(weighted_confirmed, weighted_G)
	metric1 = r2_score(weighted_death, weighted_D)
	metric2 = r2_score(weighted_hosp, weighted_IH)
	return -(theta2 * metric0 + (1 - theta2) / 2 * metric1 + (1 - theta2) / 2 * metric2) + Hiding_init / 1000000
	# return -(theta2 * metric0 + (1 - theta2) * metric1) + Hiding_init / 1000000


def fit_SIRG(confirmed0, n_0):
	np.random.seed()
	confirmed = confirmed0.copy()
	size = len(confirmed)
	optimal = minimize(loss_SIRG, [uni(beta_range[0], beta_range[1]),
	                               uni(gamma_range[0], gamma_range[1]),
	                               uni(eta_range[0], eta_range[1])],
	                   args=(confirmed, n_0, SIRG), method='L-BFGS-B', bounds=[beta_range, gamma_range, eta_range])
	current_loss = loss_SIRG(optimal.x, confirmed, n_0, SIRG)
	beta = optimal.x[0]
	gamma = optimal.x[1]
	eta = optimal.x[2]
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])

	data1 = [(confirmed[i] - G[i]) / confirmed[i] for i in range(size)]

	weights = [Geo ** n for n in range(size)]
	weights.reverse()
	sum_wt = sum(weights)
	metric1 = math.sqrt(sum([data1[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 3) * sum_wt / size)
	                    )

	return [beta, gamma, eta, metric1], current_loss


def fit_init(confirmed0, death0, hosp0, n_0, metric1, metric2, metric3):
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
	for c1 in np.arange(0.9, 1, 0.01):
		# optimal = minimize(loss, [10, 0.05, 0.01, 0.1, 0.1, 0.1, 0.02], args=(c1, confirmed, death, n_0, SIDRG_sd),
		optimal = minimize(loss1, [uni(beta_range[0], beta_range[1]),
		                           uni(gamma_range[0], gamma_range[1]),
		                           uni(sigma_range[0], sigma_range[1]),
		                           uni(a1_range[0], a1_range[1]),
		                           uni(a2_range[0], a2_range[1]),
		                           uni(a3_range[0], a3_range[1]),
		                           uni(eta_range[0], eta_range[1])],
		                   args=(c1, confirmed, death, hosp, n_0, SIDRG_sd), method='L-BFGS-B',
		                   bounds=[beta_range, gamma_range, sigma_range, a1_range, a2_range, a3_range, eta_range])
		current_loss = loss1(optimal.x, c1, confirmed, death, hosp, n_0, SIDRG_sd)
		if current_loss < min_loss:
			min_loss = current_loss
			c_max = c1
			beta = optimal.x[0]
			gamma = optimal.x[1]
			sigma = optimal.x[2]
			a1 = optimal.x[3]
			a2 = optimal.x[4]
			a3 = optimal.x[5]
			eta = optimal.x[6]

	c1 = c_max
	S = [n_0 * eta]
	I = [confirmed[0]]
	IH = [I[-1] * sigma]
	IN = [I[-1] * (1 - sigma)]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	# H = [Hiding * n_0 * eta]
	Betas = [beta]
	for i in range(1, size):
		delta = SIDRG_sd(i, [S[i - 1], I[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1], beta, gamma, sigma,
		                     a1, a2, a3, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		IH.append(IH[-1] + delta[2])
		IN.append(IN[-1] + delta[3])
		D.append(D[-1] + delta[4])
		R.append(R[-1] + delta[5])
		G.append(G[-1] + delta[6])
		# H.append(H[-1])
		Betas.append(delta[7])

	data1 = [(confirmed[i] - G[i]) / confirmed[i] for i in range(size)]
	data2 = [(death[i] - D[i]) / death[i] for i in range(size)]
	data3 = [(hosp[i] - IH[i]) / hosp[i] for i in range(size)]

	# metric1 = math.sqrt(sum([i ** 2 for i in data1]) / (len(data1) - 8))
	# metric2 = math.sqrt(sum([i ** 2 for i in data2]) / (len(data2) - 8))
	weights = [Geo ** n for n in range(size)]
	weights.reverse()
	sum_wt = sum(weights)
	metric1 = math.sqrt(sum([data1[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 8) * sum_wt / size)
	                    )
	metric2 = math.sqrt(sum([data2[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 8) * sum_wt / size)
	                    )
	metric3 = math.sqrt(sum([data3[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 8) * sum_wt / size)
	                    )

	return [beta, gamma, sigma, a1, a2, a3, eta, c1, metric1, metric2, metric3], min_loss


def fit_reopen(confirmed0, death0, hosp0, n_0, para_init, start_point, metric1, metric2, metric3):
	np.random.seed()
	confirmed = confirmed0.copy()
	death = death0.copy()
	hosp = hosp0.copy()
	size = len(confirmed)
	[beta, gamma, sigma, a1, a2, a3, eta, c1, m1, m2, m3] = para_init
	if metric2 != 0 or metric1 != 0:
		scale1 = pd.Series(np.random.normal(1, metric1, size))
		confirmed = [max(confirmed[i] * scale1[i], 1) for i in range(size)]
		scale2 = pd.Series(np.random.normal(1, metric2, size))
		death = [max(death[i] * scale2[i], 1) for i in range(size)]
		scale3 = pd.Series(np.random.normal(1, metric3, size))
		hosp = [max(hosp[i] * scale3[i], 1) for i in range(size)]

	# optimal = minimize(loss2, [0, 0.5, 1],
	optimal = minimize(loss2, [uni(0, 0.5 * eta * n_0),
	                           0.5,
	                           uni(k_range[0], k_range[1])],
	                   args=(para_init, confirmed, death, hosp, start_point, n_0, SIDRG_sd),
	                   method='L-BFGS-B', bounds=[(0, 5 * eta * n_0), (0, 5), k_range])
	min_loss = loss2(optimal.x, para_init, confirmed, death, hosp, start_point, n_0, SIDRG_sd)
	h = optimal.x[0]
	Hiding_init = optimal.x[1]
	k = optimal.x[2]
	[S0, I0, IH0, IN0, D0, R0, G0] = start_point
	S = [S0]
	I = [I0]
	IH = [IH0]
	IN = [IN0]
	D = [D0]
	R = [R0]
	G = [G0]
	H = [Hiding_init * eta * n_0]
	eta2 = eta
	for i in range(1, size):
		release = min(H[-1], h * funcmod(i))
		S[-1] += release
		H[-1] -= release
		if change_eta2:
			eta2 += release / n_0
		delta = SIDRG_sd(i, [S[i - 1], I[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1], beta, gamma,
		                     k * sigma, a1, a2, a3, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		IH.append(IH[-1] + delta[2])
		IN.append(IN[-1] + delta[3])
		D.append(D[-1] + delta[4])
		R.append(R[-1] + delta[5])
		G.append(G[-1] + delta[6])
		H.append(H[-1])
	# Betas.append(delta[7])

	data1 = [(confirmed[i] - G[i]) / confirmed[i] for i in range(size)]
	data2 = [(death[i] - D[i]) / death[i] for i in range(size)]
	data3 = [(hosp[i] - IH[i]) / hosp[i] for i in range(size)]

	# metric1 = math.sqrt(sum([i ** 2 for i in data1]) / (len(data1) - 8))
	# metric2 = math.sqrt(sum([i ** 2 for i in data2]) / (len(data2) - 8))
	weights = [Geo ** n for n in range(size)]
	weights.reverse()
	sum_wt = sum(weights)
	metric1 = math.sqrt(sum([data1[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 3) * sum_wt / size)
	                    )
	metric2 = math.sqrt(sum([data2[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 3) * sum_wt / size)
	                    )
	metric3 = math.sqrt(sum([data3[i] ** 2 * weights[i] for i in range(size)])
	                    /
	                    ((size - 8) * sum_wt / size)
	                    )

	return [h, Hiding_init, k, metric1, metric2, metric3], min_loss


def funcmod(i):
	# return 0.5 * np.log(1 + i)
	# return 1.00 * np.power(i, -0.4)
	return 1


# save SIRG parameters
def save_para_SIRG(state, para_SIRG):
	df = pd.DataFrame([para_SIRG], columns=['beta', 'gamma', 'eta', 'metric1'])
	df.to_csv(f'MT_{end_date}/{state}/{state}_para_SIRG.csv', index=False)
	print('SIRG parameters saved\n')


def fit_2_phase_MT(state, ConfirmFile, DeathFile, PopFile, dates):
	t1 = time.perf_counter()

	if not os.path.exists(f'MT_{end_date}/{state}'):
		os.makedirs(f'MT_{end_date}/{state}')

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
		if confirmed.iloc[0].loc[start_date] >= I_0 and death.iloc[0].loc[start_date] > 0:
			break
	days = list(confirmed.columns)
	days = days[days.index(start_date):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[start_date: end_date]
	confirmed_init = confirmed.loc[start_date: dates[0]].tolist()
	confirmed_reopen = confirmed.loc[dates[0]: dates[1]].tolist()
	death = death.iloc[0].loc[start_date: end_date]
	death_init = death.loc[start_date: dates[0]].tolist()
	death_reopen = death.loc[dates[0]: dates[1]].tolist()
	days_init = days[:len(confirmed_init)]
	days_reopen = days[- len(confirmed_reopen):]

	# select hospitalization data
	hosp = readHosp(state)
	if min(hosp.index) > start_date or max(hosp.index) < end_date:
		print(f'Not enough hospitalization data in {state}')
		print(f'exiting {state}\n')
		return -1
	else:
		hosp_init = hosp[start_date: dates[0]].tolist()
		hosp_reopen = hosp[dates[0]: dates[1]].tolist()

	# initial SIRG fitting
	# para_SIRG = MT_SIRG(confirmed_init, n_0)
	# plot_filename = f'MT_{end_date}/{state}/{state}_fit_SIRG.png'
	# [S, I, R, G] = plot_SIRG(plot_filename, confirmed_init, days_init, n_0, para_SIRG)

	# save SIRG parameters
	# save_para_SIRG(state, para_SIRG)

	# save simulation of SIRG fitting to csv
	# csv_filename = f'MT_{end_date}/{state}/{state}_sim_SIRG.csv'
	# save_sim_SIRG(csv_filename, [S, I, R, G], days_init)

	# initial fitting
	para_init = MT_init(confirmed_init, death_init, hosp_init, n_0)
	# [beta, gamma, sigma, a1, a2, a3, eta, c1, metric1, metric2] = para_init
	plot_filename = f'MT_{end_date}/{state}/{state}_fit_init.png'
	[S, I, IH, IN, D, R, G] = plot_init(plot_filename, confirmed_init, death_init, hosp_init, days_init, n_0, para_init)

	# save simulation of initial fitting to csv
	csv_filename = f'MT_{end_date}/{state}/{state}_sim_init.csv'
	save_sim_init(csv_filename, [S, I, IH, IN, D, R, G], days_init)

	# monte carlo for initial parameter distribution
	para_init_avg = MT_init_dist(state, G, D, IH, n_0, para_init)

	plot_para_init(state, para_init)

	# reopen fitting
	start_point = [S[-1], I[-1], IH[-1], IN[-1], D[-1], R[-1], G[-1]]
	para_reopen = MT_reopen(confirmed_reopen, death_reopen, hosp_reopen, n_0, para_init, start_point)
	[h, Hiding_init, k, metric1, metric2, metric3] = para_reopen
	plot_filename = f'MT_{end_date}/{state}/{state}_fit_reopen.png'
	[S, I, IH, IN, D, R, G, H] = plot_reopen(plot_filename, confirmed_reopen, death_reopen, hosp_reopen, n_0, para_init,
	                                         start_point, days_reopen, para_reopen)

	# save simulation of reopen fitting to csv
	csv_filename = f'MT_{end_date}/{state}/{state}_sim_reopen.csv'
	save_sim_reopen(csv_filename, [S, I, IH, IN, D, R, G, H], days_reopen)

	# monte carlo for reopen parameter distribution
	para_reopen_avg = MT_reopen_dist(state, G, D, IH, n_0, para_init, start_point, para_reopen)

	plot_para_reopen(state, para_reopen)
	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds in total for {state}\n')

	return 0


# read hospitalization data
def readHosp(state):
	df = pd.read_csv(f'data/hosp/{state}/Top_Counties_by_Cases_Full_Data_data.csv', usecols=['Date', 'Hospitalization'])
	df = df[df['Hospitalization'].notna()]

	dates = [datetime.datetime.strptime(date, '%m/%d/%Y') for date in df['Date']]
	dates = [date.strftime('%Y-%m-%d') for date in dates]

	df['Date'] = dates
	df = df.sort_values('Date')
	df = df.T
	df.columns = df.iloc[0]
	df = df.iloc[1]
	return df


# save simulation of SIRG fitting to csv
def save_sim_SIRG(csv_filename, data, days_init):
	[S, I, R, G] = data
	days = [day.strftime('%Y-%m-%d') for day in days_init]
	c0 = ['S', 'I', 'R', 'G']
	df = pd.DataFrame(data, columns=days)
	df.insert(0, 'series', c0)
	df.to_csv(csv_filename, index=False)
	print('SIRG simulation saved\n')


# save simulation of initial fitting to csv
def save_sim_init(csv_filename, data, days_init):
	# [S, I, IH, IN, D, R, G] = data
	days = [day.strftime('%Y-%m-%d') for day in days_init]
	c0 = ['S', 'I', 'IH', 'IN', 'D', 'R', 'G']
	df = pd.DataFrame(data, columns=days)
	df.insert(0, 'series', c0)
	df.to_csv(csv_filename, index=False)
	print('initial simulation saved\n')


# save simulation of reopen fitting to csv
def save_sim_reopen(csv_filename, data, days_reopen):
	# [S, I, IH, IN, D, R, G, H] = data
	days = [day.strftime('%Y-%m-%d') for day in days_reopen]
	c0 = ['S', 'I', 'IH', 'IN', 'D', 'R', 'G', 'H']
	df = pd.DataFrame(data, columns=days)
	df.insert(0, 'series', c0)
	df.to_csv(csv_filename, index=False)
	print('reopen simulation saved\n')


# reopen fitting
def MT_reopen(confirmed_reopen, death_reopen, hosp_reopen, n_0, para_init, start_point):
	para_best = []
	min_loss = 10000
	with concurrent.futures.ProcessPoolExecutor() as executor:
		t1 = time.perf_counter()
		results = [
			executor.submit(fit_reopen, confirmed_reopen, death_reopen, hosp_reopen, n_0, para_init, start_point, 0, 0,
			                0) for _
			in range(num_threads_reopen)]

		k = 0
		for f in concurrent.futures.as_completed(results):
			para_reopen, current_loss = f.result()
			k += 1
			if current_loss < min_loss:
				min_loss = current_loss
				para_best = para_reopen
				print(f'best paras updated at {k}')
		# if k % 10 == 0:
		# 	print(f'{k}/{num_threads} thread(s) completed')

		t2 = time.perf_counter()
		print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_threads_reopen, 3)} seconds per job')
	print('reopen best fitting completed\n')
	return para_best


# initial SIR fitting
def MT_SIRG(confirmed_init, n_0):
	warnings.filterwarnings('ignore', '.*double_scalars.*', )
	para_best = []
	min_loss = 10000
	with concurrent.futures.ProcessPoolExecutor() as executor:
		t1 = time.perf_counter()
		results = [executor.submit(fit_SIRG, confirmed_init, n_0) for _ in range(num_threads)]

		k = 0
		for f in concurrent.futures.as_completed(results):
			para, current_loss = f.result()
			k += 1
			if current_loss < min_loss:
				min_loss = current_loss
				para_best = para
				print(f'best paras updated at {k}')

		t2 = time.perf_counter()
		print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_threads, 3)} seconds per job')
	print('initial SIRG fitting completed\n')
	return para_best


# initial fitting
def MT_init(confirmed_init, death_init, hosp_init, n_0):
	para_best = []
	min_loss = 10000
	with concurrent.futures.ProcessPoolExecutor() as executor:
		t1 = time.perf_counter()
		results = [executor.submit(fit_init, confirmed_init, death_init, hosp_init, n_0, 0, 0, 0) for _ in
		           range(num_threads)]

		k = 0
		for f in concurrent.futures.as_completed(results):
			para, current_loss = f.result()
			k += 1
			if current_loss < min_loss:
				min_loss = current_loss
				para_best = para
				print(f'best paras updated at {k}')
		# if k % 10 == 0:
		# 	print(f'{k}/{num_threads} thread(s) completed')

		t2 = time.perf_counter()
		print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_threads, 3)} seconds per job')

	print('initial best fitting completed\n')
	return para_best


# reopen parameter distribution
def MT_reopen_dist(state, confirmed_reopen, death_reopen, hosp_reopen, n_0, para_init, start_point, para_reopen):
	[h, Hiding_init, k, metric1, metric2, metric3] = para_reopen
	paras = [para_reopen]  # store the bet fit parameters in the first row
	with concurrent.futures.ProcessPoolExecutor() as executor:
		t1 = time.perf_counter()
		results = [
			executor.submit(fit_reopen, confirmed_reopen, death_reopen, hosp_reopen, n_0, para_init, start_point,
			                metric1, metric2, metric3) for _ in range(num_threads_dist_reopen)]

		k = 0
		for f in concurrent.futures.as_completed(results):
			para, loss = f.result()
			paras.append(para)
			k += 1
			if k % 100 == 0:
				print(f'{k}/{num_threads_dist_reopen} thread(s) completed')

		t2 = time.perf_counter()
		print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_threads_dist_reopen, 3)} seconds per job')

	save_para_reopen(state, paras)

	para_avg = []
	for i in range(len(para_reopen)):
		para_avg.append(np.mean([row[i] for row in paras]))
	return para_avg


# initial parameter distribution
def MT_init_dist(state, confirmed_init, death_init, hosp_init, n_0, para_init):
	[beta, gamma, sigma, a1, a2, a3, eta, c1, metric1, metric2, metric3] = para_init
	paras = [para_init]  # store the bet fit parameters in the first row
	with concurrent.futures.ProcessPoolExecutor() as executor:
		t1 = time.perf_counter()
		results = [executor.submit(fit_init, confirmed_init, death_init, hosp_init, n_0, metric1, metric2, metric3) for
		           _ in range(num_threads_dist)]

		k = 0
		for f in concurrent.futures.as_completed(results):
			para, loss = f.result()
			paras.append(para)
			k += 1
			if k % 100 == 0:
				print(f'{k}/{num_threads_dist} thread(s) completed')

		t2 = time.perf_counter()
		print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_threads_dist, 3)} seconds per job')

	save_para_init(state, paras)

	para_avg = []
	for i in range(len(para_init)):
		para_avg.append(np.mean([row[i] for row in paras]))
	return para_avg


# save the reopen parameters distribution to CSV
def save_para_reopen(state, paras):
	para_label = ['h', 'Hiding_init', 'k', 'metric1', 'metric2', 'metric3']
	df = pd.DataFrame(paras, columns=para_label)
	df.to_csv(f'MT_{end_date}/{state}/{state}_para_reopen.csv', index=False, header=True)
	print('reopen parameters saved\n')


# save the initial parameters distribution to CSV
def save_para_init(state, paras):
	para_label = ['beta', 'gamma', 'sigma', 'a1', 'a2', 'a3', 'eta', 'c1', 'metric1', 'metric2', 'metric3']
	df = pd.DataFrame(paras, columns=para_label)
	df.to_csv(f'MT_{end_date}/{state}/{state}_para_init.csv', index=False, header=True)
	print('initial parameters saved\n')


# plot the histograms of the reopen parameters
def plot_para_reopen(state, para_reopen):
	plt.rcParams.update({'font.size': 8})
	fig = plt.figure(figsize=(10, 6))
	para_label = ['h', 'Hiding_init', 'k', 'metric1', 'metric2', 'metric3']
	df = pd.read_csv(f'MT_{end_date}/{state}/{state}_para_reopen.csv')

	for i in range(3):
		ax = fig.add_subplot(2, 2, i + 1)
		ax.set_title(f'{para_label[i]}={round(para_reopen[i], 3)}')
		data = df.iloc[:, i]
		ax.hist(data, bins='auto', alpha=0.3, histtype='stepfilled')
		ax.axvline(para_reopen[i], color='red', label=f'{para_label[i]}={round(para_reopen[i], 3)}')
		ax.axvline(np.mean(data), color='green', label=f'avg={round(np.mean(data), 3)}')
	# ax.legend()

	plt.savefig(f'MT_{end_date}/{state}/{state}_para_reopen.png', bbox_inches="tight")
	plt.close()
	print('reopen parameters figure saved\n')


# plot the histograms of the initial parameters
def plot_para_init(state, para_init):
	plt.rcParams.update({'font.size': 8})
	fig = plt.figure(figsize=(9, 9))
	para_label = ['beta', 'gamma', 'sigma', 'a1', 'a2', 'a3', 'eta', 'c1', 'metric1', 'metric2', 'metric3']
	df = pd.read_csv(f'MT_{end_date}/{state}/{state}_para_init.csv')

	for i in range(8):
		ax = fig.add_subplot(3, 3, i + 1)
		ax.set_title(f'{para_label[i]}={round(para_init[i], 3)}')
		data = df.iloc[:, i]
		ax.hist(data, bins='auto', alpha=0.3, histtype='stepfilled')
		ax.axvline(para_init[i], color='red', label=f'{para_label[i]}={round(para_init[i], 3)}')
		ax.axvline(np.mean(data), color='green', label=f'avg={round(np.mean(data), 3)}')
	# ax.legend()

	plt.savefig(f'MT_{end_date}/{state}/{state}_para_init.png', bbox_inches="tight")
	plt.close()
	print('initial parameters figure saved\n')


# plot the SIRG fitting figure and return the results
def plot_SIRG(plot_filename, confirmed, days, n_0, para_SIRG):
	size = len(confirmed)
	[beta, gamma, eta, metric1] = para_SIRG
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])

	if plot_filename != '':
		plt.rcParams.update({'font.size': 8})
		fig, ax = plt.subplots()
		fig.autofmt_xdate()
		ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
		ax.plot(days, [i / 1000 for i in G], label='G')
		plt.legend()
		plt.savefig(plot_filename, bbox_inches="tight")
		plt.close()

	return [S, I, R, G]


# plot the initial fitting figure and return the results
def plot_init(plot_filename, confirmed, death, hosp, days, n_0, para):
	size = len(confirmed)
	[beta, gamma, sigma, a1, a2, a3, eta, c1, metric1, metric2, metric3] = para
	S = [n_0 * eta]
	I = [confirmed[0]]
	IH = [I[-1] * sigma]
	IN = [I[-1] * (1 - sigma)]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	# H = [Hiding * n_0 * eta]
	# Betas = [beta]
	for i in range(1, size):
		delta = SIDRG_sd(i, [S[i - 1], I[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1], beta, gamma, sigma,
		                     a1,
		                     a2, a3, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		IH.append(IH[-1] + delta[2])
		IN.append(IN[-1] + delta[3])
		D.append(D[-1] + delta[4])
		R.append(R[-1] + delta[5])
		G.append(G[-1] + delta[6])
	# H.append(H[-1])
	# Betas.append(delta[7])

	if plot_filename != '':
		plt.rcParams.update({'font.size': 8})
		# fig, ax = plt.subplots()
		fig = plt.figure(figsize=(10, 5))
		ax = fig.add_subplot(1, 2, 1)
		ax2 = fig.add_subplot(1, 2, 2)
		ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
		ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
		ax2.plot(days, [i / 1000 for i in hosp], linewidth=5, linestyle=':', label="Hospitalization")
		ax.plot(days, [i / 1000 for i in G], label='G')
		ax2.plot(days, [i / 1000 for i in D], label='D')
		ax2.plot(days, [i / 1000 for i in IH], label='IH')
		ax.legend()
		ax2.legend()
		fig.autofmt_xdate()
		plt.savefig(plot_filename, bbox_inches="tight")
		plt.close(fig)

	return [S, I, IH, IN, D, R, G]


# plot the reopen fitting figure and return the results
def plot_reopen(plot_filename, confirmed_reopen, death_reopen, hosp_reopen, n_0, para_init, start_point, days_reopen, para_reopen):
	size = len(confirmed_reopen)
	[beta, gamma, sigma, a1, a2, a3, eta, c1, metric1, metric2, metric3] = para_init
	[h, Hiding_init, k, metric1, metric2, metric3] = para_reopen
	[S0, I0, IH0, IN0, D0, R0, G0] = start_point
	S = [S0]
	I = [I0]
	IH = [IH0]
	IN = [IN0]
	D = [D0]
	R = [R0]
	G = [G0]
	H = [Hiding_init * eta * n_0]
	eta2 = eta
	for i in range(1, size):
		# if i < 40:
		release = min(H[-1], h * funcmod(i))
		S[-1] += release
		H[-1] -= release
		if change_eta2:
			eta2 += release / n_0
		delta = SIDRG_sd(i, [S[i - 1], I[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1], beta, gamma,
		                     k * sigma,
		                     a1, a2, a3, eta2, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		IH.append(IH[-1] + delta[2])
		IN.append(IN[-1] + delta[3])
		D.append(D[-1] + delta[4])
		R.append(R[-1] + delta[5])
		G.append(G[-1] + delta[6])
		H.append(H[-1])

	if plot_filename != '':
		plt.rcParams.update({'font.size': 8})
		# fig, ax = plt.subplots()
		fig = plt.figure(figsize=(10, 5))
		ax = fig.add_subplot(1, 2, 1)
		ax2 = fig.add_subplot(1, 2, 2)
		ax.plot(days_reopen, [i / 1000 for i in confirmed_reopen], linewidth=5, linestyle=':',
		        label="Cumulative\nCases")
		ax2.plot(days_reopen, [i / 1000 for i in death_reopen], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
		ax2.plot(days_reopen, [i / 1000 for i in hosp_reopen], linewidth=5, linestyle=':', label="Hospitalization")
		ax.plot(days_reopen, [i / 1000 for i in G], label='G')
		ax2.plot(days_reopen, [i / 1000 for i in D], label='D')
		ax2.plot(days_reopen, [i / 1000 for i in IH], label='IH')
		ax.plot(days_reopen, [i / 1000 for i in H], label='H')
		ax.plot(days_reopen, [i / 1000 for i in S], label='S')
		ax.legend()
		ax2.legend()
		fig.autofmt_xdate()
		plt.savefig(plot_filename, bbox_inches="tight")
		plt.close(fig)

	return [S, I, IH, IN, D, R, G, H]


def fit_all_2_phase_MT():
	t1 = time.perf_counter()
	# plt.rcParams.update({'font.size': 8})
	# fig = plt.figure(figsize=(16, 20))
	# fig_num = 0
	# show_legend = False

	state = 'NY-New York'
	dates = ['2020-06-22', end_date]
	fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)

	# state = 'TX-Harris--Houston'
	# dates = ['2020-06-03', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	#
	# state = 'AZ-Maricopa'
	# dates = ['2020-05-28', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	state = 'CA-Los Angeles'
	dates = ['2020-06-12', end_date]
	fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	state = 'FL-Miami-Dade'
	dates = ['2020-06-03', end_date]
	fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'GA-Fulton'
	# dates = ['2020-06-12', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	state = 'IL-Cook'
	dates = ['2020-06-03', end_date]
	fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'AZ-Maricopa'
	# dates = ['2020-06-22', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'NJ-Bergen'
	# dates = ['2020-06-22', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'PA-Philadelphia'
	# dates = ['2020-06-05', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'MD-Prince Georges'
	# dates = ['2020-06-29', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'NV-Clark'
	# dates = ['2020-05-29', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'NC-Mecklenburg'
	# dates = ['2020-05-22', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'LA-Jefferson'
	# dates = ['2020-06-05', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'CA-Riverside'
	# dates = ['2020-06-12', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'FL-Broward'
	# dates = ['2020-06-12', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	state = 'TX-Dallas'
	dates = ['2020-05-22', end_date]
	fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'NJ-Hudson'
	# dates = ['2020-06-22', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'MA-Middlesex'
	# dates = ['2020-06-22', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'OH-Franklin'
	# dates = ['2020-05-21', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'VA-Fairfax'
	# dates = ['2020-06-12', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'TN-Shelby'
	# dates = ['2020-06-15', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'WI-Milwaukee'
	# dates = ['2020-07-01', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'UT-Salt Lake'
	# dates = ['2020-05-15', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)
	#
	# state = 'MN-Hennepin'
	# dates = ['2020-06-04', end_date]
	# fit_2_phase_MT(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv', dates)

	# fig_num += 1
	# dates = ['2020-05-11', end_date]
	# fit_2_phase('SC-Charleston', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)
	#
	# fig_num += 1
	# dates = ['2020-06-08', end_date]
	# fit_2_phase('MI-Oakland', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	t2 = time.perf_counter()
	print(f'{round((t2 - t1) / 60, 3)} minutes for all counties')
	return 0


def test():
	r1 = [1, 2, 3]
	r2 = [4, 5, 6]
	col = ['c2', 'c3', 'c4']
	c1 = [0, 5]
	df = pd.DataFrame([r1, r2], columns=col)
	print(df)
	df.insert(0, 'c1', c1)
	print(df)
	df.to_csv('index.csv', index=True)
	df.to_csv('no_index.csv', index=False)


def main():
	fit_all_2_phase_MT()
	# test()
	return 0


if __name__ == '__main__':
	main()
