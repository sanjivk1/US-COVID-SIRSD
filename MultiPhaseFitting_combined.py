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
from SIRfunctions import SIRG_combined
import datetime
from numpy.random import uniform as uni
import os
import warnings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# matplotlib.use('Agg')

np.set_printoptions(threshold=sys.maxsize)
Geo = 0.98

num_threads = 12
num_threads_dist = 0

# weight of G in initial fitting
theta = 0.7
# weight of G in release fitting
theta2 = 0.8

I_0 = 5
beta_range = (0.1, 100)
gamma_range = (0.04, 0.2)
gamma2_range = (0.04, 0.2)
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
end_date = '2020-08-16'
# end_date = '2020-09-23'
# end_date = '2020-09-22'
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
	c0 = ['S', 'I', 'IH', 'IN', 'D', 'R', 'G', 'H', 'beta']
	df = pd.DataFrame(data, columns=days)
	df.insert(0, 'series', c0)
	df.to_csv(csv_filename, index=False)
	print('simulation saved\n')


# save the parameters distribution to CSV
def save_para_combined(state, paras):
	para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1', 'metric1',
	              'metric2', 'metric3']
	df = pd.DataFrame(paras, columns=para_label)
	df.to_csv(f'combined_{end_date}/{state}/para.csv', index=False, header=True)
	print('parameters saved\n')


# simulate combined phase
def simulate_combined(size, SIRG, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2, eta, c1, n_0,
                      reopen_day):
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
		delta = SIRG(i, [S[i - 1], I[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1], beta, kk * gamma,
		                 gamma2, a1, kk2 * a2, a3, eta2, n_0, c1, H[-1], H0])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		IH.append(IH[-1] + delta[2])
		IN.append(IN[-1] + delta[3])
		D.append(D[-1] + delta[4])
		R.append(R[-1] + delta[5])
		G.append(G[-1] + delta[6])
		H.append(H[-1])
		betas.append(delta[7])
		if S[-1] < 0:
			result = False
			break
	return result, [S, I, IH, IN, D, R, G, H, betas]


# combined fitting
def loss_combined(point, c1, confirmed, death, hosp, n_0, SIRG, reopen_day):
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

	weights = [Geo ** n for n in range(size)]
	weights.reverse()
	weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
	weighted_G = [G[i] * weights[i] for i in range(size)]
	weighted_death = [death[i] * weights[i] for i in range(size)]
	weighted_D = [D[i] * weights[i] for i in range(size)]
	# weighted_hosp = [hosp[i] * weights[i] for i in range(size)]
	# weighted_IH = [IH[i] * weights[i] for i in range(size)]

	confirmed_derivative = np.diff(confirmed)
	G_derivative = np.diff(G)
	confirmed_derivative = [confirmed_derivative[i] * weights[i] for i in range(size - 1)]
	G_derivative = [G_derivative[i] * weights[i] for i in range(size - 1)]
	alpha = 0.5
	metric00 = r2_score(weighted_confirmed, weighted_G)
	metric01 = r2_score(confirmed_derivative, G_derivative)
	# metric0 = (alpha * metric00 + (1 - alpha) * metric01)

	weighted_hosp = hosp
	weighted_IH = IH

	metric0 = r2_score(weighted_confirmed, weighted_G)
	metric1 = r2_score(weighted_death, weighted_D)
	metric2 = r2_score(weighted_hosp, weighted_IH)
	return -(theta * metric0 + 1 * (1 - theta) / 2 * metric1 + 1 * (1 - theta) / 2 * metric2)


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
		                                   uni(gamma_range[0], gamma_range[1]),
		                                   uni(gamma2_range[0], gamma2_range[1]),
		                                   uni(a1_range[0], a1_range[1]),
		                                   uni(a2_range[0], a2_range[1]),
		                                   uni(a3_range[0], a3_range[1]),
		                                   uni(eta_range[0], eta_range[1]),
		                                   uni(0, 0.5),
		                                   0.5,
		                                   uni(k_range[0], k_range[1]),
		                                   uni(k2_range[0], k2_range[1])],
		                   args=(c1, confirmed, death, hosp, n_0, SIRG_combined, reopen_day), method='L-BFGS-B',
		                   bounds=[beta_range,
		                           gamma_range,
		                           gamma2_range,
		                           a1_range,
		                           a2_range,
		                           a3_range,
		                           eta_range,
		                           (0, 0.5),
		                           (0, 5),
		                           k_range,
		                           k2_range])
		current_loss = loss_combined(optimal.x, c1, confirmed, death, hosp, n_0, SIRG_combined, reopen_day)
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
		= simulate_combined(size, SIRG_combined, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2,
		                    eta, c1, n_0, reopen_day)

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

	return [beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3], min_loss


def funcmod(i):
	# return 0.5 * np.log(1 + i)
	# return 1.00 * np.power(i, -0.4)
	return 1


def fit_state_combined(state, ConfirmFile, DeathFile, PopFile, dates):
	t1 = time.perf_counter()

	if not os.path.exists(f'combined_{end_date}/{state}'):
		os.makedirs(f'combined_{end_date}/{state}')

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
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[start_date: end_date]
	death = death.iloc[0].loc[start_date: end_date]
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
	[S, I, IH, IN, D, R, G, H, betas] = plot_combined(state, confirmed, death, hosp, days, n_0, reopen_day, para)
	csv_file = f'combined_{end_date}/{state}/sim.csv'
	save_sim_combined(csv_file, [S, I, IH, IN, D, R, G, H, betas], days)
	save_para_combined(state, [para])
	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds in total for {state}\n')

	return 0


# plot result
def plot_combined(state, confirmed, death, hosp, days, n_0, reopen_day, para):
	[beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3] = para
	para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1', 'metric1',
	              'metric2', 'metric3']
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
		= simulate_combined(size, SIRG_combined, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2,
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
	fig.savefig(f'combined_{end_date}/{state}/sim.png', bbox_inches="tight")
	plt.close(fig)
	return [S, I, IH, IN, D, R, G, H, betas]


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


def fit_all_combined():
	t1 = time.perf_counter()

	state = 'NY-New York'
	dates = ['2020-06-22', end_date]
	fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	                   dates)

	state = 'CA-Los Angeles'
	dates = ['2020-06-12', end_date]
	fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	                   dates)

	state = 'FL-Miami-Dade'
	dates = ['2020-06-03', end_date]
	fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	                   dates)

	state = 'IL-Cook'
	dates = ['2020-06-03', end_date]
	fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	                   dates)

	state = 'TX-Dallas'
	dates = ['2020-05-22', end_date]
	fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	                   dates)

	# state = 'TX-Harris--Houston'
	# dates = ['2020-06-03', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'AZ-Maricopa'
	# dates = ['2020-05-28', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'GA-Fulton'
	# dates = ['2020-06-12', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'AZ-Maricopa'
	# dates = ['2020-06-22', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'NJ-Bergen'
	# dates = ['2020-06-22', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'PA-Philadelphia'
	# dates = ['2020-06-05', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'MD-Prince Georges'
	# dates = ['2020-06-29', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'NV-Clark'
	# dates = ['2020-05-29', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'NC-Mecklenburg'
	# dates = ['2020-05-22', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'LA-Jefferson'
	# dates = ['2020-06-05', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'CA-Riverside'
	# dates = ['2020-06-12', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'FL-Broward'
	# dates = ['2020-06-12', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'NJ-Hudson'
	# dates = ['2020-06-22', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'MA-Middlesex'
	# dates = ['2020-06-22', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'OH-Franklin'
	# dates = ['2020-05-21', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'VA-Fairfax'
	# dates = ['2020-06-12', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'TN-Shelby'
	# dates = ['2020-06-15', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'WI-Milwaukee'
	# dates = ['2020-07-01', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'UT-Salt Lake'
	# dates = ['2020-05-15', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)
	#
	# state = 'MN-Hennepin'
	# dates = ['2020-06-04', end_date]
	# fit_state_combined(state, 'data/Confirmed-counties.csv', 'data/Death-counties.csv', 'data/CountyPopulation.csv',
	#                    dates)

	t2 = time.perf_counter()
	print(f'{round((t2 - t1) / 60, 3)} minutes for all counties')


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
	fit_all_combined()
	# test()
	return 0


if __name__ == '__main__':
	main()
