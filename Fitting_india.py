import numpy as np
import pandas as pd
import time
import math
import concurrent.futures
import multiprocessing
from scipy.optimize import minimize
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score, mean_squared_error
from SIRfunctions import SEIARG, SEIARG_fixed, weighted_deviation, weighted_relative_deviation
import datetime
from numpy.random import uniform as uni
import os
import warnings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

np.set_printoptions(threshold=sys.maxsize)
Geo = 0.98

# num_threads = 200
num_threads = 100
num_CI = 1000
# num_CI = 5
start_dev = 0

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
a2_range = (0.001, 0.2)
a3_range = (0.01, 0.2)
eta_range = (0.001, 0.4)
c1_fixed = (0.9, 0.9)
c1_range = (0.8, 1)
h_range = (0, 10)
k_range = (0.1, 2)
k2_range = (0.1, 2)
I_initial_range = (0, 1)
start_date = '2021-02-01'
reopen_date = '2021-03-15'
end_date = '2021-05-03'
release_duration = 30
k_drop = 14
p_m = 1
# Hiding = 0.33
delay = 7
change_eta2 = False

fig_row = 5
fig_col = 3

states2 = ['kl', 'dl', 'tg', 'rj', 'hr', 'jk', 'ka', 'la', 'mh', 'pb', 'tn', 'up', 'ap', 'ut', 'or', 'wb', 'py', 'ch',
           'ct', 'gj', 'hp', 'mp', 'br', 'mn', 'mz', 'ga', 'an', 'as', 'jh', 'ar', 'tr', 'nl', 'ml', 'dn', 'sk',
           'unassigned', 'dd', 'dn_dd', 'ld']

states = ['kl', 'dl', 'tg', 'rj', 'hr', 'jk', 'ka', 'la', 'mh', 'pb', 'tn', 'up', 'ap', 'ut', 'or', 'wb', 'py', 'ch',
          'ct', 'gj', 'hp', 'mp', 'br', 'mn', 'mz', 'ga', 'an', 'as', 'jh', 'ar', 'tr', 'nl', 'ml', 'sk', 'dn_dd', 'ld']


def simulate_combined(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h,
                      k, k2, eta, c1, n_0, reopen_day):
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
			release = min(H[-1], r)
			S[-1] += release
			H[-1] -= release

		delta = SEIARG(i,
		               [S[i - 1], E[i - 1], I[i - 1], A[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1],
		                beta, gammaE, alpha, kk * gamma, gamma2, gamma3, a1, kk2 * a2, a3, eta2, n_0, c1, H[-1], H0])
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


def loss_combined(point, c1, confirmed, death, n_0, reopen_day):
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
	I_initial = point[14]
	S = [n_0 * eta]
	E = [0]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [Hiding_init * eta * n_0]
	# H = [0]
	result, [S, E, I, A, IH, IN, D, R, G, H, betas] \
		= simulate_combined(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
		                    a3, h, k, k2, eta, c1, n_0, reopen_day)

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

	weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
	weighted_G = [G[i] * weights[i] for i in range(size)]
	weighted_death = [death[i] * weights[i] for i in range(size)]
	weighted_D = [D[i] * weights[i] for i in range(size)]

	metric0 = r2_score(weighted_confirmed, weighted_G)
	metric1 = r2_score(weighted_death, weighted_D)

	return -(0.9 * metric0 + 0.1 * metric1)


def fit(confirmed0, death0, reopen_day_gov, n_0):
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
	for reopen_day in range(reopen_day_gov, reopen_day_gov + 14):
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
			                                   uni(k2_range[0], k2_range[1]),
			                                   uni(I_initial_range[0], I_initial_range[1])],
			                   args=(c1, confirmed, death, n_0, reopen_day), method='L-BFGS-B',
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
			                           k2_range,
			                           I_initial_range])
			current_loss = loss_combined(optimal.x, c1, confirmed, death, n_0, reopen_day)
			if current_loss < min_loss:
				# print(f'updating loss={current_loss} with c1={c1}')
				min_loss = current_loss
				c_max = c1
				reopen_max = reopen_day
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
				I_initial = optimal.x[14]

	c1 = c_max
	reopen_day = reopen_max
	S = [n_0 * eta]
	E = [0]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [Hiding_init * n_0 * eta]
	# H = [0]
	# Betas = [beta]

	result, [S, E, I, A, IH, IN, D, R, G, H, betas] \
		= simulate_combined(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
		                    a3, h, k, k2, eta, c1, n_0, reopen_day)

	# data1 = [(confirmed[i] - G[i]) / confirmed[i] for i in range(size)]
	# data2 = [(death[i] - D[i]) / death[i] for i in range(size)]

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

	# sum_wt = sum(weights)
	# metric1 = math.sqrt(sum([data1[i] ** 2 * weights[i] for i in range(size)])
	#                     /
	#                     ((size - 12) * sum_wt / size)
	#                     )
	# metric2 = math.sqrt(sum([data2[i] ** 2 * weights[i] for i in range(size)])
	#                     /
	#                     ((size - 12) * sum_wt / size)
	#                     )
	metric1 = weighted_relative_deviation(weights, confirmed, G, start_dev, 12)
	metric2 = weighted_relative_deviation(weights, death, D, start_dev, 12)

	r1 = r2_score(confirmed, G)
	r2 = r2_score(death, D)

	return [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, k, k2, eta, h, Hiding_init, c1, I_initial, metric1,
	        metric2, r1, r2, reopen_day], min_loss


def MT_fitting(confirmed, death, n_0, reopen_day_gov):
	para_best = []
	min_loss = 10000
	with concurrent.futures.ProcessPoolExecutor() as executor:
		t1 = time.perf_counter()
		results = [executor.submit(fit, confirmed, death, reopen_day_gov, n_0) for _ in range(num_threads)]

		threads = 0
		for f in concurrent.futures.as_completed(results):
			para, current_loss = f.result()
			threads += 1
			# print(f'thread {threads} returned')
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


def fit_state(state, ConfirmFile, DeathFile, PopFile):
	t1 = time.perf_counter()
	state_path = f'india/fitting_{end_date}/{state}'
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

	# for start_date in confirmed.columns[1:]:
	# 	# if confirmed.iloc[0].loc[start_date] >= I_0 and death.iloc[0].loc[start_date] > 0:
	# 	if confirmed.iloc[0].loc[start_date] >= I_0:
	# 		break

	days = list(confirmed.columns)
	days = days[days.index(start_date):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[start_date: end_date]
	death = death.iloc[0].loc[start_date: end_date]
	for i in range(len(death)):
		if death.iloc[i] == 0:
			death.iloc[i] = 0.01
	death = death.tolist()

	reopen_day_gov = days.index(datetime.datetime.strptime(reopen_date, '%Y-%m-%d'))

	# fitting
	para = MT_fitting(confirmed, death, n_0, reopen_day_gov)
	[S, E, I, A, IH, IN, D, R, G, H, betas] = plot_combined(state, confirmed, death, days, n_0, para, state_path)

	save_sim_combined([S, E, I, A, IH, IN, D, R, G, H, betas], days, state_path)

	para[-1] = days[para[-1]]
	save_para_combined([para], state_path)
	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds in total for {state}\n')

	return


def save_para_combined(paras, state_path):
	para_label = ['beta', 'gammaE', 'alpha', 'gamma', 'gamma2', 'gamma3', 'a1', 'a2', 'a3', 'k', 'k2', 'eta', 'h',
	              'Hiding_init', 'c1', 'I_initial', 'metric1', 'metric2', 'r1', 'r2', 'reopen']
	df = pd.DataFrame(paras, columns=para_label)
	df.to_csv(f'{state_path}/para.csv', index=False, header=True)

	print('parameters saved\n')


def save_sim_combined(data, days, state_path):
	days = [day.strftime('%Y-%m-%d') for day in days]
	c0 = ['S', 'E', 'I', 'A', 'IH', 'IN', 'D', 'R', 'G', 'H', 'beta']
	df = pd.DataFrame(data, columns=days)
	df.insert(0, 'series', c0)
	df.to_csv(f'{state_path}/sim.csv', index=False)
	print('simulation saved\n')


def plot_combined(state, confirmed, death, days, n_0, para, state_path):
	[beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, k, k2, eta, h, Hiding_init, c1, I_initial, metric1,
	 metric2, r1, r2, reopen_day] = para
	para_label = ['beta', 'gammaE', 'alpha', 'gamma', 'gamma2', 'gamma3', 'a1', 'a2', 'a3', 'k', 'k2', 'eta', 'h',
	              'Hiding_init', 'c1', 'I_initial', 'metric1', 'metric2', 'r1', 'r2', 'reopen']
	for i in range(len(para)):
		print(f'{para_label[i]}={para[i]} ', end=' ')
		if i % 4 == 1:
			print()

	S = [n_0 * eta]
	E = [0]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [Hiding_init * n_0 * eta]
	# H = [0]
	size = len(days)
	result, [S, E, I, A, IH, IN, D, R, G, H, betas] \
		= simulate_combined(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
		                    a3, h, k, k2, eta, c1, n_0, reopen_day)

	fig = plt.figure(figsize=(12, 10))
	fig.suptitle(state)
	ax = fig.add_subplot(321)
	# ax.set_title(state)
	ax2 = fig.add_subplot(322)
	ax3 = fig.add_subplot(323)
	ax4 = fig.add_subplot(324)
	ax5 = fig.add_subplot(325)
	ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax2.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax3.axvline(days[reopen_day], linestyle='dashed', color='tab:grey', label=days[reopen_day].strftime('%Y-%m-%d'))
	ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
	ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
	ax.plot(days, [i / 1000 for i in G], label='G')
	ax2.plot(days, [i / 1000 for i in D], label='D')
	ax3.plot(days, betas, label='beta')
	ax5.plot(days, [i / 1000 for i in S], label='S')
	ax5.plot(days, [i / 1000 for i in H], label='H')
	diff_G = pd.Series(np.diff(G))
	diff_confirmed = pd.Series(np.diff(confirmed))
	ax4.plot(days[-len(diff_confirmed):], [i / 1000 for i in diff_confirmed], label='daily new cases')
	ax4.plot(days[-len(diff_G):], [i / 1000 for i in diff_G], label='dG')
	ax.legend()
	ax2.legend()
	ax3.legend()
	ax4.legend()
	ax5.legend()
	fig.autofmt_xdate()
	fig.savefig(f'{state_path}/sim.png', bbox_inches="tight")
	# fig.savefig(f'init_only_{end_date}/{state}/sim.png', bbox_inches="tight")
	plt.close(fig)
	return [S, E, I, A, IH, IN, D, R, G, H, betas]


def fit_all():
	t1 = time.perf_counter()
	# states = ['mz', 'dn_dd', 'ld']
	for state in states:
		fit_state(state, 'india/indian_cases_confirmed_cases.csv', 'india/indian_cases_confirmed_deaths.csv',
		          'india/state_population.csv')

	t2 = time.perf_counter()
	print(f'{round((t2 - t1) / 3600, 3)} hours for all states')
	return


def tmp():
	PopFile = 'india/state_population.csv'
	df = pd.read_csv(PopFile)
	# print(states2[0] in df['state'])
	for state in states2:
		if df[df['state'] == state].empty:
			print(state)

	return


def main():
	fit_all()
	# tmp()
	return


if __name__ == '__main__':
	main()
