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
from SIRfunctions import SEIARG, SEIARG_fixed, weighted_deviation, weighted_relative_deviation
import datetime
from numpy.random import uniform as uni
import os
import warnings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

np.set_printoptions(threshold=sys.maxsize)
Geo = 0.98
num_para = 15

# num_threads = 200
num_threads = 10
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
eta_range = (0.001, 0.95)
c1_fixed = (0.9, 0.9)
c1_range = (0.8, 1)
h_range = (1 / 30, 1 / 14)
Hiding_init_range = (0.1, 0.9)
Lockdown_init_range = (0, 0.5)
k_range = (0.1, 2)
k2_range = (0.1, 2)
I_initial_range = (0, 1)
start_date = '2021-02-01'
reopen_date = '2021-03-15'
end_date = '2021-05-22'
lockdown_date = '2021-04-20'
lockdown_rate = 1 / 7
release_duration = 30
k_drop = 14
p_m = 1
# Hiding = 0.33
delay = 7
change_eta2 = False
size_ext = 180

fig_row = 5
fig_col = 3

states2 = ['kl', 'dl', 'tg', 'rj', 'hr', 'jk', 'ka', 'la', 'mh', 'pb', 'tn', 'up', 'ap', 'ut', 'or', 'wb', 'py', 'ch',
           'ct', 'gj', 'hp', 'mp', 'br', 'mn', 'mz', 'ga', 'an', 'as', 'jh', 'ar', 'tr', 'nl', 'ml', 'dn', 'sk',
           'unassigned', 'dd', 'dn_dd', 'ld']

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


def computeBeta(beta, S, S0, c1):
	return beta * (1 - c1) / (1 - c1 * (S / S0))


def simulate_combined(size, S, E, I, A, IH, IN, D, R, G, H, L, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3,
                      h, Hiding_init, eta, c1, n_0, reopen_day, lockdown_day):
	result = True
	S0 = S[0]
	H0 = H[0]
	L0 = L[0]
	# eta2 = eta * (1 - Hiding_init)
	r = h * H0
	betas = [beta]
	for i in range(1, size):

		if i > reopen_day:
			release = min(H[-1], r)
			S[-1] += release
			S0 += release
			H[-1] -= release

		if i > lockdown_day:
			retrieve = min(L[-1], L0 * lockdown_rate)
			S[-1] -= retrieve
			S0 -= retrieve
			L[-1] -= retrieve

		if S[-1] < 0:
			result = False
			break

		beta1 = computeBeta(beta, S[-1], S0, c1)

		dS = -beta1 * S[-1] * (I[-1] + A[-1]) / n_0  # dS
		dE = beta1 * S[-1] * (I[-1] + A[-1]) / n_0 - gammaE * E[-1]  # dE
		dI = (1 - alpha) * gammaE * E[-1] - (gamma + gamma2) * I[-1]  # dI
		dA = alpha * gammaE * E[-1] - gamma3 * A[-1]  # dA
		dIH = gamma * I[-1] - (a1 + a2) * IH[-1]  # dIH
		dIN = gamma2 * I[-1] - a3 * IN[-1]  # dIN
		dD = a2 * IH[-1]  # dD
		dR = a1 * IH[-1] + a3 * IN[-1] + gamma3 * A[-1]  # dR
		dG = (1 - alpha) * gammaE * E[-1]  # dG

		S.append(S[-1] + dS)
		E.append(E[-1] + dE)
		I.append(I[-1] + dI)
		A.append(A[-1] + dA)
		IH.append(IH[-1] + dIH)
		IN.append(IN[-1] + dIN)
		D.append(D[-1] + dD)
		R.append(R[-1] + dR)
		G.append(G[-1] + dG)
		H.append(H[-1])
		L.append(L[-1])
		betas.append(beta1)

		if S[-1] < 0:
			result = False
			break
	return result, [S, E, I, A, IH, IN, D, R, G, H, L, betas]


def simulate_release(size, S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
                     a3, h, Hiding_init, eta, c1, n_0, reopen_day, release_day, release_size, daily_speed):
	S = S0.copy()
	E = E0.copy()
	I = I0.copy()
	A = A0.copy()
	IH = IH0.copy()
	IN = IN0.copy()
	D = D0.copy()
	R = R0.copy()
	G = G0.copy()
	H = H0.copy()
	result = True
	H0 = H[0]
	eta2 = eta * (1 - Hiding_init)
	r = h * H0
	betas = [beta]
	HH = [release_size * n_0]
	daily_release = daily_speed * HH[-1]
	for i in range(1, size):

		if i > reopen_day:
			release = min(H[-1], r)
			S[-1] += release
			H[-1] -= release

		if i > release_day:
			release = min(daily_release, HH[-1])
			S[-1] += release
			HH[-1] -= release
			H0 += release

		delta = SEIARG(i,
		               [S[i - 1], E[i - 1], I[i - 1], A[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1],
		                beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta2, n_0, c1, H[-1], H0])
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
		if S[-1] < 0:
			result = False
			break
	return result, [S, E, I, A, IH, IN, D, R, G, H, HH, betas]


def loss_combined(point, c1, confirmed, death, n_0, reopen_day, lockdown_day):
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
	Lockdown_init = point[12]
	I_initial = point[13]
	S = [n_0 * eta * (1 - Hiding_init)]
	E = [0]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [n_0 * eta * Hiding_init]
	L = [n_0 * eta * Lockdown_init]
	# H = [0]
	result, [S, E, I, A, IH, IN, D, R, G, H, L, betas] \
		= simulate_combined(size, S, E, I, A, IH, IN, D, R, G, H, L, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
		                    a3, h, Hiding_init, eta, c1, n_0, reopen_day, lockdown_day)

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

	return -(0.8 * metric0 + 0.2 * metric1)


def fit(confirmed0, death0, reopen_day_gov, lockdown_day, n_0):
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
			                                   uni(h_range[0], h_range[1]),
			                                   uni(Hiding_init_range[0], Hiding_init_range[1]),
			                                   uni(Lockdown_init_range[0], Lockdown_init_range[1]),
			                                   uni(I_initial_range[0], I_initial_range[1])],
			                   args=(c1, confirmed, death, n_0, reopen_day, lockdown_day), method='L-BFGS-B',
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
			                           h_range,
			                           Hiding_init_range,
			                           Lockdown_init_range,
			                           I_initial_range])
			current_loss = loss_combined(optimal.x, c1, confirmed, death, n_0, reopen_day, lockdown_day)
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
				Lockdown_init = optimal.x[12]
				I_initial = optimal.x[13]

	c1 = c_max
	reopen_day = reopen_max
	S = [n_0 * eta * (1 - Hiding_init)]
	E = [0]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [n_0 * eta * Hiding_init]
	L = [n_0 * eta * Lockdown_init]
	# H = [0]
	# Betas = [beta]

	result, [S, E, I, A, IH, IN, D, R, G, H, L, betas] \
		= simulate_combined(size, S, E, I, A, IH, IN, D, R, G, H, L, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
		                    a3, h, Hiding_init, eta, c1, n_0, reopen_day, lockdown_day)

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
	metric1 = weighted_relative_deviation(weights, confirmed, G, start_dev, num_para)
	metric2 = weighted_relative_deviation(weights, death, D, start_dev, num_para)

	r1 = r2_score(confirmed, G)
	r2 = r2_score(death, D)

	return [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, h, Hiding_init, Lockdown_init, c1, I_initial,
	        metric1, metric2, r1, r2, reopen_day, lockdown_day], min_loss


def MT_fitting(confirmed, death, n_0, reopen_day_gov, lockdown_day):
	para_best = []
	min_loss = 10000
	with concurrent.futures.ProcessPoolExecutor() as executor:
		t1 = time.perf_counter()
		results = [executor.submit(fit, confirmed, death, reopen_day_gov, lockdown_day, n_0) for _ in
		           range(num_threads)]

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
	state_path = f'india/fittingV3_{end_date}/{state}'
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
	lockdown_day = days.index(datetime.datetime.strptime(lockdown_date, '%Y-%m-%d'))

	# fitting
	para = MT_fitting(confirmed, death, n_0, reopen_day_gov, lockdown_day)
	[S, E, I, A, IH, IN, D, R, G, H, L, betas] = plot_combined(state, confirmed, death, days, n_0, para, state_path)

	save_sim_combined([S, E, I, A, IH, IN, D, R, G, H, L, betas], days, state_path)

	para[-1] = days[para[-1]]
	para[-2] = days[para[-2]]
	save_para_combined([para], state_path)
	t2 = time.perf_counter()
	print(f'{round((t2 - t1) / 60, 3)} minutes in total for {state}\n')

	return


def save_para_combined(paras, state_path):
	para_label = ['beta', 'gammaE', 'alpha', 'gamma', 'gamma2', 'gamma3', 'a1', 'a2', 'a3', 'eta', 'h', 'Hiding_init',
	              'Lockdown_init', 'c1', 'I_initial', 'metric1', 'metric2', 'r1', 'r2', 'reopen', 'lockdown']
	df = pd.DataFrame(paras, columns=para_label)
	df.to_csv(f'{state_path}/para.csv', index=False, header=True)

	print('parameters saved\n')


def save_sim_combined(data, days, state_path):
	days = [day.strftime('%Y-%m-%d') for day in days]
	c0 = ['S', 'E', 'I', 'A', 'IH', 'IN', 'D', 'R', 'G', 'H', 'L', 'beta']
	df = pd.DataFrame(data, columns=days)
	df.insert(0, 'series', c0)
	df.to_csv(f'{state_path}/sim.csv', index=False)
	print('simulation saved\n')


def plot_combined(state, confirmed, death, days, n_0, para, state_path):
	[beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, h, Hiding_init, Lockdown_init, c1, I_initial, metric1,
	 metric2, r1, r2, reopen_day, lockdown_day] = para
	para_label = ['beta', 'gammaE', 'alpha', 'gamma', 'gamma2', 'gamma3', 'a1', 'a2', 'a3', 'eta', 'h', 'Hiding_init',
	              'Lockdown_init', 'c1', 'I_initial', 'metric1', 'metric2', 'r1', 'r2', 'reopen', 'lockdown']
	for i in range(len(para)):
		print(f'{para_label[i]}={round(para[i], 4)} ', end=' ')
		if i % 4 == 3:
			print()
	print()

	S = [n_0 * eta * (1 - Hiding_init)]
	E = [0]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [n_0 * eta * Hiding_init]
	L = [n_0 * eta * Lockdown_init]
	# H = [0]
	size = len(days)
	result, [S, E, I, A, IH, IN, D, R, G, H, L, betas] \
		= simulate_combined(size, S, E, I, A, IH, IN, D, R, G, H, L, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
		                    a3, h, Hiding_init, eta, c1, n_0, reopen_day, lockdown_day)

	fig = plt.figure(figsize=(12, 10))
	fig.suptitle(state_dict[state])
	ax = fig.add_subplot(321)
	# ax.set_title(state)
	ax2 = fig.add_subplot(322)
	ax3 = fig.add_subplot(323)
	ax4 = fig.add_subplot(324)
	ax5 = fig.add_subplot(325)
	ax6 = fig.add_subplot(326)
	ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax2.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax5.axvline(days[reopen_day], linestyle='dashed', color='tab:grey', label=days[reopen_day].strftime('%Y-%m-%d'))
	ax6.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax.axvline(days[lockdown_day], linestyle='dashed', color='tab:grey')
	ax2.axvline(days[lockdown_day], linestyle='dashed', color='tab:grey')
	ax5.axvline(days[lockdown_day], linestyle='dashed', color='tab:grey', label=days[lockdown_day].strftime('%Y-%m-%d'))
	ax6.axvline(days[lockdown_day], linestyle='dashed', color='tab:grey')
	ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
	ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
	ax.plot(days, [i / 1000 for i in G], label='G')
	ax2.plot(days, [i / 1000 for i in D], label='D')
	ax5.plot(days, betas, label='beta')
	ax6.plot(days, [i / 1000 for i in S], label='S')
	ax6.plot(days, [i / 1000 for i in H], label='H')
	ax6.plot(days, [i / 1000 for i in L], label='L')
	diff_G = pd.Series(np.diff(G))
	diff_confirmed = pd.Series(np.diff(confirmed))
	diff_D = pd.Series(np.diff(D))
	diff_death = pd.Series(np.diff(death))
	ax3.plot(days[-len(diff_confirmed):], [i / 1000 for i in diff_confirmed], label='daily new cases')
	ax3.plot(days[-len(diff_G):], [i / 1000 for i in diff_G], label='dG')
	ax4.plot(days[-len(diff_death):], [i / 1000 for i in diff_death], label='daily new deaths')
	ax4.plot(days[-len(diff_D):], [i / 1000 for i in diff_D], label='dD')
	ax.legend()
	ax2.legend()
	ax3.legend()
	ax4.legend()
	ax5.legend()
	ax6.legend()
	fig.autofmt_xdate()
	fig.savefig(f'{state_path}/sim.png', bbox_inches="tight")
	# fig.savefig(f'init_only_{end_date}/{state}/sim.png', bbox_inches="tight")
	plt.close(fig)
	return [S, E, I, A, IH, IN, D, R, G, H, L, betas]


def fit_all():
	t1 = time.perf_counter()
	matplotlib.use('Agg')
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


def extend_state(state, ConfirmFile, DeathFile, PopFile, ParaFile, release_size, peak_ratio, daily_speed):
	state_path = f'india/extended/{state}'
	if not os.path.exists(state_path):
		os.makedirs(state_path)
	df = pd.read_csv(ParaFile)
	beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, h, Hiding_init, c1, I_initial, metric1, metric2, r1, r2, reopen_date = \
		df.iloc[0]
	if release_size == 0:
		release_size = 1 - eta
	else:
		# release_size = min(release_size, 1 - eta)
		release_size = min(1 - eta, eta * Hiding_init * release_size)
	print(
		f'eta={round(eta, 3)} hiding={round(eta * Hiding_init, 3)} release={round(release_size, 3)} in {state_dict[state]}')

	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	dates = list(confirmed.columns)
	dates = dates[dates.index(start_date):dates.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates]
	confirmed = confirmed.iloc[0].loc[start_date: end_date]
	death = death.iloc[0].loc[start_date: end_date]
	reopen_day = days.index(datetime.datetime.strptime(reopen_date, '%Y-%m-%d'))

	d_confirmed = [confirmed[i] - confirmed[i - 1] for i in range(1, len(confirmed))]
	d_confirmed.insert(0, 0)
	d_death = [death[i] - death[i - 1] for i in range(1, len(death))]
	d_death.insert(0, 0)

	S = [n_0 * eta * (1 - Hiding_init)]
	E = [0]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [n_0 * eta * Hiding_init]
	# H = [0]
	size = len(days)
	days_ext = [days[0] + datetime.timedelta(days=i) for i in range(size + size_ext)]
	dates_ext = [d.strftime('%Y-%m-%d') for d in days_ext]

	result, [S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, betas0] \
		= simulate_combined(size + size_ext, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3,
		                    a1, a2, a3, h, Hiding_init, eta, c1, n_0, reopen_day)

	dG = [G0[i] - G0[i - 1] for i in range(1, len(G0))]
	dG.insert(0, 0)
	dD = [D0[i] - D0[i - 1] for i in range(1, len(D0))]
	dD.insert(0, 0)
	peak_dG = 0
	peak_day = 0
	for i in range(reopen_day, len(dG)):
		if peak_dG < dG[i]:
			peak_dG = dG[i]
			peak_day = i
	for i in range(peak_day, len(dG)):
		if dG[i] <= peak_ratio * peak_dG:
			release_day = i
			release_date = dates_ext[i]
			break
	# release_day = max(release_day, dates_ext.index('2021-06-01'))
	release_day = dates_ext.index('2021-06-01')

	S = [n_0 * eta * (1 - Hiding_init)]
	E = [0]
	I = [n_0 * eta * I_initial * (1 - alpha)]
	A = [n_0 * eta * I_initial * alpha]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [n_0 * eta * Hiding_init]

	result, [S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1, HH1, betas1] \
		= simulate_release(size + size_ext, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3,
		                   a1, a2, a3, h, Hiding_init, eta, c1, n_0, reopen_day, release_day, release_size, daily_speed)

	dG2 = [G1[i] - G1[i - 1] for i in range(1, len(G1))]
	dG2.insert(0, 0)
	dD2 = [D1[i] - D1[i - 1] for i in range(1, len(D1))]
	dD2.insert(0, 0)

	# fig = plt.figure(figsize=(20, 20))
	fig = plt.figure(figsize=(9, 8))
	# fig.suptitle(f'{state} eta={round(eta, 3)}')
	fig.suptitle(state_dict[state])
	# ax = fig.add_subplot(421)
	# ax.set_title('Confirmed')
	# ax2 = fig.add_subplot(422)
	# ax2.set_title('Death')
	# ax3 = fig.add_subplot(423)
	# ax3.set_title('Infected')
	# ax4 = fig.add_subplot(424)
	# ax4.set_title('Asymptomatic')
	# ax5 = fig.add_subplot(425)
	# ax5.set_title('Beta')
	# ax6 = fig.add_subplot(426)
	# ax6.set_title('Susceptible')
	# ax7 = fig.add_subplot(427)
	# ax7.set_title('New Cases')
	# ax8 = fig.add_subplot(428)
	# ax8.set_title('New Deaths')

	ax = fig.add_subplot(221)
	ax.set_title('Confirmed Cases')
	ax2 = fig.add_subplot(222)
	ax2.set_title('Deaths')
	ax7 = fig.add_subplot(223)
	ax7.set_title('Daily New Cases')
	ax8 = fig.add_subplot(224)
	ax8.set_title('Daily New Deaths')

	# ax.axvline(days_ext[reopen_day], linestyle='dashed', color='tab:grey', label=reopen_date)
	# ax2.axvline(days_ext[reopen_day], linestyle='dashed', color='tab:grey', label=reopen_date)
	# ax3.axvline(days_ext[reopen_day], linestyle='dashed', color='tab:grey', label=reopen_date)
	# ax4.axvline(days_ext[reopen_day], linestyle='dashed', color='tab:grey', label=reopen_date)
	# ax5.axvline(days_ext[reopen_day], linestyle='dashed', color='tab:grey', label=reopen_date)
	# ax6.axvline(days_ext[reopen_day], linestyle='dashed', color='tab:grey', label=reopen_date)
	# ax7.axvline(days_ext[reopen_day], linestyle='dashed', color='tab:grey', label=reopen_date)
	# ax8.axvline(days_ext[reopen_day], linestyle='dashed', color='tab:grey', label=reopen_date)
	# ax.axvline(days_ext[release_day], linestyle='dashed', color='tab:red', label=release_date)
	# ax2.axvline(days_ext[release_day], linestyle='dashed', color='tab:red', label=release_date)
	# ax3.axvline(days_ext[release_day], linestyle='dashed', color='tab:red', label=release_date)
	# ax4.axvline(days_ext[release_day], linestyle='dashed', color='tab:red', label=release_date)
	# ax5.axvline(days_ext[release_day], linestyle='dashed', color='tab:red', label=release_date)
	# ax6.axvline(days_ext[release_day], linestyle='dashed', color='tab:red', label=release_date)
	# ax7.axvline(days_ext[release_day], linestyle='dashed', color='tab:red', label=release_date)
	# ax8.axvline(days_ext[release_day], linestyle='dashed', color='tab:red', label=release_date)

	ax.axvline(days_ext[reopen_day], linestyle='dashed', color='tab:grey')
	ax2.axvline(days_ext[reopen_day], linestyle='dashed', color='tab:grey')
	ax7.axvline(days_ext[reopen_day], linestyle='dashed', color='tab:grey')
	ax8.axvline(days_ext[reopen_day], linestyle='dashed', color='tab:grey')
	ax.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')
	ax2.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')
	ax7.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')
	ax8.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')

	ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Reported", alpha=0.6)
	ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Reported", alpha=0.6)
	# ax.scatter(days, [i / 1000 for i in confirmed], s=20, linewidth=0, alpha=1, label="Reported\nCases")
	# ax2.scatter(days, [i / 1000 for i in death], s=20, linewidth=0, alpha=1, label="Reported\nDeaths")

	ax.plot(days_ext, [i / 1000 for i in G1], label='25% release', color='orangered')
	ax2.plot(days_ext, [i / 1000 for i in D1], label='25% release', color='orangered')
	ax.plot(days_ext, [i / 1000 for i in G0], label='Original\nProjection', color='green')
	ax2.plot(days_ext, [i / 1000 for i in D0], label='Original\nProjection', color='green')

	# ax3.plot(days_ext, [i / 1000 for i in I0], label="I")
	# ax3.plot(days_ext, [i / 1000 for i in I1], label="I2")

	# ax4.plot(days_ext, [i / 1000 for i in A0], label="A")
	# ax4.plot(days_ext, [i / 1000 for i in A1], label="A2")

	# ax5.plot(days_ext, betas0, label="beta")
	# ax5.plot(days_ext, betas1, label="beta2")

	# ax6.plot(days_ext, [i / 1000 for i in S0], label="S")
	# ax6.plot(days_ext, [i / 1000 for i in S1], label="S2")
	# ax6.plot(days_ext, [i / 1000 for i in H0], label="H")
	# ax6.plot(days_ext, [i / 1000 for i in HH1], label="HH")

	# ax7.plot(days_ext[1:len(d_confirmed)], [i / 1000 for i in d_confirmed[1:]], linewidth=5, linestyle=':',
	#          label="Reported\nCases", alpha=0.6)
	# ax8.plot(days_ext[1:len(d_death)], [i / 1000 for i in d_death[1:]], linewidth=5, linestyle=':',
	#          label="Reported\nDeaths", alpha=0.6)
	ax7.scatter(days_ext[1:len(d_confirmed)], [i / 1000 for i in d_confirmed[1:]], s=15, linewidth=0, alpha=0.6,
	            label="Reported")
	ax8.scatter(days_ext[1:len(d_death)], [i / 1000 for i in d_death[1:]], s=15, linewidth=0, alpha=0.6,
	            label="Reported")
	ax7.plot(days_ext[1:], [i / 1000 for i in dG2[1:]], label="25% release", color='orangered')
	ax7.plot(days_ext[1:], [i / 1000 for i in dG[1:]], label="Original\nProjection", color='green')
	ax8.plot(days_ext[1:], [i / 1000 for i in dD2[1:]], label="25% release", color='orangered')
	ax8.plot(days_ext[1:], [i / 1000 for i in dD[1:]], label="Original\nProjection", color='green')

	ax.legend()
	ax2.legend()
	# ax3.legend()
	# ax4.legend()
	# ax5.legend()
	# ax6.legend()
	ax7.legend()
	ax8.legend()

	fig.autofmt_xdate()
	fig.savefig(f'{state_path}/extended_{state}.png', bbox_inches="tight")
	# plt.show()
	plt.close(fig)

	data = [S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, betas0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1, HH1, betas1]
	c0 = ['S', 'E', 'I', 'A', 'IH', 'IN', 'D', 'R', 'G', 'H', 'betas', 'S2', 'E2', 'I2', 'A2', 'IH2', 'IN2', 'D2', 'R2',
	      'G2', 'H2', 'HH2', 'betas2']
	df = pd.DataFrame(data, columns=dates_ext)
	df.insert(0, 'series', c0)
	df.to_csv(f'{state_path}/sim.csv', index=False)
	return state, confirmed, death, G0, D0, G1, D1, release_day


def extend_india(confirmed, death, G0, D0, G1, D1, release_day):
	d_confirmed = [confirmed[i] - confirmed[i - 1] for i in range(1, len(confirmed))]
	d_confirmed.insert(0, 0)
	d_death = [death[i] - death[i - 1] for i in range(1, len(death))]
	d_death.insert(0, 0)

	dG = [G0[i] - G0[i - 1] for i in range(1, len(G0))]
	dG.insert(0, 0)
	dD = [D0[i] - D0[i - 1] for i in range(1, len(D0))]
	dD.insert(0, 0)
	dG2 = [G1[i] - G1[i - 1] for i in range(1, len(G1))]
	dG2.insert(0, 0)
	dD2 = [D1[i] - D1[i - 1] for i in range(1, len(D1))]
	dD2.insert(0, 0)

	fig = plt.figure(figsize=(9, 8))
	fig.suptitle('India')
	ax = fig.add_subplot(221)
	ax.set_title('Confirmed Cases')
	ax2 = fig.add_subplot(222)
	ax2.set_title('Deaths')
	ax7 = fig.add_subplot(223)
	ax7.set_title('Daily New Cases')
	ax8 = fig.add_subplot(224)
	ax8.set_title('Daily New Deaths')

	days_ext = [datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=i) for i in range(len(G0))]

	ax.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')
	ax2.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')
	ax7.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')
	ax8.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')

	ax.plot(days_ext[:len(confirmed)], [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Reported",
	        alpha=0.6)
	ax2.plot(days_ext[:len(death)], [i / 1000 for i in death], linewidth=5, linestyle=':', label="Reported", alpha=0.6)
	# ax.scatter(days_ext[:len(confirmed)], [i / 1000 for i in confirmed], s=20, linewidth=0, alpha=1,
	#            label="Reported\nCases")
	# ax2.scatter(days_ext[:len(death)], [i / 1000 for i in death], s=20, linewidth=0, alpha=1, label="Reported\nDeaths")

	ax.plot(days_ext, [i / 1000 for i in G1], label='25% release', color='orangered')
	ax2.plot(days_ext, [i / 1000 for i in D1], label='25% release', color='orangered')
	ax.plot(days_ext, [i / 1000 for i in G0], label='Original\nProjection', color='green')
	ax2.plot(days_ext, [i / 1000 for i in D0], label='Original\nProjection', color='green')

	# ax7.plot(days_ext[1:len(d_confirmed)], [i / 1000 for i in d_confirmed[1:]], linewidth=5, linestyle=':',
	#          label="Reported\nCases", alpha=0.6)
	# ax8.plot(days_ext[1:len(d_death)], [i / 1000 for i in d_death[1:]], linewidth=5, linestyle=':',
	#          label="Reported\nDeaths", alpha=0.6)
	ax7.scatter(days_ext[1:len(d_confirmed)], [i / 1000 for i in d_confirmed[1:]], s=15, linewidth=0, alpha=0.6,
	            label="Reported")
	ax8.scatter(days_ext[1:len(d_death)], [i / 1000 for i in d_death[1:]], s=15, linewidth=0, alpha=0.6,
	            label="Reported")

	ax7.plot(days_ext[1:], [i / 1000 for i in dG2[1:]], label="25% release", color='orangered')
	ax7.plot(days_ext[1:], [i / 1000 for i in dG[1:]], label="Original\nProjection", color='green')
	ax8.plot(days_ext[1:], [i / 1000 for i in dD2[1:]], label="25% release", color='orangered')
	ax8.plot(days_ext[1:], [i / 1000 for i in dD[1:]], label="Original\nProjection", color='green')

	ax.legend()
	ax2.legend()
	ax7.legend()
	ax8.legend()
	fig.autofmt_xdate()
	fig.savefig(f'india/extended/extended_india.png', bbox_inches="tight")
	# plt.show()
	plt.close(fig)
	return


def extend_all():
	India_G0 = []
	India_D0 = []
	India_G1 = []
	India_D1 = []
	India_confirmed = []
	India_death = []
	with concurrent.futures.ProcessPoolExecutor() as executor:
		results = [executor.submit(extend_state, state, 'india/indian_cases_confirmed_cases.csv',
		                           'india/indian_cases_confirmed_deaths.csv', 'india/state_population.csv',
		                           f'india/fittingV3_{end_date}/{state}/para.csv', 1 / 1, 0.5, 1 / 1) for state in
		           states]

		for f in concurrent.futures.as_completed(results):
			state, confirmed, death, G0, D0, G1, D1, release_day = f.result()
			India_release_day = release_day
			if len(India_G0) == 0:
				India_G0 = G0.copy()
				India_D0 = D0.copy()
				India_G1 = G1.copy()
				India_D1 = D1.copy()
				India_confirmed = confirmed.copy()
				India_death = death.copy()
			else:
				India_G0 = [India_G0[i] + G0[i] for i in range(len(G0))]
				India_D0 = [India_D0[i] + D0[i] for i in range(len(G0))]
				India_G1 = [India_G1[i] + G1[i] for i in range(len(G0))]
				India_D1 = [India_D1[i] + D1[i] for i in range(len(G0))]
				India_confirmed = [India_confirmed[i] + confirmed[i] for i in range(len(confirmed))]
				India_death = [India_death[i] + death[i] for i in range(len(death))]
	# print(state, 'finished')

	extend_india(India_confirmed, India_death, India_G0, India_D0, India_G1, India_D1, India_release_day)
	return


def main():
	fit_all()
	# extend_all()
	# tmp()
	return


if __name__ == '__main__':
	main()
