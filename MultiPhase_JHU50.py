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
from SIRfunctions import SIRG_combined, weighted_deviation, weighted_relative_deviation
import datetime
from numpy.random import uniform as uni
import os
import warnings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# matplotlib.use('Agg')

np.set_printoptions(threshold=sys.maxsize)
Geo = 0.98

num_threads = 100
# num_threads = 5
num_CI = 400
num_per_interval = 10
# num_CI = 200
# num_per_interval = 20
# num_CI = 5
start_dev = 25

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

interval_100 = False

fig_row = 5
fig_col = 3

state_reopens = {'AL': '2020-05-11', 'AZ': '2020-05-15', 'CA': '2020-05-26', 'CO': '2020-05-27', 'CT': '2020-06-17',
                 'DC': '2020-06-22', 'DE': '2020-06-19', 'KY': '2020-06-29', 'FL': '2020-06-03', 'GA': '2020-06-16',
                 'IA': '2020-05-28', 'IL': '2020-05-28', 'IN': '2020-05-21', 'LA': '2020-06-05', 'MA': '2020-06-22',
                 'MD': '2020-06-12', 'MI': '2020-06-08', 'MN': '2020-06-10', 'MO': '2020-06-16', 'NC': '2020-05-22',
                 'NE': '2020-06-22', 'NJ': '2020-06-22', 'NV': '2020-05-29', 'NY': '2020-07-06', 'OH': '2020-05-21',
                 'OK': '2020-06-01', 'PA': '2020-06-26', 'RI': '2020-06-30', 'SC': '2020-05-11', 'SD': '2020-06-01',
                 'TN': '2020-05-22', 'TX': '2020-06-03', 'UT': '2020-05-01', 'VA': '2020-07-01', 'WA': '2020-06-08',
                 'WI': '2020-07-01'}

counties_global = ['AL-Jefferson',
                   'AL-Mobile',
                   'AZ-Maricopa',
                   'AZ-Pima',
                   'AZ-Yuma',
                   'CA-Alameda',
                   'CA-Contra Costa',
                   'CA-Fresno',
                   'CA-Kern',
                   'CA-Los Angeles',
                   'CA-Orange',
                   'CA-Riverside',
                   'CA-Sacramento',
                   'CA-San Bernardino',
                   'CA-San Diego',
                   'CA-San Joaquin',
                   'CA-Santa Clara',
                   'CA-Stanislaus',
                   'CA-Tulare',
                   'CO-Adams',
                   'CO-Arapahoe',
                   'CO-Denver',
                   'CT-Fairfield',
                   'CT-Hartford',
                   'CT-New Haven',
                   'DE-New Castle',
                   'DE-Sussex',
                   'DC-District of Columbia',
                   'FL-Broward',
                   'FL-Duval',
                   'FL-Hillsborough',
                   'FL-Lee',
                   'FL-Miami-Dade',
                   'FL-Orange',
                   'FL-Palm Beach',
                   'FL-Pinellas',
                   'FL-Polk',
                   'GA-Cobb',
                   'GA-DeKalb',
                   'GA-Fulton',
                   'GA-Gwinnett',
                   'IL-Cook',
                   'IL-DuPage',
                   'IL-Kane',
                   'IL-Lake',
                   'IL-Will',
                   'IN-Lake',
                   'IN-Marion',
                   'IA-Polk',
                   'KY-Jefferson',
                   'LA-East Baton Rouge',
                   'LA-Jefferson',
                   'LA-Orleans',
                   'MD-Anne Arundel',
                   'MD-Baltimore',
                   'MD-Baltimore City',
                   'MD-Montgomery',
                   'MD-Prince George\'s',
                   'MA-Bristol',
                   'MA-Essex',
                   'MA-Hampden',
                   'MA-Middlesex',
                   'MA-Norfolk',
                   'MA-Plymouth',
                   'MA-Suffolk',
                   'MA-Worcester',
                   'MI-Kent',
                   'MI-Macomb',
                   'MI-Oakland',
                   'MI-Wayne',
                   'MN-Hennepin',
                   'MO-St. Louis',
                   'NE-Douglas',
                   'NV-Clark',
                   'NJ-Bergen',
                   'NJ-Burlington',
                   'NJ-Camden',
                   'NJ-Essex',
                   'NJ-Hudson',
                   'NJ-Mercer',
                   'NJ-Middlesex',
                   'NJ-Monmouth',
                   'NJ-Morris',
                   'NJ-Ocean',
                   'NJ-Passaic',
                   'NJ-Somerset',
                   'NJ-Union',
                   'NY-Bronx',
                   'NY-Dutchess',
                   'NY-Erie',
                   'NY-Kings',
                   'NY-Nassau',
                   'NY-New York',
                   'NY-Orange',
                   'NY-Queens',
                   'NY-Richmond',
                   'NY-Rockland',
                   'NY-Suffolk',
                   'NY-Westchester',
                   'NC-Mecklenburg',
                   'NC-Wake',
                   'OH-Cuyahoga',
                   'OH-Franklin',
                   'OK-Oklahoma',
                   'OK-Tulsa',
                   'PA-Berks',
                   'PA-Bucks',
                   'PA-Delaware',
                   'PA-Lehigh',
                   'PA-Luzerne',
                   'PA-Montgomery',
                   'PA-Northampton',
                   'PA-Philadelphia',
                   'RI-Providence',
                   'SC-Charleston',
                   'SC-Greenville',
                   'SD-Minnehaha',
                   'TN-Davidson',
                   'TN-Shelby',
                   'TX-Bexar',
                   'TX-Cameron',
                   'TX-Dallas',
                   'TX-El Paso',
                   'TX-Fort Bend',
                   'TX-Harris',
                   'TX-Hidalgo',
                   'TX-Nueces',
                   'TX-Tarrant',
                   'TX-Travis',
                   'UT-Salt Lake',
                   'VA-Fairfax',
                   'VA-Prince William',
                   'WA-King',
                   'WA-Snohomish',
                   'WI-Milwaukee',
                   ]


# save simulation of SIRG fitting to csv
def save_sim_combined(data, days, state_path):
	days = [day.strftime('%Y-%m-%d') for day in days]
	c0 = ['S', 'I', 'IH', 'IN', 'D', 'R', 'G', 'H', 'beta']
	df = pd.DataFrame(data, columns=days)
	df.insert(0, 'series', c0)
	df.to_csv(f'{state_path}/sim.csv', index=False)
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
def save_para_combined(state, paras, state_path):
	para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1', 'metric1',
	              'metric2', 'r1', 'r2', 'reopen']
	df = pd.DataFrame(paras, columns=para_label)
	df.to_csv(f'{state_path}/para.csv', index=False, header=True)
	# df.to_csv(f'init_only_{end_date}/{state}/para.csv', index=False, header=True)
	print('parameters saved\n')


# save the parameters distribution to CSV for initial phase only
def save_para_init(state, paras):
	para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1', 'metric1',
	              'metric2', 'metric3', 'r1', 'r2', 'r3']
	df = pd.DataFrame(paras, columns=para_label)
	df.to_csv(f'JHU/init_only_{end_date}/{state}/para.csv', index=False, header=True)
	# df.to_csv(f'init_only_{end_date}/{state}/para.csv', index=False, header=True)
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
def loss_combined(point, c1, confirmed, death, n_0, SIRG, reopen_day):
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
	IH = [0]
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

	metric0 = r2_score(weighted_confirmed, weighted_G)
	metric1 = r2_score(weighted_death, weighted_D)
	# return -(theta * metric0 + 1 * (1 - theta) / 2 * metric1 + 1 * (1 - theta) / 2 * metric2)
	return -(0.5 * metric0 + 0.5 * metric1)


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


def fit_combined(confirmed0, death0, days, reopen_day_gov, n_0, metric1, metric2):
	np.random.seed()
	confirmed = confirmed0.copy()
	death = death0.copy()
	size = len(confirmed)
	if metric2 != 0 or metric1 != 0:
		scale1 = pd.Series(np.random.normal(1, metric1, size))
		confirmed = [max(confirmed[i] * scale1[i], 1) for i in range(size)]
		scale2 = pd.Series(np.random.normal(1, metric2, size))
		death = [max(death[i] * scale2[i], 1) for i in range(size)]
	c_max = 0
	min_loss = 10000
	for reopen_day in range(reopen_day_gov, reopen_day_gov + 14):
		for c1 in np.arange(c1_range[0], c1_range[1], 0.01):
			# optimal = minimize(loss, [10, 0.05, 0.01, 0.1, 0.1, 0.1, 0.02], args=(c1, confirmed, death, n_0, SIDRG_sd),
			optimal = minimize(loss_combined, [uni(beta_range[0], beta_range[1]),
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
			                   args=(c1, confirmed, death, n_0, SIRG_combined, reopen_day), method='L-BFGS-B',
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
			current_loss = loss_combined(optimal.x, c1, confirmed, death, n_0, SIRG_combined, reopen_day)
			if current_loss < min_loss:
				# print(f'updating loss={current_loss} with c1={c1}')
				min_loss = current_loss
				c_max = c1
				reopen_max = reopen_day
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
	reopen_day = reopen_max
	S = [n_0 * eta]
	I = [confirmed[0]]
	IH = [0]
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

	r1 = r2_score(confirmed, G)
	r2 = r2_score(death, D)

	return [beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, r1, r2,
	        reopen_day], min_loss


def fit_CI_combined(confirmed0, death0, reopen_day, n_0):
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
	for c1 in np.arange(c1_range[0], c1_range[1], 0.01):
		# optimal = minimize(loss, [10, 0.05, 0.01, 0.1, 0.1, 0.1, 0.02], args=(c1, confirmed, death, n_0, SIDRG_sd),
		optimal = minimize(loss_combined, [uni(beta_range[0], beta_range[1]),
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
		                   args=(c1, confirmed, death, n_0, SIRG_combined, reopen_day), method='L-BFGS-B',
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
		current_loss = loss_combined(optimal.x, c1, confirmed, death, n_0, SIRG_combined, reopen_day)
		if current_loss < min_loss:
			# print(f'updating loss={current_loss} with c1={c1}')
			min_loss = current_loss
			c_max = c1
			reopen_max = reopen_day
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
	reopen_day = reopen_max
	S = [n_0 * eta]
	I = [confirmed[0]]
	IH = [0]
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

	r1 = r2_score(confirmed, G)
	r2 = r2_score(death, D)

	return [beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, r1, r2,
	        reopen_day], min_loss


def MT_CI_combined(confirmed0, death0, reopen_day, n_0, metric1, metric2):
	np.random.seed()
	confirmed = confirmed0.copy()
	death = death0.copy()
	size = len(confirmed)
	if metric2 != 0 or metric1 != 0:
		# scale1 = pd.Series(np.random.normal(1, metric1, size))
		# confirmed = [max(confirmed[i] * scale1[i], 0.01) for i in range(size)]
		# scale2 = pd.Series(np.random.normal(1, metric2, size))
		# death = [max(death[i] * scale2[i], 0.01) for i in range(size)]
		scale1 = pd.Series(np.random.normal(1, metric1, size))
		confirmed = [confirmed[i] * scale1[i] for i in range(size)]
		scale2 = pd.Series(np.random.normal(1, metric2, size))
		death = [death[i] * scale2[i] for i in range(size)]
	confirmed[0] = confirmed0[0]
	death[0] = death0[0]
	min_loss = 10000

	with concurrent.futures.ProcessPoolExecutor() as executor:
		results = [executor.submit(fit_CI_combined, confirmed, death, reopen_day, n_0) for _ in
		           range(num_per_interval)]

		for f in concurrent.futures.as_completed(results):
			para, current_loss = f.result()
			if current_loss < min_loss:
				min_loss = current_loss
				para_best = para

	return para_best, min_loss


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
		                   args=(c1, confirmed, death, hosp, n_0, SIRG_combined, reopen_day), method='L-BFGS-B',
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
		current_loss = loss_init(optimal.x, c1, confirmed, death, hosp, n_0, SIRG_combined, reopen_day)
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
	state_path = f'JHU50/combined2W_{end_date}/{state}'
	if not os.path.exists(state_path):
		os.makedirs(state_path)

	# # add the delay in dates
	# for i in range(len(dates) - 1):
	# 	date = datetime.datetime.strptime(dates[i], '%Y-%m-%d')
	# 	date += datetime.timedelta(days=delay)
	# 	dates[i] = date.strftime('%Y-%m-%d')
	print(state)
	print(dates)
	print()

	# read population
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

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

	# # select hospitalization data
	# hosp = readHosp(state)
	# hosp = hosp[:dates[1]].tolist()
	# diff_len = len(days) - len(hosp)
	# if diff_len > 0:
	# 	hosp = [0] * diff_len.extend(hosp)
	# else:
	# 	hosp = hosp[-len(days):]
	# hosp = [0.01 if h == 0 else h for h in hosp]

	reopen_day_gov = days.index(datetime.datetime.strptime(dates[0], '%Y-%m-%d'))

	# fitting
	para = MT_combined(confirmed, death, n_0, days, reopen_day_gov)
	[S, I, IH, IN, D, R, G, H, betas] = plot_combined(state, confirmed, death, days, n_0, reopen_day_gov, para,
	                                                  state_path)
	csv_file = f'{state_path}/sim.csv'
	# csv_file = f'init_only_{end_date}/{state}/sim.csv'
	save_sim_combined([S, I, IH, IN, D, R, G, H, betas], days, state_path)
	para[-1] = days[para[-1]]
	save_para_combined(state, [para], state_path)
	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds in total for {state}\n')

	return


# fit with SD for initial phase only
def fit_state_init(state, ConfirmFile, DeathFile, PopFile, dates):
	t1 = time.perf_counter()

	# if not os.path.exists(f'JHU/combined2W_{end_date}/{state}'):
	# 	os.makedirs(f'JHU/combined2W_{end_date}/{state}')
	if not os.path.exists(f'JHU/init_only_{end_date}/{state}'):
		os.makedirs(f'JHU/init_only_{end_date}/{state}')

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
	csv_file = f'JHU/init_only_{end_date}/{state}/sim.csv'
	save_sim_init(csv_file, [S, I, IH, IN, D, R, G, H, betas], days)
	save_para_init(state, [para])
	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds in total for {state}\n')

	return


# plot result
def plot_combined(state, confirmed, death, days, n_0, reopen_day_gov, para, state_path):
	[beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, r1, r2, reopen_day] = para
	para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1', 'metric1',
	              'metric2', 'r1', 'r2', 'reopen']
	for i in range(len(para)):
		print(f'{para_label[i]}={para[i]} ', end=' ')
		if i % 4 == 1:
			print()

	S = [n_0 * eta]
	I = [confirmed[0]]
	IH = [0]
	IN = [I[-1] * gamma2]
	D = [death[0]]
	R = [0]
	G = [confirmed[0]]
	H = [Hiding_init * n_0 * eta]
	size = len(days)
	result, [S, I, IH, IN, D, R, G, H, betas] \
		= simulate_combined(size, SIRG_combined, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2,
		                    eta, c1, n_0, reopen_day)

	fig = plt.figure(figsize=(6, 14))
	# fig.suptitle(state)
	ax = fig.add_subplot(411)
	ax.set_title(state)
	ax2 = fig.add_subplot(412)
	ax3 = fig.add_subplot(413)
	ax4 = fig.add_subplot(414)
	ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax2.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax3.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax4.axvline(days[reopen_day], linestyle='dashed', color='tab:grey', label=days[reopen_day].strftime('%Y-%m-%d'))
	ax.axvline(days[reopen_day_gov], linestyle='dashed', color='tab:grey', alpha=0.5)
	ax2.axvline(days[reopen_day_gov], linestyle='dashed', color='tab:grey', alpha=0.5)
	ax4.axvline(days[reopen_day_gov], linestyle='dashed', color='tab:grey', alpha=0.5,
	            label=days[reopen_day_gov].strftime('%Y-%m-%d'))
	ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
	ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
	ax.plot(days, [i / 1000 for i in G], label='G')
	ax2.plot(days, [i / 1000 for i in D], label='D')
	ax4.plot(days, betas, label='beta')
	ax3.plot(days, [i / 1000 for i in H], label='H')
	ax.legend()
	ax2.legend()
	ax3.legend()
	ax4.legend()
	fig.autofmt_xdate()
	fig.savefig(f'{state_path}/sim.png', bbox_inches="tight")
	# fig.savefig(f'init_only_{end_date}/{state}/sim.png', bbox_inches="tight")
	plt.close(fig)
	return [S, I, IH, IN, D, R, G, H, betas]


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
	fig.savefig(f'JHU/init_only_{end_date}/{state}/sim.png', bbox_inches="tight")
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
def MT_combined(confirmed, death, n_0, days, reopen_day_gov):
	para_best = []
	min_loss = 10000
	with concurrent.futures.ProcessPoolExecutor() as executor:
		t1 = time.perf_counter()
		results = [executor.submit(fit_combined, confirmed, death, days, reopen_day_gov, n_0, 0, 0) for _ in
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
def CI_combined(confirmed, death, n_0, days, reopen_day, metric1, metric2):
	paras = []
	with concurrent.futures.ProcessPoolExecutor() as executor:
		t1 = time.perf_counter()
		results = [
			executor.submit(MT_CI_combined, confirmed, death, reopen_day, n_0, metric1, metric2) for
			_ in range(num_CI)]

		threads = 0
		for f in concurrent.futures.as_completed(results):
			para, current_loss = f.result()
			threads += 1
			if threads % (num_CI // 10) == 0:
				t2 = time.perf_counter()
				print(f'{threads}/{num_CI} threads in {round((t2 - t1) / 60, 3)} minutes')
			para[-1] = days[para[-1]]
			paras.append(para)

		t2 = time.perf_counter()
		print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_CI, 3)} seconds per job')

	print('confidence interval fitting completed\n')
	return paras


def fit_all_combined():
	t1 = time.perf_counter()

	states = ['AL-Jefferson',
	          'AL-Mobile',
	          'AZ-Pima',
	          'AZ-Yuma',
	          'CA-Alameda',
	          'CA-Contra Costa',
	          'CA-Fresno',
	          'CA-Kern',
	          'CA-Sacramento',
	          'CA-San Joaquin',
	          'CA-Santa Clara',
	          'CA-Stanislaus',
	          'CA-Tulare',
	          'FL-Duval',
	          'FL-Hillsborough',
	          'FL-Lee',
	          'FL-Orange',
	          'FL-Pinellas']

	states = ['FL-Polk',
	          'GA-Cobb',
	          'KY-Jefferson',
	          'NE-Douglas',
	          'NC-Mecklenburg',
	          'NC-Wake',
	          'OK-Oklahoma',
	          'OK-Tulsa',
	          'SC-Charleston',
	          'SC-Greenville',
	          'TX-Bexar',
	          'TX-Cameron',
	          'TX-El Paso',
	          'TX-Fort Bend',
	          'TX-Hidalgo',
	          'TX-Nueces',
	          'TX-Travis']

	for state in states:
		reopen_date = state_reopens[state[:2]]
		dates = [reopen_date, end_date]
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
def CI_state_combined(state, ConfirmFile, DeathFile, PopFile):
	t1 = time.perf_counter()
	SimFile = f'JHU50/combined2W_{end_date}/{state}/sim.csv'
	ParaFile = f'JHU50/combined2W_{end_date}/{state}/para.csv'
	df = pd.read_csv(ParaFile)
	reopen_date = df.iloc[0]['reopen']
	# state_path = f'JHU50/CI_{end_date}/{state}'
	state_path = f'JHU50/CI_{num_CI}by{num_per_interval}_/{state}'
	if not os.path.exists(state_path):
		os.makedirs(state_path)

	print(state)
	print(reopen_date)
	print()

	# read population
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

	# select useful portion of simulation
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	for start_date in confirmed.columns[1:]:
		# if confirmed.iloc[0].loc[start_date] >= I_0 and death.iloc[0].loc[start_date] > 0:
		#     break
		if confirmed.iloc[0].loc[start_date] >= I_0:
			break
	confirmed = confirmed.iloc[0].loc[start_date:end_date]
	death = death.iloc[0].loc[start_date:end_date]

	for i in range(len(death)):
		if death.iloc[i] == 0:
			death.iloc[i] = 0.01
	death = death.tolist()

	# confirmed = confirmed[start_date:]
	# death = death[start_date:]
	# print(death)
	days = confirmed.index.tolist()

	df_sim = pd.read_csv(SimFile)
	G = df_sim[df_sim['series'] == 'G'].iloc[0].loc[start_date:]
	D = df_sim[df_sim['series'] == 'D'].iloc[0].loc[start_date:]

	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	reopen_day = days.index(datetime.datetime.strptime(reopen_date, '%Y-%m-%d'))

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
	# data1 = [(confirmed[i] - G[i]) / G[i] for i in range(size)]
	# data2 = [(death[i] - D[i]) / D[i] for i in range(size)]
	# data1 = [(confirmed[i] - G[i]) for i in range(size)]
	# data2 = [(death[i] - D[i]) for i in range(size)]
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
	# fitting
	paras = CI_combined(G, D, n_0, days, reopen_day, metric1, metric2)
	para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1', 'metric1',
	              'metric2', 'r1', 'r2', 'reopen']
	# paras[-1] = days[paras[-1]]
	out_df = pd.DataFrame(paras, columns=para_label)
	out_df.to_csv(f'{state_path}/paraCI.csv', index=False)
	t2 = time.perf_counter()
	print(f'{round((t2 - t1) / 60, 3)} minutes in total for {state}\n')

	return 0


# add deviations to input series and fit to generate new estimates for all parameters for all states
def CI_all_combined():
	t1 = time.perf_counter()

	states = counties_global
	states = ['NY-Westchester']

	for state in states:
		CI_state_combined(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
		                  'JHU/CountyPopulation.csv')

	t2 = time.perf_counter()
	print(f'{round((t2 - t1) / 60, 3)} minutes for all counties')

	return


def test():
	full_counties_init = ['Cook, Illinois, US',
	                      'Queens, New York, US',
	                      'Kings, New York, US',
	                      'Bronx, New York, US',
	                      'Nassau, New York, US',
	                      'Suffolk, New York, US',
	                      'Los Angeles, California, US',
	                      'Westchester, New York, US',
	                      'New York, New York, US',
	                      'Wayne, Michigan, US',
	                      'Philadelphia, Pennsylvania, US',
	                      'Middlesex, Massachusetts, US',
	                      'Hudson, New Jersey, US',
	                      'Bergen, New Jersey, US',
	                      'Suffolk, Massachusetts, US',
	                      'Essex, New Jersey, US',
	                      'Miami-Dade, Florida, US',
	                      'Passaic, New Jersey, US',
	                      'Union, New Jersey, US',
	                      'Middlesex, New Jersey, US',
	                      'Fairfield, Connecticut, US',
	                      'Richmond, New York, US',
	                      'Rockland, New York, US',
	                      'Essex, Massachusetts, US',
	                      'Prince George\'s, Maryland, US',
	                      'New Haven, Connecticut, US',
	                      'Orange, New York, US',
	                      'Oakland, Michigan, US',
	                      'Worcester, Massachusetts, US',
	                      'Providence, Rhode Island, US',
	                      'Harris, Texas, US',
	                      'Hartford, Connecticut, US',
	                      'Marion, Indiana, US',
	                      'Ocean, New Jersey, US',
	                      'Montgomery, Maryland, US',
	                      'Norfolk, Massachusetts, US',
	                      'King, Washington, US',
	                      'Monmouth, New Jersey, US',
	                      'Plymouth, Massachusetts, US',
	                      'Fairfax, Virginia, US',
	                      'Dallas, Texas, US',
	                      'Jefferson, Louisiana, US',
	                      'District of Columbia,District of Columbia,US',
	                      'Maricopa, Arizona, US',
	                      'Orleans, Louisiana, US',
	                      'Macomb, Michigan, US',
	                      'Lake, Illinois, US',
	                      'Broward, Florida, US',
	                      'Bristol, Massachusetts, US',
	                      'Morris, New Jersey, US',
	                      'Montgomery, Pennsylvania, US',
	                      'Mercer, New Jersey, US',
	                      'DuPage, Illinois, US',
	                      'Riverside, California, US',
	                      'Delaware, Pennsylvania, US',
	                      'San Diego, California, US',
	                      'Hampden, Massachusetts, US',
	                      'Camden, New Jersey, US',
	                      'Clark, Nevada, US',
	                      'Erie, New York, US',
	                      'Hennepin, Minnesota, US',
	                      'Milwaukee, Wisconsin, US',
	                      'Denver, Colorado, US',
	                      'Baltimore, Maryland, US',
	                      'Palm Beach, Florida, US',
	                      'Franklin, Ohio, US',
	                      'Bucks, Pennsylvania, US',
	                      'St. Louis, Missouri, US',
	                      'Will, Illinois, US',
	                      'Tarrant, Texas, US',
	                      'Somerset, New Jersey, US',
	                      'Kane, Illinois, US',
	                      'Orange, California, US',
	                      'Burlington, New Jersey, US',
	                      'Davidson, Tennessee, US',
	                      'Salt Lake, Utah, US',
	                      'Fulton, Georgia, US',
	                      'Baltimore City, Maryland, US',
	                      'Shelby, Tennessee, US',
	                      'Berks, Pennsylvania, US',
	                      'Arapahoe, Colorado, US',
	                      'Sussex, Delaware, US',
	                      'Dutchess, New York, US',
	                      'Prince William, Virginia, US',
	                      'Lehigh, Pennsylvania, US',
	                      'San Bernardino, California, US',
	                      'Cuyahoga, Ohio, US',
	                      'Minnehaha, South Dakota, US',
	                      'East Baton Rouge, Louisiana, US',
	                      'Kent, Michigan, US',
	                      'Polk, Iowa, US',
	                      'Snohomish, Washington, US',
	                      'Anne Arundel, Maryland, US',
	                      'DeKalb, Georgia, US',
	                      'Lake, Indiana, US',
	                      'New Castle, Delaware, US',
	                      'Northampton, Pennsylvania, US',
	                      'Gwinnett, Georgia, US',
	                      'Adams, Colorado, US',
	                      'Luzerne, Pennsylvania, US']

	full_counties_combined = ['Los Angeles, California, US',
	                          'Miami-Dade, Florida, US',
	                          'Maricopa, Arizona, US',
	                          'Cook, Illinois, US',
	                          'Harris, Texas, US',
	                          'Dallas, Texas, US',
	                          'Broward, Florida, US',
	                          'Queens, New York, US',
	                          'Kings, New York, US',
	                          'Clark, Nevada, US',
	                          'Riverside, California, US',
	                          'Bronx, New York, US',
	                          'Orange, California, US',
	                          'San Bernardino, California, US',
	                          'Bexar, Texas, US',
	                          'Suffolk, New York, US',
	                          'Nassau, New York, US',
	                          'Palm Beach, Florida, US',
	                          'Tarrant, Texas, US',
	                          'San Diego, California, US',
	                          'Hillsborough, Florida, US',
	                          'Westchester, New York, US',
	                          'Orange, Florida, US',
	                          'Philadelphia, Pennsylvania, US',
	                          'New York, New York, US',
	                          'Wayne, Michigan, US',
	                          'Kern, California, US',
	                          'Hidalgo, Texas, US',
	                          'Shelby, Tennessee, US',
	                          'Travis, Texas, US',
	                          'Duval, Florida, US',
	                          'Prince George\'s, Maryland, US',
	                          'Mecklenburg, North Carolina, US',
	                          'Middlesex, Massachusetts, US',
	                          'Fresno, California, US',
	                          'Fulton, Georgia, US',
	                          'Gwinnett, Georgia, US',
	                          'Milwaukee, Wisconsin, US',
	                          'Salt Lake, Utah, US',
	                          'Davidson, Tennessee, US',
	                          'Hennepin, Minnesota, US',
	                          'Suffolk, Massachusetts, US',
	                          'Franklin, Ohio, US',
	                          'Bergen, New Jersey, US',
	                          'Pima, Arizona, US',
	                          'Cameron, Texas, US',
	                          'El Paso, Texas, US',
	                          'Essex, New Jersey, US',
	                          'Hudson, New Jersey, US',
	                          'Montgomery, Maryland, US',
	                          'Pinellas, Florida, US',
	                          'King, Washington, US',
	                          'St. Louis, Missouri, US',
	                          'Fairfield, Connecticut, US',
	                          'Nueces, Texas, US',
	                          'Lee, Florida, US',
	                          'Marion, Indiana, US',
	                          'Middlesex, New Jersey, US',
	                          'Fairfax, Virginia, US',
	                          'Passaic, New Jersey, US',
	                          'Alameda, California, US',
	                          'Essex, Massachusetts, US',
	                          'Oakland, Michigan, US',
	                          'Sacramento, California, US',
	                          'San Joaquin, California, US',
	                          'Santa Clara, California, US',
	                          'Union, New Jersey, US',
	                          'Polk, Florida, US',
	                          'Cobb, Georgia, US',
	                          'Providence, Rhode Island, US',
	                          'DeKalb, Georgia, US',
	                          'Jefferson, Alabama, US',
	                          'Jefferson, Louisiana, US',
	                          'Cuyahoga, Ohio, US',
	                          'Baltimore, Maryland, US',
	                          'Richmond, New York, US',
	                          'Wake, North Carolina, US',
	                          'Fort Bend, Texas, US',
	                          'Stanislaus, California, US',
	                          'DuPage, Illinois, US',
	                          'Lake, Illinois, US',
	                          'Baltimore City, Maryland, US',
	                          'Rockland, New York, US',
	                          'Tulare, California, US',
	                          'Charleston, South Carolina, US',
	                          'District of Columbia,District of Columbia,US',
	                          'Contra Costa, California, US',
	                          'East Baton Rouge, Louisiana, US',
	                          'New Haven, Connecticut, US',
	                          'Hartford, Connecticut, US',
	                          'Worcester, Massachusetts, US',
	                          'Douglas, Nebraska, US',
	                          'Oklahoma, Oklahoma, US',
	                          'Polk, Iowa, US',
	                          'Tulsa, Oklahoma, US',
	                          'Macomb, Michigan, US',
	                          'Yuma, Arizona, US',
	                          'Mobile, Alabama, US',
	                          'Greenville, South Carolina, US',
	                          'Jefferson, Kentucky, US']

	counties = ['AL-Jefferson',
	            'AL-Mobile',
	            'AZ-Maricopa',
	            'AZ-Pima',
	            'AZ-Yuma',
	            'CA-Alameda',
	            'CA-Contra Costa',
	            'CA-Fresno',
	            'CA-Kern',
	            'CA-Los Angeles',
	            'CA-Orange',
	            'CA-Riverside',
	            'CA-Sacramento',
	            'CA-San Bernardino',
	            'CA-San Diego',
	            'CA-San Joaquin',
	            'CA-Santa Clara',
	            'CA-Stanislaus',
	            'CA-Tulare',
	            'CO-Adams',
	            'CO-Arapahoe',
	            'CO-Denver',
	            'CT-Fairfield',
	            'CT-Hartford',
	            'CT-New Haven',
	            'DE-New Castle',
	            'DE-Sussex',
	            'DC-District of Columbia',
	            'FL-Broward',
	            'FL-Duval',
	            'FL-Hillsborough',
	            'FL-Lee',
	            'FL-Miami-Dade',
	            'FL-Orange',
	            'FL-Palm Beach',
	            'FL-Pinellas',
	            'FL-Polk',
	            'GA-Cobb',
	            'GA-DeKalb',
	            'GA-Fulton',
	            'GA-Gwinnett',
	            'IL-Cook',
	            'IL-DuPage',
	            'IL-Kane',
	            'IL-Lake',
	            'IL-Will',
	            'IN-Lake',
	            'IN-Marion',
	            'IA-Polk',
	            'KY-Jefferson',
	            'LA-East Baton Rouge',
	            'LA-Jefferson',
	            'LA-Orleans',
	            'MD-Anne Arundel',
	            'MD-Baltimore',
	            'MD-Baltimore City',
	            'MD-Montgomery',
	            'MD-Prince George\'s',
	            'MA-Bristol',
	            'MA-Essex',
	            'MA-Hampden',
	            'MA-Middlesex',
	            'MA-Norfolk',
	            'MA-Plymouth',
	            'MA-Suffolk',
	            'MA-Worcester',
	            'MI-Kent',
	            'MI-Macomb',
	            'MI-Oakland',
	            'MI-Wayne',
	            'MN-Hennepin',
	            'MO-St. Louis',
	            'NE-Douglas',
	            'NV-Clark',
	            'NJ-Bergen',
	            'NJ-Burlington',
	            'NJ-Camden',
	            'NJ-Essex',
	            'NJ-Hudson',
	            'NJ-Mercer',
	            'NJ-Middlesex',
	            'NJ-Monmouth',
	            'NJ-Morris',
	            'NJ-Ocean',
	            'NJ-Passaic',
	            'NJ-Somerset',
	            'NJ-Union',
	            'NY-Bronx',
	            'NY-Dutchess',
	            'NY-Erie',
	            'NY-Kings',
	            'NY-Nassau',
	            'NY-New York',
	            'NY-Orange',
	            'NY-Queens',
	            'NY-Richmond',
	            'NY-Rockland',
	            'NY-Suffolk',
	            'NY-Westchester',
	            'NC-Mecklenburg',
	            'NC-Wake',
	            'OH-Cuyahoga',
	            'OH-Franklin',
	            'OK-Oklahoma',
	            'OK-Tulsa',
	            'PA-Berks',
	            'PA-Bucks',
	            'PA-Delaware',
	            'PA-Lehigh',
	            'PA-Luzerne',
	            'PA-Montgomery',
	            'PA-Northampton',
	            'PA-Philadelphia',
	            'RI-Providence',
	            'SC-Charleston',
	            'SC-Greenville',
	            'SD-Minnehaha',
	            'TN-Davidson',
	            'TN-Shelby',
	            'TX-Bexar',
	            'TX-Cameron',
	            'TX-Dallas',
	            'TX-El Paso',
	            'TX-Fort Bend',
	            'TX-Harris',
	            'TX-Hidalgo',
	            'TX-Nueces',
	            'TX-Tarrant',
	            'TX-Travis',
	            'UT-Salt Lake',
	            'VA-Fairfax',
	            'VA-Prince William',
	            'WA-King',
	            'WA-Snohomish',
	            'WI-Milwaukee',
	            ]

	counties_init = ['AZ-Maricopa', 'CA-Los Angeles', 'CA-Orange', 'CA-Riverside', 'CA-San Bernardino', 'CA-San Diego',
	                 'CO-Adams', 'CO-Arapahoe', 'CO-Denver', 'CT-Fairfield', 'CT-Hartford', 'CT-New Haven',
	                 'DC-District of Columbia', 'DE-New Castle', 'DE-Sussex', 'FL-Broward', 'FL-Miami-Dade',
	                 'FL-Palm Beach',
	                 'GA-DeKalb', 'GA-Fulton', 'GA-Gwinnett', 'IA-Polk', 'IL-Cook', 'IL-DuPage', 'IL-Kane', 'IL-Lake',
	                 'IL-Will', 'IN-Lake', 'IN-Marion', 'LA-East Baton Rouge', 'LA-Jefferson', 'LA-Orleans',
	                 'MA-Bristol',
	                 'MA-Essex', 'MA-Hampden', 'MA-Middlesex', 'MA-Norfolk', 'MA-Plymouth', 'MA-Suffolk',
	                 'MA-Worcester',
	                 'MD-Anne Arundel', 'MD-Baltimore', 'MD-Baltimore City', 'MD-Montgomery', 'MD-Prince George\'s',
	                 'MI-Kent',
	                 'MI-Macomb', 'MI-Oakland', 'MI-Wayne', 'MN-Hennepin', 'MO-St. Louis', 'NJ-Bergen', 'NJ-Burlington',
	                 'NJ-Camden', 'NJ-Essex', 'NJ-Hudson', 'NJ-Mercer',
	                 'NJ-Middlesex', 'NJ-Monmouth', 'NJ-Morris', 'NJ-Ocean', 'NJ-Passaic', 'NJ-Somerset', 'NJ-Union',
	                 'NV-Clark', 'NY-Bronx', 'NY-Dutchess', 'NY-Erie', 'NY-Kings', 'NY-Nassau', 'NY-New York',
	                 'NY-Orange',
	                 'NY-Queens', 'NY-Richmond', 'NY-Rockland', 'NY-Suffolk', 'NY-Westchester', 'OH-Cuyahoga',
	                 'OH-Franklin',
	                 'PA-Berks', 'PA-Bucks', 'PA-Delaware', 'PA-Lehigh', 'PA-Luzerne', 'PA-Montgomery',
	                 'PA-Northampton',
	                 'PA-Philadelphia', 'RI-Providence', 'SD-Minnehaha', 'TN-Davidson', 'TN-Shelby', 'TX-Dallas',
	                 'TX-Harris',
	                 'TX-Tarrant', 'UT-Salt Lake', 'VA-Fairfax', 'VA-Prince William', 'WA-King', 'WA-Snohomish',
	                 'WI-Milwaukee']

	counties2 = counties[:68]
	counties2.reverse()
	print(counties2)

	return


def hist(state):
	fig = plt.figure(figsize=(10, 8))
	para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1']
	df = pd.read_csv(f'JHU/CI_{end_date}/{state}/paraCI.csv')

	for i in range(len(para_label)):
		ax = fig.add_subplot(3, 4, i + 1)
		data = df[para_label[i]]
		ax.hist(data, bins='auto', alpha=0.3, histtype='stepfilled')
		ax.set_title(para_label[i])
	fig.subplots_adjust(hspace=0.3, wspace=0.3)
	fig.suptitle(state)
	fig.savefig(f'JHU/CI_{end_date}/{state}/hist.png', bbox_inches='tight')
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
def compute_interval(state):
	ConfirmFile = 'JHU/JHU_Confirmed-counties.csv'
	DeathFile = 'JHU/JHU_Death-counties.csv'
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	for start_date in confirmed.columns[1:]:
		# if confirmed.iloc[0].loc[start_date] >= I_0 and death.iloc[0].loc[start_date] > 0:
		if confirmed.iloc[0].loc[start_date] >= I_0:
			break
	confirmed = confirmed.iloc[0].loc[start_date:end_date]
	death = death.iloc[0].loc[start_date:end_date]

	PopFile = 'JHU/CountyPopulation.csv'
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

	ParaFile = f'JHU50/combined2W_{end_date}/{state}/para.csv'
	df = pd.read_csv(ParaFile)
	beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, r1, r2, reopen_date = df.iloc[0]

	date = datetime.datetime.strptime(reopen_date, '%Y-%m-%d')
	reopen_date = date.strftime('%Y-%m-%d')

	SimFile = f'JHU50/combined2W_{end_date}/{state}/sim.csv'
	df = pd.read_csv(SimFile)
	S0 = df[df['series'] == 'S'].iloc[0].loc[start_date:end_date]
	# I0 = df[df['series'] == 'I'].iloc[0].loc[start_date:end_date]
	# IH0 = df[df['series'] == 'IH'].iloc[0].loc[start_date:end_date]
	# IN0 = df[df['series'] == 'IN'].iloc[0].loc[start_date:end_date]
	# R0 = df[df['series'] == 'R'].iloc[0].loc[start_date:end_date]
	D0 = df[df['series'] == 'D'].iloc[0].loc[start_date:end_date]
	G0 = df[df['series'] == 'G'].iloc[0].loc[start_date:end_date]
	# H0 = df[df['series'] == 'H'].iloc[0].loc[start_date:end_date]
	days = df.columns[1:].tolist()
	days = days[days.index(start_date):]
	size = len(G0)

	reopen_day = days.index(reopen_date)
	state_path = f'JHU50/CI_{end_date}/{state}'
	# state_path = f'JHU50/CI_{num_CI}by{num_per_interval}_/{state}'
	CIFile = f'{state_path}/paraCI.csv'
	df = pd.read_csv(CIFile)
	# Gs = [G0]
	# Ds = [D0]
	Gs = []
	Ds = []

	for i in range(len(df)):
		beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, r1, r2, reopen_date = \
			df.iloc[i]
		S2 = [n_0 * eta]
		# I2 = [confirmed[0]]
		I2 = [G0[0]]
		IH2 = [0]
		IN2 = [I2[-1] * gamma2]
		# D2 = [death[0]]
		D2 = [D0[0]]
		R2 = [0]
		# G2 = [confirmed[0]]
		G2 = [G0[0]]
		H2 = [Hiding_init * eta * n_0]
		result, [S, I, IH, IN, D, R, G, H, betas] \
			= simulate_combined(size, SIRG_combined, S2, I2, IH2, IN2, D2, R2, G2, H2, beta, gamma, gamma2, a1, a2, a3,
			                    h, k, k2, eta, c1, n_0, reopen_day)
		if result:
			Gs.append(G)
			Ds.append(D)
		else:
			print('failed in', state)

	G_max = []
	G_min = []
	D_max = []
	D_min = []
	G_max2 = []
	G_min2 = []
	D_max2 = []
	D_min2 = []

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
		G_max2.append(max([G[i] for G in Gs]))
		G_min2.append(min([G[i] for G in Gs]))
		D_max2.append(max([D[i] for D in Ds]))
		D_min2.append(min([D[i] for D in Ds]))

	col = ['series']
	col.extend(days)
	df = pd.DataFrame(columns=col)

	G_max.insert(0, 'G_high')
	G_min.insert(0, 'G_low')
	D_max.insert(0, 'D_high')
	D_min.insert(0, 'D_low')
	G_max2.insert(0, 'G_high_100')
	G_min2.insert(0, 'G_low_100')
	D_max2.insert(0, 'D_high_100')
	D_min2.insert(0, 'D_low_100')
	df.loc[len(df)] = G_max
	df.loc[len(df)] = G_min
	df.loc[len(df)] = D_max
	df.loc[len(df)] = D_min
	df.loc[len(df)] = G_max2
	df.loc[len(df)] = G_min2
	df.loc[len(df)] = D_max2
	df.loc[len(df)] = D_min2
	df.to_csv(f'{state_path}/GD_high_low.csv', index=False)
	return state


# generate confidence interval of G,D for all states
def compute_interval_all():
	counties = counties_global

	# counties = ['MA-Essex']

	with concurrent.futures.ProcessPoolExecutor() as executor:
		results = [executor.submit(compute_interval, state) for state in counties]

		for f in concurrent.futures.as_completed(results):
			print(f.result(), 'computed')

	return


def compute_interval_tmp():
	counties = counties_global

	states = ['NY-Westchester']

	with concurrent.futures.ProcessPoolExecutor() as executor:
		results = [executor.submit(compute_interval, state) for state in states]

		for f in concurrent.futures.as_completed(results):
			print(f.result(), 'computed')

	return


def CI_para(state):
	df = pd.read_csv(f'JHU50/combined2W_{end_date}/{state}/para.csv')
	beta = df['beta'].iloc[0]
	gamma = df['gamma'].iloc[0]
	gamma2 = df['gamma2'].iloc[0]
	a2 = df['a2'].iloc[0]

	df2 = pd.read_csv(f'JHU50/CI_{end_date}/{state}/paraCI.csv')
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
	states = ['AZ-Maricopa',
	          'CA-Los Angeles',
	          'CA-Orange',
	          'CA-Riverside',
	          'CA-San Bernardino',
	          'CA-San Diego',
	          'CO-Adams',
	          'CO-Arapahoe',
	          'CO-Denver',
	          'CT-Fairfield',
	          'CT-Hartford',
	          'CT-New Haven',
	          'DC-District of Columbia',
	          'DE-New Castle',
	          'DE-Sussex',
	          'FL-Broward',
	          'FL-Miami-Dade',
	          'FL-Palm Beach',
	          'GA-DeKalb',
	          'GA-Fulton',
	          'GA-Gwinnett',
	          'IA-Polk',
	          'IL-Cook',
	          'IL-DuPage',
	          'IL-Kane',
	          'IL-Lake',
	          'IL-Will',
	          'IN-Lake',
	          'IN-Marion',
	          'LA-East Baton Rouge',
	          'LA-Jefferson',
	          'LA-Orleans',
	          'MA-Bristol',
	          'MA-Essex',
	          'MA-Hampden',
	          'MA-Middlesex',
	          'MA-Norfolk',
	          'MA-Plymouth',
	          'MA-Suffolk',
	          'MA-Worcester',
	          'MD-Anne Arundel',
	          'MD-Baltimore',
	          'MD-Baltimore City',
	          'MD-Montgomery',
	          'MD-Prince George\'s',
	          'MI-Kent',
	          'MI-Macomb',
	          'MI-Oakland',
	          'MI-Wayne',
	          'MN-Hennepin',
	          'MO-St. Louis',
	          'NJ-Bergen',
	          'NJ-Burlington',
	          'NJ-Camden',
	          'NJ-Essex',
	          'NJ-Hudson',
	          'NJ-Mercer',
	          'NJ-Middlesex',
	          'NJ-Monmouth',
	          'NJ-Morris',
	          'NJ-Ocean',
	          'NJ-Passaic',
	          'NJ-Somerset',
	          'NJ-Union',
	          'NV-Clark',
	          'NY-Bronx',
	          'NY-Dutchess',
	          'NY-Erie',
	          'NY-Kings',
	          'NY-Nassau',
	          'NY-New York',
	          'NY-Orange',
	          'NY-Queens',
	          'NY-Richmond',
	          'NY-Rockland',
	          'NY-Suffolk',
	          'NY-Westchester',
	          'OH-Cuyahoga',
	          'OH-Franklin',
	          'PA-Berks',
	          'PA-Bucks',
	          'PA-Delaware',
	          'PA-Lehigh',
	          'PA-Luzerne',
	          'PA-Montgomery',
	          'PA-Northampton',
	          'PA-Philadelphia',
	          'RI-Providence',
	          'SD-Minnehaha',
	          'TN-Davidson',
	          'TN-Shelby',
	          'TX-Dallas',
	          'TX-Harris',
	          'TX-Tarrant',
	          'UT-Salt Lake',
	          'VA-Fairfax',
	          'VA-Prince William',
	          'WA-King',
	          'WA-Snohomish',
	          'WI-Milwaukee']

	# states = ['TX-Harris']

	columns = ['State - County', 'beta', 'gamma', 'gamma2', 'a2']
	table = []
	for state in counties_global:
		table.append(CI_para(state))

	out_df = pd.DataFrame(table, columns=columns)
	out_df.to_csv(f'JHU50/CI_{end_date}/para_table.csv', index=False)
	# print(table)
	return


def plot_interval_all():
	counties = counties_global
	row = 7
	col = 5

	# plt.rcParams.update({'font.size': 10})
	fig = plt.figure(figsize=(16, 20))
	i = 0
	for state in counties[:35]:
		i += 1
		ax = fig.add_subplot(row, col, i)
		ax2 = plot_interval(state, ax)

	ax.legend(bbox_to_anchor=(0.5, -0.35), loc="upper right")
	ax2.legend(bbox_to_anchor=(0.5, -0.35), loc="upper left")
	fig.subplots_adjust(hspace=0.6, wspace=0.35)
	ax1 = fig.add_subplot(111, frameon=False)
	ax2 = ax1.twinx()
	ax2.set_frame_on(False)
	ax1.set_ylabel("Cumulative Cases (Thousand)")
	ax2.set_ylabel("Deaths (Thousand)")
	# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	fig.savefig(f'JHU50/CI_{end_date}/grid1.png', bbox_inches="tight")
	plt.close(fig)

	fig = plt.figure(figsize=(16, 20))
	i = 0
	for state in counties[35:70]:
		i += 1
		ax = fig.add_subplot(row, col, i)
		ax2 = plot_interval(state, ax)

	ax.legend(bbox_to_anchor=(0.5, -0.35), loc="upper right")
	ax2.legend(bbox_to_anchor=(0.5, -0.35), loc="upper left")
	fig.subplots_adjust(hspace=0.6, wspace=0.35)
	ax1 = fig.add_subplot(111, frameon=False)
	ax2 = ax1.twinx()
	ax2.set_frame_on(False)
	ax1.set_ylabel("Cumulative Cases (Thousand)")
	ax2.set_ylabel("Deaths (Thousand)")
	# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	fig.savefig(f'JHU50/CI_{end_date}/grid2.png', bbox_inches="tight")
	plt.close(fig)

	fig = plt.figure(figsize=(16, 20))
	i = 0
	for state in counties[70:105]:
		i += 1
		ax = fig.add_subplot(row, col, i)
		ax2 = plot_interval(state, ax)

	ax.legend(bbox_to_anchor=(0.5, -0.35), loc="upper right")
	ax2.legend(bbox_to_anchor=(0.5, -0.35), loc="upper left")
	fig.subplots_adjust(hspace=0.6, wspace=0.35)
	ax1 = fig.add_subplot(111, frameon=False)
	ax2 = ax1.twinx()
	ax2.set_frame_on(False)
	ax1.set_ylabel("Cumulative Cases (Thousand)")
	ax2.set_ylabel("Deaths (Thousand)")
	# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	fig.savefig(f'JHU50/CI_{end_date}/grid3.png', bbox_inches="tight")
	plt.close(fig)

	fig = plt.figure(figsize=(16, 20))
	i = 0
	for state in counties[105:]:
		i += 1
		ax = fig.add_subplot(row, col, i)
		ax2 = plot_interval(state, ax)

	ax.legend(bbox_to_anchor=(0.5, -0.35), loc="upper right")
	ax2.legend(bbox_to_anchor=(0.5, -0.35), loc="upper left")
	fig.subplots_adjust(hspace=0.6, wspace=0.35)
	ax1 = fig.add_subplot(111, frameon=False)
	ax2 = ax1.twinx()
	ax2.set_frame_on(False)
	ax1.set_ylabel("Cumulative Cases (Thousand)")
	ax2.set_ylabel("Deaths (Thousand)")
	# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	fig.savefig(f'JHU50/CI_{end_date}/grid4.png', bbox_inches="tight")
	plt.close(fig)

	return


def plot_interval(state, ax):
	print('plotting', state)
	ParaFile = f'JHU50/combined2W_{end_date}/{state}/para.csv'
	df = pd.read_csv(ParaFile)
	beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, r1, r2, reopen_date = df.iloc[0]

	df = pd.read_csv(f'JHU50/CI_{end_date}/{state}/GD_high_low.csv')
	# df = pd.read_csv(f'JHU50/CI_{num_CI}by{num_per_interval}_/{state}/GD_high_low.csv')
	dates = df.columns[1:].tolist()
	start_date = dates[0]
	reopen_day = dates.index(reopen_date)
	if interval_100:
		G_high = df[df['series'] == 'G_high_100'].iloc[0].iloc[1:]
		G_low = df[df['series'] == 'G_low_100'].iloc[0].iloc[1:]
		D_high = df[df['series'] == 'D_high_100'].iloc[0].iloc[1:]
		D_low = df[df['series'] == 'D_low_100'].iloc[0].iloc[1:]
	else:
		G_high = df[df['series'] == 'G_high'].iloc[0].iloc[1:]
		G_low = df[df['series'] == 'G_low'].iloc[0].iloc[1:]
		D_high = df[df['series'] == 'D_high'].iloc[0].iloc[1:]
		D_low = df[df['series'] == 'D_low'].iloc[0].iloc[1:]

	df = pd.read_csv(f'JHU50/combined2W_{end_date}/{state}/sim.csv')
	G = df[df['series'] == 'G'].iloc[0].loc[start_date:]
	D = df[df['series'] == 'D'].iloc[0].loc[start_date:]

	ConfirmFile = 'JHU/JHU_Confirmed-counties.csv'
	DeathFile = 'JHU/JHU_Death-counties.csv'
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[start_date: end_date]
	death = death.iloc[0].loc[start_date: end_date]

	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates]
	ax2 = ax.twinx()
	ax.plot([], ' ', color='black', label="Left Axis\n(Thousand):")
	ax2.plot([], ' ', color='black', label="Right Axis\n(Thousand):")
	ax.set_title(state)
	ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax.fill_between(days, [g / 1000 for g in G_high], [g / 1000 for g in G_low], facecolor='orange', alpha=0.4,
	                label='CI_G')
	ax.plot(days, [i / 1000 for i in confirmed], linewidth=2, linestyle=':', label="Cumulative\nCases", color='red')
	ax.plot(days, [g / 1000 for g in G.tolist()], linewidth=0.5, label='G', color='red', alpha=1)

	ax2.plot(days, [i / 1000 for i in death], linewidth=2, linestyle=':', label="Cumulative\nDeaths", color='blue')
	ax2.plot(days, [d / 1000 for d in D.tolist()], linewidth=0.5, label='D', color='blue')
	ax2.fill_between(days, [d / 1000 for d in D_high], [d / 1000 for d in D_low], facecolor='royalblue', alpha=0.4,
	                 label='CI_D')
	l, u = ax.get_ylim()
	ax.set_ylim(0, u)
	l, u = ax2.get_ylim()
	ax2.set_ylim(0, u * 2)
	ax.set_xlim(days[0], days[-1])

	date_form = DateFormatter("%m-%d")
	ax.xaxis.set_major_formatter(date_form)
	plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

	return ax2


def plot_interval_tmp():
	states = ['NY-Westchester']
	for state in states:
		fig, ax = plt.subplots()
		ax2 = plot_interval(state, ax)
		plt.show()
		plt.close(fig)

	return


def random_tester():
	# states = ['GA-Cobb',
	#           'IN-Marion',
	#           'PA-Lehigh',
	#           'DC-District of Columbia',
	#           'IL-Cook',
	states = ['NY-Westchester']

	# states = ['IL-Cook']

	for state in states:
		random_state(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv', 'JHU/CountyPopulation.csv')
	return


def random_state(state, ConfirmFile, DeathFile, PopFile):
	t1 = time.perf_counter()
	SimFile = f'JHU50/combined2W_{end_date}/{state}/sim.csv'
	ParaFile = f'JHU50/combined2W_{end_date}/{state}/para.csv'
	df = pd.read_csv(ParaFile)
	reopen_date = df.iloc[0]['reopen']
	state_path = f'JHU50/CI_{end_date}/{state}'
	if not os.path.exists(state_path):
		os.makedirs(state_path)

	print(state)
	print(reopen_date)

	# read population
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

	# select useful portion of simulation
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	for start_date in confirmed.columns[1:]:
		# if confirmed.iloc[0].loc[start_date] >= I_0 and death.iloc[0].loc[start_date] > 0:
		#     break
		if confirmed.iloc[0].loc[start_date] >= I_0:
			break
	confirmed = confirmed.iloc[0].loc[start_date:end_date]
	death = death.iloc[0].loc[start_date:end_date]

	for i in range(len(death)):
		if death.iloc[i] == 0:
			death.iloc[i] = 0.01
	# death = death.tolist()

	# confirmed = confirmed[start_date:]
	# death = death[start_date:]
	# print(death)
	days = confirmed.index.tolist()

	df_sim = pd.read_csv(SimFile)
	G = df_sim[df_sim['series'] == 'G'].iloc[0].loc[start_date:]
	D = df_sim[df_sim['series'] == 'D'].iloc[0].loc[start_date:]

	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	reopen_day = days.index(datetime.datetime.strptime(reopen_date, '%Y-%m-%d'))

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
	# MA_confirmed = confirmed.rolling(7, min_periods=1).mean()
	# MA_death = death.rolling(7, min_periods=1).mean()
	# data1 = [(confirmed[i] - G[i]) / G[i] for i in range(size)]
	# data2 = [(death[i] - D[i]) / D[i] for i in range(size)]
	# data1 = [(confirmed[i] - G[i]) for i in range(size)]
	# data2 = [(death[i] - D[i]) for i in range(size)]
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
	# wt_data1 = [(confirmed[i] - G[i]) / G[i] for i in range(size)]
	# wt_data2 = [(death[i] - D[i]) / D[i] for i in range(size)]
	# print('weighted data1:')
	# print(wt_data1)
	# print('weighted data2:')
	# print(wt_data2)

	# fig = plt.figure()
	# ax = fig.add_subplot()
	# ax.plot(days, wt_data1, label='wt data1')
	# ax.plot(days, wt_data2, label='wt data2')
	# ax.legend()
	# plt.show()
	# plt.close(fig)

	print(f'metric1={round(metric1, 4)}')
	# print(f'new metric1={round(weighted_relative_deviation(weights, confirmed, G, start_dev, 12), 4)}')
	print(f'metric2={round(metric2, 4)}')
	# print(f'new metric2={round(weighted_relative_deviation(weights, death, D, start_dev, 12), 4)}')
	Gs = []
	Ds = []
	with concurrent.futures.ProcessPoolExecutor() as executor:
		results = [executor.submit(MT_random, G, D, metric1, metric2) for _ in range(num_CI)]

		for f in concurrent.futures.as_completed(results):
			G2, D2 = f.result()
			Gs.append(G2)
			Ds.append(D2)

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

	fig = plt.figure()
	fig.suptitle(state)
	ax = fig.add_subplot(2, 1, 1)
	ax2 = fig.add_subplot(2, 1, 2)
	for i in range(4):
		ax.plot(days, Gs[i * num_CI // 4])
		ax2.plot(days, Ds[i * num_CI // 4])
	ax.plot(days, G, linestyle=':')
	ax.plot(days, G_max, linestyle=':')
	ax.plot(days, G_min, linestyle=':')
	ax2.plot(days, D, linestyle=':')
	ax2.plot(days, D_max, linestyle=':')
	ax2.plot(days, D_min, linestyle=':')

	plt.show()
	plt.close(fig)
	return


def MT_random(confirmed0, death0, metric1, metric2):
	np.random.seed()
	confirmed = confirmed0.copy()
	death = death0.copy()
	size = len(confirmed)
	# scale1 = pd.Series(np.random.normal(1, metric1, size))
	# confirmed = [max(confirmed[i] * scale1[i], 0.01) for i in range(size)]
	# scale2 = pd.Series(np.random.normal(1, metric2, size))
	# death = [max(death[i] * scale2[i], 0.01) for i in range(size)]
	scale1 = pd.Series(np.random.normal(1, metric1, size))
	confirmed = [confirmed[i] * scale1[i] for i in range(size)]
	scale2 = pd.Series(np.random.normal(1, metric2, size))
	death = [death[i] * scale2[i] for i in range(size)]
	# confirmed[0] = max(confirmed[0], 0.01)
	# death[0] = max(death[0], 0.01)
	return confirmed, death


def main():
	# fit_all_init()

	# fit_all_combined()
	# CI_all_combined()

	# hist_all()

	# compute_interval_all()
	# plot_interval_all()
	CI_para_all()

	# test()
	# compute_interval_tmp()
	# plot_interval_tmp()
	# random_tester()
	return 0


if __name__ == '__main__':
	main()
