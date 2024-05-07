import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score, mean_squared_error
from SIRfunctions import SIRG_sd, SIRG, computeBeta
import datetime
from matplotlib.dates import DateFormatter

"""
Authors: Yi Zhang, Mohit Hota, Sanjiv Kapoor

This code is motivated by https://github.com/Lewuathe/COVID19-SIR
"""

np.set_printoptions(threshold=sys.maxsize)
Geo = 0.8
theta = 0.7
I_0 = 50
beta_range = (0.1, 100)
gamma_range = (0.04, 0.08)
eta_range = (0.001, 0.1)
c1_fixed = (0.9, 0.9)
c1_range = (0, 0.98)
h_range = (0, 10)
end_date = '2020-07-10'
reopen_date = '2020-05-15'
p_m = 1


def forecast(state, n_0, para_sd, para, file):
	# plt.clf()
	# plt.rcParams.update({'font.size': 14})
	fig, ax = plt.subplots()
	plt.rcParams.update({'font.size': 14})
	fig.autofmt_xdate()
	ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	ax.set_ylabel('Cases (Thousands)')
	start_date = para_sd[0]

	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	days = list(confirmed.columns)
	days = days[days.index(start_date):]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[start_date:]
	size = len(confirmed)
	plt.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative cases")

	# SIRG_sd
	beta = para_sd[1]
	gamma = para_sd[2]
	eta = para_sd[3]
	c1 = para_sd[4]
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	print('SIRG-SD')
	print('beta=', beta, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
	print('****************************')
	print('c1=', c1)
	print('****************************')
	for i in range(1, size):
		delta = SIRG_sd(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
	plt.plot(days, [i / 1000 for i in G], label='SIRG-SD')

	# SIRG
	beta = para[1]
	gamma = para[2]
	eta = para[3]
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	print('SIRG')
	print('beta=', beta, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
	for i in range(1, size):
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
	plt.plot(days, [i / 1000 for i in G], label='SIRG')

	plt.axvline(datetime.datetime.strptime(end_date, '%Y-%m-%d'), linestyle='dashed', color='tab:grey', label=end_date)
	plt.title(state + ' forecast')
	plt.legend()
	plt.show()


def forecast2(state, n_0, para_sd, file):
	# plt.clf()
	# plt.rcParams.update({'font.size': 14})
	# fig, ax = plt.subplots()
	# plt.rcParams.update({'font.size': 14})
	# fig.autofmt_xdate()
	# ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	# ax.set_ylabel('Cases (Thousands)')
	start_date = para_sd[0]

	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	days = list(confirmed.columns)
	days = days[days.index(start_date):]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[start_date:]
	size = len(confirmed)
	# plt.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative cases")

	# SIRG_sd
	beta = para_sd[1]
	gamma = para_sd[2]
	eta = para_sd[3]
	c1 = para_sd[4]
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	print('SIRG-SD')
	print('beta=', beta, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
	print('****************************')
	print('c1=', c1)
	print('****************************')
	for i in range(1, size):
		delta = SIRG_sd(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
	plt.plot(days, [i / 1000 for i in G], label='SIRG-SD')


# does not optimize c1
def fit(file, state, n_0, SIRG, method):
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	for d in confirmed.columns[1:]:
		if confirmed.iloc[0].loc[d] >= I_0:
			break
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[d: end_date]
	optimal = minimize(loss, [10, 0.05, 0.02, 0.9], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
	                   bounds=[beta_range, gamma_range, eta_range, c1_fixed])
	beta = optimal.x[0]
	gamma = optimal.x[1]
	eta = optimal.x[2]
	c1 = optimal.x[3]
	current_loss = loss(optimal.x, confirmed, n_0, SIRG)
	# print(f'beta = {round(beta, 8)} , gamma = {round(gamma, 8)} , '
	#       f'eta = {round(eta, 8)}, StartDate = {d} ')

	size = len(confirmed)
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])

	if method == 'SIR-SD':
		plt.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative cases")
	plt.plot(days, [i / 1000 for i in G], label=method)
	# plt.plot(days, [i / 1000 for i in S], label="S")
	# plt.plot(days, [i / 1000 for i in I], label="I")
	# plt.plot(days, [i / 1000 for i in R], label="R")
	plt.title(state)

	confirmed_derivative = np.diff(confirmed)
	G_derivative = np.diff(G)
	weights = [Geo ** (n - 1) for n in range(1, size)]
	weights.reverse()
	confirmed_derivative *= weights
	G_derivative *= weights
	metric0 = r2_score(confirmed, G)
	metric1 = r2_score(confirmed_derivative, G_derivative)
	if method == 'SIR-SD':
		print(method)
		print('beta=', beta, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
		print('****************************')
		print('c1=', c1)
		print('****************************')
		print('start date=', d)
		print(f'R2: {metric0} and {metric1}')
		print('loss=', current_loss)
		print()

	return [d, beta, gamma, eta]


# # optimizes c1
# def fit2(file, state, n_0, SIRG, method):
# 	df = pd.read_csv(file)
# 	confirmed = df[df.iloc[:, 0] == state]
#
# 	for d in confirmed.columns[1:]:
# 		if confirmed.iloc[0].loc[d] >= I_0:
# 			break
# 	days = list(confirmed.columns)
# 	days = days[days.index(d):days.index(end_date) + 1]
# 	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
# 	confirmed = confirmed.iloc[0].loc[d: end_date]
# 	optimal = minimize(loss, [10, 0.05, 0.02, 0.3], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
# 	                   bounds=[beta_range, gamma_range, eta_range, c1_range])
# 	beta = optimal.x[0]
# 	gamma = optimal.x[1]
# 	eta = optimal.x[2]
# 	c1 = optimal.x[3]
# 	current_loss = loss(optimal.x, confirmed, n_0, SIRG)
# 	# print(f'beta = {round(beta, 8)} , gamma = {round(gamma, 8)} , '
# 	#       f'eta = {round(eta, 8)}, StartDate = {d} ')
#
# 	size = len(confirmed)
# 	S = [n_0 * eta]
# 	I = [confirmed[0]]
# 	R = [0]
# 	G = [confirmed[0]]
# 	for i in range(1, size):
# 		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0, c1])
# 		S.append(S[-1] + delta[0])
# 		I.append(I[-1] + delta[1])
# 		R.append(R[-1] + delta[2])
# 		G.append(G[-1] + delta[3])
#
# 	if method == 'SIR-SD':
# 		plt.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative cases")
# 	plt.plot(days, [i / 1000 for i in G], label=method)
# 	# plt.plot(days, [i / 1000 for i in S], label="S")
# 	plt.plot(days, [i / 1000 for i in I], label="I")
# 	plt.plot(days, [i / 1000 for i in R], label="R")
# 	plt.title(state)
#
# 	confirmed_derivative = np.diff(confirmed)
# 	G_derivative = np.diff(G)
# 	weights = [Geo ** (n - 1) for n in range(1, size)]
# 	weights.reverse()
# 	confirmed_derivative *= weights
# 	G_derivative *= weights
# 	metric0 = r2_score(confirmed, G)
# 	metric1 = r2_score(confirmed_derivative, G_derivative)
# 	if method == 'SIR-SD' or True:
# 		print(method)
# 		print('beta=', beta, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
# 		print('****************************')
# 		print('c1=', c1)
# 		print('****************************')
# 		print('start date=', d)
# 		print(f'R2: {metric0} and {metric1}')
# 		print('loss=', current_loss)
# 		print()
# 	return c1


# optimizes c1
def fit2(file, state, n_0, SIRG, method):
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]

	for d in confirmed.columns[1:]:
		if confirmed.iloc[0].loc[d] >= I_0:
			break
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[d: end_date]
	c_max = 0
	min_loss = 10000
	for c1 in np.arange(0, 1, 0.01):
		optimal = minimize(loss, [10, 0.05, 0.02, c1], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
		                   bounds=[beta_range, gamma_range, eta_range, (c1, c1)])
		current_loss = loss(optimal.x, confirmed, n_0, SIRG)
		if current_loss < min_loss:
			min_loss = current_loss
			c_max = c1
	# print(f'beta = {round(beta, 8)} , gamma = {round(gamma, 8)} , '
	#       f'eta = {round(eta, 8)}, StartDate = {d} ')

	optimal = minimize(loss, [10, 0.05, 0.02, c_max], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
	                   bounds=[beta_range, gamma_range, eta_range, (c_max, c_max)])
	beta = optimal.x[0]
	gamma = optimal.x[1]
	eta = optimal.x[2]
	c1 = optimal.x[3]
	size = len(confirmed)
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
	if method == 'SIR-SD':
		plt.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative cases")
	plt.plot(days, [i / 1000 for i in G], label=method)
	# plt.plot(days, [i / 1000 for i in S], label="S")
	plt.plot(days, [i / 1000 for i in I], label="I")
	plt.plot(days, [i / 1000 for i in R], label="R")
	plt.title(state)

	confirmed_derivative = np.diff(confirmed)
	G_derivative = np.diff(G)
	weights = [Geo ** (n - 1) for n in range(1, size)]
	weights.reverse()
	confirmed_derivative *= weights
	G_derivative *= weights
	metric0 = r2_score(confirmed, G)
	metric1 = r2_score(confirmed_derivative, G_derivative)
	if method == 'SIR-SD' or True:
		print(method)
		print('beta=', beta, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
		print('****************************')
		print('c1=', c1)
		print('****************************')
		print('start date=', d)
		print(f'R2: {metric0} and {metric1}')
		print('loss=', min_loss)
		print()
	return [d, beta, gamma, eta, c1]


# fits until reopen day
def fit3(file, state, n_0, SIRG, method):
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]

	for d in confirmed.columns[1:]:
		if confirmed.iloc[0].loc[d] >= I_0:
			break
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed2 = confirmed.iloc[0].loc[d: end_date]
	plt.plot(days, [i / 1000 for i in confirmed2], linewidth=5, linestyle=':', label="Cumulative cases")
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(reopen_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[d: reopen_date]
	c_max = 0
	min_loss = 10000
	for c1 in np.arange(0.8, 1, 0.01):
		optimal = minimize(loss, [10, 0.05, 0.02, c1], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
		                   bounds=[beta_range, gamma_range, eta_range, (c1, c1)])
		current_loss = loss(optimal.x, confirmed, n_0, SIRG)
		if current_loss < min_loss:
			min_loss = current_loss
			c_max = c1
	# print(f'beta = {round(beta, 8)} , gamma = {round(gamma, 8)} , '
	#       f'eta = {round(eta, 8)}, StartDate = {d} ')

	optimal = minimize(loss, [10, 0.05, 0.02, c_max], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
	                   bounds=[beta_range, gamma_range, eta_range, (c_max, c_max)])
	beta = optimal.x[0]
	gamma = optimal.x[1]
	eta = optimal.x[2]
	c1 = optimal.x[3]
	size = len(confirmed)
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])

	# plt.plot(days, [i / 1000 for i in G], label=method)
	# plt.plot(days, [i / 1000 for i in S], label="S")
	# plt.plot(days, [i / 1000 for i in I], label="I")
	# plt.plot(days, [i / 1000 for i in R], label="R")
	plt.axvline(datetime.datetime.strptime(reopen_date, '%Y-%m-%d'), linestyle='dashed', color='tab:grey',
	            label=reopen_date)
	plt.title(state)

	confirmed_derivative = np.diff(confirmed)
	G_derivative = np.diff(G)
	weights = [Geo ** (n - 1) for n in range(1, size)]
	weights.reverse()
	confirmed_derivative *= weights
	G_derivative *= weights
	metric0 = r2_score(confirmed, G)
	metric1 = r2_score(confirmed_derivative, G_derivative)
	if method == 'SIR-SD' or True:
		print(method)
		print('beta=', beta, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
		print('****************************')
		print('c1=', c1)
		print('****************************')
		print('start date=', d)
		print(f'R2: {metric0} and {metric1}')
		print('loss=', min_loss)
		print()
	return [d, beta, gamma, eta, c1]


# linear release after reopen day
def fit_reopen(file, state, n_0, SIRG, para):
	d = para[0]
	beta = para[1]
	gamma = para[2]
	eta = para[3]
	c1 = para[4]
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[d: reopen_date]
	size = len(confirmed)
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		print(computeBeta(beta, eta, n_0, S[-1], c1) * eta / gamma)
		delta = SIRG(i, [S[-1], I[-1], R[-1], G[-1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
	print(f'****** reopen on {reopen_date} ******')
	confirmed = df[df.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[reopen_date: end_date]
	optimal = minimize(loss2, [0], args=(beta, gamma, eta, c1, confirmed, S[-1], I[-1], R[-1], G[-1], n_0, SIRG),
	                   method='L-BFGS-B', bounds=[(0, eta / 50)])
	h = optimal.x[0]
	size = len(confirmed)
	# S[-1] += h * n_0
	# eta = S[-1] / n_0
	for i in range(1, size):
		print(computeBeta(beta, eta, n_0, S[-1], c1) * eta / gamma)
		# if i < 40:
		S[-1] += h * n_0
		delta = SIRG(i, [S[-1], I[-1], R[-1], G[-1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
	plt.plot(days, [i / 1000 for i in G], label='G')
	plt.plot(days, [i / 1000 for i in I], label='I')
	plt.plot(days, [i / 1000 for i in R], label='R')
	# plt.plot(days, [i / 1000 for i in S], label='S')
	# plt.axhline(y=eta * n_0 / 1000, color='grey', label='S0', linestyle='--')
	print(f'reopen fit:\nh={round(h, 5)} at {round(h / eta * 100, 3)}%')


# linear release after reopen day with new beta
def fit_reopen3(file, state, n_0, SIRG, para):
	d = para[0]
	beta = para[1]
	gamma = para[2]
	eta = para[3]
	c1 = para[4]
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[d: reopen_date]
	size = len(confirmed)
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		# print(computeBeta(beta, eta, n_0, S[-1], c1) * eta / gamma)
		delta = SIRG(i, [S[-1], I[-1], R[-1], G[-1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
	print(f'****** reopen on {reopen_date} ******')
	confirmed = df[df.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[reopen_date: end_date]
	min_loss = 1000000
	# for c1 in np.arange(c1, c1+.00001, 0.01):
	for beta in np.arange(1, 200, 0.1):
		optimal = minimize(loss2, [0.00000001],
		                   args=(beta, gamma, eta, c1, confirmed, S[-1], I[-1], R[-1], G[-1], n_0, SIRG),
		                   method='L-BFGS-B', bounds=[(0, eta / 100)])
		# optimal = minimize(loss, [10, 0.05, 0.02, c1], args=(confirmed, n_0, SIRG), method='L-BFGS-B',bounds=[beta_range, gamma_range, eta_range, (c1, c1)])
		h = optimal.x[0]
		current_loss = loss2(optimal.x, beta, gamma, eta, c1, confirmed, S[-1], I[-1], R[-1], G[-1], n_0, SIRG)
		# current_loss = loss2(optimal.x, confirmed, n_0, SIRG)
		if current_loss < min_loss:
			min_loss = current_loss
			beta_max = beta
			h_max = optimal.x[0]
	# optimal = minimize(loss2, [0.0001], args=(beta, gamma, eta, c1, confirmed, S[-1], I[-1], R[-1], G[-1], n_0, SIRG), method='L-BFGS-B', bounds=[h_range])
	h = h_max  # optimal.x[0]
	beta = beta_max
	# c1 = c_max
	size = len(confirmed)
	# S[-1] += h * n_0
	# eta = S[-1] / n_0
	for i in range(1, size):
		# print(computeBeta(beta, eta, n_0, S[-1], c1) * eta / gamma)
		# print(computeBeta(beta, eta, n_0, S[-1], c1))
		# if i < 40:
		S[-1] += h * n_0
		delta = SIRG(i, [S[-1], I[-1], R[-1], G[-1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
	plt.plot(days, [i / 1000 for i in G], label='G')
	plt.plot(days, [i / 1000 for i in I], label='I')
	plt.plot(days, [i / 1000 for i in R], label='R')
	# plt.plot(days, [i / 1000 for i in S], label='S')
	# plt.axhline(y=eta * n_0 / 1000, color='grey', label='S0', linestyle='--')
	print(f'reopen fit:\nh={round(h, 5)} at {round(h / eta * 100, 3)}%')
	print('c1=', c1, 'beta=', beta)


# release by mobility
def fit_reopen2(file, mob_file, state, n_0, SIRG, para):
	d = para[0]
	beta = para[1]
	gamma = para[2]
	eta = para[3]
	c1 = para[4]
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[d: reopen_date]
	size = len(confirmed)
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		# print(computeBeta(beta, eta, n_0, S[-1], c1) * eta / gamma)
		delta = SIRG(i, [S[-1], I[-1], R[-1], G[-1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
	print(f'****** reopen on {reopen_date} ******')
	confirmed = df[df.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[reopen_date: end_date]
	mobility = pd.read_csv(mob_file)
	mobility = mobility[mobility.iloc[:, 0] == state]

	s_mobility = mobility.rolling(window=7, axis=1).mean()
	mobility = s_mobility.iloc[0].loc[reopen_date: end_date]

	# mobility = mobility.iloc[0].loc[reopen_date: end_date]
	optimal = minimize(loss_mobility, [1],
	                   args=(beta, gamma, eta, c1, confirmed, mobility, S[-1], I[-1], R[-1], G[-1], n_0, SIRG),
	                   method='L-BFGS-B', bounds=[h_range])

	h = optimal.x[0]
	size = len(confirmed)
	# S[-1] += h * n_0
	# eta = S[-1] / n_0
	for i in range(1, size):
		# print(computeBeta(beta, eta, n_0, S[-1], c1) * eta / gamma)
		# if i < 40:
		print(f'S={S[-1]}')
		print(f'release={h * (mobility[i] - mobility[i - 1]) * n_0}')
		S[-1] += h * (mobility[i] - mobility[i - 1]) * n_0 * eta
		delta = SIRG(i, [S[-1], I[-1], R[-1], G[-1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
	plt.plot(days, [i / 1000 for i in G], label='G')
	plt.plot(days, [i / 1000 for i in I], label='I')
	plt.plot(days, [i / 1000 for i in R], label='R')
	# plt.plot(days, [i / 1000 for i in S], label='S')
	# plt.axhline(y=eta * n_0 / 1000, color='grey', label='S0', linestyle='--')
	print(f'reopen fit:\nh={round(h, 5)}')


def loss(point, confirmed, n_0, SIRG):
	size = len(confirmed)
	beta = point[0]
	gamma = point[1]
	eta = point[2]
	c1 = point[3]
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
		if S[-1] < 0:
			return 1000000

	confirmed_derivative = np.diff(confirmed)
	G_derivative = np.diff(G)
	weights = [Geo ** (n - 1) for n in range(1, size)]
	weights.reverse()
	confirmed_derivative *= weights
	G_derivative *= weights

	metric0 = r2_score(confirmed, G)
	metric1 = r2_score(confirmed_derivative, G_derivative)
	return - (theta * metric0 + (1 - theta) * metric1)


def loss2(point, beta, gamma, eta, c1, confirmed, S0, I0, R0, G0, n_0, SIRG):
	h = point[0]
	size = len(confirmed)
	# S = [S0 + h * n_0]
	S = [S0]
	I = [I0]
	R = [R0]
	G = [G0]
	# eta = S[-1] / n_0
	for i in range(1, size):
		# if i < 40:
		S[-1] += h * n_0
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
		if S[-1] < 0:
			return 1000000

	confirmed_derivative = np.diff(confirmed)
	G_derivative = np.diff(G)
	weights = [Geo ** (n - 1) for n in range(1, size)]
	weights.reverse()
	confirmed_derivative *= weights
	G_derivative *= weights

	metric0 = r2_score(confirmed, G)
	metric1 = r2_score(confirmed_derivative, G_derivative)
	return - (theta * metric0 + (1 - theta) * metric1)


def loss_mobility(point, beta, gamma, eta, c1, confirmed, mobility, S0, I0, R0, G0, n_0, SIRG):
	h = point[0]
	size = len(confirmed)
	# S = [S0 + h * n_0]
	S = [S0]
	I = [I0]
	R = [R0]
	G = [G0]
	# eta = S[-1] / n_0
	for i in range(1, size):
		# if i < 40:
		S[i - 1] += h * (mobility[i] - mobility[i - 1]) * n_0 * eta
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0, c1])
		S.append(S[i - 1] + delta[0])
		I.append(I[i - 1] + delta[1])
		R.append(R[i - 1] + delta[2])
		G.append(G[i - 1] + delta[3])
		if S[i] < 0:
			return 1000000

	confirmed_derivative = np.diff(confirmed)
	G_derivative = np.diff(G)
	weights = [Geo ** (n - 1) for n in range(1, size)]
	weights.reverse()
	confirmed_derivative *= weights
	G_derivative *= weights

	metric0 = r2_score(confirmed, G)
	metric1 = r2_score(confirmed_derivative, G_derivative)
	return - (theta * metric0 + (1 - theta) * metric1)


# does not optimize c1
def fit_state(state):
	df = pd.read_csv('data/population.csv')
	n_0 = df[df['INIT'] == state].iloc[0]['POP']
	plt.rcParams.update({'font.size': 14})
	fig, ax = plt.subplots()
	fig.autofmt_xdate()
	ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	ax.set_ylabel('Cases (Thousands)')
	print(state)
	figFileName = f'figures/{state}_fitting.png'
	fit('data/Confirmed-US.csv', state, n_0, SIRG_sd, 'SIR-SD')
	fit('data/Confirmed-US.csv', state, n_0, SIRG, 'SIR')
	plt.legend()
	# plt.savefig(figFileName)
	plt.show()


# optimizes c1
def fit_state2(state):
	df = pd.read_csv('data/population.csv')
	n_0 = df[df['INIT'] == state].iloc[0]['POP']
	plt.rcParams.update({'font.size': 14})
	fig, ax = plt.subplots()
	fig.autofmt_xdate()
	ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	ax.set_ylabel('Cases (Thousands)')
	print(state)
	figFileName = f'figures/{state}_fitting.png'
	para_sd = fit2('data/Confirmed-US.csv', state, n_0, SIRG_sd, 'SIR-SD')
	para = fit('data/Confirmed-US.csv', state, n_0, SIRG, 'SIR')
	plt.legend()
	# plt.savefig(figFileName)
	# plt.show()
	forecast(state, n_0, para_sd, para, 'data/Confirmed-US.csv')
	return para_sd[4]


def fit_state_reopen(state):
	df = pd.read_csv('data/population.csv')
	n_0 = df[df['INIT'] == state].iloc[0]['POP']
	plt.rcParams.update({'font.size': 14})
	fig, ax = plt.subplots()
	fig.autofmt_xdate()
	ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	ax.set_ylabel('Cases (Thousands)')
	print(state)
	figFileName = f'figures/{state}_fitting.png'
	para_sd = fit3('data/Confirmed-US.csv', state, n_0, SIRG_sd, 'SIR-SD')
	forecast2(state, n_0, para_sd, 'data/Confirmed-US.csv')
	fit_reopen('data/Confirmed-US.csv', state, n_0, SIRG_sd, para_sd)
	plt.legend()
	# plt.savefig(figFileName)
	plt.show()


def fit_county_reopen(state):
	df = pd.read_csv('data/CountyPopulation.csv')
	n_0 = df[df['COUNTY'] == state].iloc[0]['POP']
	plt.rcParams.update({'font.size': 14})
	fig, ax = plt.subplots()
	fig.autofmt_xdate()
	ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	ax.set_ylabel('Cases (Thousands)')
	print(state)
	figFileName = f'figures/{state}_fitting.png'
	para_sd = fit3('data/Confirmed-counties.csv', state, n_0, SIRG_sd, 'SIR-SD')
	forecast2(state, n_0, para_sd, 'data/Confirmed-counties.csv')
	# fit_reopen2('data/Confirmed-counties.csv', 'data/Mobility-counties.csv', state, n_0, SIRG_sd, para_sd)
	fit_reopen('data/Confirmed-counties.csv', state, n_0, SIRG_sd, para_sd)
	plt.legend()
	# plt.savefig(figFileName)
	plt.show()


def fit_state_mobility(state):
	df = pd.read_csv('data/population.csv')
	n_0 = df[df['INIT'] == state].iloc[0]['POP']
	plt.rcParams.update({'font.size': 14})
	fig, ax = plt.subplots()
	fig.autofmt_xdate()
	ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	ax.set_ylabel('Cases (Thousands)')
	print(state)
	df = pd.read_csv('data/Confirmed-US.csv')
	confirmed = df[df.iloc[:, 0] == state]
	for d in confirmed.columns[1:]:
		if confirmed.iloc[0].loc[d] >= I_0:
			break
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[d: end_date]
	df = pd.read_csv('data/Mobility-states.csv')
	mobility = df[df.iloc[:, 0] == state]
	mobility = mobility.iloc[0].loc[d: end_date]
	c_max = 0
	min_loss = 10000
	for c1 in np.arange(0, 1, 1):
		optimal = minimize(loss_mobility, [10, 0.05, 0.02, c1], args=(confirmed, mobility, n_0, SIRG),
		                   method='L-BFGS-B', bounds=[beta_range, gamma_range, eta_range, (c1, c1)])
		current_loss = loss(optimal.x, confirmed, n_0, SIRG_sd)
		if current_loss < min_loss:
			min_loss = current_loss
			c_max = c1

	optimal = minimize(loss_mobility, [10, 0.05, 0.02, c_max], args=(confirmed, mobility, n_0, SIRG), method='L-BFGS-B',
	                   bounds=[beta_range, gamma_range, eta_range, (c_max, c_max)])
	beta = optimal.x[0]
	gamma = optimal.x[1]
	eta = optimal.x[2]
	c1 = optimal.x[3]
	size = len(confirmed)
	S = [n_0 * eta * (1 + p_m * mobility[0])]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0] + n_0 * eta * p_m * (mobility[i] - mobility[i - 1]))
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
	plt.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative cases")
	plt.plot(days, [i / 1000 for i in G], label='G')
	plt.plot(days, [i / 1000 for i in S], label="S")
	plt.plot(days, [i / 1000 for i in I], label="I")
	plt.plot(days, [i / 1000 for i in R], label="R")
	plt.title(state)
	confirmed_derivative = np.diff(confirmed)
	G_derivative = np.diff(G)
	weights = [Geo ** (n - 1) for n in range(1, size)]
	weights.reverse()
	confirmed_derivative *= weights
	G_derivative *= weights
	metric0 = r2_score(confirmed, G)
	metric1 = r2_score(confirmed_derivative, G_derivative)
	print('beta=', beta, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
	print('****************************')
	print('c1=', c1)
	print('****************************')
	print('start date=', d)
	print(f'R2: {metric0} and {metric1}')
	print('loss=', min_loss)
	print()
	plt.legend()
	plt.show()
	return [d, beta, gamma, eta, c1]


# does not optimize c1
def fit_county(state):
	df = pd.read_csv('data/CountyPopulation.csv')
	n_0 = df[df['COUNTY'] == state].iloc[0]['POP']
	# plt.rcParams.update({'font.size': 14})
	# fig, ax = plt.subplots()
	# fig.autofmt_xdate()
	# ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	# ax.set_ylabel('Cases (Thousands)')
	print(state)
	figFileName = f'figures/{state}_fitting.png'
	fit('data/Confirmed-US-counties.csv', state, n_0, SIRG_sd, 'SIR-SD')
	fit('data/Confirmed-US-counties.csv', state, n_0, SIRG, 'SIR')


# plt.legend()
# plt.savefig(figFileName)
# plt.show()


# optimizes c1
def fit_county2(state):
	df = pd.read_csv('data/CountyPopulation.csv')
	n_0 = df[df['COUNTY'] == state].iloc[0]['POP']
	plt.rcParams.update({'font.size': 14})
	fig, ax = plt.subplots()
	fig.autofmt_xdate()
	ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	ax.set_ylabel('Cases (Thousands)')
	print(state)
	figFileName = f'figures/{state}_fitting.png'
	para_sd = fit2('data/Confirmed-counties.csv', state, n_0, SIRG_sd, 'SIR-SD')
	para = fit('data/Confirmed-counties.csv', state, n_0, SIRG, 'SIR')
	plt.legend()
	# plt.savefig(figFileName)
	# plt.show()
	forecast(state, n_0, para_sd, para, 'data/Confirmed-counties.csv')
	return para_sd[4]


# optimizes c1
def fit_city2(state):
	df = pd.read_csv('data/CityPopulation.csv')
	n_0 = df[df['CITY'] == state].iloc[0]['POP']
	plt.rcParams.update({'font.size': 14})
	fig, ax = plt.subplots()
	fig.autofmt_xdate()
	ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	ax.set_ylabel('Cases (Thousands)')
	print(state)
	figFileName = f'figures/{state}_fitting.png'
	para_sd = fit2('data/Confirmed-cities.csv', state, n_0, SIRG_sd, 'SIR-SD')
	para = fit('data/Confirmed-cities.csv', state, n_0, SIRG, 'SIR')
	plt.legend()
	# plt.savefig(figFileName)
	plt.show()
	return para_sd[4]


def fit_all_counties():
	df = pd.read_csv('data/CountyPopulation.csv')
	counties = list(df.iloc[:, 0])
	cs = []
	dens = []
	for s in counties:
		df = pd.read_csv('data/CountyPopulation.csv')
		pop = df[df['COUNTY'] == s].iloc[0]['POP']
		area = df[df['COUNTY'] == s].iloc[0]['AREA']
		dens.append(pop / area)
		print('density', dens[-1])
		c1 = fit_county2(s)
		cs.append(c1)
		plt.scatter(dens[-1], cs[-1])
		plt.annotate('  ' + s, (dens[-1], cs[-1]))

	print('number of counties', len(counties))
	plt.ylabel('c')
	plt.xlabel('density')
	plt.show()


def fit_all_cities():
	# plt.rcParams.update({'font.size': 14})
	df = pd.read_csv('data/CityPopulation.csv')
	counties = list(df.iloc[:, 0])
	cs = []
	dens = []
	for s in counties:
		df = pd.read_csv('data/CityPopulation.csv')
		pop = df[df['CITY'] == s].iloc[0]['POP']
		area = df[df['CITY'] == s].iloc[0]['AREA']
		dens.append(pop / area)
		# print('density', dens[-1])
		c1 = fit_city2(s)
		cs.append(c1)
	# plt.scatter(dens[-1], cs[-1])
	# if s == 'San Francisco' or s == 'New York':
	# 	plt.annotate(s, (dens[-1] - 2000, cs[-1]))
	# else:
	# 	plt.annotate('  ' + s, (dens[-1], cs[-1]))

	print('number of cities', len(counties))
	# plt.ylabel('c')
	# plt.xlabel('Density (persons / square mile)')
	# plt.savefig('figures/density.png')
	# plt.show()
	for i in range(len(counties)):
		print(counties[i], 'c1=', cs[i])


def c_vs_density():
	plt.rcParams.update({'font.size': 14})
	fig, ax = plt.subplots()
	c1 = {'New York': 0.94, 'Philadelphia': 0.95, 'San Francisco': 0.98,
	      'Chicago': 0.81, 'Boston': 0.58}
	df = pd.read_csv('data/CityPopulation.csv')
	cs = []
	dens = []
	for index, row in df.iterrows():
		city = row['CITY']
		dens.append(row['POP'] / row['AREA'])
		cs.append(c1[city])
		print(city, 'density=', dens[-1], 'c1=', cs[-1])
		ax.scatter(dens[-1], cs[-1])
		ax.annotate('  ' + city, (dens[-1], cs[-1]))
	m, b = np.polyfit(dens, cs, 1)
	y = [m * d + b for d in dens]
	# print(mean_squared_error(y, dens))
	minx = min(dens)
	maxx = max(dens)
	x = np.linspace(minx, maxx, 100)
	y = [m * d + b for d in x]
	# ax.plot(dens, y, label=f'y={round(m, 2)}x+{round(b, 2)}')
	ax.plot(x, y)
	x1, x2 = ax.get_xlim()
	ax.set_xlim(x1, x2 * 1.25)
	ax.set_ylabel('c1')
	ax.set_xlabel('Density (persons / square mile)')
	fig.savefig('figures/cs.png')
	plt.show()


def main():
	# fit_state2('MN')
	# fit_state2('FL')
	# fit_state2('IL')
	fit_state2('NY')
	# fit_state2('CA')
	# fit_state2('NJ')
	# fit_county2('NJ-Bergen')
	# fit_all_counties()
	# fit_all_cities()
	# fit_city2('Chicago')
	# fit_city2('New York')
	# c_vs_density()
	# fit_county2('FL-Miami-Dade') # 5/10
	# fit_county2('FL-Broward')
	# fit_county2('TX-Harris--Houston')
	# fit_county2('FL-Palm Beach')
	# fit_county2('TX-Dallas')
	# fit_county2('AZ-Maricopa') # 5/20

	# fit_state_reopen('FL')
	# fit_state_mobility('TX')

	# fit_county_reopen('CA-Los Angeles')
	# fit_county_reopen('NY-New York')

	# fit_county_reopen('FL-Miami-Dade')
	# fit_county_reopen('TX-Harris--Houston')


if __name__ == '__main__':
	main()
