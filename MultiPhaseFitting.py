import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score, mean_squared_error
from SIRfunctions import SIRG_sd, SIRG, computeBeta
import datetime
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

np.set_printoptions(threshold=sys.maxsize)
Geo = 0.8

# weight of G in initial fitting
theta = 0.95
# weight of G in release fitting
theta2 = 1

I_0 = 50
beta_range = (0.1, 100)
gamma_range = (0.04, 0.08)
eta_range = (0.001, 0.05)
c1_fixed = (0.9, 0.9)
c1_range = (0, 0.98)
h_range = (0, 10)
end_date = '2020-08-19'
p_m = 1
Hiding = 0.33
delay = 7
change_eta2 = False

fig_row = 7
fig_col = 4


# initial fitting
def loss(point, c1, confirmed, n_0, SIRG):
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


# multiple release with h
def loss2(point, beta, gamma, eta, c1, confirmed, S0, I0, R0, G0, H0, n_0, SIRG):
	h = point[0]
	size = len(confirmed)
	S = [S0]
	I = [I0]
	R = [R0]
	G = [G0]
	H = [H0]
	eta2 = eta
	for i in range(1, size):
		# if i < 40:
		release = min(H[-1], h * funcmod(i))
		S[-1] += release
		H[-1] -= release
		if change_eta2:
			eta2 += release / n_0
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta2, n_0, c1])
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


# one release phase with h and Hiding
def loss3(point, beta, gamma, eta, c1, confirmed, S0, I0, R0, G0, n_0, SIRG):
	h = point[0]
	Hiding_init = point[1]
	size = len(confirmed)
	S = [S0]
	I = [I0]
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
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta2, n_0, c1])
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
	return - (theta2 * metric0 + (1 - theta2) * metric1) + Hiding_init / 100000


# fits until reopen day
def fit_init(file, state, n_0, SIRG, method, reopen_date, ax):
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	for d in confirmed.columns[1:]:
		if confirmed.iloc[0].loc[d] >= I_0:
			break
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed2 = confirmed.iloc[0].loc[d: end_date]
	ax.plot(days, [i / 1000 for i in confirmed2], linewidth=5, linestyle=':', label="Cumulative\nCases")
	# ax3.plot(days, [i / 1000 for i in confirmed2], linewidth=5, linestyle=':', label="Cumulative\nCases")
	# days = list(confirmed.columns)
	# days = days[days.index(d):days.index(reopen_date) + 1]
	# days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[d: reopen_date]
	c_max = 0
	min_loss = 10000
	for c1 in np.arange(0.8, 1, 0.01):
		# optimal = minimize(loss, [10, 0.05, 0.02, c1], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
		#                    bounds=[beta_range, gamma_range, eta_range, (c1, c1)])
		# optimal = minimize(loss, [10, 0.05, 0.02, c1], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
		#                    bounds=[beta_range, gamma_range, eta_range, (c1, c1)])
		optimal = minimize(loss, [10, 0.05, 0.02], args=(c1, confirmed, n_0, SIRG), method='L-BFGS-B',
		                   bounds=[beta_range, gamma_range, eta_range])
		current_loss = loss(optimal.x, c1, confirmed, n_0, SIRG)
		if current_loss < min_loss:
			min_loss = current_loss
			c_max = c1
	# print(f'beta = {round(beta, 8)} , gamma = {round(gamma, 8)} , '
	#       f'eta = {round(eta, 8)}, StartDate = {d} ')

	# optimal = minimize(loss, [10, 0.05, 0.02, c_max], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
	#                    bounds=[beta_range, gamma_range, eta_range, (c_max, c_max)])
	optimal = minimize(loss, [10, 0.05, 0.02], args=(c_max, confirmed, n_0, SIRG), method='L-BFGS-B',
	                   bounds=[beta_range, gamma_range, eta_range])
	beta = optimal.x[0]
	gamma = optimal.x[1]
	eta = optimal.x[2]
	# c1 = optimal.x[3]
	c1 = c_max
	size = len(confirmed)
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	H = [Hiding * n_0 * eta]
	Betas = [beta]
	for i in range(1, size):
		# print(computeBeta(beta, eta, n_0, S[-1], c1))
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
		H.append(H[-1])
		Betas.append(delta[4])

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
		print(f'c1={round(c1, 3)}')
		print(f'\n****** {d} to {reopen_date} ******')
		print(f'R2: {round(metric0, 5)} and {round(metric1, 5)}')
		print('loss=', round(min_loss, 5))
	return [S, I, R, G, H, Betas, d, beta, gamma, eta, c1, 0, 0, metric0, metric1]


# multiple linear release after reopen day
def fit_phase(file, state, n_0, SIRG, para, start, end):
	# plt.axvline(datetime.datetime.strptime(start, '%Y-%m-%d'), linestyle='dashed', color='tab:grey',
	#             label=start)
	[S, I, R, G, H, Betas, d, beta, gamma, eta, c1, h] = para
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	print(f'\n****** {start} to {end} ******')
	confirmed = df[df.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[start: end]
	size = len(confirmed)
	optimal = minimize(loss2, [0],
	                   args=(beta, gamma, eta, c1, confirmed, S[-1], I[-1], R[-1], G[-1], H[-1], n_0, SIRG),
	                   method='L-BFGS-B',
	                   bounds=[(0, H[-1])])
	h = optimal.x[0]
	# size = len(confirmed)
	eta2 = eta
	for i in range(1, size):
		# print(computeBeta(beta, eta, n_0, S[-1], c1) * eta / gamma)
		# if i < 40:
		release = min(H[-1], h * funcmod(i))
		S[-1] += release
		H[-1] -= release
		if change_eta2:
			eta2 += release / n_0
		# print(computeBeta(beta, eta, n_0, S[-1], c1))
		delta = SIRG(i, [S[-1], I[-1], R[-1], G[-1], beta, gamma, eta2, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
		H.append(H[-1])
		Betas.append(delta[4])

	confirmed_derivative = np.diff(confirmed)
	G2 = G[len(G) - size:]
	G_derivative = np.diff(G2)
	weights = [Geo ** (n - 1) for n in range(1, size)]
	weights.reverse()
	confirmed_derivative *= weights
	G_derivative *= weights
	metric0 = r2_score(confirmed, G2)
	metric1 = r2_score(confirmed_derivative, G_derivative)
	print(f'h={round(h, 5)} at {round(h / (n_0 * eta) * 100, 3)}%')
	print(f'R2: {round(metric0, 5)} and {round(metric1, 5)}')
	print('loss=', round(- (theta * metric0 + (1 - theta) * metric1), 5))
	return [S, I, R, G, H, Betas, d, beta, gamma, eta2, c1, h]


# one linear release phase after reopen day
def fit_release(file, state, n_0, SIRG, para, start, end):
	[S, I, R, G, H, Betas, d, beta, gamma, eta, c1, h, Hiding_init, metric0, metric1] = para
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	print(f'\n****** {start} to {end} ******')
	confirmed = df[df.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[start: end]
	size = len(confirmed)
	optimal = minimize(loss3, [0, 0.00001],
	                   args=(beta, gamma, eta, c1, confirmed, S[-1], I[-1], R[-1], G[-1], n_0, SIRG), method='L-BFGS-B',
	                   bounds=[(0, 5 * eta * n_0), (0, 5)])
	h = optimal.x[0]
	Hiding_init = optimal.x[1] * eta * n_0
	for i in range(len(H)):
		H[i] = Hiding_init
	# size = len(confirmed)
	eta2 = eta
	for i in range(1, size):
		# print(computeBeta(beta, eta, n_0, S[-1], c1) * eta / gamma)
		# if i < 40:
		release = min(H[-1], h * funcmod(i))
		S[-1] += release
		H[-1] -= release
		if change_eta2:
			eta2 += release / n_0
		# print(computeBeta(beta, eta, n_0, S[-1], c1))
		delta = SIRG(i, [S[-1], I[-1], R[-1], G[-1], beta, gamma, eta2, n_0, c1])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])
		H.append(H[-1])
		Betas.append(delta[4])

	confirmed_derivative = np.diff(confirmed)
	G2 = G[len(G) - size:]
	G_derivative = np.diff(G2)
	weights = [Geo ** (n - 1) for n in range(1, size)]
	weights.reverse()
	confirmed_derivative *= weights
	G_derivative *= weights
	metric0 = r2_score(confirmed, G2)
	metric1 = r2_score(confirmed_derivative, G_derivative)
	print(f'h={round(h, 5)} at {round(h / n_0 * 100, 5)}%')
	print(f'Hiding={round(Hiding_init, 5)} at {round(Hiding_init / (n_0 * eta) * 100, 3)}%')
	print(f'R2: {round(metric0, 5)} and {round(metric1, 5)}')
	print('loss=', round(- (theta2 * metric0 + (1 - theta2) * metric1), 5))
	return [S, I, R, G, H, Betas, d, beta, gamma, eta2, c1, h, Hiding_init, metric0, metric1]


def funcmod(i):
	# return 0.5 * np.log(1 + i)
	# return 1.00 * np.power(i, -0.4)
	return 1


def forecast(para, days, dates, SIRG, n_0, fig, ax, state, file):
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	for d in confirmed.columns[1:]:
		if confirmed.iloc[0].loc[d] >= I_0:
			break

	confirmed2 = confirmed.iloc[0].loc[d: end_date]
	diff_confirmed2 = np.diff(confirmed2)
	diff_confirmed2 = pd.Series(diff_confirmed2)
	diff_confirmed2 = diff_confirmed2.rolling(7)
	diff_confirmed2 = diff_confirmed2.mean()
	diff_confirmed = [float('nan')]
	diff_confirmed.extend(diff_confirmed2)

	[S, I, R, G, H, Betas, d, beta, gamma, eta, c1, h, Hiding_init, metric0, metric1] = para
	for i in range(len(dates)):
		date = datetime.datetime.strptime(dates[i], '%Y-%m-%d')
		ax.axvline(date, linestyle='dashed', color='tab:grey')
	# ax.axvline(date, linestyle='dashed', color='tab:grey', label=date.strftime('%y-%m-%d'))
	# ax3.axvline(date, linestyle='dashed', color='tab:grey', label=date.strftime('%y-%m-%d'))
	length = datetime.datetime.strptime(dates[-1], '%Y-%m-%d') - datetime.datetime.strptime(dates[-2], '%Y-%m-%d')
	length = length.days
	# i = length
	# while I[-1] > 100:

	eta2 = eta
	days2 = days.copy()

	days = list(confirmed.columns)
	days = days[days.index(d):days.index(dates[-1]) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]

	# for i in range(length + 1, length + 100):
	# 	# i += 1
	# 	release = min(H[-1], h * funcmod(i))
	# 	S[-1] += release
	# 	H[-1] -= release
	# 	if change_eta2:
	# 		eta2 += release / n_0
	# 	delta = SIRG(len(S), [S[-1], I[-1], R[-1], G[-1], beta, gamma, eta2, n_0, c1])
	# 	S.append(S[-1] + delta[0])
	# 	I.append(I[-1] + delta[1])
	# 	R.append(R[-1] + delta[2])
	# 	G.append(G[-1] + delta[3])
	# 	H.append(H[-1])
	# 	Betas.append(delta[4])
	# 	days.append(days[-1] + datetime.timedelta(days=1))

	ax.plot(days, [i / 1000 for i in G], label='Cases\nPredicted')
	# ax3.plot(days, [i / 1000 for i in G], label='Cases\nPredicted')
	# ax.bar(days, [i / 1000 for i in G], label='Cases\nPredicted')
	ax.plot(days, [i / 1000 for i in I], label='Active\nInfectious')
	# ax3.plot(days, [i / 1000 for i in I], label='Active\nInfectious')
	diff_G = [0]
	diff_G.extend(np.diff(G))

	# ax2.plot(days2[len(days2) - len(diff_confirmed):], [i / 1000 for i in diff_confirmed], color='tab:red', label='Reported\nDaily')
	# lines, labels = ax.get_legend_handles_labels()
	# lines2, labels2 = ax2.get_legend_handles_labels()
	# ax2.legend(lines + lines2, labels + labels2, loc=0)
	# plt.plot(days, [i / 1000 for i in R], label='R')

	ax.plot(days, [i / 1000 for i in S], label='S')
	# ax3.plot(days, [i / 1000 for i in S], label='S')
	ax.plot(days, [i / 1000 for i in H], label='H')
	if state == 'MN-Hennepin':
		ax.legend(bbox_to_anchor=(1.7, 1), loc="upper right")
	# plt.legend()
	# plt.subplots_adjust(left=0.25, right=0.75)

	ax2 = ax.twinx()
	# ax2.set_ylabel('New Cases (Thousands)')
	ax2.plot([], [], ' ', color='black', label="Right Axis")
	ax2.plot(days2, [i / 1000 for i in diff_confirmed], label='New\nCases')
	ax2.bar(days, [i / 1000 for i in diff_G], color='tab:red', label='Forecast\nNew\nCases', alpha=0.2)

	if state == 'MN-Hennepin':
		ax2.legend(bbox_to_anchor=(1.7, 1), loc="upper left")

	# ax2.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
	# ax.legend(bbox_to_anchor=(-0.1, 1), loc="upper right")

	# ax4 = ax3.twinx()
	# ax4.set_ylabel('\u03B2')
	# ax4.plot(days, Betas, label='\u03B2', color='tab:red')
	# ax4.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
	# ax3.legend(bbox_to_anchor=(-0.1, 1), loc="upper right")

	[low, up] = ax.get_ylim()
	ax.set_ylim(0, up)
	ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

	[low, up] = ax2.get_ylim()
	ax2.set_ylim(0, up)
	ax2.xaxis.set_major_formatter(DateFormatter("%m/%d"))

	# [low, up] = ax4.get_ylim()
	# ax4.set_ylim(0, up)

	# [low, up] = ax3.get_ylim()
	# ax3.set_ylim(0, up)
	# ax3.xaxis.set_major_formatter(DateFormatter("%m/%d"))

	# fig.autofmt_xdate()
	# plt.savefig(f'figures/twinaxes{state}.png', bbox_inches="tight")
	# plt.show()
	return [S, I, R, G, H, Betas, days]


def fit_multi_phase(state, file, PopFile, dates):
	for i in range(len(dates) - 1):
		date = datetime.datetime.strptime(dates[i], '%Y-%m-%d')
		date += datetime.timedelta(days=delay)
		dates[i] = date.strftime('%Y-%m-%d')
	print()
	print(state)
	print(dates)
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
	# n_0 = n_0 / (1 + Hiding)
	# plt.rcParams.update({'font.size': 14})
	fig, (ax, ax3) = plt.subplots(2, 1, figsize=(8, 10))
	fig.suptitle(state)
	# fig.autofmt_xdate()
	# ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	ax.set_ylabel('Cases (Thousands)')
	ax3.set_ylabel('Cases (Thousands)')
	para_sd = fit_init(file, state, n_0, SIRG_sd, 'SIR-SD', dates[0], ax, ax3)

	for i in range(1, len(dates)):
		para_sd = fit_phase(file, state, n_0, SIRG_sd, para_sd, dates[i - 1], dates[i])
	[S, I, R, G, H, Betas, d, beta, gamma, eta, c1, h] = para_sd
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	[S, I, R, G, H, Betas, days] = forecast(para_sd, days, dates, SIRG_sd, n_0, fig, ax, ax3, state, file)
	# print(Betas)
	# diff_G = [0]
	# diff_G.extend(np.diff(G))
	# plt.plot(days, [i / 1000 for i in diff_G], label='G\'')
	# plt.show()
	return 0


# fit initial and one release phase
def fit_2_phase(state, file, PopFile, dates, df_out, fig, fig_num):
	for i in range(len(dates) - 1):
		date = datetime.datetime.strptime(dates[i], '%Y-%m-%d')
		date += datetime.timedelta(days=delay)
		dates[i] = date.strftime('%Y-%m-%d')
	print()
	print(state)
	print(dates)
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
	dens = df[df.iloc[:, 0] == state].iloc[0]['DENS']
	# n_0 = n_0 / (1 + Hiding)
	# plt.rcParams.update({'font.size': 14})
	ax = fig.add_subplot(fig_row, fig_col, fig_num + 1)
	ax.set_title(state)
	ax.plot([], [], ' ', color='black', label="Left Axis")
	# fig.autofmt_xdate()
	# ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	# ax.set_ylabel('Cases (Thousands)')
	# ax3.set_ylabel('Cases (Thousands)')
	row = [state, n_0]

	# fit the initial phase
	para_sd = fit_init(file, state, n_0, SIRG_sd, 'SIR-SD', dates[0], ax)
	[S, I, R, G, H, Betas, d, beta, gamma, eta, c1, h, Hiding_init, metric0, metric1] = para_sd
	row.extend([beta, gamma, eta, c1, metric0, metric1])
	size_first = len(S)

	# fit the release phase
	para_sd = fit_release(file, state, n_0, SIRG_sd, para_sd, dates[0], dates[1])
	[S, I, R, G, H, Betas, d, beta, gamma, eta, c1, h, Hiding_init, metric0, metric1] = para_sd
	row.extend([Hiding_init, h, metric0, metric1, h / n_0])

	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	[S, I, R, G, H, Betas, days] = forecast(para_sd, days, dates, SIRG_sd, n_0, fig, ax, state, file)

	confirmed2 = confirmed.iloc[0].loc[d: end_date]
	row.extend(peak_ratio(S, I, G, confirmed2, n_0 * eta, size_first))
	row.append(h / eta / n_0 / dens)
	df_out.loc[len(df_out)] = row
	# print(df_out)
	# print(Betas)
	# diff_G = [0]
	# diff_G.extend(np.diff(G))
	# plt.plot(days, [i / 1000 for i in diff_G], label='G\'')
	# plt.show()
	return 0


def peak_ratio(S, I, G, confirmed, s_0, size_first):
	I_peak = max(I[:size_first - 1])
	I_release = I[size_first - 2]
	S_release = S[size_first - 2]

	diff_G = [0]
	diff_G.extend(np.diff(G))
	peak1_sim = np.nanmax(diff_G[:size_first - 1])
	peak2_sim = np.nanmax(diff_G[size_first - 1:])

	diff_confirmed2 = np.diff(confirmed)
	diff_confirmed2 = pd.Series(diff_confirmed2)
	diff_confirmed2 = diff_confirmed2.rolling(7)
	diff_confirmed2 = diff_confirmed2.mean()
	diff_confirmed = [float('nan')]
	diff_confirmed.extend(diff_confirmed2)
	peak1_act = np.nanmax(diff_confirmed[:size_first - 1])
	peak2_act = np.nanmax(diff_confirmed[size_first - 1:])
	# print(diff_confirmed[:size_first])

	return [I_release / I_peak, S_release / s_0, peak2_sim / peak1_sim, peak2_act / peak1_act]


def print_df(df, outfile):
	df.to_csv(outfile, index=False)


def fit_all_2_phase():
	df = pd.DataFrame(
		columns=['county', 'population', 'beta', 'gamma', 'eta', 'c1', 'R2_1', 'R2_2', 'hiding', 'daily release',
		         'R2_3', 'R2_4', 'h/POP', 'I ratio', 'S left', 'peak ratio sim', 'peak ratio act', 'r/(eta*n_0)'])

	plt.rcParams.update({'font.size': 8})
	fig = plt.figure(figsize=(16, 20))
	fig_num = 0

	dates = ['2020-05-28', end_date]
	fit_2_phase('AZ-Maricopa', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-12', end_date]
	fit_2_phase('CA-Los Angeles', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-03', end_date]
	fit_2_phase('FL-Miami-Dade', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-12', end_date]
	fit_2_phase('GA-Fulton', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-03', end_date]
	fit_2_phase('IL-Cook', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-22', end_date]
	fit_2_phase('NY-New York', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-03', end_date]
	fit_2_phase('TX-Harris--Houston', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig,
	            fig_num)

	fig_num += 1
	dates = ['2020-06-22', end_date]
	fit_2_phase('NJ-Bergen', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-05', end_date]
	fit_2_phase('PA-Philadelphia', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-29', end_date]
	fit_2_phase('MD-Prince Georges', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig,
	            fig_num)

	fig_num += 1
	dates = ['2020-05-29', end_date]
	fit_2_phase('NV-Clark', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-05-22', end_date]
	fit_2_phase('NC-Mecklenburg', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-05', end_date]
	fit_2_phase('LA-Jefferson', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-12', end_date]
	fit_2_phase('CA-Riverside', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-12', end_date]
	fit_2_phase('FL-Broward', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-05-22', end_date]
	fit_2_phase('TX-Dallas', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-22', end_date]
	fit_2_phase('NJ-Hudson', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-22', end_date]
	fit_2_phase('MA-Middlesex', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-05-21', end_date]
	fit_2_phase('OH-Franklin', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-12', end_date]
	fit_2_phase('VA-Fairfax', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	# fig_num += 1
	# dates = ['2020-05-11', end_date]
	# fit_2_phase('SC-Charleston', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)
	#
	# fig_num += 1
	# dates = ['2020-06-08', end_date]
	# fit_2_phase('MI-Oakland', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-15', end_date]
	fit_2_phase('TN-Shelby', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-07-01', end_date]
	fit_2_phase('WI-Milwaukee', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-05-15', end_date]
	fit_2_phase('UT-Salt Lake', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	fig_num += 1
	dates = ['2020-06-04', end_date]
	fit_2_phase('MN-Hennepin', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates, df, fig, fig_num)

	print_df(df, 'data/ReleaseResult.csv')
	plt.subplots_adjust(wspace=0.3)
	# fig.autofmt_xdate()
	plt.savefig(f'figures/grid.png', bbox_inches="tight")


# plt.show()


def plot_bar(infile):
	color_dict = {'AZ-Maricopa': 'lime', 'CA-Los Angeles': 'orange', 'FL-Miami-Dade': 'lime', 'GA-Fulton': 'lime',
	              'IL-Cook': 'crimson', 'LA-Jefferson': 'lime', 'MD-Prince Georges': 'crimson', 'MN-Hennepin': 'lime',
	              'NV-Clark': 'lime', 'NJ-Bergen': 'crimson', 'NY-New York': 'crimson', 'NC-Mecklenburg': 'lime',
	              'PA-Philadelphia': 'orange', 'TX-Harris--Houston': 'lime', 'CA-Riverside': 'orange',
	              'FL-Broward': 'lime', 'TX-Dallas': 'lime', 'NJ-Hudson': 'crimson', 'MA-Middlesex': 'crimson',
	              'OH-Franklin': 'lime', 'VA-Fairfax': 'crimson', 'SC-Charleston': 'lime', 'MI-Oakland': 'crimson',
	              'TN-Shelby': 'lime', 'WI-Milwaukee': 'orange', 'UT-Salt Lake': 'orange'}
	df = pd.read_csv(infile, usecols=['county', 'I ratio', 'S left', 'peak ratio sim', 'peak ratio act', 'r/(eta*n_0)'])
	counties = df['county']
	x = df['I ratio']
	# y = df['S left']
	y = df['r/(eta*n_0)']
	sim = df['peak ratio sim']
	act = df['peak ratio act']

	fig = plt.figure()
	ax1 = fig.add_subplot(121, projection='3d')
	ax2 = fig.add_subplot(122, projection='3d')

	ax1.set_xlabel('I_reopen / I_peak')
	# ax1.set_ylabel('S left')
	ax1.set_ylabel('r/(eta*n_0)')
	ax1.set_zlabel('simulated peak ratio')
	ax2.set_xlabel('I_reopen / I_peak')
	# ax2.set_ylabel('S left')
	ax2.set_ylabel('r/(eta*n_0)')
	ax2.set_zlabel('actual peak ratio')

	for i in range(len(counties)):
		if not (counties[i] in color_dict):
			print(f'\'{counties[i]}\'')
		ax1.bar3d(x[i], y[i], 0, 0.01, 0.000002, sim[i], alpha=0.6, color=color_dict.setdefault(counties[i], 'black'))
		ax2.bar3d(x[i], y[i], 0, 0.01, 0.000002, act[i], alpha=0.6, color=color_dict.setdefault(counties[i], 'black'))

	plt.show()


def main():
	# dates = ['2020-05-01', '2020-05-22', '2020-06-03', end_date]
	# fit_multi_phase('TX-Harris--Houston', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates)

	# dates = ['2020-05-08', '2020-06-11', end_date]
	# fit_multi_phase('FL-Miami-Dade', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates)

	# dates = ['2020-05-28', '2020-06-29', end_date]
	# fit_multi_phase('IL-Cook', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates)

	# dates = ['2020-05-08', '2020-05-25', end_date]
	# fit_multi_phase('CA-Los Angeles', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates)

	# dates = ['2020-06-08', '2020-06-22', '2020-07-06', end_date]
	# fit_multi_phase('NY-New York', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates)

	# dates = ['2020-05-28', end_date]
	# fit_multi_phase('AZ-Maricopa', 'data/Confirmed-counties.csv', 'data/CountyPopulation.csv', dates)

	# fit_all_2_phase()
	plot_bar('data/ReleaseResult.csv')


if __name__ == '__main__':
	main()
