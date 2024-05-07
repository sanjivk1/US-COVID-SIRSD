import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from csaps import csaps
import numpy as np
import pandas as pd
import datetime
from SIRfunctions import SIRG_combined as SIRG, make_datetime, make_datetime_end
from matplotlib.dates import DateFormatter
import os

delay = 7


def NextPhase(diff_G, start):
	downward = False
	i = 0
	for i in range(start, len(diff_G) - 14):

		downward = True
		counter = 0
		for j in range(14):
			if diff_G[i + j] < diff_G[i + j + 1]:
				if j == 0:
					downward = False
					break
				counter += 1
				if counter > 3:
					downward = False
					break
		if downward:
			break

	return downward, i


def read_para(SimFolder):
	ParaFile = f'{SimFolder}/para.csv'
	df = pd.read_csv(ParaFile)
	beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3 = df.iloc[0]
	return beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3


def read_sim(SimFolder):
	SimFile = f'{SimFolder}/sim.csv'
	df = pd.read_csv(SimFile)
	S = df[df['series'] == 'S'].iloc[0].iloc[1:]
	I = df[df['series'] == 'I'].iloc[0].iloc[1:]
	IH = df[df['series'] == 'IH'].iloc[0].iloc[1:]
	IN = df[df['series'] == 'IN'].iloc[0].iloc[1:]
	D = df[df['series'] == 'D'].iloc[0].iloc[1:]
	R = df[df['series'] == 'R'].iloc[0].iloc[1:]
	G = df[df['series'] == 'G'].iloc[0].iloc[1:]
	H = df[df['series'] == 'H'].iloc[0].iloc[1:]
	days = S.index

	return S, I, IH, IN, D, R, G, H, days


def release_on(S0, I0, IH0, IN0, D0, R0, G0, H0, size, beta, gamma, gamma2, eta, n_0, c1, k, k2, a1, a2, a3, h,
               release_day):
	S = [S0[0]]
	I = [I0[0]]
	IH = [IH0[0]]
	IN = [IN0[0]]
	D = [D0[0]]
	R = [R0[0]]
	G = [G0[0]]
	H = [H0[0]]
	H0 = H0[0]
	eta2 = eta
	kk = 1
	kk2 = 1
	r = h * H0
	betas = [beta]
	i = 0
	for i in range(1, size):
		# while i <= release_day or I[i] > 50:
		# 	i += 1
		if i > release_day:
			release = min(H[i - 1], r)
			S[i - 1] += release
			H[i - 1] -= release
			kk = k
			kk2 = k2
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
	return S, I, IH, IN, D, R, G, H


def read_data(state, ConfirmFile, PopFile, start_date, reopen_date):
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[start_date:reopen_date]
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
	return confirmed, n_0


def CDC(state, reopen_date, ax3):
	print(state)
	plt.rcParams.update({'font.size': 8})
	fig = plt.figure(figsize=(10, 1.5))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	end_date = '2020-08-31'
	CDC_date = '2020-04-15'

	SimFolder = f'combined2W_{end_date}/{state}'
	ConfirmFile = 'data/Confirmed-counties.csv'
	PopFile = 'data/CountyPopulation.csv'
	S, I, IH, IN, D, R, G, H, days = read_sim(SimFolder)
	reopen_day = days.get_loc(reopen_date)

	start_date = days[0]
	CDC_index = days.get_loc(CDC_date)
	size = len(S)

	for i in range(size):
		if H[i - 1] > H[i]:
			break
	release_index = i
	# print(release_index)
	confirmed, n_0 = read_data(state, ConfirmFile, PopFile, days[0], days[-1])
	beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3 = read_para(
		SimFolder)

	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	diff_G = pd.Series(np.diff(confirmed))

	# release date
	ax1.axvline(x=days[reopen_day], color='grey', linestyle='dashed', label=reopen_date)
	ax3.axvline(x=days[reopen_day], color='grey', linestyle='dashed', label='Reopen date')

	x = np.arange(len(diff_G))
	# ax1.scatter(days[1:], diff_G, label='dG', s=5, color='green')
	ax1.bar(days[1:], diff_G, label='New cases', color='red', alpha=0.5)
	ax3.bar(days[1:], diff_G, label='New cases', color='red', alpha=0.5)
	MA_diff_G = diff_G.rolling(14, min_periods=1).mean()
	# ax1.plot(MA_diff_G, label='MA_dG')

	xs = np.arange(0, len(diff_G), 1)
	ys = csaps(x, MA_diff_G, xs, smooth=0.1)
	ax1.plot(days[1:], ys, label='smooth')
	ax3.plot(days[1:], ys, label='Smoothed new cases')

	# look for downward trajectory
	downward1, ph2 = NextPhase(ys, CDC_index)
	# downward1, ph2 = NextPhase(ys, 0)

	if downward1:
		# ax1.axvline(x=days[ph2], color='blue')
		# ax1.axvline(x=days[ph2 + 14], color='blue')
		ax1.axvspan(days[ph2], days[ph2 + 14], facecolor='green', alpha=0.15)
		ax3.axvspan(days[ph2], days[ph2 + 14], facecolor='green', alpha=0.15)

	downward2, ph3 = NextPhase(ys, ph2 + 15)
	if downward2:
		# ax1.axvline(x=days[ph3], color='red')
		# ax1.axvline(x=days[ph3 + 14], color='red')
		ax1.axvspan(days[ph3], days[ph3 + 14], facecolor='green', alpha=0.25)
		ax3.axvspan(days[ph3], days[ph3 + 14], facecolor='green', alpha=0.25)

	G0 = G.copy()
	# ax2.plot(make_datetime(start_date, len(G)), [i / 1000 for i in G], label='G')
	# ax2.plot(make_datetime(start_date, len(H)), [i / 1000 for i in H], label='H')

	# release based on CDC guideline
	if downward2:
		release_day = ph3 + 21
		S, I, IH, IN, D, R, G, H = release_on(S, I, IH, IN, D, R, G, H, size, beta, gamma, gamma2, eta, n_0, c1, k, k2,
		                                      a1, a2, a3, h, release_day)

		ax2.plot(make_datetime(start_date, size), [i / 1000 for i in G[:size]], label='G_CDC')
	# ax2.plot(make_datetime(start_date, size), [i / 1000 for i in I[:size]], label='I_CDC')
	# ax2.plot(make_datetime(start_date, len(H)), [i / 1000 for i in H], label='H_CDC')

	release_day = release_index + 7
	S, I, IH, IN, D, R, G, H = release_on(S, I, IH, IN, D, R, G, H, size, beta, gamma, gamma2, eta, n_0, c1, k, k2,
	                                      a1, a2, a3, h, release_day)
	# ax2.plot(days, [i / 1000 for i in G], label='1 week')
	ax2.plot(make_datetime(start_date, size), [i / 1000 for i in G[:size]], label='1 week')

	release_day = release_index + 30
	S, I, IH, IN, D, R, G, H = release_on(S, I, IH, IN, D, R, G, H, size, beta, gamma, gamma2, eta, n_0, c1, k, k2,
	                                      a1, a2, a3, h, release_day)
	# ax2.plot(days, [i / 1000 for i in G], label='1 month')
	ax2.plot(make_datetime(start_date, size), [i / 1000 for i in G[:size]], label='1 month')

	ax2.fill_between(make_datetime(start_date, len(G0)), [i / 1000 for i in G0], 0, label='G', facecolor='red',
	                 alpha=0.1)
	ax1.legend()
	ax2.legend()
	l, h = ax1.get_ylim()
	ax1.set_ylim(0, h)
	l, h = ax2.get_ylim()
	ax2.set_ylim(0, h)
	fig.suptitle(state)
	fig.autofmt_xdate()
	fig.savefig(f'CDC/CDC_{state}.png', bbox_inches="tight")
	# plt.show()
	plt.close(fig)

	ax3.set_title(state)
	l, h = ax3.get_ylim()
	ax3.set_ylim(0, h)
	date_form = DateFormatter("%m-%d")
	ax3.xaxis.set_major_formatter(date_form)
	plt.setp(ax3.get_xticklabels(), rotation=25, ha='right')
	return


def CDC_all():
	plt.rcParams.update({'font.size': 8})
	fig = plt.figure(figsize=(14, 18))
	col = 4
	row = 6
	i = 0

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'AZ-Maricopa'
	reopen_date = '2020-05-28'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'CA-Los Angeles'
	reopen_date = '2020-06-12'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'CA-Riverside'
	reopen_date = '2020-06-12'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'FL-Broward'
	reopen_date = '2020-06-12'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'FL-Miami-Dade'
	reopen_date = '2020-06-03'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'GA-Fulton'
	reopen_date = '2020-06-12'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'IL-Cook'
	reopen_date = '2020-06-03'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'LA-Jefferson'
	reopen_date = '2020-06-05'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'MA-Middlesex'
	reopen_date = '2020-06-22'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'MD-Prince Georges'
	reopen_date = '2020-06-29'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'MN-Hennepin'
	reopen_date = '2020-06-04'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NC-Mecklenburg'
	reopen_date = '2020-05-22'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NJ-Bergen'
	reopen_date = '2020-06-22'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NJ-Hudson'
	reopen_date = '2020-06-22'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NV-Clark'
	reopen_date = '2020-05-29'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NY-New York'
	reopen_date = '2020-06-22'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'OH-Franklin'
	reopen_date = '2020-05-21'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'PA-Philadelphia'
	reopen_date = '2020-06-05'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'TN-Shelby'
	reopen_date = '2020-06-15'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'TX-Dallas'
	reopen_date = '2020-05-22'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'TX-Harris--Houston'
	reopen_date = '2020-06-03'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'UT-Salt Lake'
	reopen_date = '2020-05-15'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'VA-Fairfax'
	reopen_date = '2020-06-12'
	CDC(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'WI-Milwaukee'
	reopen_date = '2020-07-01'
	CDC(state, reopen_date, ax)

	ax.legend(bbox_to_anchor=(0, -0.3), loc="upper left")
	fig.subplots_adjust(hspace=0.4, wspace=0.35)
	# plt.rcParams.update({'font.size': 10})
	fig.savefig(f'CDC/CDC_grid.png', bbox_inches="tight")
	plt.close(fig)

	return


def delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax):
	print(f'delay {release_delay} days, {release_speed} time release, end on {sim_end_date}', state)

	# ax = fig.add_subplot()
	end_date = '2020-08-31'

	SimFolder = f'combined2W_{end_date}/{state}'
	ConfirmFile = 'data/Confirmed-counties.csv'
	PopFile = 'data/CountyPopulation.csv'
	S, I, IH, IN, D, R, G, H, days = read_sim(SimFolder)
	reopen_day = days.get_loc(reopen_date)

	start_date = days[0]
	size = len(S)

	for i in range(size):
		if H[i - 1] > H[i]:
			break
	release_index = i

	confirmed, n_0 = read_data(state, ConfirmFile, PopFile, days[0], days[-1])
	beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3 = read_para(
		SimFolder)

	# print(H[release_index - 1], H[release_index], H[release_index + 1])

	# days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]

	days = make_datetime_end(start_date, sim_end_date)
	size = len(days)

	G0 = G.copy()
	I0 = I.copy()

	release_day = release_index
	S1, I1, IH1, IN1, D1, R1, G1, H1 \
		= release_on(S, I, IH, IN, D, R, G, H, size, beta, gamma, gamma2, eta, n_0, c1, k, k2, a1, a2, a3, h,
		             release_day)

	release_day = release_index + release_delay
	S2, I2, IH2, IN2, D2, R2, G2, H2 \
		= release_on(S, I, IH, IN, D, R, G, H, size, beta, gamma, gamma2, eta, n_0, c1, k, k2, a1, a2, a3,
		             release_speed * h, release_day)
	# print(H[release_index - 1], H[release_index], H[release_index + 1])
	if H2[-1] > 0:
		print('H left:', H2[-1])
	ax.plot(days, [i / 1000 for i in G2], label='G_delay')
	# ax.plot(days, [i / 1000 for i in I2], label='I_delay')

	ax.fill_between(days, [i / 1000 for i in G1], 0, label='G', facecolor='red', alpha=0.1)
	# ax.plot(days, [i / 1000 for i in I1], label='I', color='red')

	l, h = ax.get_ylim()
	ax.set_ylim(0, h)
	ax.set_title(state)
	date_form = DateFormatter("%m-%d")
	ax.xaxis.set_major_formatter(date_form)
	plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
	return (G2[-1] - G2[release_index]) / (G1[-1] - G1[release_index])


def delay_all():
	if not os.path.exists('CDC/delay'):
		os.makedirs('CDC/delay')
	fig = plt.figure(figsize=(14, 18))
	col = 4
	row = 6
	i = 0
	release_delay = 30
	release_speed = 1
	sim_end_date = '2020-09-30'

	table = []

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'AZ-Maricopa'
	reopen_date = '2020-05-28'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'CA-Los Angeles'
	reopen_date = '2020-06-12'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'CA-Riverside'
	reopen_date = '2020-06-12'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'FL-Broward'
	reopen_date = '2020-06-12'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'FL-Miami-Dade'
	reopen_date = '2020-06-03'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'GA-Fulton'
	reopen_date = '2020-06-12'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'IL-Cook'
	reopen_date = '2020-06-03'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'LA-Jefferson'
	reopen_date = '2020-06-05'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'MA-Middlesex'
	reopen_date = '2020-06-22'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'MD-Prince Georges'
	reopen_date = '2020-06-29'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'MN-Hennepin'
	reopen_date = '2020-06-04'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NC-Mecklenburg'
	reopen_date = '2020-05-22'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NJ-Bergen'
	reopen_date = '2020-06-22'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NJ-Hudson'
	reopen_date = '2020-06-22'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NV-Clark'
	reopen_date = '2020-05-29'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NY-New York'
	reopen_date = '2020-06-22'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'OH-Franklin'
	reopen_date = '2020-05-21'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'PA-Philadelphia'
	reopen_date = '2020-06-05'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'TN-Shelby'
	reopen_date = '2020-06-15'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'TX-Dallas'
	reopen_date = '2020-05-22'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'TX-Harris--Houston'
	reopen_date = '2020-06-03'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'UT-Salt Lake'
	reopen_date = '2020-05-15'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'VA-Fairfax'
	reopen_date = '2020-06-12'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'WI-Milwaukee'
	reopen_date = '2020-07-01'
	ratio = delay_sim(state, reopen_date, release_delay, release_speed, sim_end_date, ax)
	table.append([state, ratio])

	ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
	fig.subplots_adjust(hspace=0.4, wspace=0.35)
	fig.savefig('CDC/delay/delay_grid.png', bbox_inches="tight")

	print('average ratio =', np.mean([row[1] for row in table]))
	out_df = pd.DataFrame(table, columns=['State - County', 'Ratio'])
	out_df.to_csv('CDC/delay/improvement.csv', index=False)

	return


def main():
	CDC_all()
	# delay_all()
	return


if __name__ == "__main__":
	main()
