import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from SIRfunctions import SIRG_sd as SIRG
from SIRfunctions import computeBeta
import os
import datetime

file = 'data/Confirmed-counties.csv'
end_date = '2020-09-23'

# FL-Miami-Dade
# Beta = 6.92419823003761
# gamma = 0.04
# eta = 0.04487935915566999
# population = 2716940
# fig_file = 'figures/FL-Miami-Dade_'
# c1 = 0.98
# Hiding = 129210.92644

# IL
# Beta = 10.0168032429735
# gamma = 0.04
# eta = 0.0240239881550231
# population = 12671821
# fig_file = 'figures/IL_'
# c1 = 0.9

# NY
# Beta = 19.183888243279583
# gamma = 0.04400678511547825
# eta = 0.022664506710965527
# population = 23628065
# fig_file = 'figures/NY_'
# c1 = 0.9

# New York City
# Beta = 11.721344692621557
# gamma = 0.06057114273052704
# eta = 0.04675160234765425
# population = 8336817
# fig_file = 'figures/NY_'
# c1 = 0.94

# Chicago
# Beta = 7.219294381528623
# gamma = 0.08
# eta = 0.03603237406213456
# population = 2693976
# fig_file = 'figures/IL_'
# c1 = 0.81

initial_infection = 68
# fraction of susceptible going hiding
alpha = 0.5
# the peaks after each release needs to stay below this fraction of the initial peak
TH = 0.6


def gradual_release_simulator(S0, I0, R0, G0, eta, beta, gamma, c1, n_0, start_day, release_day, hiding, daily_release):
	S = S0.copy()
	I = I0.copy()
	R = R0.copy()
	G = G0.copy()
	peak_day = start_day
	current_day = start_day
	peak = False
	sim_result = True
	release_check = False

	while True:

		# update SIR for the current day
		if current_day > start_day:
			delta = SIRG(current_day,
			             [S[current_day - 1], I[current_day - 1], R[current_day - 1], G[current_day - 1], beta,
			              gamma, eta, n_0, c1])
			S.append(S[-1] + delta[0])
			I.append(I[-1] + delta[1])
			R.append(R[-1] + delta[2])
			G.append(G[-1] + delta[3])

		# record the peak
		if not peak and I[current_day] < I[current_day - 1] and current_day > release_day:
			peak = True
			peak_day = current_day - 1

		if release_day != -1 and current_day >= release_day and hiding > 0:
			h = min(hiding, daily_release)
			S[current_day] += h
			hiding -= h

		# end the simulation
		if current_day > start_day + 100 and I[current_day] < 100:
			break

		current_day += 1

	return S, I, R, G, peak_day, sim_result


def get_beta(beta, eta, n_0, S):
	beta_t = []
	for i in range(len(S)):
		beta_t.append(computeBeta(beta, eta, n_0, S[i], c1))
	return beta_t


def plot_simulation(S, I, R, G):
	plt.clf()
	plt.rcParams.update({'font.size': 14})
	beta_t = get_beta(Beta, eta, population, S)

	fig, ax = plt.subplots()
	S = [i / 1000 for i in S]
	I = [i / 1000 for i in I]
	ax.plot(S, label='S', color='tab:green')
	ax.plot(I, label='I', color='tab:red')
	ax.set_ylabel('simulation (thousand)')
	ax2 = ax.twinx()
	ax2.plot(beta_t, label='beta_t', color='tab:blue')
	ax2.set_ylabel('beta_t')
	lines_1, labels_1 = ax.get_legend_handles_labels()
	lines_2, labels_2 = ax2.get_legend_handles_labels()

	lines = lines_1 + lines_2
	labels = labels_1 + labels_2

	ax.legend(lines, labels, loc=0)
	plt.savefig(fig_file)
	plt.show()


def figure1():
	plt.clf()
	plt.rcParams.update({'font.size': 14})
	rate = 0.015
	S_0 = population * eta
	hiding = S_0 / (1 - alpha) * alpha
	daily_release = hiding * rate
	S, I, R, G, peak_day, sim_result = gradual_release_simulator([S_0], [initial_infection], [0],
	                                                             [initial_infection], eta, Beta, gamma, population,
	                                                             0, -1, hiding, daily_release)
	plt.plot([i / 1000 for i in I], label='No release')
	delta_I = np.diff(I)
	release_day = np.argmax(delta_I) + 15
	S, I, R, G, peak_day2, sim_result = gradual_release_simulator([S_0], [initial_infection], [0],
	                                                              [initial_infection], eta, Beta, gamma, population,
	                                                              0, release_day, hiding, daily_release)
	plt.plot([i / 1000 for i in I], label='2 weeks after\nnew cases peak')
	S, I, R, G, peak_day2, sim_result = gradual_release_simulator([S_0], [initial_infection], [0],
	                                                              [initial_infection], eta, Beta, gamma, population,
	                                                              0, peak_day + 15, hiding, daily_release)
	plt.plot([i / 1000 for i in I], label='2 weeks after\nactive cases peak')
	plt.xlabel('Day')
	plt.ylabel('Active cases (Thousands)')
	plt.legend()
	plt.savefig(fig_file + 'figure1.png')
	plt.show()


def figure6(state):
	folder = f'figures/MT_{end_date}/{state}'
	if not os.path.exists(folder):
		os.makedirs(folder)
	[Beta, gamma, eta, c1, h, Hiding, population, release_day, I_0] = read_parameters(state)
	plt.clf()
	plt.rcParams.update({'font.size': 14})
	rates = [0.25, 0.5, 0.75, 1]
	S_0 = population * eta
	# hiding = S_0 / (1 - alpha) * alpha
	hiding = Hiding
	p = []

	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	for d in confirmed.columns[1:]:
		if confirmed.iloc[0].loc[d] >= initial_infection:
			break
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed2 = confirmed.iloc[0].loc[d: end_date]
	p.append(plt.plot([i / 1000 for i in confirmed2], linewidth=5, linestyle=':', label='Cumulative'))

	# S, I, R, G, peak_day, sim_result = \
	# 	gradual_release_simulator([S_0], [initial_infection], [0], [initial_infection], eta, Beta, gamma, c1,
	# 	                          population, 0, -1, hiding, 0)
	S, I, R, G, peak_day, sim_result = \
		gradual_release_simulator([S_0], [I_0], [0], [I_0], eta, Beta, gamma, c1, population, 0, -1, hiding, 0)
	# plt.plot([i / 1000 for i in I], label='No release')
	S = S[:peak_day + 1]
	I = I[:peak_day + 1]
	R = R[:peak_day + 1]
	G = G[:peak_day + 1]
	for rate in rates:
		daily_release = h * rate

		S2, I2, R2, G2, peak_day2, sim_result = \
			gradual_release_simulator(S, I, R, G, eta, Beta, gamma, c1, population, peak_day, release_day, hiding,
			                          daily_release)
		p.append(plt.plot([i / 1000 for i in G2], label=f'\u03BA={round(rate * 100, 1)}%'))

	plt.xlabel('Day')
	plt.ylabel('Active cases (Thousands)')
	plt.legend()
	# plt.savefig(fig_file + 'figure6.png')
	plt.title(state)
	plt.tight_layout()
	plt.show()


def read_parameters(state):
	df = pd.read_csv(f'MT_{end_date}/{state}/{state}_para_init.csv')
	df = df.iloc[0]
	[Beta, gamma, sigma, a1, a2, a3, eta, c1, metric1, metric2] = df

	df = pd.read_csv(f'MT_{end_date}/{state}/{state}_para_reopen.csv')
	df = df.iloc[0]
	[h, Hiding_init, k, metric1, metric2] = df

	df = pd.read_csv('data/CountyPopulation.csv')
	population = df[df.iloc[:, 0] == state].iloc[0]['POP']
	Hiding = eta * population * Hiding_init

	df = pd.read_csv(f'MT_{end_date}/{state}/{state}_sim_init.csv')
	release_day = len(df.columns) - 2
	I_0 = df.iloc[1, 1]

	return [Beta, gamma, eta, c1, h, Hiding, population, release_day, I_0]


def main():
	# result, S, I, R, G, peak_day, r1, r2, r3, p1, p2, p3 = SIR_simulator(Beta, gamma, eta, population, alpha, TH)
	# figure1()
	# figure6('TX-Harris--Houston')
	figure6('IL-Cook')


if __name__ == "__main__":
	main()
