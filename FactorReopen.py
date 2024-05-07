import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
from SIRfunctions import SIRG_sd as SIRG
from SIRfunctions import computeBeta
from SIRfunctions import c
import os
import matplotlib.ticker as mtick

end_date = '2020-08-01'

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

initial_infection = 1
# fraction of susceptible going hiding
alpha = 0.5
# the peaks after each release needs to stay below this fraction of the initial peak
TH = 0.6


def factor_release_simulator(S0, I0, R0, G0, H0, eta, beta, gamma, c1, n_0, start_day, release_day, hiding, factor):
	S = S0.copy()
	I = I0.copy()
	R = R0.copy()
	G = G0.copy()
	H = H0.copy()
	peak_day = start_day
	current_day = start_day
	peak = False
	sim_result = True
	release_complete = False
	release_end = release_day

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
			H.append(0)

		# record the peak
		if not peak and I[current_day] < I[current_day - 1] and current_day > release_day:
			peak = True
			peak_day = current_day - 1

		if release_day != -1 and current_day >= release_day and hiding > 0:
			S_p = (gamma * n_0) / (gamma * c(eta, c1) + (1 - c(eta, c1) * eta) * beta)
			# h = factor * gamma * n_0 / fxns.computeBeta(Beta, eta, n_0, S[current_day]) - S[current_day]
			release = factor * (S_p - S[current_day])
			release = max(0, release)
			release = min(hiding, release)
			# h = max(0, h)
			S[current_day] += release
			H[current_day] = release
			hiding -= release

		if not release_complete and hiding == 0:
			release_end = current_day
			release_complete = True

		# end the simulation
		if current_day > start_day + 100 and I[current_day] < 100:
			break

		current_day += 1

	return S, I, R, G, H, peak_day, release_end, sim_result


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


def figure7(state):
	folder = f'figures/MT_{end_date}/{state}'
	if not os.path.exists(folder):
		os.makedirs(folder)
	[Beta, gamma, eta, c1, Hiding, population] = read_parameters(state)

	plt.clf()
	# plt.rcParams.update({'font.size': 14})
	factors = [1, 0.75, 0.5, 0.25]
	# factors = [1]
	S_0 = population * eta
	# hiding = S_0 / (1 - alpha) * alpha
	hiding = Hiding
	# fig, ax1 = plt.subplots()
	# fig = plt.figure()
	ax1 = plt.subplot()
	# print(fig.axes)
	ax2 = ax1.twinx()
	S, I, R, G, H, peak_day, release_end, sim_result \
		= factor_release_simulator([S_0], [initial_infection], [0], [initial_infection], [0], eta, Beta, gamma, c1,
		                           population, 0, -1, hiding, 0)
	# ax1.plot([i / 1000 for i in I], label='No release')
	for factor in factors:
		S, I, R, G, H, peak_day2, release_end, sim_result \
			= factor_release_simulator([S_0], [initial_infection], [0], [initial_infection], [0], eta, Beta, gamma, c1,
			                           population, 0, peak_day, hiding, factor)

		ax1.plot([i / 1000 for i in I], label=f'I, \u03B1={factor}')
		ax2.plot([h / 1000 for h in H], linestyle='dashdot', label=f'dH, \u03B1={factor}')
		print(release_end)
	# plt.plot([s / 1000 for s in S], label=f'S,factor={factor}')

	lines_1, labels_1 = ax1.get_legend_handles_labels()
	lines_2, labels_2 = ax2.get_legend_handles_labels()
	ymin, ymax = ax1.get_ylim()
	ax1.set_ylim(0, ymax)
	ymin, ymax = ax2.get_ylim()
	ax2.set_ylim(0, ymax * 1.3)
	lines = lines_1 + lines_2
	labels = labels_1 + labels_2
	ax1.legend(lines, labels, loc=0)

	ax1.set_xlabel('Day')
	ax1.set_ylabel('Active cases (Thousands)')
	ax2.set_ylabel('Release (Thousands)')
	pos = ax2.get_position()
	ax2.set_position((pos.x0, pos.y0, pos.width * 0.95, pos.height))
	# fig.savefig(fig_file + 'figure7.png')
	plt.title(state)
	# plt.tight_layout()
	plt.show()


def figure7b():
	plt.clf()
	fig, ax = plt.subplots()
	matplotlib.rcParams.update({'font.size': 14})
	factor = 1
	# factors = [0.5]
	beta_range = [0.8, 1, 1.2, 1.4]
	alphas = np.arange(0.1, 0.8, 0.01)
	for b in beta_range:
		beta = Beta * b
		EoR = []
		for alpha in alphas:
			S_0 = population * eta
			hiding = S_0 / (1 - alpha) * alpha
			S, I, R, G, H, peak_day, release_end, sim_result = factor_release_simulator([S_0], [initial_infection], [0],
			                                                                            [initial_infection], [0], eta,
			                                                                            beta,
			                                                                            gamma, population, 0, -1,
			                                                                            hiding, 0)
			S, I, R, G, H, peak_day2, release_end, sim_result = factor_release_simulator([S_0], [initial_infection],
			                                                                             [0],
			                                                                             [initial_infection], [0], eta,
			                                                                             beta, gamma, population, 0,
			                                                                             peak_day, hiding, factor)
			EoR.append(release_end)
		ax.plot([a * 100 for a in alphas], EoR, label=str(round(b * 100)) + '% \u03B2')

	# plt.title('\u03B1 release')
	ax.set_xlabel('Lockdown (%)', fontsize=14)
	ax.set_ylabel('End of release (Day)', fontsize=14)
	ax.legend()
	pos = ax.get_position()
	ax.set_position((pos.x0 + 0.05, pos.y0, pos.width * 0.95, pos.height))
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	figFileName = fig_file + 'figure7b.png'
	fig.savefig(figFileName)
	plt.show()


def figure89():
	plt.clf()
	plt.rcParams.update({'font.size': 14})
	delays = np.arange(0, 40, 1)
	factors = np.arange(0.2, 1.01, 0.05)
	S_0 = population * eta
	hiding = S_0 / (1 - alpha) * alpha
	release_ends = []

	for i in range(len(factors)):
		factor = factors[i]
		release_ends.append([])
		for j in range(len(delays)):
			delay = delays[j]
			S, I, R, G, H, peak_day, release_end, sim_result = factor_release_simulator([S_0], [initial_infection], [0],
			                                                                            [initial_infection], [0], eta,
			                                                                            Beta,
			                                                                            gamma, population, 0, -1,
			                                                                            hiding,
			                                                                            0)
			S, I, R, G, H, peak_day2, release_end, sim_result = factor_release_simulator([S_0], [initial_infection],
			                                                                             [0],
			                                                                             [initial_infection], [0], eta,
			                                                                             Beta,
			                                                                             gamma, population, 0,
			                                                                             peak_day + delay, hiding,
			                                                                             factor)
			release_ends[i].append(release_end)

	# figure 8
	for j in range(0, len(delays), 5):
		delay = delays[j]
		data = []
		for i in range(len(factors)):
			data.append(release_ends[i][j])
		plt.plot(factors, data, label='delay=' + str(delay))

	plt.xlabel('Factor')
	plt.ylabel('Release End')
	plt.legend(loc='upper center', bbox_to_anchor=(0.6, 1.1), ncol=3)
	plt.savefig(fig_file + 'figure8.png')
	plt.show()

	# figure 9
	for i in range(0, len(factors), 2):
		factor = factors[i]
		data = []
		for j in range(len(delays)):
			data.append(release_ends[i][j])
		plt.plot(delays, data, label=f'\u03B1={round(factor, 1)}')

	plt.xlabel('Delay')
	plt.ylabel('Release End')
	plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.1), ncol=3)
	plt.savefig(fig_file + 'figure9.png')
	plt.show()


def read_parameters(state):
	df = pd.read_csv(f'MT/{state}/{state}_para_init.csv')
	df = df.iloc[0]
	[Beta, gamma, sigma, a1, a2, a3, eta, c1, metric1, metric2] = df

	df = pd.read_csv(f'MT/{state}/{state}_para_reopen.csv')
	df = df.iloc[0]
	[h, Hiding_init, k, metric1, metric2] = df

	df = pd.read_csv('data/CountyPopulation.csv')
	population = df[df.iloc[:, 0] == state].iloc[0]['POP']
	Hiding = eta * population * Hiding_init

	return [Beta, gamma, eta, c1, Hiding, population]


def main():
	# figure7('CA-Los Angeles')
	figure7('TX-Dallas')

	# figure7b()
	# figure89()
	return


if __name__ == "__main__":
	main()
