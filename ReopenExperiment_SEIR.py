import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from SIRfunctions import SEIRG_sd as SEIRG
from SIRfunctions import computeBeta



# IL
Beta = 38.01584528776274
betaEI = 0.16666666666666666
gamma = 0.10581614103069933
eta = 0.0186628445258563
population = 12671821
fig_file = 'figures/IL_'
c1 = 0.91

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

opa = 0.1


def overlay(alpha, beta_0, g, N, S_0, c, I_p1, S2, h):
	b1 = beta_0 * (1 - c * S_0 / N)
	k1 = b1 * I_p1 / g / N
	k2 = b1 * h / g / N * (g * c / b1 + 1)
	C1 = 1 - g * N / b1 / I_p1 * np.log(g * N / (g * c + b1)) \
	     + g * c * g * N / b1 / I_p1 / (g * c + b1) + g * N / I_p1 / (g * c + b1)
	C2 = C1 + g * N / b1 / I_p1 * np.log(h)

	# t1 = - g * N / b1 / I_p1 * np.log(1 - 1 / (np.exp(k1 * (1 - alpha) + k2)))
	# t2 = - (g * c + b1) / b1 / I_p1 * h / (np.exp(k1 * (1 - alpha) + k2) - 1)
	# t3 = - (g * c + b1) / b1 / I_p1 * h / (k1 + k2) * (1 + alpha * k1 / (k1 + k2)) + g * N / b1 / I_p1 * (
	# 		1 - (k1 * (1 - alpha) + k2))

	# A = g * N / b1 / I_p1
	# B = (g * c + b1) / b1 / I_p1
	# x = k1 * (1 - alpha) + k2
	# ret2 = C1 + 1 / k1 * np.log(h) - 1 / k1 * np.log(np.exp(k2) - 1) - 1 / k2 * np.exp(k2) / (np.exp(k2) - 1) * (
	# 		k1 * (1 - alpha) + k2 * k2 * (1 - alpha) * (1 - alpha) / 2) - k2 / k1 / (np.exp(k2) - 1) * (
	# 			       1 - np.exp(k2) / (np.exp(k2) - 1) * (k1 * (1 - alpha) + k1 * k1 * (1 - alpha) * (1 - alpha) / 2))

	# actual
	ret = C2 - g * N / b1 / I_p1 * (k1 * (1 - alpha) + k2) - g * N / b1 / I_p1 * np.log(
		1 - 1 / (np.exp(k1 * (1 - alpha) + k2))) - (g * c + b1) / b1 / I_p1 * h / (np.exp(k1 * (1 - alpha) + k2) - 1)

	return ret


# simulate a release given the date and previous data
def release_simulator(S0, E0, I0, R0, G0, eta, beta, betaEI, gamma, n_0, start_day, release_day, hiding, threshold):
	S = S0.copy()
	E = E0.copy()
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
			if computeBeta(beta, eta, n_0, S[-1], c1) <= 0:
				sim_result = False
				break
			# point = [S[current_day - 1], I[current_day - 1], R[current_day - 1], G[current_day - 1], beta, gamma, eta, n_0]
			delta = SEIRG(current_day, [S[current_day - 1], E[current_day - 1], I[current_day - 1], R[current_day - 1], G[current_day - 1], beta, betaEI, gamma, eta, n_0, c1])
			S.append(S[-1] + delta[0])
			E.append(E[-1] + delta[1])
			I.append(I[-1] + delta[2])
			R.append(R[-1] + delta[3])
			G.append(G[-1] + delta[4])

		# record the peak
		if not peak and I[current_day] < I[current_day - 1] and current_day > release_day:
			peak = True
			peak_day = current_day - 1

		if current_day == release_day:
			if not release_check:
				# fail the simulation if release day is too early
				sim_result = False
				break
			else:
				# release the hiding
				S[current_day] += hiding

		# start checking the new peak only after I is below the threshold
		if not release_check and I[current_day] <= threshold:
			release_check = True

		# fail the simulation if threshold is exceeded
		if release_check and I[current_day] > threshold:
			sim_result = False
			break

		# end the simulation
		if current_day > start_day + 100 and I[current_day] < 100:
			break

		current_day += 1

	return S, E, I, R, G, peak_day, sim_result


# simulate a 3-release
def SIR_simulator(beta, betaEI, gamma, eta, population, alpha, TH):
	hiding = population * eta / (1 - alpha) * alpha
	r1 = 0
	r2 = 0
	r3 = 0
	p1 = 0
	p2 = 0
	p3 = 0

	# simulate the natural peak
	S, E, I, R, G, peak_day, result = release_simulator([population * eta], [initial_infection], [initial_infection], [0], [initial_infection],
	                                                 eta, beta, betaEI, gamma, population, 0, -1, 0, population * eta)
	if not result:
		return -1, S, E, I, R, G, peak_day, r1, r2, r3, p1, p2, p3
	TH_size = I[peak_day] * TH

	# first release
	S = S[:peak_day + 1]
	E = E[:peak_day + 1]
	I = I[:peak_day + 1]
	R = R[:peak_day + 1]
	G = G[:peak_day + 1]
	r1 = peak_day + 1
	while True:
		S2, E2, I2, R2, G2, p1, result = release_simulator(S, E, I, R, G, eta, beta, betaEI, gamma, population, peak_day, r1,
		                                               hiding / 3, TH_size)

		if result:
			S = S2
			E = E2
			I = I2
			R = R2
			G = G2
			break

		if r1 > peak_day + 1000:
			break

		r1 += 1

	if not result:
		# print('failed to get peak 1')
		return 1, S, E, I, R, G, peak_day, r1, r2, r3, p1, p2, p3

	# second release
	S = S[:p1 + 1]
	E = E[:p1 + 1]
	I = I[:p1 + 1]
	R = R[:p1 + 1]
	G = G[:p1 + 1]
	r2 = p1 + 1
	while True:
		S2, E2, I2, R2, G2, p2, result = release_simulator(S, E, I, R, G, eta, beta, betaEI, gamma, population, p1, r2, hiding / 3,
		                                               TH_size)

		if result:
			S = S2
			E = E2
			I = I2
			R = R2
			G = G2
			break

		if r2 > p1 + 1000:
			break

		r2 += 1

	if not result:
		# print('failed to get peak 2')
		return 2, S, E, I, R, G, peak_day, r1, r2, r3, p1, p2, p3

	# third release
	S = S[:p2 + 1]
	E = E[:p2 + 1]
	I = I[:p2 + 1]
	R = R[:p2 + 1]
	G = G[:p2 + 1]
	r3 = p2 + 1
	while True:
		S2, E2, I2, R2, G2, p3, result = release_simulator(S, E, I, R, G, eta, beta, betaEI, gamma, population, p2, r3,
		                                               hiding / 3, TH_size)

		if result:
			S = S2
			E = E2
			I = I2
			R = R2
			G = G2
			break

		if r3 > p2 + 1000:
			break

		r3 += 1

	if not result:
		# print('failed to get peak 3')
		return 3, S, E, I, R, G, peak_day, r1, r2, r3, p1, p2, p3

	return 0, S, E, I, R, G, peak_day, r1, r2, r3, p1, p2, p3


def get_beta(beta, eta, n_0, S):
	beta_t = []
	for i in range(len(S)):
		beta_t.append(computeBeta(beta, eta, n_0, S[i], c1))
	return beta_t


def plot_simulation(S, E, I, R, G, peak_day, TH):
	plt.clf()
	plt.rcParams.update({'font.size': 14})
	peak_I = I[peak_day]
	beta_t = get_beta(Beta, eta, population, S)

	ax = plt.subplot()
	ax.plot([i / 1000 for i in S], label='S')
	ax.plot([i / 1000 for i in E], label='E')
	ax.plot([i / 1000 for i in I], label='I')
	# ax.plot([i / 1000 for i in R], label='R')
	ax.plot([i / 1000 for i in G], label='G')
	ax.axhline(peak_I * TH / 1000, linestyle=':', color='tab:grey')
	ax.set_ylabel('simulation (thousand)')

	ax2 = ax.twinx()
	ax2.plot(beta_t, '--', label='beta_t', color='tab:green')
	ax2.set_ylabel('beta_t')
	l1, l2 = ax2.get_ylim()
	ax2.set_ylim(0, l2)

	lines_1, labels_1 = ax.get_legend_handles_labels()
	lines_2, labels_2 = ax2.get_legend_handles_labels()
	lines = lines_1 + lines_2
	labels = labels_1 + labels_2
	ax2.legend(lines, labels)
	# ax.legend(lines, labels, loc=0)

	# plt.savefig(fig_file)
	plt.show()


def figure2():
	plt.clf()
	plt.rcParams.update({'font.size': 14})
	# THs = [0.6, 0.7, 0.8, 0.9, 1]
	THs = [1]
	p = []
	for i in range(len(THs)):
		TH = THs[i]
		result, S, I, R, G, peak_day, r1, r2, r3, p1, p2, p3 = SIR_simulator(Beta, gamma, eta, population, alpha, TH)
		print(r3)
		p.append(plt.plot([i / 1000 for i in I], label=f'TH={round(TH * 100)}%'))

	# filling
	for i in reversed(range(len(THs))):
		# if i == len(THs) - 1:
		if i == 0:
			y = p[i][0].get_ydata()
			plt.fill_between(range(0, len(y)), y, color='white')
			plt.fill_between(range(0, len(y)), y, color=p[i][0].get_color(), alpha=opa)
		else:
			y1 = p[i][0].get_ydata()
			y2 = p[i - 1][0].get_ydata()[:len(y1)]
			plt.fill_between(range(0, len(y1)), y1, y2, where=(y1 > y2), color='white')
			plt.fill_between(range(0, len(y1)), y1, y2, where=(y1 > y2), color=p[i][0].get_color(), alpha=opa)

	plt.xlabel('Day')
	plt.ylabel('Active cases (Thousands)')
	plt.legend()
	figFileName = fig_file + 'figure2.png'
	# plt.savefig(figFileName)
	plt.show()


def figure3():
	plt.clf()
	plt.rcParams.update({'font.size': 14})
	THs = [0.6, 0.7, 0.8, 0.9, 1]
	p = []
	for TH in THs:
		result, S, I, R, G, peak_day, r1, r2, r3, p1, p2, p3 = SIR_simulator(Beta, gamma, eta, population, alpha, TH)

		p.append(plt.plot([i / 1000 for i in G], label=f'TH={round(TH * 100)}%'))

	# filling
	for i in reversed(range(len(THs))):
		# if i == len(THs) - 1:
		if i == 0:
			y = p[i][0].get_ydata()
			plt.fill_between(range(0, len(y)), y, color='white')
			plt.fill_between(range(0, len(y)), y, color=p[i][0].get_color(), alpha=opa)
		else:
			y1 = p[i][0].get_ydata()
			y2 = p[i - 1][0].get_ydata()[:len(y1)]
			plt.fill_between(range(0, len(y1)), y1, y2, where=(y1 > y2), color='white')
			plt.fill_between(range(0, len(y1)), y1, y2, where=(y1 > y2), color=p[i][0].get_color(), alpha=opa)

	plt.xlabel('Day')
	plt.ylabel('Cumulative cases (Thousands)')
	plt.legend()
	figFileName = fig_file + 'figure3.png'
	plt.savefig(figFileName)
	# plt.show()


def figure4():
	plt.clf()
	plt.rcParams.update({'font.size': 14})
	beta_range = [0.8, 1, 1.2, 1.4]
	beta_range = [1]
	# THs = [1, 0.9, 0.8, 0.7, 0.6]
	THs = np.arange(0.5, 1.01, 0.01)
	THs = [0.75]
	for b in beta_range:
		rat = []
		# z = []
		for TH in THs:
			result, S, I, R, G, peak_day, r1, r2, r3, p1, p2, p3 = SIR_simulator(b * Beta, gamma, eta, population,
			                                                                     alpha, TH)
			rat.append(I[r1] / I[peak_day] * 100)
			print(rat)
			# z.append(100 * overlay(TH, b * Beta, gamma, population, population * eta, fxns.c(eta), I[peak_day],
			#                        S[r1] - population * eta / (1 - alpha) * alpha / 3,
			#                        population * eta / (1 - alpha) * alpha / 3))

		# band graph
		x = [element * 100 for element in THs]
		y = rat
		plt.plot(x, y, label=str(round(b * 100)) + '% \u03B2')
		# plt.plot(x, z, label=str(round(b * 100)) + '% Beta linear')

	plt.ylabel('Reopen at (%) of Peak')
	plt.xlabel('Peak Threshold (%)')
	plt.legend()
	figFileName = fig_file + 'figure4.png'
	# plt.savefig(figFileName)
	plt.show()


def figure5():
	plt.clf()
	plt.rcParams.update({'font.size': 14})
	# beta_range = [0.8, 1, 1.2, 1.4]
	beta_range = [1]
	# THs = np.arange(0.5, 1.01, 0.01)
	THs = [0.75]
	for b in beta_range:
		rat = []
		for TH in THs:
			result, S, I, R, G, peak_day, r1, r2, r3, p1, p2, p3 = SIR_simulator(b * Beta, gamma, eta, population,
			                                                                     alpha, TH)
			rat.append(r1 - peak_day)
			print(rat)
			# print(f'peak day={peak_day} r1={r1} when TH={round(TH, 2)}')
		# band graph
		x = [round(element * 100) for element in THs]
		y = rat
		plt.plot(x, y, label=str(round(b * 100)) + '% \u03B2')

	plt.ylabel('Delay of reopen (Days)')
	plt.xlabel('Peak Threshold (%)')
	plt.legend()
	figFileName = fig_file + 'figure5.png'
	# plt.savefig(figFileName)
	plt.show()


def figure11():
	plt.clf()
	plt.rcParams.update({'font.size': 14})
	alphas = np.arange(0.4, 0.8, 0.01)
	beta_range = [0.8, 1, 1.2, 1.4]
	TH = 0.75
	for b in beta_range:
		rat = []
		for alpha in alphas:
			result, S, I, R, G, peak_day, r1, r2, r3, p1, p2, p3 = SIR_simulator(b * Beta, gamma, eta, population,
			                                                                     alpha, TH)
			# if len(rat) > 10 and r1 / peak_day * 100 <= rat[-1] and rat[-1] > 300:
			# 	break
			if len(rat) > 10 and r1 - peak_day <= rat[-1] and rat[-1] > 100:
				break
			# rat.append(r1 / peak_day * 100)
			rat.append(r1 - peak_day)
		# if result != 0:
		# 	print(f'failed at peak{result} when {b}beta alpha={alpha}')
		# band graph
		x = [round(element * 100) for element in alphas]
		x = x[:len(rat)]
		y = rat
		plt.plot(x, y, label=str(round(b * 100)) + '% \u03B2')
	# ax.plot(rat, label=f'beta={b}')

	plt.xlabel('Lockdown (%)')
	plt.ylabel('Delay of reopen (Days)')
	plt.legend()
	figFileName = fig_file + 'figure11.png'
	plt.savefig(figFileName)
	# plt.show()


def figure11b():
	plt.clf()
	plt.rcParams.update({'font.size': 14})
	alphas = np.arange(0.1, 0.8, 0.01)
	beta_range = [0.8, 1, 1.2, 1.4]
	TH = 1
	for b in beta_range:
		rat = []
		for alpha in alphas:
			result, S, I, R, G, peak_day, r1, r2, r3, p1, p2, p3 = SIR_simulator(b * Beta, gamma, eta, population,
			                                                                     alpha, TH)
			# if len(rat) > 10 and r1 / peak_day * 100 <= rat[-1] and rat[-1] > 300:
			# 	break
			if len(rat) > 10 and p3 <= rat[-1] and rat[-1] > 100:
				break
			# rat.append(r1 / peak_day * 100)
			rat.append(r3)
		# if result != 0:
		# 	print(f'failed at peak{result} when {b}beta alpha={alpha}')
		# band graph
		x = [round(element * 100) for element in alphas]
		x = x[:len(rat)]
		y = rat
		plt.plot(x, y, label=str(round(b * 100)) + '% \u03B2')
	# ax.plot(rat, label=f'beta={b}')

	plt.xlabel('Lockdown (%)')
	plt.ylabel('End of release (Day)')
	plt.legend()
	figFileName = fig_file + 'figure11b.png'
	plt.savefig(figFileName)
	# plt.show()


def main():
	result, S, E, I, R, G, peak_day, r1, r2, r3, p1, p2, p3 = SIR_simulator(Beta, betaEI, gamma, eta, population, 0.5, 0.75)
	# result = 0: success
	# result = 1: failure at first release
	# result = 2: failure at second release
	# result = 3: failure at third release
	# print(r1, peak_day)
	plot_simulation(S, E, I, R, G, peak_day, TH)

	# figure2()
	# figure3()
	# figure4()
	# figure5()
	# figure11()
	# figure11b()


if __name__ == "__main__":
	main()
