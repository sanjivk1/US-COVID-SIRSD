import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import FactorReopen as FR
import ReopenExperiment as RE

# IL
# Beta = 10.0168032429735
# gamma = 0.04
# eta = 0.0240239881550231
# population = 12671821
# fig_file = 'figures/IL_'

# NY
# Beta = 19.183888243279583
# gamma = 0.04400678511547825
# eta = 0.022664506710965527
# population = 23628065
# fig_file = 'figures/NY_'

# New York City
Beta = 11.721344692621557
gamma = 0.06057114273052704
eta = 0.04675160234765425
population = 8336817
fig_file = 'figures/NY_'
c1 = 0.94

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

opa = 0.1


def comparision1():
	plt.clf()
	plt.rcParams.update({'font.size': 14})
	THs = np.arange(0.5, 1.01, 0.01)
	# THs = [0.5]
	alpha_list = []
	delays = []
	loss_3P = []
	loss_GR = []
	for TH in THs:
		print(round(TH, 2))
		S_0 = population * eta
		hiding = S_0 / (1 - alpha) * alpha
		result, S, I, R, G, peak_day, r1, r2, r3, p1, p2, p3 = RE.SIR_simulator(Beta, gamma, eta, population, alpha, TH)
		# print(p3)
		# print(I[p3] / I[peak_day])
		# print(sum(I[peak_day:p3]))
		# print(sum(I[p3:]))
		# plt.plot([s / 1000 for s in S], label=f'S:TH={round(TH * 100)}%')
		# plt.plot([i / 1000 for i in I], label=f'I:TH={round(TH * 100)}%')
		cum_hiding = []
		for d in range(0, len(I)):
			if d == 0:
				cum_hiding.append(hiding)
			else:
				cum_hiding.append(cum_hiding[-1])
			if d == r1:
				cum_hiding[-1] -= hiding / 3
			if d == r2:
				cum_hiding[-1] -= hiding / 3
			if d == r3:
				cum_hiding[-1] -= hiding / 3
		loss_3P.append(sum(cum_hiding))
		# print('3-phase\nintegral=', sum(cum_hiding))
		# plt.plot([i / 1000 for i in cum_hiding], label=f'hiding:TH={round(TH * 100)}%')

		S, I, R, G, H, peak_day, release_end, sim_result = FR.factor_release_simulator([S_0], [initial_infection], [0],
		                                                                               [initial_infection], [0], eta,
		                                                                               Beta,
		                                                                               gamma, population, 0, -1, hiding,
		                                                                               0)
		factor = TH
		while factor > 0:
			S, I, R, G, H, peak_day2, release_end, sim_result = FR.factor_release_simulator([S_0], [initial_infection],
			                                                                                [0],
			                                                                                [initial_infection], [0],
			                                                                                eta,
			                                                                                Beta, gamma, population, 0,
			                                                                                peak_day, hiding, factor)
			peakI = I[peak_day:release_end + 1]
			avg = round(sum(peakI) / len(peakI) / I[peak_day], 6)
			if avg <= TH:
				break
			factor -= 0.001
		# print(release_end)
		# print(I[release_end] / I[peak_day])
		# print(sum(I[peak_day:release_end]))
		# print(sum(I[release_end:]))

		cum_hiding = []
		for d in range(0, len(I)):
			if d == 0:
				cum_hiding.append(hiding - H[d])
			else:
				cum_hiding.append(cum_hiding[-1] - H[d])
		loss_GR.append(sum(cum_hiding))
		# print('gradual\nintegral=', sum(cum_hiding))
		alpha_list.append(factor)
		delays.append(p3 - release_end)
		# plt.plot([i / 1000 for i in S], label=f'S:\u03B1={factor}')
		# plt.plot([i / 1000 for i in I], label=f'I:\u03B1={round(factor, 3)}')
		# plt.plot([i / 1000 for i in cum_hiding], label=f'hiding:\u03B1={round(factor, 3)}')
		# plt.xlabel('Day')
		# plt.ylabel('Active cases (Thousands)')
		# plt.legend()
		# plt.title("TH="+str(round(TH, 2)))
		# plt.show()
	THs = [i * 100 for i in THs]
	loss_3P = [i / 1000000 for i in loss_3P]
	loss_GR = [i / 1000000 for i in loss_GR]
	p1 = plt.plot(THs, loss_GR, label='Gradual')
	p2 = plt.plot(THs, loss_3P, label='3-phase')
	plt.fill_between(THs, loss_GR, color=p1[0].get_color(), alpha= opa)
	plt.fill_between(THs, loss_GR, loss_3P, color=p2[0].get_color(), alpha= opa)
	plt.xlabel('Peak Threshold (%)')
	plt.ylabel('Loss (million person days)')
	plt.legend()
	plt.savefig(fig_file + 'loss.png')
	plt.show()

	# plt.clf()
	# plt.plot(THs, alpha_list)
	# plt.xlabel('TH')
	# plt.ylabel('\u03B1')
	# plt.savefig(fig_file + 'alpha.png')
	# plt.show()


	# plt.clf()
	# THs = [round(i * 100) for i in THs]
	# fig, ax1 = plt.subplots()
	# ax1.plot(THs, delays, label='Difference in release end', color='tab:red')
	# ax2 = ax1.twinx()
	# ax2.plot(THs, alpha_list, label='\u03B1', color='tab:blue')
	# lines_1, labels_1 = ax1.get_legend_handles_labels()
	# lines_2, labels_2 = ax2.get_legend_handles_labels()
	# lines = lines_1 + lines_2
	# labels = labels_1 + labels_2
	# ax1.legend(lines, labels)
	# ax1.set_xlabel("TH (%)")
	# ax1.set_ylabel('Days')
	# ax2.set_ylabel('\u03B1')
	# plt.show()


def main():
	comparision1()


if __name__ == "__main__":
	main()
