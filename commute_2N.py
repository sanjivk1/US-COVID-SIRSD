import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from SIRfunctions import SEIARG, SEIARG_school
from matplotlib.dates import DateFormatter
import os

test_ratio = 0.9
school_beta_multiplier = 2
city_beta_multiplier = 1


class Node:

	# n_0 = 0
	# beta = 0
	# alpha = 0
	# gamma = 0
	# gamma2 = 0
	# gamma3 = 0
	# a1 = 0
	# a2 = 0
	# a3 = 0
	# h = 0
	# Hiding_init = 0
	# k = 0
	# k2 = 0
	# eta = 0
	# c1 = 0
	# state = ''

	def pop(self, day):
		return self.S[day] + self.E[day] + self.I[day] + self.A[day] + self.IH[day] + self.IN[day] + self.D[day] + \
		       self.R[day]

	def read_para(self, SimFolder):
		ParaFile = f'{SimFolder}/{self.state}/para.csv'
		df = pd.read_csv(ParaFile)
		beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3 = \
			df.iloc[0]
		return [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1]

	def read_pop(self, PopFile):
		df = pd.read_csv(PopFile)
		n_0 = df[df.iloc[:, 0] == self.state].iloc[0]['POP']
		return n_0

	def __init__(self, state, SimFolder, PopFile, initial_infection):
		self.state = state
		self.test_rate = test_ratio
		self.n_0 = self.read_pop(PopFile)

		[self.beta,
		 self.gammaE,
		 self.alpha,
		 self.gamma,
		 self.gamma2,
		 self.gamma3,
		 self.a1,
		 self.a2,
		 self.a3,
		 self.h,
		 self.Hiding_init,
		 self.k,
		 self.k2,
		 self.eta,
		 self.c1] = self.read_para(SimFolder)

		self.S = [self.n_0 * self.eta]
		self.E = [0]
		self.I = [initial_infection]
		self.A = [0]
		self.IH = [0]
		self.IN = [0]
		self.Q = [0]
		self.D = [0]
		self.R = [0]
		self.G = [initial_infection]
		self.GA = [initial_infection]
		self.H = [self.Hiding_init * self.n_0 * self.eta]
		self.Betas = []

	def commute_out(self, commuter):
		self.S[-1] -= commuter.S[-1]
		self.E[-1] -= commuter.E[-1]
		self.I[-1] -= commuter.I[-1]
		self.A[-1] -= commuter.A[-1]
		return commuter

	def sim_with_com(self, day, commuter):
		# self.Q.append(0)
		S0 = self.S[-1]
		S1 = commuter.S[-1]
		E0 = self.E[-1]
		E1 = commuter.E[-1]
		I0 = self.I[-1]
		I1 = commuter.I[-1]
		A0 = self.A[-1]
		A1 = commuter.A[-1]
		delta = SEIARG_school(day,
		               [S0 + S1, E0 + E1, I0 + I1, A0 + A1, self.IH[-1], self.IN[-1], self.D[-1], self.R[-1],
		                self.G[-1], self.beta, self.gammaE, self.alpha, self.gamma, self.gamma2, self.gamma3, self.a1,
		                self.a2, self.a3, self.eta, self.n_0, self.c1, self.H[-1], self.H[0]])

		self.S.append(S0 + delta[0] / 2 * S0 / (S0 + S1))
		commuter.S.append(S1 + delta[0] / 2 * S1 / (S0 + S1))

		self.E.append(E0 * (1 - self.gammaE / 2) - delta[0] / 2 * S0 / (S0 + S1))
		commuter.E.append(E1 * (1 - self.gammaE / 2) - delta[0] / 2 * S1 / (S0 + S1))

		self.I.append(I0 * (1 - (self.gamma + self.gamma2) / 2) + E0 * self.gammaE / 2 * (1 - self.alpha))
		commuter.I.append(I1 * (1 - (self.gamma + self.gamma2) / 2) + E1 * self.gammaE / 2 * (1 - self.alpha))

		self.A.append(A0 * (1 - self.gamma3 / 2) + E0 * self.gammaE / 2 * self.alpha)
		commuter.A.append(A1 * (1 - self.gamma3 / 2) + E1 * self.gammaE / 2 * self.alpha)

		self.IH.append(self.IH[-1] + delta[4] / 2)
		self.IN.append(self.IN[-1] + delta[5] / 2)
		self.D.append(self.D[-1] + delta[6] / 2)
		self.R.append(self.R[-1] + delta[7] / 2)
		self.G.append(self.G[-1] + delta[8] / 2)
		self.GA.append(self.GA[-1] + delta[8] / (1 - self.alpha) / 2)
		self.H.append(self.H[-1])
		self.Betas.append(delta[9])

		# Q
		self.IH[-1] += self.Q[-1] * self.gamma / 2
		self.IN[-1] += self.Q[-1] * self.gamma2 / 2
		self.Q.append(self.Q[-1] * (1 - (self.gamma + self.gamma2) / 2))

	def sim(self, day):
		# self.Q.append(0)
		delta = SEIARG_school(day,
		               [self.S[-1], self.E[- 1], self.I[- 1], self.A[- 1], self.IH[- 1], self.IN[- 1], self.D[- 1],
		                self.R[- 1], self.G[- 1], self.beta, self.gammaE, self.alpha, self.gamma, self.gamma2,
		                self.gamma3, self.a1, self.a2, self.a3, self.eta, self.n_0, self.c1, self.H[-1], self.H[0]])

		self.S.append(self.S[-1] + delta[0] / 2)
		self.E.append(self.E[-1] + delta[1] / 2)
		self.I.append(self.I[-1] + delta[2] / 2)
		self.A.append(self.A[-1] + delta[3] / 2)
		self.IH.append(self.IH[-1] + delta[4] / 2)
		self.IN.append(self.IN[-1] + delta[5] / 2)
		self.D.append(self.D[-1] + delta[6] / 2)
		self.R.append(self.R[-1] + delta[7] / 2)
		self.G.append(self.G[-1] + delta[8] / 2)
		self.GA.append(self.GA[-1] + delta[8] / (1 - self.alpha) / 2)
		self.H.append(self.H[-1])
		self.Betas.append(delta[9])

		# Q
		self.IH[-1] += self.Q[-1] * self.gamma / 2
		self.IN[-1] += self.Q[-1] * self.gamma2 / 2
		self.Q.append(self.Q[-1] * (1 - (self.gamma + self.gamma2) / 2))

	def test(self, day, commuter):
		# self.IH[day] += (self.I[day] + commuter.I[day]) * self.test_rate * self.gamma / (self.gamma + self.gamma2)
		# self.IN[day] += (self.I[day] + commuter.I[day]) * self.test_rate * self.gamma2 / (self.gamma + self.gamma2)
		self.Q[day] += (self.I[day] + commuter.I[day]) * self.test_rate

		self.IN[day] += (self.A[day] + commuter.A[day]) * self.test_rate
		self.G[day] += (self.A[day] + commuter.A[day]) * self.test_rate

		self.I[day] -= self.I[day] * self.test_rate
		self.A[day] -= self.A[day] * self.test_rate
		commuter.I[day] -= commuter.I[day] * self.test_rate
		commuter.A[day] -= commuter.A[day] * self.test_rate


class School:

	# n_0 = 0
	# beta = 0
	# alpha = 0
	# gamma = 0
	# gamma2 = 0
	# gamma3 = 0
	# a1 = 0
	# a2 = 0
	# a3 = 0
	# h = 0
	# Hiding_init = 0
	# k = 0
	# k2 = 0
	# eta = 0
	# c1 = 0
	# state = ''

	def pop(self, day):
		return self.S[day] + self.E[day] + self.I[day] + self.A[day] + self.IH[day] + self.IN[day] + self.D[day] + \
		       self.R[day]

	def read_para(self, SimFolder):
		ParaFile = f'{SimFolder}/{self.state}/para.csv'
		df = pd.read_csv(ParaFile)
		beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3 = \
			df.iloc[0]
		return [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1]

	def read_pop(self, PopFile):
		df = pd.read_csv(PopFile)
		n_0 = df[df.iloc[:, 0] == self.state].iloc[0]['POP']
		return n_0

	def __init__(self, state, SimFolder, PopFile, initial_infection):
		self.state = state
		self.test_rate = test_ratio
		self.n_0 = self.read_pop(PopFile)

		[self.beta,
		 self.gammaE,
		 self.alpha,
		 self.gamma,
		 self.gamma2,
		 self.gamma3,
		 self.a1,
		 self.a2,
		 self.a3,
		 self.h,
		 self.Hiding_init,
		 self.k,
		 self.k2,
		 self.eta,
		 self.c1] = self.read_para(SimFolder)

		self.S = [self.n_0 * self.eta]
		self.E = [0]
		self.I = [initial_infection]
		self.A = [0]
		self.IH = [0]
		self.IN = [0]
		self.Q = [0]
		self.D = [0]
		self.R = [0]
		self.G = [initial_infection]
		self.GA = [initial_infection]
		self.H = [self.Hiding_init * self.n_0 * self.eta]
		self.Betas = []

	def commute_out(self, commuter):
		self.S[-1] -= commuter.S[-1]
		self.E[-1] -= commuter.E[-1]
		self.I[-1] -= commuter.I[-1]
		self.A[-1] -= commuter.A[-1]
		return commuter

	def sim_with_com(self, day, commuter):
		# self.Q.append(0)
		S0 = self.S[-1]
		S1 = commuter.S[-1]
		E0 = self.E[-1]
		E1 = commuter.E[-1]
		I0 = self.I[-1]
		I1 = commuter.I[-1]
		A0 = self.A[-1]
		A1 = commuter.A[-1]
		delta = SEIARG_school(day,
		                      [S0 + S1, E0 + E1, I0 + I1, A0 + A1, self.IH[-1], self.IN[-1], self.D[-1], self.R[-1],
		                       self.G[-1], self.beta, self.gammaE, self.alpha, self.gamma, self.gamma2, self.gamma3,
		                       self.a1, self.a2, self.a3, self.eta, self.n_0, self.c1, self.H[-1], self.H[0]])

		self.S.append(S0 + delta[0] / 2 * S0 / (S0 + S1))
		commuter.S.append(S1 + delta[0] / 2 * S1 / (S0 + S1))

		self.E.append(E0 * (1 - self.gammaE / 2) - delta[0] / 2 * S0 / (S0 + S1))
		commuter.E.append(E1 * (1 - self.gammaE / 2) - delta[0] / 2 * S1 / (S0 + S1))

		self.I.append(I0 * (1 - (self.gamma + self.gamma2) / 2) + E0 * self.gammaE / 2 * (1 - self.alpha))
		commuter.I.append(I1 * (1 - (self.gamma + self.gamma2) / 2) + E1 * self.gammaE / 2 * (1 - self.alpha))

		self.A.append(A0 * (1 - self.gamma3 / 2) + E0 * self.gammaE / 2 * self.alpha)
		commuter.A.append(A1 * (1 - self.gamma3 / 2) + E1 * self.gammaE / 2 * self.alpha)

		self.IH.append(self.IH[-1] + delta[4] / 2)
		self.IN.append(self.IN[-1] + delta[5] / 2)
		self.D.append(self.D[-1] + delta[6] / 2)
		self.R.append(self.R[-1] + delta[7] / 2)
		self.G.append(self.G[-1] + delta[8] / 2)
		self.GA.append(self.GA[-1] + delta[8] / (1 - self.alpha) / 2)
		self.H.append(self.H[-1])
		self.Betas.append(delta[9])

		# Q
		self.IH[-1] += self.Q[-1] * self.gamma / 2
		self.IN[-1] += self.Q[-1] * self.gamma2 / 2
		self.Q.append(self.Q[-1] * (1 - (self.gamma + self.gamma2) / 2))

	def sim(self, day):
		# self.Q.append(0)
		delta = SEIARG_school(day,
		                      [self.S[-1], self.E[- 1], self.I[- 1], self.A[- 1], self.IH[- 1], self.IN[- 1],
		                       self.D[- 1], self.R[- 1], self.G[- 1], self.beta, self.gammaE, self.alpha, self.gamma,
		                       self.gamma2, self.gamma3, self.a1, self.a2, self.a3, self.eta, self.n_0, self.c1,
		                       self.H[-1], self.H[0]])

		self.S.append(self.S[-1] + delta[0] / 2)
		self.E.append(self.E[-1] + delta[1] / 2)
		self.I.append(self.I[-1] + delta[2] / 2)
		self.A.append(self.A[-1] + delta[3] / 2)
		self.IH.append(self.IH[-1] + delta[4] / 2)
		self.IN.append(self.IN[-1] + delta[5] / 2)
		self.D.append(self.D[-1] + delta[6] / 2)
		self.R.append(self.R[-1] + delta[7] / 2)
		self.G.append(self.G[-1] + delta[8] / 2)
		self.GA.append(self.GA[-1] + delta[8] / (1 - self.alpha) / 2)
		self.H.append(self.H[-1])
		self.Betas.append(delta[9])

		# Q
		self.IH[-1] += self.Q[-1] * self.gamma / 2
		self.IN[-1] += self.Q[-1] * self.gamma2 / 2
		self.Q.append(self.Q[-1] * (1 - (self.gamma + self.gamma2) / 2))

	def test(self, day, commuter):
		# self.IH[day] += (self.I[day] + commuter.I[day]) * self.test_rate * self.gamma / (self.gamma + self.gamma2)
		# self.IN[day] += (self.I[day] + commuter.I[day]) * self.test_rate * self.gamma2 / (self.gamma + self.gamma2)
		self.Q[day] += (self.I[day] + commuter.I[day]) * self.test_rate

		self.IN[day] += (self.A[day] + commuter.A[day]) * self.test_rate

		self.G[day] += (self.A[day] + commuter.A[day]) * self.test_rate

		self.I[day] -= self.I[day] * self.test_rate
		self.A[day] -= self.A[day] * self.test_rate
		commuter.I[day] -= commuter.I[day] * self.test_rate
		commuter.A[day] -= commuter.A[day] * self.test_rate


class Commuter:
	def __init__(self, S0, E0, I0, A0):
		self.S = [S0]
		self.E = [E0]
		self.I = [I0]
		self.A = [A0]

	def pop(self, day):
		return self.S[day] + self.E[day] + self.I[day] + self.A[day]

	def sim(self, day):
		self.S.append(self.S[-1])
		self.E.append(self.E[-1])
		self.I.append(self.I[-1])
		self.A.append(self.A[-1])


def with_testing(ax, ax2):
	school = School('school', '2N', '2N/Population.csv', 5)
	city = Node('IL-Cook', '2N', '2N/Population.csv', 5)
	city.S[0] *= 1.5
	city.eta *= 1.5
	commuter = Commuter(school.S[-1] * 0.8, 0, 0, 0)
	school.commute_out(commuter)

	school.beta *= school_beta_multiplier
	city.beta *= city_beta_multiplier

	days = 200

	for i in range(days * 2):
		if i % 2 == 0:
			city.sim(i + 1)
			school.sim_with_com(i + 1, commuter)
			school.test(i + 1, commuter)

			# print('Travelling to school')
			print(
				f'{city.pop(i + 1)}+{school.pop(i + 1)}+{commuter.pop(i + 1)}={city.pop(i + 1) + school.pop(i + 1) + commuter.pop(i + 1)}\n')
		else:
			city.sim_with_com(i + 1, commuter)
			school.sim(i + 1)

		# print('Travelling to city')
		print(
			f'{city.pop(i + 1)}+{school.pop(i + 1)}+{commuter.pop(i + 1)}={city.pop(i + 1) + school.pop(i + 1) + commuter.pop(i + 1)}\n')

	print(len(city.S))
	print(len(city.Q))

	# ax.plot(school.S, label='S st')
	# ax.plot(school.E, label='E')
	ax.plot([school.IH[i] + city.IH[i] for i in range(len(school.I))], label='sum IH t')
	# ax.plot(school.I, label='I st')
	# ax.plot(school.A, label='A st')
	# ax.plot(school.Q, label='Q st')
	# ax.plot(school.GA, label='GA')
	# ax.plot(school.IH, label='IH st')
	# ax.plot(school.IN, label='IN')
	# ax.plot(school.D, label='D')
	# ax.legend()
	# ax.set_title('school with testing')
	print('testing peak sum IH', max([school.IH[i] + city.IH[i] for i in range(len(school.I))]))
	# print('testing school peak I', max(school.I))
	# print('testing school peak IH', max(school.IH))
	# print('testing school peak G', max(school.G))
	# print('testing school peak GA', max(school.GA))

	# ax2.plot(city.S, label='S')
	# ax2.plot(city.E, label='E')
	# ax2.plot(city.I, label='I ct')
	# ax2.plot(city.A, label='A ct')
	# ax2.plot(city.Q, label='Q ct')
	# ax2.plot(city.G, label='G')
	ax2.plot(city.IH, label='IH ct')
	# ax2.plot(city.IN, label='IN')
	# ax2.plot(city.D, label='D')
	# ax2.legend()
	# ax2.set_title('city with testing')
	# print('testing city peak G', max(city.G))
	# print('testing city peak GA', max(city.GA))
	# print('testing city peak IH', max(city.IH))

	# ax_beta.plot(city.Betas, label='city testing')
	# ax_beta.plot(school.Betas, label='school testing')

	return


def without_testing(ax, ax2):
	school = School('school', '2N', '2N/Population.csv', 5)
	city = Node('IL-Cook', '2N', '2N/Population.csv', 5)
	city.S[0] *= 1.5
	city.eta *= 1.5
	commuter = Commuter(school.S[-1] * 0.8, 0, 0, 0)
	school.commute_out(commuter)

	school.beta *= school_beta_multiplier
	city.beta *= city_beta_multiplier

	days = 200

	for i in range(days * 2):
		if i % 2 == 0:
			city.sim(i + 1)
			school.sim_with_com(i + 1, commuter)
		# school.test(i + 1, commuter)

		# print('Travelling to school')
		# print(f'{city.pop(i + 1)}+{school.pop(i + 1)}+{commuter.pop(i + 1)}={city.pop(i + 1) + school.pop(i + 1) + commuter.pop(i + 1)}\n')
		else:
			city.sim_with_com(i + 1, commuter)
			school.sim(i + 1)

	# print('Travelling to city')
	# print(f'{city.pop(i + 1)}+{school.pop(i + 1)}+{commuter.pop(i + 1)}={city.pop(i + 1) + school.pop(i + 1) + commuter.pop(i + 1)}\n')

	# ax.plot(school.S, label='S snt')
	# ax.plot(school.E, label='E')
	ax.plot([school.IH[i] + city.IH[i] for i in range(len(school.I))], label='sum IH nt')
	# ax.plot(school.I, label='I snt')
	# ax.plot(school.A, label='A snt')
	# ax.plot(school.Q, label='Q snt')
	# ax.plot(school.G, label='G')
	# ax.plot(school.IH, label='IH snt')
	# ax.plot(school.IN, label='IN')
	# ax.plot(school.D, label='D')
	# ax.legend()
	# ax.set_title('school without testing')
	print('no testing peak sum IH', max([school.IH[i] + city.IH[i] for i in range(len(school.I))]))
	# print('no testing school peak I', max(school.I))
	# print('no testing school peak IH', max(school.IH))
	# print('no testing school peak G', max(school.G))
	# print('no testing school peak GA', max(school.GA))

	# ax2.plot(city.S, label='S')
	# ax2.plot(city.E, label='E')
	# ax2.plot(city.I, label='I cnt')
	# ax2.plot(city.A, label='A cnt')
	# ax2.plot(city.Q, label='Q cnt')
	# ax2.plot(city.G, label='G')
	ax2.plot(city.IH, label='IH cnt')
	# ax2.plot(city.IN, label='IN')
	# ax2.plot(city.D, label='D')
	# ax2.legend()
	# ax2.set_title('city without testing')
	# print('no testing city peak G', max(city.G))
	# print('no testing city peak GA', max(city.GA))
	# print('no testing city peak IH', max(city.IH))

	# ax_beta.plot(city.Betas, label='city no testing')
	# ax_beta.plot(school.Betas, label='school no testing')

	return


def main():
	fig = plt.figure()
	ax = fig.add_subplot(121)
	# ax2 = fig.add_subplot(232)
	ax3 = fig.add_subplot(122)
	# ax4 = fig.add_subplot(234)
	# ax5 = fig.add_subplot(235)
	with_testing(ax, ax3)
	without_testing(ax, ax3)
	fig.suptitle(f'{test_ratio} testing in school')
	# ax5.set_title('beta')
	# ax5.legend()
	ax.legend()
	ax3.legend()
	plt.show()


if __name__ == "__main__":
	main()
