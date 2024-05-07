import numpy as np
import pandas as pd
import datetime
import math
import os
import matplotlib.pyplot as plt
from SIRfunctions import SEIARG
import sympy as sym

end_date = '2020-08-31'
test_ratio = 0.05


def read_sim(SimFolder):
	SimFile = f'{SimFolder}/sim.csv'
	df = pd.read_csv(SimFile)
	S = df[df['series'] == 'S'].iloc[0].iloc[1:]
	E = df[df['series'] == 'E'].iloc[0].iloc[1:]
	I = df[df['series'] == 'I'].iloc[0].iloc[1:]
	A = df[df['series'] == 'A'].iloc[0].iloc[1:]
	IH = df[df['series'] == 'IH'].iloc[0].iloc[1:]
	IN = df[df['series'] == 'IN'].iloc[0].iloc[1:]
	D = df[df['series'] == 'D'].iloc[0].iloc[1:]
	R = df[df['series'] == 'R'].iloc[0].iloc[1:]
	G = df[df['series'] == 'G'].iloc[0].iloc[1:]
	H = df[df['series'] == 'H'].iloc[0].iloc[1:]
	days = S.index

	return S, E, I, A, IH, IN, D, R, G, H, days


def read_data(state, ConfirmFile, PopFile, start_date, reopen_date):
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[start_date:reopen_date]
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
	return confirmed, n_0


def read_para(SimFolder):
	ParaFile = f'{SimFolder}/para.csv'
	df = pd.read_csv(ParaFile)
	beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3 = \
		df.iloc[0]
	return beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3


# simulate combined phase
def simulate_combined(size, SIRG, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
                      a3, h, k, k2, eta, c1, n_0, reopen_day):
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
			release = min(H[-1], r)
			S[-1] += release
			H[-1] -= release
		delta = SIRG(i,
		             [S[i - 1], E[i - 1], I[i - 1], A[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1], beta,
		              gammaE, alpha, kk * gamma, gamma2, gamma3, a1, kk2 * a2, a3, eta2, n_0, c1, H[-1], H0])
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
		betas.append(delta[9])
		if S[-1] < 0:
			result = False
			break
	return result, [S, E, I, A, IH, IN, D, R, G, H, betas]


# simulate combined phase with testing
def simulate_testing(size, SIRG, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3,
                     h, k, k2, eta, c1, n_0, reopen_day, test_day, test_ratio):
	result = True
	H0 = H[0]
	eta2 = eta
	kk = 1
	kk2 = 1
	r = h * H0
	betas = [beta]
	Q = [0]
	for i in range(1, size):

		if i > reopen_day:
			kk = k
			kk2 = k2
			release = min(H[-1], r)
			S[-1] += release
			H[-1] -= release
		IH.append(IH[-1])
		IN.append(IN[-1])
		G.append(G[-1])
		if i >= test_day:
			Q.append(I[-1] * test_ratio)
			IN[-1] += A[-1] * test_ratio
			G[-1] += A[-1] * test_ratio
			I[-1] = I[-1] * (1 - test_ratio)
			A[-1] = A[-1] * (1 - test_ratio)
			IH[-1] += Q[-1] * (kk * gamma) / (kk * gamma + gamma2)
			IN[-1] += Q[-1] * gamma2 / (kk * gamma + gamma2)
		else:
			Q.append(0)

		delta = SIRG(i,
		             [S[i - 1], E[i - 1], I[i - 1], A[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1], beta,
		              gammaE, alpha, kk * gamma, gamma2, gamma3, a1, kk2 * a2, a3, eta2, n_0, c1, H[-1], H0])
		S.append(S[-1] + delta[0])
		E.append(E[-1] + delta[1])
		I.append(I[-1] + delta[2])
		A.append(A[-1] + delta[3])
		# IH.append(IH[-1] + delta[2])
		# IN.append(IN[-1] + delta[3])
		IH[-1] += delta[4]
		IN[-1] += delta[5]
		D.append(D[-1] + delta[6])
		R.append(R[-1] + delta[7])
		# G.append(G[-1] + delta[7])
		G[-1] += delta[8]
		H.append(H[-1])
		betas.append(delta[9])

		if S[-1] < 0:
			result = False
			break
	return result, [S, E, I, A, IH, IN, D, R, G, H, Q, betas]


# simulate without release
def simulate_no_release(size, SIRG, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2, eta, c1, n_0):
	result = True
	H0 = H[0]
	eta2 = eta
	kk = 1
	kk2 = 1
	r = h * H0
	betas = [beta]
	for i in range(1, size):

		# if i > reopen_day:
		# 	kk = k
		# 	kk2 = k2
		# 	release = min(H[-1], r)
		# 	S[-1] += release
		# 	H[-1] -= release
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


def Testing(state, reopen_date):
	print(state)
	SimFolder = f'ASYM/combined2W_{end_date}/{state}'
	ConfirmFile = 'JHU/JHU_Confirmed-counties.csv'
	PopFile = 'JHU/CountyPopulation.csv'
	S, E, I, A, IH, IN, D, R, G, H, days = read_sim(SimFolder)
	reopen_day = days.get_loc(reopen_date)

	start_date = days[0]
	size = len(S)

	for i in range(size):
		if H[i - 1] > H[i]:
			break
	release_index = i

	confirmed, n_0 = read_data(state, ConfirmFile, PopFile, days[0], days[-1])
	beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3 \
		= read_para(SimFolder)

	S0 = S[:1].tolist()
	E0 = E[:1].tolist()
	I0 = I[:1].tolist()
	A0 = A[:1].tolist()
	IH0 = IH[:1].tolist()
	IN0 = IN[:1].tolist()
	D0 = D[:1].tolist()
	R0 = R[:1].tolist()
	G0 = G[:1].tolist()
	H0 = H[:1].tolist()
	Q0 = [0]

	result, [S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, betas0] \
		= simulate_combined(size, SEIARG, S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, beta, gammaE, alpha, gamma, gamma2,
		                    gamma3, a1, a2, a3, h, k, k2, eta, c1, n_0, reopen_day)

	fig = plt.figure()
	ax = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax3 = fig.add_subplot(223)
	ax4 = fig.add_subplot(224)
	ax.set_title('Cumulative cases')
	ax2.set_title('Active cases')
	ax3.set_title('Asymptomatic cases')
	ax4.set_title('Hospitalization')
	ax.plot([i / 1000 for i in G0], label='No testing')
	ax2.plot([i / 1000 for i in I0], label='No testing')
	ax3.plot([i / 1000 for i in A0], label='No testing')
	ax4.plot([i / 1000 for i in IH0], label='No testing')

	S1 = S[:1].tolist()
	E1 = E[:1].tolist()
	I1 = I[:1].tolist()
	A1 = A[:1].tolist()
	IH1 = IH[:1].tolist()
	IN1 = IN[:1].tolist()
	D1 = D[:1].tolist()
	R1 = R[:1].tolist()
	G1 = G[:1].tolist()
	H1 = H[:1].tolist()
	test_day = reopen_day
	# test_day = 10
	# print(I0[test_day])

	result, [S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1, Q1, betas1] \
		= simulate_testing(size, SEIARG, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1, beta, gammaE, alpha, gamma, gamma2,
		                   gamma3, a1, a2, a3, h, k, k2, eta, c1, n_0, reopen_day, test_day, test_ratio)

	# print(G1[-1] - G1[reopen_day], G0[-1] - G0[reopen_day])
	ax.plot([i / 1000 for i in G1], label='Testing')
	ax2.plot([i / 1000 for i in I1], label='Testing')
	ax3.plot([i / 1000 for i in A1], label='Testing')
	ax4.plot([i / 1000 for i in IH1], label='Testing')
	ax.axvline(test_day, color='grey', linestyle=':')
	ax2.axvline(test_day, color='grey', linestyle=':')
	ax3.axvline(test_day, color='grey', linestyle=':')
	ax4.axvline(test_day, color='grey', linestyle=':')
	# ax.legend()
	# ax2.legend()
	# ax3.legend()
	ax4.legend(bbox_to_anchor=(1, -0.15), loc="upper right")
	fig.subplots_adjust(hspace=0.3)
	fig.suptitle(f'{state} testing rate = {test_ratio}')
	fig.savefig(f'ASYM/Testing/ratio={test_ratio}/Testing_{state}_{test_ratio}.png', bbox_inches="tight")
	# plt.show()
	plt.close(fig)

	return G1[-1] - G1[reopen_day], G0[-1] - G0[reopen_day]


def main():
	# state = 'NY-New York'
	# reopen_date = '2020-06-22'
	# Testing(state, reopen_date)
	#
	# state = 'IL-Cook'
	# reopen_date = '2020-06-03'
	# Testing(state, reopen_date)
	#
	# state = 'FL-Miami-Dade'
	# reopen_date = '2020-06-03'
	# Testing(state, reopen_date)
	print('test ratio=', test_ratio)
	if not os.path.exists(f'ASYM/Testing/ratio={test_ratio}/'):
		os.makedirs(f'ASYM/Testing/ratio={test_ratio}/')
	Table = []

	state = 'NY-New York'
	reopen_date = '2020-06-22'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'CA-Los Angeles'
	reopen_date = '2020-06-12'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'FL-Miami-Dade'
	reopen_date = '2020-06-03'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'IL-Cook'
	reopen_date = '2020-06-03'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'TX-Dallas'
	reopen_date = '2020-05-22'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'TX-Harris'
	reopen_date = '2020-06-03'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'AZ-Maricopa'
	reopen_date = '2020-05-28'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'GA-Fulton'
	reopen_date = '2020-06-12'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'NJ-Bergen'
	reopen_date = '2020-06-22'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'PA-Philadelphia'
	reopen_date = '2020-06-05'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'MD-Prince George\'s'
	reopen_date = '2020-06-29'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'NV-Clark'
	reopen_date = '2020-05-29'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'NC-Mecklenburg'
	reopen_date = '2020-05-22'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'LA-Jefferson'
	reopen_date = '2020-06-05'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'CA-Riverside'
	reopen_date = '2020-06-12'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'FL-Broward'
	reopen_date = '2020-06-12'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'NJ-Hudson'
	reopen_date = '2020-06-22'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'MA-Middlesex'
	reopen_date = '2020-06-22'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'OH-Franklin'
	reopen_date = '2020-05-21'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'VA-Fairfax'
	reopen_date = '2020-06-12'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'TN-Shelby'
	reopen_date = '2020-06-15'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'WI-Milwaukee'
	reopen_date = '2020-07-01'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'UT-Salt Lake'
	reopen_date = '2020-05-15'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	state = 'MN-Hennepin'
	reopen_date = '2020-06-04'
	new_test, new_no_test = Testing(state, reopen_date)
	Table.append([state, new_test, new_no_test, new_test / new_no_test])

	out_df = pd.DataFrame(Table, columns=['State - County', 'new_test', 'new_no_test', 'ratio'])
	out_df.to_csv(f'ASYM/Testing/ratio={test_ratio}/ratio={test_ratio}.csv', index=False)

	return


def matrix_inv():
	# sx, sy, rho = sym.symbols('gamma_e gamma beta')
	# matrix = sym.Matrix([[-sx, rho],
	#                      [sx, -sy]])
	# matrix1 = sym.Matrix([[-sx, 0],
	#                       [sx, -sy]])
	# matrixT = sym.Matrix([[0, rho],
	#                       [0, 0]])
	# matrix2 = matrix1.inv()
	# matrixN = matrixT.multiply(matrix2)
	# sym.pprint(matrix1)
	# sym.pprint(matrix2)
	# eigen = matrixN.eigenvals()
	# sym.pprint(sym.simplify(eigen))

	# b, g, S, I = sym.symbols('beta gamma S I')
	# matrix = sym.Matrix([[-b * I, -b * S],
	#                     [b * I, b * I - g]])
	# eigen = matrix.eigenvals()
	# sym.pprint(sym.simplify(eigen))

	b_u, b_v, S_u, S_v, N_u, N_v, g_u, g_v, T_uv, T_vu = sym.symbols(
		'beta_u beta_v S_u S_v N_u N_v gamma_u gamma_v T_uv T_vu')
	c1, c2, I_u = sym.symbols('c_1 c_2 I_u')
	N_u = 8419000
	N_v = 8419000
	I_u = 10
	b_u = 0.4
	b_v = 0.4
	S_u = N_u * 1
	S_v = N_v * 1
	g_u = 0.1
	g_v = 0.1
	T_vu = 0
	T_uv = 0.0089 * N_u
	matrix = sym.Matrix([[b_u * S_u / N_u - g_u - T_uv / N_u, T_vu / N_v],
	                     [T_uv / N_u, b_v * S_v / N_v - g_v - T_vu / N_v]])
	eigen_value = matrix.eigenvals()
	print('eigen values:')
	sym.pprint(sym.simplify(eigen_value))
	print('\neigen vectors:')
	eigen_vec = matrix.eigenvects()
	sym.pprint(eigen_vec)
	# print(len(eigen_vec))
	# print(eigen_vec[0])
	system = eigen_vec[0][2][0]
	# print(type(system))
	system = system.col_insert(1, eigen_vec[1][2][0])
	system = system.col_insert(2, sym.Matrix([I_u, 0]))
	# sym.pprint(system)
	sol = sym.solve_linear_system(system, c1, c2)
	print('\nconstants:')
	sym.pprint(sym.simplify(sol))
	c1 = sol[c1]
	c2 = sol[c2]
	vec1 = eigen_vec[0][2][0]
	vec2 = eigen_vec[1][2][0]
	print(c1, c2)
	print(vec1, vec2)
	lambda1 = list(eigen_value.keys())[0]
	lambda2 = list(eigen_value.keys())[1]
	print(lambda1, lambda2)
	# eta, beta, gamma, gamma2, a1, a2, a3 = sym.symbols('eta beta gamma gamma_2 a_1 a_2 a_3')
	# matrix = sym.Matrix([[eta * beta - gamma - gamma2, 0, 0],
	#                      [gamma, -a1 - a2, 0],
	#                      [gamma2, 0, -a3]])
	# eigen_value = matrix.eigenvals()
	# sym.pprint(sym.simplify(eigen_value))
	# eigen_vec = matrix.eigenvects()
	# sym.pprint(eigen_vec)
	# print(eigen_vec)
	I_us = []
	I_vs = []
	i_range = np.arange(0, 6, 0.1)
	for i in i_range:
		out_Matrix = calcu_I(i, lambda1, lambda2, vec1, vec2, c1, c2)
		I_us.append(out_Matrix[0])
		I_vs.append(out_Matrix[1])
		print(out_Matrix[0], out_Matrix[1])

	fig = plt.figure()
	ax = fig.add_subplot()
	# ax.plot(i_range, [i / 1 for i in I_us], label='I_u')
	ax.plot(i_range, [i / 1 for i in I_vs], label='I_v')
	ax.plot(i_range, [(b_v * S_v / N_v - g_v - T_vu / N_v) * i for i in I_vs], label='internal')
	ax.plot(i_range, [(T_uv / N_u) * i for i in I_us], label='external')
	ax.legend()
	plt.show()
	return


def calcu_I(t, lambda1, lambda2, vec1, vec2, c1, c2):
	print(t)
	out_Matrix = c1 * math.exp(lambda1 * t) * vec1 + c2 * math.exp(lambda2 * t) * vec2
	# print(out_Matrix)
	return out_Matrix


def matrix_test():
	x, y, z = sym.symbols('x y z')
	# system = sym.Matrix([[1, 1, z],
	#                      [2, 3, 5]])
	system = sym.Matrix([[1], [2]])
	system = system.col_insert(1, sym.Matrix([1, 3]))
	system = system.col_insert(2, sym.Matrix([z, 5]))
	sol = sym.solve_linear_system(system, x, y)
	print(sol)
	sym.pprint(sol)
	return


if __name__ == "__main__":
	# main()
	matrix_inv()
# matrix_test()
