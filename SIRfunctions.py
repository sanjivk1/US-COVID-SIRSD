import numpy as np
import datetime
import pandas as pd
import math

expon = 1


def weighted_relative_deviation(weights0, cases0, sim0, start_value, num_para):
	for i in range(len(cases0)):
		if cases0[i] >= start_value:
			break
	cases = cases0[i:]
	sim = sim0[i:]
	weights = weights0[i:]
	size = len(cases)
	sum_wt = sum(weights)
	data = [(cases[i] - sim[i]) / sim[i] for i in range(size)]
	metric = math.sqrt(sum([data[i] ** 2 * weights[i] for i in range(size)])
	                   /
	                   ((size - num_para) * sum_wt / size)
	                   )
	return metric


def weighted_deviation(weights0, cases0, sim0, start_value, num_para):
	for i in range(len(cases0)):
		if cases0[i] >= start_value:
			break
	cases = cases0[i:]
	sim = sim0[i:]
	weights = weights0[i:]
	size = len(cases)
	sum_wt = sum(weights)
	data = [(cases[i] - sim[i]) for i in range(size)]
	metric = math.sqrt(sum([data[i] ** 2 * weights[i] for i in range(size)])
	                   /
	                   ((size - num_para) * sum_wt / size)
	                   )
	return metric


def c(eta, c1):
	return c1 / eta


def SIRG_sd(t, y):
	S = y[0]
	I = y[1]
	# R = y[2]
	# G = y[3]
	beta = y[4]
	gamma = y[5]
	eta = y[6]
	n_0 = y[7]
	c1 = y[8]
	beta1 = computeBeta(beta, eta, n_0, S, c1)
	return [-beta1 * S * I / n_0, beta1 * S * I / n_0 - gamma * I, gamma * I, beta1 * S * I / n_0, beta1, 0, 0, 0, 0]


def SIDRG_sd(t, y):
	S = y[0]
	I = y[1]
	IH = y[2]
	IN = y[3]
	D = y[4]
	R = y[5]
	G = y[6]
	beta = y[7]
	gamma = y[8]
	sigma = y[9]
	a1 = y[10]
	a2 = y[11]
	a3 = y[12]
	eta = y[13]
	n_0 = y[14]
	c1 = y[15]
	beta1 = computeBeta(beta, eta, n_0, S, c1)
	return [-beta1 * S * I / n_0,  # dS
	        beta1 * S * I / n_0 - gamma * I,  # dI
	        sigma * gamma * I - (a1 + a2) * IH,  # dIH
	        (1 - sigma) * gamma * I - a3 * IN,  # dIN
	        a2 * IH,  # dD
	        a1 * IH + a3 * IN,  # dR
	        beta1 * S * I / n_0,  # dG
	        beta1,
	        0, 0, 0, 0, 0, 0, 0, 0]


def SIRG_new(t, y):
	S = y[0]
	I = y[1]
	IH = y[2]
	IN = y[3]
	D = y[4]
	R = y[5]
	G = y[6]
	beta = y[7]
	gamma = y[8]
	gamma2 = y[9]
	a1 = y[10]
	a2 = y[11]
	a3 = y[12]
	eta = y[13]
	n_0 = y[14]
	c1 = y[15]
	beta1 = computeBeta(beta, eta, n_0, S, c1)
	return [-beta1 * S * I / n_0,  # dS
	        beta1 * S * I / n_0 - (gamma + gamma2) * I,  # dI
	        gamma * I - (a1 + a2) * IH,  # dIH
	        gamma2 * I - a3 * IN,  # dIN
	        a2 * IH,  # dD
	        a1 * IH + a3 * IN,  # dR
	        beta1 * S * I / n_0,  # dG
	        beta1,
	        0, 0, 0, 0, 0, 0, 0, 0]


def SIRG_combined(t, y):
	S = y[0]
	I = y[1]
	IH = y[2]
	IN = y[3]
	D = y[4]
	R = y[5]
	G = y[6]
	beta = y[7]
	gamma = y[8]
	gamma2 = y[9]
	a1 = y[10]
	a2 = y[11]
	a3 = y[12]
	eta = y[13]
	n_0 = y[14]
	c1 = y[15]
	H = y[16]
	H0 = y[17]
	beta1 = computeBeta_combined(beta, eta, n_0, S, I, H, c1, H0)
	return [-beta1 * S * I / n_0,  # dS
	        beta1 * S * I / n_0 - (gamma + gamma2) * I,  # dI
	        gamma * I - (a1 + a2) * IH,  # dIH
	        gamma2 * I - a3 * IN,  # dIN
	        a2 * IH,  # dD
	        a1 * IH + a3 * IN,  # dR
	        beta1 * S * I / n_0,  # dG
	        beta1,
	        0, 0, 0, 0, 0, 0, 0, 0]


def SIARG(t, y):
	S = y[0]
	I = y[1]
	A = y[2]
	IH = y[3]
	IN = y[4]
	D = y[5]
	R = y[6]
	G = y[7]
	beta = y[8]
	alpha = y[9]
	gamma = y[10]
	gamma2 = y[11]
	gamma3 = y[12]
	a1 = y[13]
	a2 = y[14]
	a3 = y[15]
	eta = y[16]
	n_0 = y[17]
	c1 = y[18]
	H = y[19]
	H0 = y[20]
	beta1 = computeBeta_combined(beta, eta, n_0, S, I, H, c1, H0)
	return [-beta1 * S * (I + A) / n_0,  # dS
	        (1 - alpha) * beta1 * S * (I + A) / n_0 - (gamma + gamma2) * I,  # dI
	        alpha * beta1 * S * (I + A) / n_0 - gamma3 * A,  # dA
	        gamma * I - (a1 + a2) * IH,  # dIH
	        gamma2 * I - a3 * IN,  # dIN
	        a2 * IH,  # dD
	        a1 * IH + a3 * IN + gamma3 * A,  # dR
	        (1 - alpha) * beta1 * S * (I + A) / n_0,  # dG
	        beta1,
	        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def SEIARG_SG(t, y):
	S = y[0]
	E = y[1]
	I = y[2]
	A = y[3]
	IH = y[4]
	IN = y[5]
	D = y[6]
	R = y[7]
	G = y[8]
	beta = y[9]
	gammaE = y[10]
	alpha = y[11]
	gamma = y[12]
	gamma2 = y[13]
	gamma3 = y[14]
	a1 = y[15]
	a2 = y[16]
	a3 = y[17]
	eta = y[18]
	n_0 = y[19]
	c1 = y[20]
	H0 = y[21]
	beta1 = computeBeta_SG(beta, eta, n_0, S, c1, H0)
	return [-beta1 * S * (I + A) / n_0,  # dS
	        beta1 * S * (I + A) / n_0 - gammaE * E,  # dE
	        (1 - alpha) * gammaE * E - (gamma + gamma2) * I,  # dI
	        alpha * gammaE * E - gamma3 * A,  # dA
	        gamma * I - (a1 + a2) * IH,  # dIH
	        gamma2 * I - a3 * IN,  # dIN
	        a2 * IH,  # dD
	        a1 * IH + a3 * IN + gamma3 * A,  # dR
	        (1 - alpha) * gammaE * E,  # dG
	        beta1,
	        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def SEIARG(t, y):
	S = y[0]
	E = y[1]
	I = y[2]
	A = y[3]
	IH = y[4]
	IN = y[5]
	D = y[6]
	R = y[7]
	G = y[8]
	beta = y[9]
	gammaE = y[10]
	alpha = y[11]
	gamma = y[12]
	gamma2 = y[13]
	gamma3 = y[14]
	a1 = y[15]
	a2 = y[16]
	a3 = y[17]
	eta = y[18]
	n_0 = y[19]
	c1 = y[20]
	H = y[21]
	H0 = y[22]
	beta1 = computeBeta_combined(beta, eta, n_0, S, I, H, c1, H0)
	# print(f'dS={-beta1 * S * (I + A) / n_0}, S={S}, E={E}, I={I}, A={A}')
	return [-beta1 * S * (I + A) / n_0,  # dS
	        beta1 * S * (I + A) / n_0 - gammaE * E,  # dE
	        (1 - alpha) * gammaE * E - (gamma + gamma2) * I,  # dI
	        alpha * gammaE * E - gamma3 * A,  # dA
	        gamma * I - (a1 + a2) * IH,  # dIH
	        gamma2 * I - a3 * IN,  # dIN
	        a2 * IH,  # dD
	        a1 * IH + a3 * IN + gamma3 * A,  # dR
	        (1 - alpha) * gammaE * E,  # dG
	        beta1,
	        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def SEIARG_fixed(t, y):
	S = y[0]
	E = y[1]
	I = y[2]
	A = y[3]
	IH = y[4]
	IN = y[5]
	D = y[6]
	R = y[7]
	G = y[8]
	beta = y[9]
	gammaE = y[10]
	alpha = y[11]
	gamma = y[12]
	gamma2 = y[13]
	gamma3 = y[14]
	a1 = y[15]
	a2 = y[16]
	a3 = y[17]
	eta = y[18]
	n_0 = y[19]
	c1 = y[20]
	H = y[21]
	H0 = y[22]
	beta1 = beta
	# beta1 = computeBeta_combined(beta, eta, n_0, S, I, H, c1, H0)
	# print(f'dS={-beta1 * S * (I + A) / n_0}, S={S}, E={E}, I={I}, A={A}')
	return [-beta1 * S * (I + A) / n_0,  # dS
	        beta1 * S * (I + A) / n_0 - gammaE * E,  # dE
	        (1 - alpha) * gammaE * E - (gamma + gamma2) * I,  # dI
	        alpha * gammaE * E - gamma3 * A,  # dA
	        gamma * I - (a1 + a2) * IH,  # dIH
	        gamma2 * I - a3 * IN,  # dIN
	        a2 * IH,  # dD
	        a1 * IH + a3 * IN + gamma3 * A,  # dR
	        (1 - alpha) * gammaE * E,  # dG
	        beta1,
	        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def SIRG(t, y):
	S = y[0]
	I = y[1]
	R = y[2]
	G = y[3]
	beta = y[4]
	gamma = y[5]
	eta = y[6]
	n_0 = y[7]
	# print('SIR', S, I, R, G)
	# print(-beta * S * I / n_0, beta * S * I / n_0 - gamma * I, gamma * I, beta * S * I / n_0)
	return [-beta * S * I / n_0, beta * S * I / n_0 - gamma * I, gamma * I, beta * S * I / n_0, 0, 0, 0, 0]


def SEIRG_sd(t, y):
	S = y[0]
	E = y[1]
	I = y[2]
	R = y[3]
	G = y[4]
	beta = y[5]
	betaEI = y[6]
	gamma = y[7]
	eta = y[8]
	n_0 = y[9]
	c1 = y[10]
	beta1 = computeBeta(beta, eta, n_0, S, c1)
	return [-beta1 * S * I / n_0,
	        beta1 * S * I / n_0 - betaEI * E,
	        betaEI * E - gamma * I,
	        gamma * I,
	        betaEI * E]


def SEIRG(t, y):
	S = y[0]
	E = y[1]
	I = y[2]
	R = y[3]
	G = y[4]
	beta = y[5]
	betaEI = y[6]
	gamma = y[7]
	eta = y[8]
	n_0 = y[9]
	# c1 = y[10]
	# beta1 = computeBeta(beta, eta, n_0, S, c1)
	return [-beta * S * I / n_0,
	        beta * S * I / n_0 - betaEI * E,
	        betaEI * E - gamma * I,
	        gamma * I,
	        betaEI * E]


def computeBeta(beta, eta, n_0, S, c1):
	if S / n_0 < eta:
		beta1 = beta * (1 - c1) / (1 - c1 / eta * (S / n_0))
	else:
		beta1 = beta
	return beta1


def computeBeta_combined(beta, eta, n_0, S, I, H, c1, H0):
	# beta1 = beta * ((1 - c1 * (eta * n_0) / (n_0 - H0 / eta))
	#                 /
	#                 (1 - c1 * S / (n_0 - H / eta)))
	# beta1 = beta * ((1 - c1 * (eta * n_0 + H - H0) / (n_0 - H / eta))
	#                 /
	#                 (1 - c1 * S / (n_0 - H / eta)))
	# beta1 = beta * (1 - c1 * I / (eta * n_0 + H0 - H))
	beta1 = beta * (1 - c1) / (1 - c1 * (S / (eta * n_0 + H0 - H)))

	return beta1


def computeBeta_SG(beta, eta, n_0, S, c1, H0):
	beta1 = beta * (1 - c1) / (1 - c1 * (S / (eta * n_0 + H0)))

	return beta1


def weighting(S, Geo):
	size = len(S)
	weights = [Geo ** n for n in range(size)]
	weights.reverse()
	weighted_S = [S[i] * weights[i] for i in range(size)]
	# print(weights)
	return weighted_S


def make_datetime(start_date, size):
	dates = [datetime.datetime.strptime(start_date, '%Y-%m-%d')]
	for i in range(1, size):
		dates.append(dates[0] + datetime.timedelta(days=i))
	return dates


def make_datetime_end(start_date, end_date):
	return pd.date_range(start_date, end_date)
