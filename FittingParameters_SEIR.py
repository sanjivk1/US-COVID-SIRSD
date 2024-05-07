import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score, mean_squared_error
from SIRfunctions import SEIRG_sd, SEIRG
import datetime
from matplotlib.dates import DateFormatter

"""
Authors: Yi Zhang, Mohit Hota, Sanjiv Kapoor

This code is motivated by https://github.com/Lewuathe/COVID19-SIR
"""

np.set_printoptions(threshold=sys.maxsize)
Geo = 0.8
I_0 = 50
theta = 0.9
beta_range = (0.1, 200)
betaEI_range = (1 / 6, 1 / 4)
gamma_range = (1 / 14, 1 / 4)
eta_range = (0.001, 0.05)
c1_fixed = (0.9, 0.9)
c1_range = (0, 0.98)
iterations = 50000
end_date = '2020-05-01'


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
	plt.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="CUM")

	# SEIRG_sd
	beta = para_sd[1]
	betaEI = para_sd[2]
	gamma = para_sd[3]
	eta = para_sd[4]
	c1 = para_sd[5]
	S = [n_0 * eta]
	E = [confirmed[0]]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	print('SEIRG-SD')
	print('beta=', beta, 'betaEI=', betaEI, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
	print('****************************')
	print('c1=', c1)
	print('****************************')
	for i in range(1, size):
		delta = SEIRG_sd(i, [S[i - 1], E[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, betaEI, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		E.append(E[-1] + delta[1])
		I.append(I[-1] + delta[2])
		R.append(R[-1] + delta[3])
		G.append(G[-1] + delta[4])
	plt.plot(days, [i / 1000 for i in G], label='SEIRG-SD')

	# SEIRG
	beta = para[1]
	betaEI = para[2]
	gamma = para[3]
	eta = para[4]
	S = [n_0 * eta]
	E = [confirmed[0]]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	print('SEIRG')
	print('beta=', beta, 'betaEI=', betaEI, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
	for i in range(1, size):
		delta = SEIRG(i, [S[i - 1], E[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, betaEI, gamma, eta, n_0])
		S.append(S[-1] + delta[0])
		E.append(E[-1] + delta[1])
		I.append(I[-1] + delta[2])
		R.append(R[-1] + delta[3])
		G.append(G[-1] + delta[4])
	plt.plot(days, [i / 1000 for i in G], label='SEIRG')

	plt.axvline(datetime.datetime.strptime(end_date, '%Y-%m-%d'), linestyle='dashed', color='tab:grey', label=end_date)
	plt.title(state + ' forecast')
	plt.legend()
	plt.show()
	return 0


# does not optimize c1
def fit(file, state, n_0, SEIRG, method):
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]
	for d in confirmed.columns[1:]:
		if confirmed.iloc[0].loc[d] >= I_0:
			break
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[d: end_date]
	optimal = minimize(loss, [50, 0.2, 0.3, 0.02, 0.9], args=(confirmed, n_0, SEIRG), method='L-BFGS-B',
	                   bounds=[beta_range, betaEI_range, gamma_range, eta_range, c1_fixed])
	beta = optimal.x[0]
	betaEI = optimal.x[1]
	gamma = optimal.x[2]
	eta = optimal.x[3]
	c1 = optimal.x[4]
	current_loss = loss(optimal.x, confirmed, n_0, SEIRG)
	# print(f'beta = {round(beta, 8)} , gamma = {round(gamma, 8)} , '
	#       f'eta = {round(eta, 8)}, StartDate = {d} ')

	size = len(confirmed)
	S = [n_0 * eta]
	E = [confirmed[0]]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SEIRG(i, [S[i - 1], E[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, betaEI, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		E.append(E[-1] + delta[1])
		I.append(I[-1] + delta[2])
		R.append(R[-1] + delta[3])
		G.append(G[-1] + delta[4])

	if method == 'SEIR-SD':
		plt.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="CUM")
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
	if method == 'SEIR-SD':
		print(method)
		print('beta=', beta, 'betaEI=', betaEI, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
		print('****************************')
		print('c1=', c1)
		print('****************************')
		print('start date=', d)
		print(f'R2: {metric0} and {metric1}')
		print('loss=', current_loss)
		print()

	return [d, beta, betaEI, gamma, eta]


# optimizes c1
def fit2(file, state, n_0, SEIRG, method):
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
	for c1 in np.arange(0.8, 1, 0.01):
		optimal = minimize(loss, [100, 0.2, 0.3, 0.02, c1], args=(confirmed, n_0, SEIRG), method='L-BFGS-B',
		                   bounds=[beta_range, betaEI_range, gamma_range, eta_range, (c1, c1)])
		current_loss = loss(optimal.x, confirmed, n_0, SEIRG)
		if current_loss < min_loss:
			min_loss = current_loss
			c_max = c1
	# print(f'beta = {round(beta, 8)} , gamma = {round(gamma, 8)} , '
	#       f'eta = {round(eta, 8)}, StartDate = {d} ')

	optimal = minimize(loss, [100, 0.2, 0.3, 0.02, c_max], args=(confirmed, n_0, SEIRG), method='L-BFGS-B',
	                   bounds=[beta_range, betaEI_range, gamma_range, eta_range, (c_max, c_max)])
	beta = optimal.x[0]
	betaEI = optimal.x[1]
	gamma = optimal.x[2]
	eta = optimal.x[3]
	c1 = optimal.x[4]
	size = len(confirmed)
	S = [n_0 * eta]
	E = [confirmed[0]]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SEIRG(i, [S[i - 1], E[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, betaEI, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		E.append(E[-1] + delta[1])
		I.append(I[-1] + delta[2])
		R.append(R[-1] + delta[3])
		G.append(G[-1] + delta[4])

	if method == 'SEIR-SD':
		plt.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="CUM")
	plt.plot(days, [i / 1000 for i in G], label=method)
	# plt.plot(days, [i / 1000 for i in S], label="S")
	plt.plot(days, [i / 1000 for i in E], label="E")
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
	if method == 'SEIR-SD' or True:
		print(method)
		print('beta=', beta, 'betaEI=', betaEI, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
		print('****************************')
		print('c1=', c1)
		print('****************************')
		print('start date=', d)
		print(f'R2: {metric0} and {metric1}')
		print('loss=', min_loss)
		print()
	return [d, beta, betaEI, gamma, eta, c1]


# optimizes c1
def fit2_MC(file, state, n_0, SEIRG, method):
	df = pd.read_csv(file)
	confirmed = df[df.iloc[:, 0] == state]

	for d in confirmed.columns[1:]:
		if confirmed.iloc[0].loc[d] >= I_0:
			break
	days = list(confirmed.columns)
	days = days[days.index(d):days.index(end_date) + 1]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	confirmed = confirmed.iloc[0].loc[d: end_date]
	maximizer = []
	min_loss = 10000

	for i in range(iterations):
		incubation = np.random.uniform(4, 6)
		infectious = np.random.uniform(4, 14)
		R0 = np.random.uniform(1, 6)
		gamma = 1 / infectious
		betaEI = 1 / incubation
		eta = np.random.uniform(0, 0.05)
		# eta = 1
		beta = R0 * gamma / eta
		c1 = np.random.uniform(0.8, 1)
		current_loss = loss([beta, betaEI, gamma, eta, c1], confirmed, n_0, SEIRG)
		if current_loss < min_loss:
			min_loss = current_loss
			maximizer = [beta, betaEI, gamma, eta, c1]
	beta = maximizer[0]
	betaEI = maximizer[1]
	gamma = maximizer[2]
	eta = maximizer[3]
	c1 = maximizer[4]
	size = len(confirmed)
	S = [n_0 * eta]
	E = [confirmed[0]]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SEIRG(i, [S[i - 1], E[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, betaEI, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		E.append(E[-1] + delta[1])
		I.append(I[-1] + delta[2])
		R.append(R[-1] + delta[3])
		G.append(G[-1] + delta[4])

	if method == 'SEIR-SD':
		plt.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="CUM")
	plt.plot(days, [i / 1000 for i in G], label=method)
	# plt.plot(days, [i / 1000 for i in S], label="S")
	plt.plot(days, [i / 1000 for i in E], label="E")
	plt.plot(days[:(len(I) - round(1 / betaEI))], [i / 1000 for i in I[:(len(I) - round(1 / betaEI))]], label="I")
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
	if method == 'SEIR-SD' or True:
		print(method)
		print('beta=', beta, 'betaEI=', betaEI, 'gamma=', gamma, 'eta=', eta, 'R0=', beta * eta / gamma)
		print('****************************')
		print('c1=', c1)
		print('****************************')
		print('start date=', d)
		print(f'R2: {metric0} and {metric1}')
		print('loss=', min_loss)
		print()
	return [d, beta, betaEI, gamma, eta, c1]


def loss(point, confirmed, n_0, SEIRG):
	size = len(confirmed)
	beta = point[0]
	betaEI = point[1]
	gamma = point[2]
	eta = point[3]
	c1 = point[4]
	S = [n_0 * eta]
	E = [confirmed[0]]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SEIRG(i, [S[i - 1], E[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, betaEI, gamma, eta, n_0, c1])
		S.append(S[-1] + delta[0])
		E.append(E[-1] + delta[1])
		I.append(I[-1] + delta[2])
		R.append(R[-1] + delta[3])
		G.append(G[-1] + delta[4])
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
	fit('data/Confirmed-US.csv', state, n_0, SEIRG_sd, 'SEIR-SD')
	fit('data/Confirmed-US.csv', state, n_0, SEIRG, 'SEIR')
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
	para_sd = fit2_MC('data/Confirmed-US.csv', state, n_0, SEIRG_sd, 'SEIR-SD')
	para = fit('data/Confirmed-US.csv', state, n_0, SEIRG, 'SEIR')
	plt.legend()
	# plt.savefig(figFileName)
	# plt.show()
	forecast(state, n_0, para_sd, para, 'data/Confirmed-US.csv')
	return para_sd[4]


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
	fit('data/Confirmed-US-counties.csv', state, n_0, SEIRG_sd, 'SEIR-SD')
	fit('data/Confirmed-US-counties.csv', state, n_0, SEIRG, 'SEIR')


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
	para_sd = fit2('data/Confirmed-counties.csv', state, n_0, SEIRG_sd, 'SEIR-SD')
	para = fit('data/Confirmed-counties.csv', state, n_0, SEIRG, 'SEIR')
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
	para_sd = fit2('data/Confirmed-cities.csv', state, n_0, SEIRG_sd, 'SEIR-SD')
	para = fit('data/Confirmed-cities.csv', state, n_0, SEIRG, 'SEIR')
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
	fit_state2('FL')
	# fit_state2('IL')
	# fit_state2('NY')
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


if __name__ == '__main__':
	main()
