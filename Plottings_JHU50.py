import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import pearsonr, ttest_ind, ttest_rel
import concurrent.futures
import os
import math
# import warnings
import datetime
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sklearn.metrics import r2_score, mean_squared_error
from SIRfunctions import SIRG, weighting
from numpy.random import uniform as uni
import numpy.polynomial.polynomial as poly

# theta = 0.4
num_row = 7
num_col = 4
end_date = '2020-08-31'
# Geo = 0.98
beta_range = (0.1, 100)
gamma_range = (0.04, 0.25)
eta_range = (0.001, 0.1)
Geo_range = np.arange(0.98, 1.01, 0.05)
theta_range = np.arange(1, 1.01, 0.05)
num_threads = 200
log_axis = True

delay = 7

color_dict = {'AL': 'lime', 'AZ': 'lime', 'CA': 'orange', 'CO': 'red', 'CT': 'red', 'DC': 'orange', 'DE': 'red',
              'FL': 'lime', 'GA': 'lime', 'IA': 'lime', 'IL': 'red', 'IN': 'lime', 'KY': 'orange', 'LA': 'lime',
              'MA': 'red', 'MD': 'lime', 'MI': 'orange', 'MN': 'lime', 'MO': 'lime', 'NC': 'lime', 'NE': 'lime',
              'NJ': 'red', 'NV': 'orange', 'NY': 'red', 'OH': 'lime', 'OK': 'lime', 'PA': 'red', 'RI': 'red',
              'SC': 'lime', 'SD': 'lime', 'TN': 'orange', 'TX': 'lime', 'UT': 'lime', 'VA': 'red', 'WA': 'orange',
              'WI': 'orange'}

state_reopens = {'AL': '2020-05-11', 'AZ': '2020-05-15', 'CA': '2020-05-26', 'CO': '2020-05-27', 'CT': '2020-06-17',
                 'DC': '2020-06-22', 'DE': '2020-06-19', 'KY': '2020-06-29', 'FL': '2020-06-03', 'GA': '2020-06-16',
                 'IA': '2020-05-28', 'IL': '2020-05-28', 'IN': '2020-05-21', 'LA': '2020-06-05', 'MA': '2020-06-22',
                 'MD': '2020-06-12', 'MI': '2020-06-08', 'MN': '2020-06-10', 'MO': '2020-06-16', 'NC': '2020-05-22',
                 'NE': '2020-06-22', 'NJ': '2020-06-22', 'NV': '2020-05-29', 'NY': '2020-07-06', 'OH': '2020-05-21',
                 'OK': '2020-06-01', 'PA': '2020-06-26', 'RI': '2020-06-30', 'SC': '2020-05-11', 'SD': '2020-06-01',
                 'TN': '2020-05-22', 'TX': '2020-06-03', 'UT': '2020-05-01', 'VA': '2020-07-01', 'WA': '2020-06-08',
                 'WI': '2020-07-01'}

counties_global = ['AL-Jefferson',
                   'AL-Mobile',
                   'AZ-Maricopa',
                   'AZ-Pima',
                   'AZ-Yuma',
                   'CA-Alameda',
                   'CA-Contra Costa',
                   'CA-Fresno',
                   'CA-Kern',
                   'CA-Los Angeles',
                   'CA-Orange',
                   'CA-Riverside',
                   'CA-Sacramento',
                   'CA-San Bernardino',
                   'CA-San Diego',
                   'CA-San Joaquin',
                   'CA-Santa Clara',
                   'CA-Stanislaus',
                   'CA-Tulare',
                   'CO-Adams',
                   'CO-Arapahoe',
                   'CO-Denver',
                   'CT-Fairfield',
                   'CT-Hartford',
                   'CT-New Haven',
                   'DE-New Castle',
                   'DE-Sussex',
                   'DC-District of Columbia',
                   'FL-Broward',
                   'FL-Duval',
                   'FL-Hillsborough',
                   'FL-Lee',
                   'FL-Miami-Dade',
                   'FL-Orange',
                   'FL-Palm Beach',
                   'FL-Pinellas',
                   'FL-Polk',
                   'GA-Cobb',
                   'GA-DeKalb',
                   'GA-Fulton',
                   'GA-Gwinnett',
                   'IL-Cook',
                   'IL-DuPage',
                   'IL-Kane',
                   'IL-Lake',
                   'IL-Will',
                   'IN-Lake',
                   'IN-Marion',
                   'IA-Polk',
                   'KY-Jefferson',
                   'LA-East Baton Rouge',
                   'LA-Jefferson',
                   'LA-Orleans',
                   'MD-Anne Arundel',
                   'MD-Baltimore',
                   'MD-Baltimore City',
                   'MD-Montgomery',
                   'MD-Prince George\'s',
                   'MA-Bristol',
                   'MA-Essex',
                   'MA-Hampden',
                   'MA-Middlesex',
                   'MA-Norfolk',
                   'MA-Plymouth',
                   'MA-Suffolk',
                   'MA-Worcester',
                   'MI-Kent',
                   'MI-Macomb',
                   'MI-Oakland',
                   'MI-Wayne',
                   'MN-Hennepin',
                   'MO-St. Louis',
                   'NE-Douglas',
                   'NV-Clark',
                   'NJ-Bergen',
                   'NJ-Burlington',
                   'NJ-Camden',
                   'NJ-Essex',
                   'NJ-Hudson',
                   'NJ-Mercer',
                   'NJ-Middlesex',
                   'NJ-Monmouth',
                   'NJ-Morris',
                   'NJ-Ocean',
                   'NJ-Passaic',
                   'NJ-Somerset',
                   'NJ-Union',
                   'NY-Bronx',
                   'NY-Dutchess',
                   'NY-Erie',
                   'NY-Kings',
                   'NY-Nassau',
                   'NY-New York',
                   'NY-Orange',
                   'NY-Queens',
                   'NY-Richmond',
                   'NY-Rockland',
                   'NY-Suffolk',
                   'NY-Westchester',
                   'NC-Mecklenburg',
                   'NC-Wake',
                   'OH-Cuyahoga',
                   'OH-Franklin',
                   'OK-Oklahoma',
                   'OK-Tulsa',
                   'PA-Berks',
                   'PA-Bucks',
                   'PA-Delaware',
                   'PA-Lehigh',
                   'PA-Luzerne',
                   'PA-Montgomery',
                   'PA-Northampton',
                   'PA-Philadelphia',
                   'RI-Providence',
                   'SC-Charleston',
                   'SC-Greenville',
                   'SD-Minnehaha',
                   'TN-Davidson',
                   'TN-Shelby',
                   'TX-Bexar',
                   'TX-Cameron',
                   'TX-Dallas',
                   'TX-El Paso',
                   'TX-Fort Bend',
                   'TX-Harris',
                   'TX-Hidalgo',
                   'TX-Nueces',
                   'TX-Tarrant',
                   'TX-Travis',
                   'UT-Salt Lake',
                   'VA-Fairfax',
                   'VA-Prince William',
                   'WA-King',
                   'WA-Snohomish',
                   'WI-Milwaukee',
                   ]


def plot_GD(state, ax, folder):
	# calculate initial hiding size
	df = pd.read_csv('JHU/CountyPopulation.csv')
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
	df = pd.read_csv(folder + f'{state}/{state}_para_init.csv')
	eta = df.iloc[0].loc['eta']
	df = pd.read_csv(folder + f'{state}/{state}_para_reopen.csv')
	Hiding_init = df.iloc[0].loc['Hiding_init'] * eta * n_0

	# read dates
	ax.set_title(state)
	df_init = pd.read_csv(folder + f'{state}/{state}_sim_init.csv')
	df_reopen = pd.read_csv(folder + f'{state}/{state}_sim_reopen.csv')
	days_init = df_init.columns[1:-1]
	days_reopen = df_reopen.columns[1:]

	# read reported data
	df_G_data = pd.read_csv(f'JHU/JHU_Confirmed-counties.csv')
	G_data = df_G_data[df_G_data['county'] == state].loc[:, days_init[0]: days_reopen[-1]]
	G_data = list(G_data.iloc[0])
	df_D_data = pd.read_csv(f'JHU/JHU_Death-counties.csv')
	D_data = df_D_data[df_D_data['county'] == state].loc[:, days_init[0]: days_reopen[-1]]
	D_data = list(D_data.iloc[0])

	# read simulation data
	G_init = df_init[df_init['series'] == 'G'].loc[:, days_init]
	G_reopen = df_reopen[df_reopen['series'] == 'G'].loc[:, days_reopen]
	D_init = df_init[df_init['series'] == 'D'].loc[:, days_init]
	D_reopen = df_reopen[df_reopen['series'] == 'D'].loc[:, days_reopen]
	G = pd.concat([G_init, G_reopen], axis=1)
	G = list(G.iloc[0])
	D = pd.concat([D_init, D_reopen], axis=1)
	D = list(D.iloc[0])
	H = df_reopen[df_reopen['series'] == 'H'].loc[:, days_reopen]
	H_init = [Hiding_init] * len(days_init)
	H_init.extend(list(H.iloc[0]))
	H = H_init

	# prepare dates
	days_init = [datetime.datetime.strptime(day, '%Y-%m-%d') for day in days_init]
	days_reopen = [datetime.datetime.strptime(day, '%Y-%m-%d') for day in days_reopen]
	days = days_init.copy()
	days.extend(days_reopen)

	# plot
	ax2 = ax.twinx()
	ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	ax.xaxis.set_major_locator(mdates.MonthLocator())
	ax2.xaxis.set_major_formatter(DateFormatter("%m/%d"))
	ax2.xaxis.set_major_locator(mdates.MonthLocator())
	ax.plot([], ' ', color='black', label="Left Axis")
	ax2.plot([], ' ', color='black', label="Right Axis")
	ax.axvline(days_reopen[0], linestyle='dashed', color='tab:grey')
	ax.plot(days, [i / 1000 for i in G_data], linewidth=5, linestyle=':', label='confirmed', color='tab:red',
	        alpha=0.5)
	ax.plot(days, [i / 1000 for i in G], label='G', color='tab:red')
	ax.plot(days, [i / 1000 for i in H], label='H', color='tab:blue')

	ax2.plot(days, [i / 1000 for i in D_data], linewidth=5, linestyle=':', label='death', color='tab:grey',
	         alpha=0.5)
	ax2.plot(days, [i / 1000 for i in D], label='D', color='tab:grey')

	bottom, top = ax.get_ylim()
	ax.set_ylim(0, top)
	bottom, top = ax2.get_ylim()
	ax2.set_ylim(0, top * 1.5)

	return ax2


def plot_grid(folder):
	plt.rcParams.update({'font.size': 8})
	fig = plt.figure(figsize=(16, 32))
	states = ['AZ-Maricopa', 'CA-Los Angeles', 'CA-Riverside', 'FL-Broward', 'FL-Miami-Dade',
	          'GA-Fulton', 'IL-Cook', 'LA-Jefferson', 'MA-Middlesex', 'MD-Prince George\'s', 'MN-Hennepin',
	          'NC-Mecklenburg', 'NJ-Bergen', 'NJ-Hudson', 'NV-Clark', 'NY-New York', 'OH-Franklin', 'PA-Philadelphia',
	          'TN-Shelby', 'TX-Dallas', 'TX-Harris', 'UT-Salt Lake', 'VA-Fairfax', 'WI-Milwaukee']
	# states = ['IL-Cook', 'TX-Harris']

	for i in range(len(states)):
		state = states[i]
		ax = fig.add_subplot(num_row, num_col, i + 1)
		# fig.autofmt_xdate()
		ax2 = plot_GD(state, ax, folder)
		if i == len(states) - 1:
			ax.legend(bbox_to_anchor=(1.7, 1), loc="upper right")
			ax2.legend(bbox_to_anchor=(1.7, 1), loc="upper left")

	# fig.autofmt_xdate()
	fig.savefig(folder + 'grid.png', bbox_inches="tight")
	plt.close(fig)
	return 0


def plot_state_combined(ax, state, ConfirmFile, DeathFile, PopFile, dates):
	reopen_date = dates[0]
	HospFile = f'JHU/hosp/{state}/Top_Counties_by_Cases_Full_Data_data.csv'

	# read population
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

	# read simulation
	SimFile = f'JHU/combined2W_{end_date}/{state}/sim.csv'
	df = pd.read_csv(SimFile)
	days = df.columns[1:].tolist()
	start_date = days[0]
	reopen_day = days.index(reopen_date) + 7

	S = df[df['series'] == 'S'].iloc[0].iloc[1:]
	G = df[df['series'] == 'G'].iloc[0].iloc[1:]
	D = df[df['series'] == 'D'].iloc[0].iloc[1:]
	H = df[df['series'] == 'H'].iloc[0].iloc[1:]
	I = df[df['series'] == 'I'].iloc[0].iloc[1:]

	# read confirmed and deaths
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[start_date: end_date]
	death = death.iloc[0].loc[start_date: end_date]

	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	ax2 = ax.twinx()
	ax.plot([], ' ', color='black', label="Left Axis")
	ax2.plot([], ' ', color='black', label="Right Axis")
	ax.set_title(state)
	ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases", color='red',
	        alpha=0.5)
	ax.plot(days, [g / 1000 for g in G.tolist()], label='G', color='red')
	ax.plot(days, [i / 1000 for i in I.tolist()], label='I', color='green')
	# ax.plot(days, [h / 1000 for h in H.tolist()], label='H', color='orange')
	ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Cumulative\nDeaths", color='blue',
	         alpha=0.5)
	ax2.plot(days, [d / 1000 for d in D.tolist()], label='D', color='blue')
	l, u = ax.get_ylim()
	ax.set_ylim(0, u)
	l, u = ax2.get_ylim()
	ax2.set_ylim(0, u * 1.5)

	return ax2


# plot G, D, I_H for all states in a grid
def plot_combined():
	plt.rcParams.update({'font.size': 8})
	fig = plt.figure(figsize=(20, 30))
	fig_num = 0

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'NY-New York'
	dates = ['2020-06-22', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'CA-Los Angeles'
	dates = ['2020-06-12', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'FL-Miami-Dade'
	dates = ['2020-06-03', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'IL-Cook'
	dates = ['2020-06-03', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'TX-Dallas'
	dates = ['2020-05-22', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'TX-Harris'
	dates = ['2020-06-03', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'GA-Fulton'
	dates = ['2020-06-12', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'AZ-Maricopa'
	# dates = ['2020-06-22', end_date]
	dates = ['2020-05-28', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'NJ-Bergen'
	dates = ['2020-06-22', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'PA-Philadelphia'
	dates = ['2020-06-05', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'MD-Prince George\'s'
	dates = ['2020-06-29', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'NV-Clark'
	dates = ['2020-05-29', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'NC-Mecklenburg'
	dates = ['2020-05-22', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'LA-Jefferson'
	dates = ['2020-06-05', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'CA-Riverside'
	dates = ['2020-06-12', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'FL-Broward'
	dates = ['2020-06-12', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'NJ-Hudson'
	dates = ['2020-06-22', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'MA-Middlesex'
	dates = ['2020-06-22', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'OH-Franklin'
	dates = ['2020-05-21', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'VA-Fairfax'
	dates = ['2020-06-12', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'TN-Shelby'
	dates = ['2020-06-15', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'WI-Milwaukee'
	dates = ['2020-07-01', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'UT-Salt Lake'
	dates = ['2020-05-15', end_date]
	plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                    'JHU/CountyPopulation.csv', dates)

	fig_num += 1
	ax = fig.add_subplot(num_row, num_col, fig_num)
	state = 'MN-Hennepin'
	dates = ['2020-06-04', end_date]
	ax2 = plot_state_combined(ax, state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
	                          'JHU/CountyPopulation.csv', dates)

	ax.legend(bbox_to_anchor=(1.5, 1), loc="upper right")
	ax2.legend(bbox_to_anchor=(1.5, 1), loc="upper left")

	# fig.autofmt_xdate()
	fig.savefig(f'JHU/combined2W_{end_date}/grid.png', bbox_inches="tight")
	# plt.show()
	plt.close(fig)

	return 0


def read_bar(state, SimFolder, ConfirmFile, PopFile):
	ParaFile = f'{SimFolder}/para.csv'
	SimFile = f'{SimFolder}/sim.csv'

	df = pd.read_csv(ParaFile)
	beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, r1, r2, reopen_date = df.iloc[0]

	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

	S, I, R, D, G, H, days = read_sim(state, SimFolder)
	start_date = days[0]
	# df = pd.read_csv(SimFile)
	# S = df[df['series'] == 'S'].iloc[0].iloc[1:]
	# start_date = S.index[0]
	S1 = S.loc[:reopen_date]

	# G = df[df['series'] == 'G'].iloc[0].iloc[1:]
	G1 = G.loc[:reopen_date]
	G2 = G.loc[reopen_date:]
	diff_G1 = np.diff(G1)
	diff_G2 = np.diff(G2)
	peak1 = max(diff_G1)
	peak2 = max(diff_G2)

	# I = df[df['series'] == 'I'].iloc[0].iloc[1:]
	I1 = I.loc[:reopen_date]
	I_max = max(I1)

	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[start_date:end_date]
	confirmed1 = confirmed.loc[:reopen_date]
	confirmed2 = confirmed.loc[reopen_date:]
	diff_confirmed1 = np.diff(confirmed1)
	diff_confirmed2 = np.diff(confirmed2)
	peakc1 = max(diff_confirmed1)
	peakc2 = max(diff_confirmed2)
	# if 15 < peak2 / peak1 < 20:
	# 	print(state)
	# print(I1.iloc[-1] / I1.iloc[0], S1[-1] / S1[0], peak2 / peak1)

	return [I1.loc[reopen_date] / I_max, S1[-1] / S1[0], peak2 / peak1, peakc2 / peakc1, H[0] / S[0], state]


def plot_against_safeGraph():
	fig = plt.figure()
	fig.set_figheight(16)
	fig.set_figwidth(12)
	axes = fig.subplots(4, 2)
	# fig.suptitle('Daily Visit Increase per 1000')
	NAICSs = ['7211', '7224', '7225', '7139', '7131', '8131', '8121']
	for i, NAICS in enumerate(NAICSs):
		ax = axes[i // 2][i % 2]
		plot_PIR_against_NAICS(NAICS, ax)

	slope_df = pd.read_csv('Safe Graph/slope.csv')
	slopes = []
	points = []
	ax = axes[-1][-1]
	for state in counties_global:
		point = read_bar(state, f'JHU50/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
		                 'JHU/CountyPopulation.csv')
		slope = slope_df[slope_df['county'] == state].iloc[0, 1]
		# print(state, slope)
		slopes.append(slope)
		points.append(np.log(point[2]))
		# slopes.append(slope)
		# points.append(point[2])
		ax.scatter(slopes[-1], points[-1], c=color_dict[state[:2]], alpha=0.6)

	m, b = np.polyfit(slopes, points, 1)
	ax.plot(np.array(slopes), m * np.array(slopes) + b, label=f'\u03C1={round(np.corrcoef(slopes, points)[0][1], 3)}')
	# print(np.corrcoef(range(10), range(0, 20, 2))[0][1])
	ax.legend()
	ax.set_title('Summation of 7 Categories')
	ax.set_xlabel('Daily Visit Increase per 1000')
	ax.set_ylabel('log(PIR)')
	plt.subplots_adjust(hspace=0.3)
	fig.savefig('Safe Graph/slope.png', bbox_inches="tight")
	plt.close(fig)

	return


def plot_PIR_against_NAICS(NAICS, ax):
	NAICS_dict = {'7211': 'Traveler Accommodation',
	              '7213': 'Rooming and Boarding Houses, Dormitories, and Workers\' Camps',
	              '7224': 'Drinking Places (Alcoholic Beverages)',
	              '7225': 'Restaurants and Other Eating Places',
	              '7139': 'Other Amusement and Recreation Industries',
	              '7131': 'Amusement Parks and Arcades',
	              '8131': 'Religious Organizations',
	              '8121': 'Personal Care Services'}
	start_dt = datetime.datetime(2020, 1, 1)
	dts = [start_dt + datetime.timedelta(i) for i in range(365)]
	start_dt = datetime.datetime(2020, 4, 1)
	end_dt = datetime.datetime(2020, 7, 1)
	start_index = dts.index(start_dt)
	end_index = dts.index(end_dt)
	visit_df = pd.read_csv(f'Safe Graph/NAICS/{NAICS_dict[NAICS]} visits.csv')
	points = []
	slopes = []
	for county in counties_global:
		row = list(visit_df[visit_df['county'] == county].iloc[0, start_index:end_index + 1])
		m, b = np.polyfit(range(len(row)), row, 1)
		point = read_bar(county, f'JHU50/combined2W_{end_date}/{county}', 'JHU/JHU_Confirmed-counties.csv',
		                 'JHU/CountyPopulation.csv')
		slopes.append(m)
		points.append(np.log(point[2]))
		ax.scatter(slopes[-1], points[-1], c=color_dict[county[:2]], alpha=0.6)
	m, b = np.polyfit(slopes, points, 1)
	ax.plot(np.array(slopes), m * np.array(slopes) + b, label=f'\u03C1={round(np.corrcoef(slopes, points)[0][1], 3)}')
	ax.legend()
	ax.set_title(f'{NAICS_dict[NAICS]}')
	ax.set_xlabel('Daily Visit Increase per 1000')
	ax.set_ylabel('log(PIR)')
	return


def plot_bar():
	if not os.path.exists('JHU50/3D'):
		os.makedirs('JHU50/3D')
	points = []
	states = ['AL-Jefferson',
	          'AL-Mobile',
	          'AZ-Maricopa',
	          'AZ-Pima',
	          'AZ-Yuma',
	          'CA-Alameda',
	          'CA-Contra Costa',
	          'CA-Fresno',
	          'CA-Kern',
	          'CA-Los Angeles',
	          'CA-Orange',
	          'CA-Riverside',
	          'CA-Sacramento',
	          'CA-San Bernardino',
	          'CA-San Diego',
	          'CA-San Joaquin',
	          'CA-Santa Clara',
	          'CA-Stanislaus',
	          'CA-Tulare',
	          'CO-Adams',
	          'CO-Arapahoe',
	          'CO-Denver',
	          'CT-Fairfield',
	          'CT-Hartford',
	          'CT-New Haven',
	          'DE-New Castle',
	          'DE-Sussex',
	          'DC-District of Columbia',
	          'FL-Broward',
	          'FL-Duval',
	          'FL-Hillsborough',
	          'FL-Lee',
	          'FL-Miami-Dade',
	          'FL-Orange',
	          'FL-Palm Beach',
	          'FL-Pinellas',
	          'FL-Polk',
	          'GA-Cobb',
	          'GA-DeKalb',
	          'GA-Fulton',
	          'GA-Gwinnett',
	          'IL-Cook',
	          'IL-DuPage',
	          'IL-Kane',
	          'IL-Lake',
	          'IL-Will',
	          'IN-Lake',
	          'IN-Marion',
	          'IA-Polk',
	          'KY-Jefferson',
	          'LA-East Baton Rouge',
	          'LA-Jefferson',
	          'LA-Orleans',
	          'MD-Anne Arundel',
	          'MD-Baltimore',
	          'MD-Baltimore City',
	          'MD-Montgomery',
	          'MD-Prince George\'s',
	          'MA-Bristol',
	          'MA-Essex',
	          'MA-Hampden',
	          'MA-Middlesex',
	          'MA-Norfolk',
	          'MA-Plymouth',
	          'MA-Suffolk',
	          'MA-Worcester',
	          'MI-Kent',
	          'MI-Macomb',
	          'MI-Oakland',
	          'MI-Wayne',
	          'MN-Hennepin',
	          'MO-St. Louis',
	          'NE-Douglas',
	          'NV-Clark',
	          'NJ-Bergen',
	          'NJ-Burlington',
	          'NJ-Camden',
	          'NJ-Essex',
	          'NJ-Hudson',
	          'NJ-Mercer',
	          'NJ-Middlesex',
	          'NJ-Monmouth',
	          'NJ-Morris',
	          'NJ-Ocean',
	          'NJ-Passaic',
	          'NJ-Somerset',
	          'NJ-Union',
	          'NY-Bronx',
	          'NY-Dutchess',
	          'NY-Erie',
	          'NY-Kings',
	          'NY-Nassau',
	          'NY-New York',
	          'NY-Orange',
	          'NY-Queens',
	          'NY-Richmond',
	          'NY-Rockland',
	          'NY-Suffolk',
	          'NY-Westchester',
	          'NC-Mecklenburg',
	          'NC-Wake',
	          'OH-Cuyahoga',
	          'OH-Franklin',
	          'OK-Oklahoma',
	          'OK-Tulsa',
	          'PA-Berks',
	          'PA-Bucks',
	          'PA-Delaware',
	          'PA-Lehigh',
	          'PA-Luzerne',
	          'PA-Montgomery',
	          'PA-Northampton',
	          'PA-Philadelphia',
	          'RI-Providence',
	          'SC-Charleston',
	          'SC-Greenville',
	          'SD-Minnehaha',
	          'TN-Davidson',
	          'TN-Shelby',
	          'TX-Bexar',
	          'TX-Cameron',
	          'TX-Dallas',
	          'TX-El Paso',
	          'TX-Fort Bend',
	          'TX-Harris',
	          'TX-Hidalgo',
	          'TX-Nueces',
	          'TX-Tarrant',
	          'TX-Travis',
	          'UT-Salt Lake',
	          'VA-Fairfax',
	          'VA-Prince William',
	          'WA-King',
	          'WA-Snohomish',
	          'WI-Milwaukee',
	          ]

	for state in states:
		point = read_bar(state, f'JHU50/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
		                 'JHU/CountyPopulation.csv')
		points.append(point)

	# color_dict = {'AZ-Maricopa': 'lime', 'CA-Los Angeles': 'orange', 'FL-Miami-Dade': 'lime', 'GA-Fulton': 'lime',
	#               'IL-Cook': 'crimson', 'LA-Jefferson': 'lime', 'MD-Prince George\'s': 'crimson', 'MN-Hennepin': 'lime',
	#               'NV-Clark': 'lime', 'NJ-Bergen': 'crimson', 'NY-New York': 'crimson', 'NC-Mecklenburg': 'lime',
	#               'PA-Philadelphia': 'orange', 'TX-Harris': 'lime', 'CA-Riverside': 'orange',
	#               'FL-Broward': 'lime', 'TX-Dallas': 'lime', 'NJ-Hudson': 'crimson', 'MA-Middlesex': 'crimson',
	#               'OH-Franklin': 'lime', 'VA-Fairfax': 'crimson', 'SC-Charleston': 'lime', 'MI-Oakland': 'crimson',
	#               'TN-Shelby': 'lime', 'WI-Milwaukee': 'orange', 'UT-Salt Lake': 'orange'}

	plt.rcParams.update({'font.size': 8})
	# 3D
	fig = plt.figure()
	ax2 = fig.add_subplot(projection='3d')
	ax2.set_xlabel('log(PAR)' if log_axis else 'PAR')
	ax2.set_ylabel('log(SLR)' if log_axis else 'SLR')
	ax2.set_zlabel('PIR')
	for i in range(len(points)):
		state = points[i][-1]
		ax2.bar3d(np.log(points[i][0]) if log_axis else points[i][0],
		          np.log(points[i][1]) if log_axis else points[i][1],
		          0,
		          0.2 if log_axis else 0.02,
		          0.05 if log_axis else 0.02,
		          points[i][3],
		          alpha=0.6, color=color_dict.setdefault(state[:2], 'grey'))
	fig.savefig('JHU50/3D/3D.png', bbox_inches="tight")
	plt.close(fig)

	# 3D log
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.set_xlabel('log(PAR)' if log_axis else 'PAR')
	ax.set_ylabel('log(SLR)' if log_axis else 'SLR')
	ax.set_zlabel('log(PIR)')
	for i in range(len(points)):
		state = points[i][-1]
		ax.bar3d(np.log(points[i][0]) if log_axis else points[i][0],
		         np.log(points[i][1]) if log_axis else points[i][1],
		         0,
		         0.2 if log_axis else 0.02,
		         0.05 if log_axis else 0.02,
		         math.log(points[i][3]), alpha=0.6,
		         color=color_dict.setdefault(state[:2], 'grey'))
	xl, xh = ax.get_xlim()
	yl, yh = ax.get_ylim()
	xs = np.linspace(xl, xh, 100)
	ys = np.linspace(yl, yh, 100)
	zs = [0] * 100
	X, Y = np.meshgrid(xs, ys)
	X, Z = np.meshgrid(xs, zs)
	ax.plot_surface(X, Y, Z, alpha=0.3)
	ax.set_xlim(xl, xh)
	ax.set_ylim(yl, yh)
	ax.view_init(elev=5, azim=315)
	fig.savefig('JHU50/3D/3D_log.png', bbox_inches="tight")
	# plt.show()
	plt.close(fig)

	plt.rcParams.update({'font.size': 10})
	# 2D using S_left
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.axhline(y=0, linewidth=0.5, color='black')
	for i in range(len(points)):
		state = points[i][-1]
		ax.scatter(np.log(points[i][1]) if log_axis else points[i][1],
		           math.log(points[i][3]),
		           alpha=0.6,
		           color=color_dict.setdefault(state[:2], 'grey'))

	X = [np.log(points[i][1]) if log_axis else points[i][1] for i in range(len(points))]
	Y = [np.log(points[i][3]) for i in range(len(points))]
	rho = np.corrcoef(X, Y)
	C = poly.polyfit(X, Y, 1)
	X2 = np.arange(min(X), max(X), (max(X) - min(X)) / 100)
	Y2 = poly.polyval(X2, C)
	ax.plot(X2, Y2, color='grey', label=f'\u03C1={round(rho[0][1], 4)}')

	ax.set_xlabel('log(SLR)' if log_axis else 'SLR')
	ax.set_ylabel('log(PIR)')
	ax.legend()
	fig.savefig('JHU50/3D/log_S_left.png', bbox_inches="tight")
	# plt.show()
	plt.close(fig)
	t_score = ttest_rel([np.log(points[i][1]) for i in range(len(points))],
	                    [np.log(points[i][3]) for i in range(len(points))])
	print('log(SLR), log(PIR) t-score=', t_score)

	# 2D using I_reopen/I_peak
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.axhline(y=0, linewidth=0.5, color='black')
	for i in range(len(points)):
		state = points[i][-1]
		ax.scatter(np.log(points[i][0]) if log_axis else points[i][0],
		           math.log(points[i][3]),
		           alpha=0.6,
		           color=color_dict.setdefault(state[:2], 'grey'))

	X = [np.log(points[i][0]) if log_axis else points[i][0] for i in range(len(points))]
	Y = [np.log(points[i][3]) for i in range(len(points))]
	rho = np.corrcoef(X, Y)
	C = poly.polyfit(X, Y, 1)
	X2 = np.arange(min(X), max(X), (max(X) - min(X)) / 100)
	Y2 = poly.polyval(X2, C)
	ax.plot(X2, Y2, color='grey', label=f'\u03C1={round(rho[0][1], 4)}')

	ax.set_xlabel('log(PAR)' if log_axis else 'PAR')
	ax.set_ylabel('log(PIR)')
	ax.legend()
	fig.savefig('JHU50/3D/log_I.png', bbox_inches="tight")
	# plt.show()
	plt.close(fig)
	t_score = ttest_rel([np.log(points[i][0]) for i in range(len(points))],
	                    [np.log(points[i][3]) for i in range(len(points))])
	print('log(PAR), log(PIR) t-score=', t_score)
	return 0


def loss(point, confirmed, n_0, SIRG, theta, Geo):
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
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0])
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


def fit_SIR_MT(state, SimFolder, ConfirmFile, PopFile, dates):
	if not os.path.exists(f'JHU/SIR/{state}'):
		os.makedirs(f'JHU/SIR/{state}')

	print(state)

	S, I, R, D, G, H, days = read_sim(state, SimFolder)
	start_date = days[0]
	reopen_date = dates[0]

	confirmed, n_0 = read_data(state, ConfirmFile, PopFile, start_date, reopen_date)

	para_best = []
	min_loss = 1000001
	with concurrent.futures.ProcessPoolExecutor() as executor:
		results = [executor.submit(fit_SIR, confirmed, n_0) for _ in range(num_threads)]
		threads = 0
		for f in concurrent.futures.as_completed(results):
			threads += 1
			current_loss, para = f.result()
			if current_loss < min_loss:
				print(f'updated at {threads}')
				min_loss = current_loss
				para_best = para
	# optimal = minimize(loss, [10, 0.05, 0.02], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
	#                    bounds=[beta_range, gamma_range, eta_range])

	beta = para_best[0]
	gamma = para_best[1]
	eta = para_best[2]
	theta = para_best[3]
	Geo = para_best[4]
	size = len(confirmed)
	S = [n_0 * eta]
	I = [confirmed[0]]
	R = [0]
	G = [confirmed[0]]
	for i in range(1, size):
		delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0])
		S.append(S[-1] + delta[0])
		I.append(I[-1] + delta[1])
		R.append(R[-1] + delta[2])
		G.append(G[-1] + delta[3])

	# print('beta:', beta_range[0], beta, beta_range[1])
	# print('gamma:', gamma_range[0], gamma, gamma_range[1])
	# print('eta:', eta_range[0], eta, eta_range[1])

	# save simulation
	c0 = ['S', 'I', 'R', 'G']
	df = pd.DataFrame([S, I, R, G], columns=days[:len(S)])
	df.insert(0, 'series', c0)
	df.to_csv(f'JHU/SIR/{state}/sim.csv', index=False)

	# save parameters
	para_label = ['beta', 'gamma', 'eta', 'theta', 'Geo', 'RMSE', 'R2']
	RMSE = math.sqrt(mean_squared_error(confirmed, G))
	r2 = r2_score(confirmed, G)
	para_best = np.append(para_best, RMSE)
	para_best = np.append(para_best, r2)
	# para_best.append(RMSE)
	df = pd.DataFrame([para_best], columns=para_label)
	df.to_csv(f'JHU/SIR/{state}/para.csv', index=False)

	days = days[:size]
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.set_title(state)
	ax.plot(days, G, label='G')
	ax.plot(days, confirmed, label='confirm')
	fig.autofmt_xdate()
	ax.legend()
	fig.savefig(f'JHU/SIR/{state}/fit.png', bbox_inches="tight")
	plt.close(fig)
	# plt.show()

	return 0


def fit_SIR(confirmed, n_0):
	# S, I, R, G, days = read_sim(state, SimFolder)
	# start_date = days[0]
	# # reopen_date = dates[0]
	np.random.seed()

	para_best = []
	min_loss = 1000001
	for Geo in Geo_range:
		for theta in theta_range:
			optimal = minimize(loss, [uni(beta_range[0], beta_range[1]),
			                          uni(gamma_range[0], gamma_range[1]),
			                          uni(eta_range[0], eta_range[1])],
			                   args=(confirmed, n_0, SIRG, theta, Geo), method='L-BFGS-B',
			                   bounds=[beta_range, gamma_range, eta_range])
			current_loss = loss(optimal.x, confirmed, n_0, SIRG, theta, Geo)
			if current_loss < min_loss:
				min_loss = current_loss
				[beta, gamma, eta] = optimal.x
				para_best = [beta, gamma, eta, theta, Geo]
	# beta = optimal.x[0]
	# gamma = optimal.x[1]
	# eta = optimal.x[2]
	# size = len(confirmed)
	# S = [n_0 * eta]
	# I = [confirmed[0]]
	# R = [0]
	# G = [confirmed[0]]
	# for i in range(1, size):
	# 	delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0])
	# 	S.append(S[-1] + delta[0])
	# 	I.append(I[-1] + delta[1])
	# 	R.append(R[-1] + delta[2])
	# 	G.append(G[-1] + delta[3])
	#
	# days = days[:size]
	# days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	# fig = plt.figure()
	# ax = fig.add_subplot()
	# ax.plot(days, G, label='G')
	# ax.plot(days, confirmed, label='confirm')
	# fig.autofmt_xdate()
	# ax.legend()
	# plt.show()
	# c0 = ['S', 'I', 'R', 'G']
	# print('beta:', beta_range[0], beta, beta_range[1])
	# print('gamma:', gamma_range[0], gamma, gamma_range[1])
	# print('eta:', eta_range[0], eta, eta_range[1])
	return current_loss, para_best


def read_sim(state, SimFolder):
	# ParaFile = f'{SimFolder}/para.csv'
	SimFile = f'{SimFolder}/sim.csv'
	df = pd.read_csv(SimFile)
	S = df[df['series'] == 'S'].iloc[0].iloc[1:]
	I = df[df['series'] == 'I'].iloc[0].iloc[1:]
	R = df[df['series'] == 'R'].iloc[0].iloc[1:]
	D = df[df['series'] == 'D'].iloc[0].iloc[1:]
	G = df[df['series'] == 'G'].iloc[0].iloc[1:]
	H = df[df['series'] == 'H'].iloc[0].iloc[1:]
	days = S.index

	return S, I, R, D, G, H, days


def read_SIR(state, SimFolder):
	# ParaFile = f'{SimFolder}/para.csv'
	SimFile = f'{SimFolder}/sim.csv'
	df = pd.read_csv(SimFile)
	S = df[df['series'] == 'S'].iloc[0].iloc[1:]
	I = df[df['series'] == 'I'].iloc[0].iloc[1:]
	R = df[df['series'] == 'R'].iloc[0].iloc[1:]
	G = df[df['series'] == 'G'].iloc[0].iloc[1:]
	days = S.index

	return S, I, R, G, days


def read_data(state, ConfirmFile, PopFile, start_date, reopen_date):
	delay = 7
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]

	# apply the delay to reopen date
	reopen_date = confirmed.columns[confirmed.columns.get_loc(reopen_date) + delay]
	confirmed = confirmed.iloc[0].loc[start_date:reopen_date]
	df = pd.read_csv(PopFile)
	n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
	return confirmed, n_0


def fit_all_SIR():
	state = 'NY-New York'
	dates = ['2020-06-22', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'CA-Los Angeles'
	dates = ['2020-06-12', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'FL-Miami-Dade'
	dates = ['2020-06-03', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'IL-Cook'
	dates = ['2020-06-03', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'TX-Dallas'
	dates = ['2020-05-22', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'TX-Harris'
	dates = ['2020-06-03', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'GA-Fulton'
	dates = ['2020-06-12', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'AZ-Maricopa'
	dates = ['2020-05-28', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'NJ-Bergen'
	dates = ['2020-06-22', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'PA-Philadelphia'
	dates = ['2020-06-05', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'MD-Prince George\'s'
	dates = ['2020-06-29', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'NV-Clark'
	dates = ['2020-05-29', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'NC-Mecklenburg'
	dates = ['2020-05-22', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'LA-Jefferson'
	dates = ['2020-06-05', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'CA-Riverside'
	dates = ['2020-06-12', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'FL-Broward'
	dates = ['2020-06-12', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'NJ-Hudson'
	dates = ['2020-06-22', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'MA-Middlesex'
	dates = ['2020-06-22', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'OH-Franklin'
	dates = ['2020-05-21', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'VA-Fairfax'
	dates = ['2020-06-12', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'TN-Shelby'
	dates = ['2020-06-15', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'WI-Milwaukee'
	dates = ['2020-07-01', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'UT-Salt Lake'
	dates = ['2020-05-15', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	state = 'MN-Hennepin'
	dates = ['2020-06-04', end_date]
	fit_SIR_MT(state, f'JHU/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
	           'JHU/CountyPopulation.csv', dates)

	return 0


# compare SD with SIR
def compare_state(state, out_df, ax):
	print(state)
	SimFolder = f'JHU/combined2W_{end_date}/{state}'
	# SimFolder = f'init_only_{end_date}/{state}'
	S, I, R, D, G, H, days = read_sim(state, SimFolder)
	start_date = days[0]

	SIRFolder = f'JHU/SIR/{state}'
	S2, I2, R2, G2, days2 = read_SIR(state, SIRFolder)
	reopen_date = days2[-1]

	ConfirmFile = 'JHU/JHU_Confirmed-counties.csv'
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	DeathFile = 'JHU/JHU_Death-counties.csv'
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	confirmed2 = confirmed.iloc[0].loc[start_date: reopen_date]
	death2 = death.iloc[0].loc[start_date: reopen_date]
	confirmed = confirmed.iloc[0].loc[start_date: end_date]
	death = death.iloc[0].loc[start_date: end_date]

	df3 = pd.read_csv(f'JHU/init_only_{end_date}/{state}/sim.csv')
	Gi = df3[df3['series'] == 'G'].iloc[0].iloc[1:]

	# death2.index = make_datetime(start_date, len(death2.index))
	Geo = 0.98
	# weighted_Gi = weighting(Gi.iloc[:len(confirmed2)], Geo)
	weighted_Gi = weighting(Gi, Geo)
	weighted_G2 = weighting(G2, Geo)
	weighted_G = weighting(G, Geo)
	weighted_D = weighting(D, Geo)
	weighted_confirmed2 = weighting(confirmed2, Geo)
	weighted_confirmed = weighting(confirmed, Geo)
	weighted_death = weighting(death, Geo)
	out_df.append([state, math.sqrt(mean_squared_error(G, confirmed)), math.sqrt(mean_squared_error(D, death)),
	               math.sqrt(mean_squared_error(weighted_G, weighted_confirmed)),
	               math.sqrt(mean_squared_error(weighted_D, weighted_death)),
	               math.sqrt(mean_squared_error(Gi, confirmed2)),
	               math.sqrt(mean_squared_error(G2, confirmed2)),
	               r2_score(G, confirmed),
	               r2_score(D, death),
	               r2_score(weighted_G, weighted_confirmed),
	               r2_score(weighted_D, weighted_death)]
	              )
	days2 = make_datetime(days[0], len(days2))
	# plt.plot(range(len(G2)), confirmed2, label='Cumulative Cases')
	# plt.plot(range(len(G2)), Gi, label='SD')
	# plt.plot(range(len(G2)), G2, label='SIR')
	ax.plot(days2, [i / 1000 for i in confirmed2], linewidth=3, linestyle=':', label='Cumulative Cases')
	ax.plot(days2, [i / 1000 for i in Gi], label='SD')
	ax.plot(days2, [i / 1000 for i in G2], label='SIR')
	# ax.legend()
	ax.set_title(state)
	date_form = DateFormatter("%m-%d")
	ax.xaxis.set_major_formatter(date_form)
	plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
	# fig.autofmt_xdate()
	# plt.show()
	fig = plt.figure()
	ax2 = fig.add_subplot()
	ax2.plot(days2, [i / 1000 for i in confirmed2], linewidth=3, linestyle=':', label='Cumulative Cases')
	ax2.plot(days2, [i / 1000 for i in Gi], label='SD')
	ax2.plot(days2, [i / 1000 for i in G2], label='SIR')
	ax2.set_title(state)
	ax2.set_ylabel('Cases (Thousand)')
	ax2.legend()
	fig.autofmt_xdate()
	fig.savefig(f'JHU/comparison/comparison_{state}.png', bbox_inches="tight")
	plt.close(fig)
	return


def make_datetime(start_date, size):
	dates = [datetime.datetime.strptime(start_date, '%Y-%m-%d')]
	for i in range(1, size):
		dates.append(dates[0] + datetime.timedelta(days=i))
	return dates


def compare_all():
	out_df = []
	# out_df = columns = ['state', 'RSME_G', 'RSME_D']
	fig = plt.figure(figsize=[14, 18])
	num_fig = 0
	row = 6
	col = 4

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'AZ-Maricopa'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'CA-Los Angeles'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'CA-Riverside'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'FL-Broward'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'FL-Miami-Dade'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'GA-Fulton'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'IL-Cook'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'LA-Jefferson'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'MA-Middlesex'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'MD-Prince George\'s'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'MN-Hennepin'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'NC-Mecklenburg'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'NJ-Bergen'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'NJ-Hudson'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'NV-Clark'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'NY-New York'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'OH-Franklin'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'PA-Philadelphia'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'TN-Shelby'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'TX-Dallas'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'TX-Harris'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'UT-Salt Lake'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'VA-Fairfax'
	compare_state(state, out_df, ax)

	num_fig += 1
	ax = fig.add_subplot(row, col, num_fig)
	state = 'WI-Milwaukee'
	compare_state(state, out_df, ax)

	ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
	# fig.autofmt_xdate()
	fig.subplots_adjust(hspace=0.4, wspace=0.25)

	fig.add_subplot(111, frameon=False)
	plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	plt.ylabel("Cases (Thousand)")

	fig.savefig('JHU/comparison/comparison.png', bbox_inches="tight")

	out_df = pd.DataFrame(out_df,
	                      columns=['state', 'RMSE_G', 'RMSE_D', 'RMSE_wtd_G', 'RMSE_wtd_D', 'RMSE_G2', 'RMSE_SIR',
	                               'R2_G', 'R2_D', 'R2_wtd_G', 'R2_wtd,D'])
	out_df.to_csv('JHU/comparison/RMSE.csv', index=False)
	# print(out_df)
	return


def RMSE_all():
	if not os.path.exists('JHU50/comparison'):
		os.makedirs('JHU50/comparison')
	out_df = []

	for state in counties_global:
		RMSE_state(state, out_df)

	out_df = pd.DataFrame(out_df,
	                      columns=['State - County', 'RMSE_wtd_G', 'RMSE_wtd_D', 'R2_wtd_G', 'R2_wtd,D'])
	out_df.to_csv('JHU50/comparison/RMSE.csv', index=False)
	# print(out_df)
	return


def RMSE_state(state, out_df):
	print(state)
	SimFolder = f'JHU50/combined2W_{end_date}/{state}'
	df = pd.read_csv(f'{SimFolder}/sim.csv')
	dates = df.columns[1:].tolist()
	start_date = dates[0]
	G = df[df['series'] == 'G'].iloc[0].iloc[1:]
	D = df[df['series'] == 'D'].iloc[0].iloc[1:]
	df = pd.read_csv(f'{SimFolder}/para.csv')
	[beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, r1, r2, reopen_date] = df.iloc[
		0]
	reopen_day = dates.index(reopen_date)
	ConfirmFile = 'JHU/JHU_Confirmed-counties.csv'
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	DeathFile = 'JHU/JHU_Death-counties.csv'
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[start_date: end_date]
	death = death.iloc[0].loc[start_date: end_date]
	Geo = 0.98
	size = len(G)
	size1 = reopen_day
	size2 = size - size1
	weights1 = [Geo ** n for n in range(size1)]
	weights1.reverse()
	weights2 = [Geo ** n for n in range(size2)]
	weights2.reverse()
	weights = weights1
	weights.extend(weights2)

	weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
	weighted_G = [G[i] * weights[i] for i in range(size)]
	weighted_death = [death[i] * weights[i] for i in range(size)]
	weighted_D = [D[i] * weights[i] for i in range(size)]

	out_df.append([state,
	               math.sqrt(mean_squared_error(weighted_G, weighted_confirmed)),
	               math.sqrt(mean_squared_error(weighted_D, weighted_death)),
	               r2_score(weighted_G, weighted_confirmed),
	               r2_score(weighted_D, weighted_death)]
	              )

	return


def interval_GD(state, reopen_date, ax):
	# read simulation
	SimFile = f'JHU/combined2W_{end_date}/{state}/sim.csv'
	df = pd.read_csv(SimFile)
	days = df.columns[1:].tolist()
	start_date = days[0]
	reopen_day = days.index(reopen_date) + 7
	G = df[df['series'] == 'G'].iloc[0].iloc[1:]
	D = df[df['series'] == 'D'].iloc[0].iloc[1:]

	# read confirmed and deaths
	ConfirmFile = 'JHU/JHU_Confirmed-counties.csv'
	DeathFile = 'JHU/JHU_Death-counties.csv'
	df = pd.read_csv(ConfirmFile)
	confirmed = df[df.iloc[:, 0] == state]
	df2 = pd.read_csv(DeathFile)
	death = df2[df2.iloc[:, 0] == state]
	confirmed = confirmed.iloc[0].loc[start_date: end_date]
	death = death.iloc[0].loc[start_date: end_date]

	# read interval
	IntervalFile = f'JHU/CI_{end_date}/{state}/GD_high_low.csv'
	df = pd.read_csv(IntervalFile)
	G_high = df[df['series'] == 'G_high'].iloc[0].iloc[1:]
	G_low = df[df['series'] == 'G_low'].iloc[0].iloc[1:]
	D_high = df[df['series'] == 'D_high'].iloc[0].iloc[1:]
	D_low = df[df['series'] == 'D_low'].iloc[0].iloc[1:]

	# plot
	days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
	ax2 = ax.twinx()
	ax.plot([], ' ', color='black', label="Left Axis\n(Thousand):")
	ax2.plot([], ' ', color='black', label="Right Axis\n(Thousand):")
	ax.set_title(state)
	ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
	ax.fill_between(days, [g / 1000 for g in G_high], [g / 1000 for g in G_low], facecolor='orange', alpha=0.4,
	                label='CI_G')
	ax.plot(days, [i / 1000 for i in confirmed], linewidth=2, linestyle=':', label="Cumulative\nCases", color='red')
	ax.plot(days, [g / 1000 for g in G.tolist()], linewidth=0.5, label='G', color='red', alpha=1)

	ax2.plot(days, [i / 1000 for i in death], linewidth=2, linestyle=':', label="Cumulative\nDeaths", color='blue')
	ax2.plot(days, [d / 1000 for d in D.tolist()], linewidth=0.5, label='D', color='blue')
	ax2.fill_between(days, [d / 1000 for d in D_high], [d / 1000 for d in D_low], facecolor='royalblue', alpha=0.4,
	                 label='CI_D')
	l, u = ax.get_ylim()
	ax.set_ylim(0, u)
	l, u = ax2.get_ylim()
	ax2.set_ylim(0, u * 2)
	ax.set_xlim(days[0], days[-1])

	date_form = DateFormatter("%m-%d")
	ax.xaxis.set_major_formatter(date_form)
	plt.setp(ax.get_xticklabels(), rotation=25, ha='right')

	return ax2


# confidence intervals for G and D grid figure
def interval_GD_all():
	fig = plt.figure(figsize=(14, 18))
	col = 4
	row = 6
	i = 0

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'AZ-Maricopa'
	reopen_date = '2020-05-28'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'CA-Los Angeles'
	reopen_date = '2020-06-12'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'CA-Riverside'
	reopen_date = '2020-06-12'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'FL-Broward'
	reopen_date = '2020-06-12'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'FL-Miami-Dade'
	reopen_date = '2020-06-03'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'GA-Fulton'
	reopen_date = '2020-06-12'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'IL-Cook'
	reopen_date = '2020-06-03'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'LA-Jefferson'
	reopen_date = '2020-06-05'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'MA-Middlesex'
	reopen_date = '2020-06-22'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'MD-Prince George\'s'
	reopen_date = '2020-06-29'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'MN-Hennepin'
	reopen_date = '2020-06-04'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NC-Mecklenburg'
	reopen_date = '2020-05-22'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NJ-Bergen'
	reopen_date = '2020-06-22'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NJ-Hudson'
	reopen_date = '2020-06-22'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NV-Clark'
	reopen_date = '2020-05-29'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'NY-New York'
	reopen_date = '2020-06-22'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'OH-Franklin'
	reopen_date = '2020-05-21'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'PA-Philadelphia'
	reopen_date = '2020-06-05'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'TN-Shelby'
	reopen_date = '2020-06-15'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'TX-Dallas'
	reopen_date = '2020-05-22'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'TX-Harris'
	reopen_date = '2020-06-03'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'UT-Salt Lake'
	reopen_date = '2020-05-15'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'VA-Fairfax'
	reopen_date = '2020-06-12'
	ax2 = interval_GD(state, reopen_date, ax)

	i += 1
	ax = fig.add_subplot(row, col, i)
	state = 'WI-Milwaukee'
	reopen_date = '2020-07-01'
	ax2 = interval_GD(state, reopen_date, ax)

	ax.legend(bbox_to_anchor=(0.5, -0.3), loc="upper right")
	ax2.legend(bbox_to_anchor=(0.5, -0.3), loc="upper left")

	# fig.autofmt_xdate()
	fig.subplots_adjust(hspace=0.4, wspace=0.35)

	ax1 = fig.add_subplot(111, frameon=False)
	ax2 = ax1.twinx()
	ax2.set_frame_on(False)
	ax1.set_ylabel("Cumulative Cases (Thousand)")
	ax2.set_ylabel("Deaths (Thousand)")
	# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

	plt.rcParams.update({'font.size': 10})
	fig.savefig(f'JHU/CI_{end_date}/grid.png', bbox_inches="tight")
	# plt.show()
	plt.close(fig)
	return


def peak_by_H():
	if not os.path.exists('JHU50/3D'):
		os.makedirs('JHU50/3D')
	points = []
	states = ['AL-Jefferson',
	          'AL-Mobile',
	          'AZ-Maricopa',
	          'AZ-Pima',
	          'AZ-Yuma',
	          'CA-Alameda',
	          'CA-Contra Costa',
	          'CA-Fresno',
	          'CA-Kern',
	          'CA-Los Angeles',
	          'CA-Orange',
	          'CA-Riverside',
	          'CA-Sacramento',
	          'CA-San Bernardino',
	          'CA-San Diego',
	          'CA-San Joaquin',
	          'CA-Santa Clara',
	          'CA-Stanislaus',
	          'CA-Tulare',
	          'CO-Adams',
	          'CO-Arapahoe',
	          'CO-Denver',
	          'CT-Fairfield',
	          'CT-Hartford',
	          'CT-New Haven',
	          'DE-New Castle',
	          'DE-Sussex',
	          'DC-District of Columbia',
	          'FL-Broward',
	          'FL-Duval',
	          'FL-Hillsborough',
	          'FL-Lee',
	          'FL-Miami-Dade',
	          'FL-Orange',
	          'FL-Palm Beach',
	          'FL-Pinellas',
	          'FL-Polk',
	          'GA-Cobb',
	          'GA-DeKalb',
	          'GA-Fulton',
	          'GA-Gwinnett',
	          'IL-Cook',
	          'IL-DuPage',
	          'IL-Kane',
	          'IL-Lake',
	          'IL-Will',
	          'IN-Lake',
	          'IN-Marion',
	          'IA-Polk',
	          'KY-Jefferson',
	          'LA-East Baton Rouge',
	          'LA-Jefferson',
	          'LA-Orleans',
	          'MD-Anne Arundel',
	          'MD-Baltimore',
	          'MD-Baltimore City',
	          'MD-Montgomery',
	          'MD-Prince George\'s',
	          'MA-Bristol',
	          'MA-Essex',
	          'MA-Hampden',
	          'MA-Middlesex',
	          'MA-Norfolk',
	          'MA-Plymouth',
	          'MA-Suffolk',
	          'MA-Worcester',
	          'MI-Kent',
	          'MI-Macomb',
	          'MI-Oakland',
	          'MI-Wayne',
	          'MN-Hennepin',
	          'MO-St. Louis',
	          'NE-Douglas',
	          'NV-Clark',
	          'NJ-Bergen',
	          'NJ-Burlington',
	          'NJ-Camden',
	          'NJ-Essex',
	          'NJ-Hudson',
	          'NJ-Mercer',
	          'NJ-Middlesex',
	          'NJ-Monmouth',
	          'NJ-Morris',
	          'NJ-Ocean',
	          'NJ-Passaic',
	          'NJ-Somerset',
	          'NJ-Union',
	          'NY-Bronx',
	          'NY-Dutchess',
	          'NY-Erie',
	          'NY-Kings',
	          'NY-Nassau',
	          'NY-New York',
	          'NY-Orange',
	          'NY-Queens',
	          'NY-Richmond',
	          'NY-Rockland',
	          'NY-Suffolk',
	          'NY-Westchester',
	          'NC-Mecklenburg',
	          'NC-Wake',
	          'OH-Cuyahoga',
	          'OH-Franklin',
	          'OK-Oklahoma',
	          'OK-Tulsa',
	          'PA-Berks',
	          'PA-Bucks',
	          'PA-Delaware',
	          'PA-Lehigh',
	          'PA-Luzerne',
	          'PA-Montgomery',
	          'PA-Northampton',
	          'PA-Philadelphia',
	          'RI-Providence',
	          'SC-Charleston',
	          'SC-Greenville',
	          'SD-Minnehaha',
	          'TN-Davidson',
	          'TN-Shelby',
	          'TX-Bexar',
	          'TX-Cameron',
	          'TX-Dallas',
	          'TX-El Paso',
	          'TX-Fort Bend',
	          'TX-Harris',
	          'TX-Hidalgo',
	          'TX-Nueces',
	          'TX-Tarrant',
	          'TX-Travis',
	          'UT-Salt Lake',
	          'VA-Fairfax',
	          'VA-Prince William',
	          'WA-King',
	          'WA-Snohomish',
	          'WI-Milwaukee',
	          ]

	for state in states:
		point = read_bar(state, f'JHU50/combined2W_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
		                 'JHU/CountyPopulation.csv')
		points.append(point)

	# color_dict = {'AZ-Maricopa': 'lime', 'CA-Los Angeles': 'orange', 'FL-Miami-Dade': 'lime', 'GA-Fulton': 'lime',
	#               'IL-Cook': 'crimson', 'LA-Jefferson': 'lime', 'MD-Prince George\'s': 'crimson', 'MN-Hennepin': 'lime',
	#               'NV-Clark': 'lime', 'NJ-Bergen': 'crimson', 'NY-New York': 'crimson', 'NC-Mecklenburg': 'lime',
	#               'PA-Philadelphia': 'orange', 'TX-Harris': 'lime', 'CA-Riverside': 'orange',
	#               'FL-Broward': 'lime', 'TX-Dallas': 'lime', 'NJ-Hudson': 'crimson', 'MA-Middlesex': 'crimson',
	#               'OH-Franklin': 'lime', 'VA-Fairfax': 'crimson', 'SC-Charleston': 'lime', 'MI-Oakland': 'crimson',
	#               'TN-Shelby': 'lime', 'WI-Milwaukee': 'orange', 'UT-Salt Lake': 'orange'}

	plt.rcParams.update({'font.size': 8})

	# 3D log
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.set_xlabel('log(PAR)' if log_axis else 'PAR')
	ax.set_ylabel('log(HSR)' if log_axis else 'HSR')
	ax.set_zlabel('log(PIR)')
	for i in range(len(points)):
		state = points[i][-1]
		ax.bar3d(np.log(points[i][0]) if log_axis else points[i][0],
		         np.log(points[i][4]) if log_axis else points[i][4],
		         0,
		         0.25 if log_axis else 0.02,
		         0.06 if log_axis else 0.1,
		         math.log(points[i][3]),
		         alpha=0.6, color=color_dict.setdefault(state[:2], 'grey'))
	xl, xh = ax.get_xlim()
	yl, yh = ax.get_ylim()
	xs = np.linspace(xl, xh, 100)
	ys = np.linspace(yl, yh, 100)
	zs = [0] * 100
	X, Y = np.meshgrid(xs, ys)
	X, Z = np.meshgrid(xs, zs)
	ax.plot_surface(X, Y, Z, alpha=0.3)
	ax.set_xlim(xl, xh)
	ax.set_ylim(yl, yh)
	ax.view_init(elev=5, azim=315)
	fig.savefig('JHU50/3D/3D_H.png', bbox_inches="tight")
	# plt.show()
	plt.close(fig)

	plt.rcParams.update({'font.size': 10})
	# 2D using I_reopen/I_peak
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.axhline(y=0, linewidth=0.5, color='black')
	for i in range(len(points)):
		state = points[i][-1]
		ax.scatter(np.log(points[i][4]) if log_axis else points[i][4],
		           np.log(points[i][3]),
		           alpha=0.6,
		           color=color_dict.setdefault(state[:2], 'grey'))
		if points[i][4] > 4:
			print(state)

	X = [np.log(points[i][4]) if log_axis else points[i][4] for i in range(len(points))]
	Y = [np.log(points[i][3]) for i in range(len(points))]
	rho = np.corrcoef(X, Y)
	C = poly.polyfit(X, Y, 1)
	X2 = np.arange(min(X), max(X), (max(X) - min(X)) / 100)
	Y2 = poly.polyval(X2, C)
	ax.plot(X2, Y2, color='grey', label=f'\u03C1={round(rho[0][1], 4)}')
	ax.set_xlabel('log(HSR)' if log_axis else 'HSR')
	ax.set_ylabel('log(PIR)')
	ax.legend()
	fig.savefig('JHU50/3D/log_H.png', bbox_inches="tight")
	# plt.show()
	plt.close(fig)
	t_score = ttest_rel([np.log(points[i][4]) for i in range(len(points))],
	                    [np.log(points[i][3]) for i in range(len(points))])
	print('log(HSR), log(PIR) t-score=', t_score)

	return


def color_table():
	table_color = {'lime': '{\cellcolor[rgb]{0,1,0}green}', 'orange': '{\cellcolor[rgb]{1, 0.647, 0}yellow}',
	               'red': '{\cellcolor[rgb]{0.863, 0.078, 0.235} red}'}
	out_table = []
	for state in counties_global:
		clr = table_color[color_dict[state[:2]]]
		df = pd.read_csv(f'JHU50/combined2W_{end_date}/{state}/para.csv')
		reopen_date = df.iloc[0]['reopen']
		# print(state, clr, reopen_date)
		out_table.append([state, clr, reopen_date])

	out_df = pd.DataFrame(out_table, columns=['County', 'Color', 'Date'])
	out_df.to_csv(f'JHU50/combined2W_{end_date}/color_table.csv', index=False)
	return


def main():
	# plot_combined()
	plot_bar()
	peak_by_H()

	# color_table()

	# fit_all_SIR()
	# plot_grid('MT_2020-08-01/')
	# compare_all()
	# RMSE_all()
	# interval_GD_all()

	# plot_against_safeGraph()

	return


if __name__ == '__main__':
	main()
