import datetime
import pandas as pd
import numpy as np

start_date = '07-01-2020'
end_date = '07-03-2021'


def parse_UK():
	states = ['England', 'Northern Ireland', 'Scotland', 'Wales']
	df_G = []
	df_D = []
	for state in states:
		df_G.append([state])
		df_D.append([state])

	start_dt = datetime.datetime.strptime(start_date, '%m-%d-%Y')
	end_dt = datetime.datetime.strptime(end_date, '%m-%d-%Y')
	dts = []
	dt = start_dt
	while dt <= end_dt:
		dts.append(dt)
		dt = dt + datetime.timedelta(days=1)
	dts_str = [dt.strftime('%m-%d-%Y') for dt in dts]

	for dt in dts_str:
		df = pd.read_csv(f'D:/GIT/JHU-COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/{dt}.csv', usecols=['Province_State', 'Country_Region', 'Confirmed', 'Deaths'])
		for i in range(len(states)):
			state = states[i]
			row = df[(df['Province_State'] == state) & (df['Country_Region'] == 'United Kingdom')].iloc[0]
			df_G[i].append(row['Confirmed'])
			df_D[i].append(row['Deaths'])
	cols = ['state']
	dts_str = [dt.strftime('%Y-%m-%d') for dt in dts]
	cols.extend(dts_str)
	df_G = pd.DataFrame(df_G, columns=cols)
	df_D = pd.DataFrame(df_D, columns=cols)
	df_G.to_csv('UK/UK_confirmed.csv', index=False)
	df_D.to_csv('UK/UK_death.csv', index=False)
	return


def main():
	parse_UK()
	return


if __name__ == '__main__':
	main()