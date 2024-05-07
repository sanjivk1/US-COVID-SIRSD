import json
import pandas as pd
import numpy as np
"""
Created on Mon May  3 10:05:35 2021

@author: thejeswarreddynarravala
"""

"""
Download the raw data from the 
http://projects.datameet.org/covid19/mohfw/#cases-by-state by the file name mohfw.json
"""


f = open("india/mohfw.json")

data = json.load(f)

rows = data['rows']
no_of_rows = len(rows)

cases_india = pd.DataFrame(columns={'date', 'state', 'confirmed_cases', 'confirmed_death'})

for i in range(no_of_rows):
	state = rows[i]['value']['state']
	confirmed_cases = rows[i]['value']['confirmed']
	confirmed_death = rows[i]['value']['death']
	date = rows[i]['value']['_id'][0:10]
	cases_india = cases_india.append(
		{'date': date, 'state': state, 'confirmed_cases': confirmed_cases, 'confirmed_death': confirmed_death},
		ignore_index=True)

cases_india['date'] = pd.to_datetime(cases_india['date'])
cases_india['date'] = cases_india['date'].dt.strftime('%Y-%m-%d')
cols = cases_india.date.unique().tolist()
ind = cases_india.state.unique().tolist()
df = pd.DataFrame(index=ind, columns=cols)
df2 = pd.DataFrame(index=ind, columns=cols)

for i in range(no_of_rows):
	state = rows[i]['value']['state']
	confirmed_cases = rows[i]['value']['confirmed']
	confirmed_death = rows[i]['value']['death']
	date = rows[i]['value']['_id'][0:10]
	df.iloc[ind.index(state), cols.index(date)] = confirmed_cases
	df2.iloc[ind.index(state), cols.index(date)] = confirmed_death

df.index.name = "state"
df2.index.name = "state"
df = df.replace(np.nan, 0)
df2 = df2.replace(np.nan, 0)
df.to_csv('india/indian_cases_confirmed_cases.csv')
df2.to_csv('india/indian_cases_confirmed_deaths.csv')
f.close()
