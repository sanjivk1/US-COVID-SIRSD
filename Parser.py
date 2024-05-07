import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
from datetime import datetime as dt

state_abbr = {'Alabama': 'AL',
              'Alaska': 'AK',
              'Arizona': 'AZ',
              'Arkansas': 'AR',
              'California': 'CA',
              'Colorado': 'CO',
              'Connecticut': 'CT',
              'Delaware': 'DE',
              'District of Columbia': 'DC',
              'Florida': 'FL',
              'Georgia': 'GA',
              'Hawaii': 'HI',
              'Idaho': 'ID',
              'Illinois': 'IL',
              'Indiana': 'IN',
              'Iowa': 'IA',
              'Kansas': 'KS',
              'Kentucky': 'KY',
              'Louisiana': 'LA',
              'Maine': 'ME',
              'Maryland': 'MD',
              'Massachusetts': 'MA',
              'Michigan': 'MI',
              'Minnesota': 'MN',
              'Mississippi': 'MS',
              'Missouri': 'MO',
              'Montana': 'MT',
              'Nebraska': 'NE',
              'Nevada': 'NV',
              'New Hampshire': 'NH',
              'New Jersey': 'NJ',
              'New Mexico': 'NM',
              'New York': 'NY',
              'North Carolina': 'NC',
              'North Dakota': '	ND',
              'Ohio': 'OH',
              'Oklahoma': 'OK',
              'Oregon': 'OR',
              'Pennsylvania': 'PA',
              'Rhode Island': 'RI',
              'South Carolina': 'SC',
              'South Dakota': 'SD',
              'Tennessee': 'TN',
              'Texas': 'TX',
              'Utah': 'UT',
              'Vermont': 'VT',
              'Virginia': 'VA',
              'Washington': 'WA',
              'West Virginia': 'WV',
              'Wisconsin': 'WI',
              'Wyoming': 'WY'
              }


def Parse1P3A_state_confirmed():
	states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS',
	          'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV',
	          'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

	df = pd.read_csv('data/cases1P3A.csv',
	                 usecols=['confirmed_date', 'state_name', 'confirmed_count'])
	df.columns = ['date', 'state', 'confirmed']
	df = df[df['state'].isin(states)]

	start_date = min(df['date'])
	end_date = max(df['date'])
	dates = pd.date_range(start_date, end_date).strftime('%Y-%m-%d')
	cols = ['state']
	cols.extend(dates)
	l = len(cols)
	out_df = pd.DataFrame(columns=cols)

	for s in states:
		row = [0] * l
		row[0] = s
		current_row = len(out_df)
		out_df.loc[current_row] = row
		print(s)
		df2 = df[df['state'] == s]

		for d in range(1, l):
			if d > 1:
				out_df.loc[current_row].iloc[d] = out_df.loc[current_row].iloc[d - 1]
			for index, row in df2[df2['date'] == cols[d]].iterrows():
				out_df.loc[current_row].iloc[d] += row['confirmed']

	out_df.to_csv('data/Confirmed-US.csv', index=False)


def Parse1P3A_county_death():
	print('parsing county death')

	# counties = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS',
	#           'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV',
	#           'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
	# counties = ['CA-Los Angeles', 'CA-Riverside', 'FL-Miami-Dade', 'FL-Broward', 'NY-New York', 'NY-Nassau', 'NY-Suffolk', 'NY-Westchester', 'NY-Rockland', 'NY-Orange', 'NJ-Hudson',
	#             'NJ-Bergen', 'NJ-Essex', 'NJ-Passaic', 'NJ-Middlesex', 'NJ-Union', 'IL-Cook',
	#             'MA-Middlesex', 'MA-Suffolk', 'MA-Essex', 'MA-Worcester',
	#             'TX-Harris--Houston', 'FL-Palm Beach', 'TX-Dallas', 'AZ-Maricopa']

	counties = ['CA-Los Angeles', 'CA-Riverside', 'FL-Miami-Dade', 'FL-Broward', 'NY-New York', 'NY-Suffolk',
	            'TX-Dallas', 'TX-Harris--Houston', 'NJ-Hudson', 'NJ-Bergen', 'IL-Cook', 'AZ-Maricopa', 'GA-Fulton',
	            'PA-Philadelphia', 'MD-Prince Georges', 'NV-Clark', 'NC-Mecklenburg', 'LA-Jefferson', 'MN-Hennepin',
	            'MA-Middlesex', 'OH-Franklin', 'VA-Fairfax', 'SC-Charleston', 'MI-Oakland', 'TN-Shelby', 'WI-Milwaukee',
	            'UT-Salt Lake']

	df = pd.read_csv('data/cases1P3A.csv',
	                 usecols=['confirmed_date', 'state_name', 'county_name', 'death_count'])
	df.columns = ['date', 'state', 'county', 'death']

	start_date = min(df['date'])
	end_date = max(df['date'])
	dates = pd.date_range(start_date, end_date).strftime('%Y-%m-%d')
	cols = ['county']
	cols.extend(dates)
	l = len(cols)
	out_df = pd.DataFrame(columns=cols)

	for i in range(len(counties)):
		state, county = counties[i].split('-', 1)
		print(state, county)
		current_row = [counties[i]]
		current_row.extend([0] * (l - 1))
		a = df['state'] == state
		b = df['county'] == county
		df2 = df[a & b]

		for d in range(1, l):
			if d > 1:
				current_row[d] = current_row[d - 1]
			for index, row in df2[df2['date'] == cols[d]].iterrows():
				current_row[d] += row['death']
		out_df.loc[i] = current_row

	out_df.to_csv('data/Death-counties.csv', index=False)
	print('county death parsed\n')
	return 0


def Parse1P3A_county_confirmed():
	print('parsing county confirmed')

	# counties = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS',
	#           'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV',
	#           'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
	# counties = ['CA-Los Angeles', 'CA-Riverside', 'FL-Miami-Dade', 'FL-Broward', 'NY-New York', 'NY-Nassau', 'NY-Suffolk', 'NY-Westchester', 'NY-Rockland', 'NY-Orange', 'NJ-Hudson',
	#             'NJ-Bergen', 'NJ-Essex', 'NJ-Passaic', 'NJ-Middlesex', 'NJ-Union', 'IL-Cook',
	#             'MA-Middlesex', 'MA-Suffolk', 'MA-Essex', 'MA-Worcester',
	#             'TX-Harris--Houston', 'FL-Palm Beach', 'TX-Dallas', 'AZ-Maricopa']

	counties = ['CA-Los Angeles', 'CA-Riverside', 'FL-Miami-Dade', 'FL-Broward', 'NY-New York', 'NY-Suffolk',
	            'TX-Dallas', 'TX-Harris--Houston', 'NJ-Hudson', 'NJ-Bergen', 'IL-Cook', 'AZ-Maricopa', 'GA-Fulton',
	            'PA-Philadelphia', 'MD-Prince Georges', 'NV-Clark', 'NC-Mecklenburg', 'LA-Jefferson', 'MN-Hennepin',
	            'MA-Middlesex', 'OH-Franklin', 'VA-Fairfax', 'SC-Charleston', 'MI-Oakland', 'TN-Shelby', 'WI-Milwaukee',
	            'UT-Salt Lake']

	df = pd.read_csv('data/cases1P3A.csv',
	                 usecols=['confirmed_date', 'state_name', 'county_name', 'confirmed_count'])
	df.columns = ['date', 'state', 'county', 'confirmed']

	start_date = min(df['date'])
	end_date = max(df['date'])
	dates = pd.date_range(start_date, end_date).strftime('%Y-%m-%d')
	cols = ['county']
	cols.extend(dates)
	l = len(cols)
	out_df = pd.DataFrame(columns=cols)

	for i in range(len(counties)):
		state, county = counties[i].split('-', 1)
		print(state, county)
		current_row = [counties[i]]
		current_row.extend([0] * (l - 1))
		a = df['state'] == state
		b = df['county'] == county
		df2 = df[a & b]

		for d in range(1, l):
			if d > 1:
				current_row[d] = current_row[d - 1]
			for index, row in df2[df2['date'] == cols[d]].iterrows():
				current_row[d] += row['confirmed']
		out_df.loc[i] = current_row

	out_df.to_csv('data/Confirmed-counties.csv', index=False)
	print('county confirmed parsed\n')
	return 0


def Parse1P3A_city_confirmed():
	# counties = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS',
	#           'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV',
	#           'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
	counties = ['NY,New York', 'PA,Philadelphia', 'MI,Wayne--Detroit', 'VA,Fairfax', 'VA,Fairfax City']

	df = pd.read_csv('data/cases1P3A.csv',
	                 usecols=['confirmed_date', 'state_name', 'county_name', 'confirmed_count'])
	df.columns = ['date', 'state', 'county', 'confirmed']

	start_date = min(df['date'])
	end_date = max(df['date'])
	dates = pd.date_range(start_date, end_date).strftime('%Y-%m-%d')
	cols = ['county']
	cols.extend(dates)
	l = len(cols)
	out_df = pd.DataFrame(columns=cols)
	for s in counties:
		state, county = s.split(',')
		row = [0] * l
		row[0] = county
		current_row = len(out_df)
		out_df.loc[current_row] = row
		print(state, county)
		a = df['state'] == state
		b = df['county'] == county
		df2 = df[a & b]

		for d in range(1, l):
			if d > 1:
				out_df.loc[current_row].iloc[d] = out_df.loc[current_row].iloc[d - 1]
			for index, row in df2[df2['date'] == cols[d]].iterrows():
				out_df.loc[current_row].iloc[d] += row['confirmed']

	out_df.to_csv('data/Confirmed-cities.csv', index=False)


def Parse1P3A_city_death():
	# counties = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS',
	#           'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV',
	#           'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
	counties = ['NY,New York', 'PA,Philadelphia', 'MI,Wayne--Detroit', 'VA,Fairfax', 'VA,Fairfax City']

	df = pd.read_csv('data/cases1P3A.csv',
	                 usecols=['confirmed_date', 'state_name', 'county_name', 'death_count'])
	df.columns = ['date', 'state', 'county', 'death']

	start_date = min(df['date'])
	end_date = max(df['date'])
	dates = pd.date_range(start_date, end_date).strftime('%Y-%m-%d')
	cols = ['county']
	cols.extend(dates)
	l = len(cols)
	out_df = pd.DataFrame(columns=cols)
	for s in counties:
		state, county = s.split(',')
		row = [0] * l
		row[0] = county
		current_row = len(out_df)
		out_df.loc[current_row] = row
		print(state, county)
		a = df['state'] == state
		b = df['county'] == county
		df2 = df[a & b]

		for d in range(1, l):
			if d > 1:
				out_df.loc[current_row].iloc[d] = out_df.loc[current_row].iloc[d - 1]
			for index, row in df2[df2['date'] == cols[d]].iterrows():
				out_df.loc[current_row].iloc[d] += row['death']

	out_df.to_csv('data/Death-cities.csv', index=False)


def Parse_mobility():
	df = pd.read_csv('data/Global_Mobility_Report.csv',
	                 usecols=['country_region_code', 'sub_region_1', 'sub_region_2', 'iso_3166_2_code', 'date',
	                          'retail_and_recreation_percent_change_from_baseline',
	                          'transit_stations_percent_change_from_baseline',
	                          'workplaces_percent_change_from_baseline'])
	df = df[df['country_region_code'] == 'US']
	states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS',
	          'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV',
	          'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
	s = states[0]
	df2 = df[df['iso_3166_2_code'] == 'US-' + s]
	col = ['State']
	col.extend(df2['date'])
	out_df = pd.DataFrame(columns=col)
	for i in range(len(states)):
		s = states[i]
		df2 = df[df['iso_3166_2_code'] == 'US-' + s]
		mob = (np.array(df2['retail_and_recreation_percent_change_from_baseline']) + np.array(
			df2['transit_stations_percent_change_from_baseline']) + np.array(
			df2['workplaces_percent_change_from_baseline'])) / 300 + 1
		mob /= max(mob)
		mob -= 1
		entry = [s]
		entry.extend(mob)
		out_df.loc[i] = entry
	out_df.to_csv('data/Mobility-states.csv', index=False)


def Parse_county_mobility():
	df = pd.read_csv('data/Global_Mobility_Report.csv',
	                 usecols=['country_region_code', 'sub_region_1', 'sub_region_2', 'date',
	                          'retail_and_recreation_percent_change_from_baseline',
	                          'transit_stations_percent_change_from_baseline',
	                          'workplaces_percent_change_from_baseline'])
	df = df[df['country_region_code'] == 'US']
	states = [['Florida', 'Miami-Dade County', 'FL-Miami-Dade'], ['Texas', 'Harris County', 'TX-Harris--Houston']]
	s = states[0]
	df2 = df[df['sub_region_1'] == s[0]]
	df2 = df2[df2['sub_region_2'] == s[1]]
	col = ['county']
	col.extend(df2['date'])
	out_df = pd.DataFrame(columns=col)
	for i in range(len(states)):
		s = states[i][0]
		c = states[i][1]
		df2 = df[df['sub_region_1'] == s]
		df2 = df2[df2['sub_region_2'] == c]

		mob = (np.array(df2['retail_and_recreation_percent_change_from_baseline']) + np.array(
			df2['transit_stations_percent_change_from_baseline']) + np.array(
			df2['workplaces_percent_change_from_baseline'])) / 300 + 1
		mob /= max(mob)
		mob -= 1
		entry = [states[i][2]]
		entry.extend(mob)
		out_df.loc[i] = entry
	out_df.to_csv('data/Mobility-counties.csv', index=False)


def ParseJHU_county_confirmed():
	print('Parsing JHU confirmed')
	# counties = ['Cook, Illinois, US',
	#             'Queens, New York, US',
	#             'Kings, New York, US',
	#             'Bronx, New York, US',
	#             'Nassau, New York, US',
	#             'Suffolk, New York, US',
	#             'Los Angeles, California, US',
	#             'Westchester, New York, US',
	#             'New York, New York, US',
	#             'Wayne, Michigan, US',
	#             'Philadelphia, Pennsylvania, US',
	#             'Middlesex, Massachusetts, US',
	#             'Hudson, New Jersey, US',
	#             'Bergen, New Jersey, US',
	#             'Suffolk, Massachusetts, US',
	#             'Essex, New Jersey, US',
	#             'Miami-Dade, Florida, US',
	#             'Passaic, New Jersey, US',
	#             'Union, New Jersey, US',
	#             'Middlesex, New Jersey, US',
	#             'Fairfield, Connecticut, US',
	#             'Richmond, New York, US',
	#             'Rockland, New York, US',
	#             'Essex, Massachusetts, US',
	#             'Prince George\'s, Maryland, US',
	#             'New Haven, Connecticut, US',
	#             'Orange, New York, US',
	#             'Oakland, Michigan, US',
	#             'Worcester, Massachusetts, US',
	#             'Providence, Rhode Island, US',
	#             'Harris, Texas, US',
	#             'Hartford, Connecticut, US',
	#             'Marion, Indiana, US',
	#             'Ocean, New Jersey, US',
	#             'Montgomery, Maryland, US',
	#             'Norfolk, Massachusetts, US',
	#             'King, Washington, US',
	#             'Monmouth, New Jersey, US',
	#             'Plymouth, Massachusetts, US',
	#             'Fairfax, Virginia, US',
	#             'Dallas, Texas, US',
	#             'Jefferson, Louisiana, US',
	#             'District of Columbia,District of Columbia,US',
	#             'Maricopa, Arizona, US',
	#             'Orleans, Louisiana, US',
	#             'Macomb, Michigan, US',
	#             'Lake, Illinois, US',
	#             'Broward, Florida, US',
	#             'Bristol, Massachusetts, US',
	#             'Morris, New Jersey, US',
	#             'Montgomery, Pennsylvania, US',
	#             'Mercer, New Jersey, US',
	#             'DuPage, Illinois, US',
	#             'Riverside, California, US',
	#             'Delaware, Pennsylvania, US',
	#             'San Diego, California, US',
	#             'Hampden, Massachusetts, US',
	#             'Camden, New Jersey, US',
	#             'Clark, Nevada, US',
	#             'Erie, New York, US',
	#             'Hennepin, Minnesota, US',
	#             'Milwaukee, Wisconsin, US',
	#             'Denver, Colorado, US',
	#             'Baltimore, Maryland, US',
	#             'Palm Beach, Florida, US',
	#             'Franklin, Ohio, US',
	#             'Bucks, Pennsylvania, US',
	#             'St. Louis, Missouri, US',
	#             'Will, Illinois, US',
	#             'Tarrant, Texas, US',
	#             'Somerset, New Jersey, US',
	#             'Kane, Illinois, US',
	#             'Orange, California, US',
	#             'Burlington, New Jersey, US',
	#             'Davidson, Tennessee, US',
	#             'Salt Lake, Utah, US',
	#             'Fulton, Georgia, US',
	#             'Baltimore City, Maryland, US',
	#             'Shelby, Tennessee, US',
	#             'Berks, Pennsylvania, US',
	#             'Arapahoe, Colorado, US',
	#             'Sussex, Delaware, US',
	#             'Dutchess, New York, US',
	#             'Prince William, Virginia, US',
	#             'Lehigh, Pennsylvania, US',
	#             'San Bernardino, California, US',
	#             'Cuyahoga, Ohio, US',
	#             'Minnehaha, South Dakota, US',
	#             'East Baton Rouge, Louisiana, US',
	#             'Kent, Michigan, US',
	#             'Polk, Iowa, US',
	#             'Snohomish, Washington, US',
	#             'Anne Arundel, Maryland, US',
	#             'DeKalb, Georgia, US',
	#             'Lake, Indiana, US',
	#             'New Castle, Delaware, US',
	#             'Northampton, Pennsylvania, US',
	#             'Gwinnett, Georgia, US',
	#             'Adams, Colorado, US',
	#             'Luzerne, Pennsylvania, US']
	counties = ['Cook, Illinois, US',
	            'Queens, New York, US',
	            'Kings, New York, US',
	            'Bronx, New York, US',
	            'Nassau, New York, US',
	            'Suffolk, New York, US',
	            'Los Angeles, California, US',
	            'Westchester, New York, US',
	            'New York, New York, US',
	            'Wayne, Michigan, US',
	            'Philadelphia, Pennsylvania, US',
	            'Middlesex, Massachusetts, US',
	            'Hudson, New Jersey, US',
	            'Bergen, New Jersey, US',
	            'Suffolk, Massachusetts, US',
	            'Essex, New Jersey, US',
	            'Miami-Dade, Florida, US',
	            'Passaic, New Jersey, US',
	            'Union, New Jersey, US',
	            'Middlesex, New Jersey, US',
	            'Fairfield, Connecticut, US',
	            'Richmond, New York, US',
	            'Rockland, New York, US',
	            'Essex, Massachusetts, US',
	            'Prince George\'s, Maryland, US',
	            'New Haven, Connecticut, US',
	            'Orange, New York, US',
	            'Oakland, Michigan, US',
	            'Worcester, Massachusetts, US',
	            'Providence, Rhode Island, US',
	            'Harris, Texas, US',
	            'Hartford, Connecticut, US',
	            'Marion, Indiana, US',
	            'Ocean, New Jersey, US',
	            'Montgomery, Maryland, US',
	            'Norfolk, Massachusetts, US',
	            'King, Washington, US',
	            'Monmouth, New Jersey, US',
	            'Plymouth, Massachusetts, US',
	            'Fairfax, Virginia, US',
	            'Dallas, Texas, US',
	            'Jefferson, Louisiana, US',
	            'District of Columbia,District of Columbia,US',
	            'Maricopa, Arizona, US',
	            'Orleans, Louisiana, US',
	            'Macomb, Michigan, US',
	            'Lake, Illinois, US',
	            'Broward, Florida, US',
	            'Bristol, Massachusetts, US',
	            'Morris, New Jersey, US',
	            'Montgomery, Pennsylvania, US',
	            'Mercer, New Jersey, US',
	            'DuPage, Illinois, US',
	            'Riverside, California, US',
	            'Delaware, Pennsylvania, US',
	            'San Diego, California, US',
	            'Hampden, Massachusetts, US',
	            'Camden, New Jersey, US',
	            'Clark, Nevada, US',
	            'Erie, New York, US',
	            'Hennepin, Minnesota, US',
	            'Milwaukee, Wisconsin, US',
	            'Denver, Colorado, US',
	            'Baltimore, Maryland, US',
	            'Palm Beach, Florida, US',
	            'Franklin, Ohio, US',
	            'Bucks, Pennsylvania, US',
	            'St. Louis, Missouri, US',
	            'Will, Illinois, US',
	            'Tarrant, Texas, US',
	            'Somerset, New Jersey, US',
	            'Kane, Illinois, US',
	            'Orange, California, US',
	            'Burlington, New Jersey, US',
	            'Davidson, Tennessee, US',
	            'Salt Lake, Utah, US',
	            'Fulton, Georgia, US',
	            'Baltimore City, Maryland, US',
	            'Shelby, Tennessee, US',
	            'Berks, Pennsylvania, US',
	            'Arapahoe, Colorado, US',
	            'Sussex, Delaware, US',
	            'Dutchess, New York, US',
	            'Prince William, Virginia, US',
	            'Lehigh, Pennsylvania, US',
	            'San Bernardino, California, US',
	            'Cuyahoga, Ohio, US',
	            'Minnehaha, South Dakota, US',
	            'East Baton Rouge, Louisiana, US',
	            'Kent, Michigan, US',
	            'Polk, Iowa, US',
	            'Snohomish, Washington, US',
	            'Anne Arundel, Maryland, US',
	            'DeKalb, Georgia, US',
	            'Lake, Indiana, US',
	            'New Castle, Delaware, US',
	            'Northampton, Pennsylvania, US',
	            'Gwinnett, Georgia, US',
	            'Adams, Colorado, US',
	            'Luzerne, Pennsylvania, US',
	            'Bexar, Texas, US',
	            'Hillsborough, Florida, US',
	            'Orange, Florida, US',
	            'Kern, California, US',
	            'Hidalgo, Texas, US',
	            'Travis, Texas, US',
	            'Duval, Florida, US',
	            'Mecklenburg, North Carolina, US',
	            'Fresno, California, US',
	            'Pima, Arizona, US',
	            'Cameron, Texas, US',
	            'El Paso, Texas, US',
	            'Pinellas, Florida, US',
	            'Nueces, Texas, US',
	            'Lee, Florida, US',
	            'Alameda, California, US',
	            'Sacramento, California, US',
	            'San Joaquin, California, US',
	            'Santa Clara, California, US',
	            'Polk, Florida, US',
	            'Cobb, Georgia, US',
	            'Jefferson, Alabama, US',
	            'Wake, North Carolina, US',
	            'Fort Bend, Texas, US',
	            'Stanislaus, California, US',
	            'Tulare, California, US',
	            'Charleston, South Carolina, US',
	            'Contra Costa, California, US',
	            'Douglas, Nebraska, US',
	            'Oklahoma, Oklahoma, US',
	            'Tulsa, Oklahoma, US',
	            'Yuma, Arizona, US',
	            'Mobile, Alabama, US',
	            'Greenville, South Carolina, US',
	            'Jefferson, Kentucky, US']
	df = pd.read_csv('JHU/time_series_covid19_confirmed_US.csv')
	df = df.iloc[:, 10:]
	cols = df.columns.tolist()
	cols[0] = 'county'
	for i in range(1, len(cols)):
		date = dt.strptime(cols[i], '%m/%d/%y')
		cols[i] = date.strftime('%Y-%m-%d')
	df.columns = cols
	df = df[df['county'].isin(counties)]
	for i in range(len(df)):
		s = df.iloc[i, 0]
		s = s.split(',')
		s = state_abbr[s[1].strip()] + '-' + s[0].strip()
		print(s)
		df.iloc[i, 0] = s
		for j in range(2, len(df.iloc[i])):
			if df.iloc[i, j] < df.iloc[i, j - 1]:
				df.iloc[i, j] = df.iloc[i, j - 1]
	df.to_csv('JHU/JHU_Confirmed-counties.csv', index=False)


def ParseJHU_county_death():
	print('Parsing JHU death')
	# counties = ['Cook, Illinois, US',
	#             'Queens, New York, US',
	#             'Kings, New York, US',
	#             'Bronx, New York, US',
	#             'Nassau, New York, US',
	#             'Suffolk, New York, US',
	#             'Los Angeles, California, US',
	#             'Westchester, New York, US',
	#             'New York, New York, US',
	#             'Wayne, Michigan, US',
	#             'Philadelphia, Pennsylvania, US',
	#             'Middlesex, Massachusetts, US',
	#             'Hudson, New Jersey, US',
	#             'Bergen, New Jersey, US',
	#             'Suffolk, Massachusetts, US',
	#             'Essex, New Jersey, US',
	#             'Miami-Dade, Florida, US',
	#             'Passaic, New Jersey, US',
	#             'Union, New Jersey, US',
	#             'Middlesex, New Jersey, US',
	#             'Fairfield, Connecticut, US',
	#             'Richmond, New York, US',
	#             'Rockland, New York, US',
	#             'Essex, Massachusetts, US',
	#             'Prince George\'s, Maryland, US',
	#             'New Haven, Connecticut, US',
	#             'Orange, New York, US',
	#             'Oakland, Michigan, US',
	#             'Worcester, Massachusetts, US',
	#             'Providence, Rhode Island, US',
	#             'Harris, Texas, US',
	#             'Hartford, Connecticut, US',
	#             'Marion, Indiana, US',
	#             'Ocean, New Jersey, US',
	#             'Montgomery, Maryland, US',
	#             'Norfolk, Massachusetts, US',
	#             'King, Washington, US',
	#             'Monmouth, New Jersey, US',
	#             'Plymouth, Massachusetts, US',
	#             'Fairfax, Virginia, US',
	#             'Dallas, Texas, US',
	#             'Jefferson, Louisiana, US',
	#             'District of Columbia,District of Columbia,US',
	#             'Maricopa, Arizona, US',
	#             'Orleans, Louisiana, US',
	#             'Macomb, Michigan, US',
	#             'Lake, Illinois, US',
	#             'Broward, Florida, US',
	#             'Bristol, Massachusetts, US',
	#             'Morris, New Jersey, US',
	#             'Montgomery, Pennsylvania, US',
	#             'Mercer, New Jersey, US',
	#             'DuPage, Illinois, US',
	#             'Riverside, California, US',
	#             'Delaware, Pennsylvania, US',
	#             'San Diego, California, US',
	#             'Hampden, Massachusetts, US',
	#             'Camden, New Jersey, US',
	#             'Clark, Nevada, US',
	#             'Erie, New York, US',
	#             'Hennepin, Minnesota, US',
	#             'Milwaukee, Wisconsin, US',
	#             'Denver, Colorado, US',
	#             'Baltimore, Maryland, US',
	#             'Palm Beach, Florida, US',
	#             'Franklin, Ohio, US',
	#             'Bucks, Pennsylvania, US',
	#             'St. Louis, Missouri, US',
	#             'Will, Illinois, US',
	#             'Tarrant, Texas, US',
	#             'Somerset, New Jersey, US',
	#             'Kane, Illinois, US',
	#             'Orange, California, US',
	#             'Burlington, New Jersey, US',
	#             'Davidson, Tennessee, US',
	#             'Salt Lake, Utah, US',
	#             'Fulton, Georgia, US',
	#             'Baltimore City, Maryland, US',
	#             'Shelby, Tennessee, US',
	#             'Berks, Pennsylvania, US',
	#             'Arapahoe, Colorado, US',
	#             'Sussex, Delaware, US',
	#             'Dutchess, New York, US',
	#             'Prince William, Virginia, US',
	#             'Lehigh, Pennsylvania, US',
	#             'San Bernardino, California, US',
	#             'Cuyahoga, Ohio, US',
	#             'Minnehaha, South Dakota, US',
	#             'East Baton Rouge, Louisiana, US',
	#             'Kent, Michigan, US',
	#             'Polk, Iowa, US',
	#             'Snohomish, Washington, US',
	#             'Anne Arundel, Maryland, US',
	#             'DeKalb, Georgia, US',
	#             'Lake, Indiana, US',
	#             'New Castle, Delaware, US',
	#             'Northampton, Pennsylvania, US',
	#             'Gwinnett, Georgia, US',
	#             'Adams, Colorado, US',
	#             'Luzerne, Pennsylvania, US']
	counties = ['Cook, Illinois, US',
	            'Queens, New York, US',
	            'Kings, New York, US',
	            'Bronx, New York, US',
	            'Nassau, New York, US',
	            'Suffolk, New York, US',
	            'Los Angeles, California, US',
	            'Westchester, New York, US',
	            'New York, New York, US',
	            'Wayne, Michigan, US',
	            'Philadelphia, Pennsylvania, US',
	            'Middlesex, Massachusetts, US',
	            'Hudson, New Jersey, US',
	            'Bergen, New Jersey, US',
	            'Suffolk, Massachusetts, US',
	            'Essex, New Jersey, US',
	            'Miami-Dade, Florida, US',
	            'Passaic, New Jersey, US',
	            'Union, New Jersey, US',
	            'Middlesex, New Jersey, US',
	            'Fairfield, Connecticut, US',
	            'Richmond, New York, US',
	            'Rockland, New York, US',
	            'Essex, Massachusetts, US',
	            'Prince George\'s, Maryland, US',
	            'New Haven, Connecticut, US',
	            'Orange, New York, US',
	            'Oakland, Michigan, US',
	            'Worcester, Massachusetts, US',
	            'Providence, Rhode Island, US',
	            'Harris, Texas, US',
	            'Hartford, Connecticut, US',
	            'Marion, Indiana, US',
	            'Ocean, New Jersey, US',
	            'Montgomery, Maryland, US',
	            'Norfolk, Massachusetts, US',
	            'King, Washington, US',
	            'Monmouth, New Jersey, US',
	            'Plymouth, Massachusetts, US',
	            'Fairfax, Virginia, US',
	            'Dallas, Texas, US',
	            'Jefferson, Louisiana, US',
	            'District of Columbia,District of Columbia,US',
	            'Maricopa, Arizona, US',
	            'Orleans, Louisiana, US',
	            'Macomb, Michigan, US',
	            'Lake, Illinois, US',
	            'Broward, Florida, US',
	            'Bristol, Massachusetts, US',
	            'Morris, New Jersey, US',
	            'Montgomery, Pennsylvania, US',
	            'Mercer, New Jersey, US',
	            'DuPage, Illinois, US',
	            'Riverside, California, US',
	            'Delaware, Pennsylvania, US',
	            'San Diego, California, US',
	            'Hampden, Massachusetts, US',
	            'Camden, New Jersey, US',
	            'Clark, Nevada, US',
	            'Erie, New York, US',
	            'Hennepin, Minnesota, US',
	            'Milwaukee, Wisconsin, US',
	            'Denver, Colorado, US',
	            'Baltimore, Maryland, US',
	            'Palm Beach, Florida, US',
	            'Franklin, Ohio, US',
	            'Bucks, Pennsylvania, US',
	            'St. Louis, Missouri, US',
	            'Will, Illinois, US',
	            'Tarrant, Texas, US',
	            'Somerset, New Jersey, US',
	            'Kane, Illinois, US',
	            'Orange, California, US',
	            'Burlington, New Jersey, US',
	            'Davidson, Tennessee, US',
	            'Salt Lake, Utah, US',
	            'Fulton, Georgia, US',
	            'Baltimore City, Maryland, US',
	            'Shelby, Tennessee, US',
	            'Berks, Pennsylvania, US',
	            'Arapahoe, Colorado, US',
	            'Sussex, Delaware, US',
	            'Dutchess, New York, US',
	            'Prince William, Virginia, US',
	            'Lehigh, Pennsylvania, US',
	            'San Bernardino, California, US',
	            'Cuyahoga, Ohio, US',
	            'Minnehaha, South Dakota, US',
	            'East Baton Rouge, Louisiana, US',
	            'Kent, Michigan, US',
	            'Polk, Iowa, US',
	            'Snohomish, Washington, US',
	            'Anne Arundel, Maryland, US',
	            'DeKalb, Georgia, US',
	            'Lake, Indiana, US',
	            'New Castle, Delaware, US',
	            'Northampton, Pennsylvania, US',
	            'Gwinnett, Georgia, US',
	            'Adams, Colorado, US',
	            'Luzerne, Pennsylvania, US',
	            'Bexar, Texas, US',
	            'Hillsborough, Florida, US',
	            'Orange, Florida, US',
	            'Kern, California, US',
	            'Hidalgo, Texas, US',
	            'Travis, Texas, US',
	            'Duval, Florida, US',
	            'Mecklenburg, North Carolina, US',
	            'Fresno, California, US',
	            'Pima, Arizona, US',
	            'Cameron, Texas, US',
	            'El Paso, Texas, US',
	            'Pinellas, Florida, US',
	            'Nueces, Texas, US',
	            'Lee, Florida, US',
	            'Alameda, California, US',
	            'Sacramento, California, US',
	            'San Joaquin, California, US',
	            'Santa Clara, California, US',
	            'Polk, Florida, US',
	            'Cobb, Georgia, US',
	            'Jefferson, Alabama, US',
	            'Wake, North Carolina, US',
	            'Fort Bend, Texas, US',
	            'Stanislaus, California, US',
	            'Tulare, California, US',
	            'Charleston, South Carolina, US',
	            'Contra Costa, California, US',
	            'Douglas, Nebraska, US',
	            'Oklahoma, Oklahoma, US',
	            'Tulsa, Oklahoma, US',
	            'Yuma, Arizona, US',
	            'Mobile, Alabama, US',
	            'Greenville, South Carolina, US',
	            'Jefferson, Kentucky, US']
	df = pd.read_csv('JHU/time_series_covid19_deaths_US.csv')
	df = df.iloc[:, 10:]
	cols = df.columns.tolist()
	cols[0] = 'county'
	for i in range(2, len(cols)):
		date = dt.strptime(cols[i], '%m/%d/%y')
		cols[i] = date.strftime('%Y-%m-%d')
	df.columns = cols
	df = df[df['county'].isin(counties)]
	for i in range(len(df)):
		s = df.iloc[i, 0]
		s = s.split(',')
		s = state_abbr[s[1].strip()] + '-' + s[0].strip()
		print(s)
		df.iloc[i, 0] = s
		for j in range(3, len(df.iloc[i])):
			if df.iloc[i, j] < df.iloc[i, j - 1]:
				df.iloc[i, j] = df.iloc[i, j - 1]
	df.to_csv('JHU/JHU_Death-counties.csv', index=False)


def tmp():
	new_counties_full = ['Cook, Illinois, US',
	                     'Queens, New York, US',
	                     'Kings, New York, US',
	                     'Bronx, New York, US',
	                     'Nassau, New York, US',
	                     'Suffolk, New York, US',
	                     'Los Angeles, California, US',
	                     'Westchester, New York, US',
	                     'New York, New York, US',
	                     'Wayne, Michigan, US',
	                     'Philadelphia, Pennsylvania, US',
	                     'Middlesex, Massachusetts, US',
	                     'Hudson, New Jersey, US',
	                     'Bergen, New Jersey, US',
	                     'Suffolk, Massachusetts, US',
	                     'Essex, New Jersey, US',
	                     'Miami-Dade, Florida, US',
	                     'Passaic, New Jersey, US',
	                     'Union, New Jersey, US',
	                     'Middlesex, New Jersey, US',
	                     'Fairfield, Connecticut, US',
	                     'Richmond, New York, US',
	                     'Rockland, New York, US',
	                     'Essex, Massachusetts, US',
	                     'Prince George\'s, Maryland, US',
	                     'New Haven, Connecticut, US',
	                     'Orange, New York, US',
	                     'Oakland, Michigan, US',
	                     'Worcester, Massachusetts, US',
	                     'Providence, Rhode Island, US',
	                     'Harris, Texas, US',
	                     'Hartford, Connecticut, US',
	                     'Marion, Indiana, US',
	                     'Ocean, New Jersey, US',
	                     'Montgomery, Maryland, US',
	                     'Norfolk, Massachusetts, US',
	                     'King, Washington, US',
	                     'Monmouth, New Jersey, US',
	                     'Plymouth, Massachusetts, US',
	                     'Fairfax, Virginia, US',
	                     'Dallas, Texas, US',
	                     'Jefferson, Louisiana, US',
	                     'District of Columbia,District of Columbia,US',
	                     'Maricopa, Arizona, US',
	                     'Orleans, Louisiana, US',
	                     'Macomb, Michigan, US',
	                     'Lake, Illinois, US',
	                     'Broward, Florida, US',
	                     'Bristol, Massachusetts, US',
	                     'Morris, New Jersey, US',
	                     'Montgomery, Pennsylvania, US',
	                     'Mercer, New Jersey, US',
	                     'DuPage, Illinois, US',
	                     'Riverside, California, US',
	                     'Delaware, Pennsylvania, US',
	                     'San Diego, California, US',
	                     'Hampden, Massachusetts, US',
	                     'Camden, New Jersey, US',
	                     'Clark, Nevada, US',
	                     'Erie, New York, US',
	                     'Hennepin, Minnesota, US',
	                     'Milwaukee, Wisconsin, US',
	                     'Denver, Colorado, US',
	                     'Baltimore, Maryland, US',
	                     'Palm Beach, Florida, US',
	                     'Franklin, Ohio, US',
	                     'Bucks, Pennsylvania, US',
	                     'St. Louis, Missouri, US',
	                     'Will, Illinois, US',
	                     'Tarrant, Texas, US',
	                     'Somerset, New Jersey, US',
	                     'Kane, Illinois, US',
	                     'Orange, California, US',
	                     'Burlington, New Jersey, US',
	                     'Davidson, Tennessee, US',
	                     'Salt Lake, Utah, US',
	                     'Fulton, Georgia, US',
	                     'Baltimore City, Maryland, US',
	                     'Shelby, Tennessee, US',
	                     'Berks, Pennsylvania, US',
	                     'Arapahoe, Colorado, US',
	                     'Sussex, Delaware, US',
	                     'Dutchess, New York, US',
	                     'Prince William, Virginia, US',
	                     'Lehigh, Pennsylvania, US',
	                     'San Bernardino, California, US',
	                     'Cuyahoga, Ohio, US',
	                     'Minnehaha, South Dakota, US',
	                     'East Baton Rouge, Louisiana, US',
	                     'Kent, Michigan, US',
	                     'Polk, Iowa, US',
	                     'Snohomish, Washington, US',
	                     'Anne Arundel, Maryland, US',
	                     'DeKalb, Georgia, US',
	                     'Lake, Indiana, US',
	                     'New Castle, Delaware, US',
	                     'Northampton, Pennsylvania, US',
	                     'Gwinnett, Georgia, US',
	                     'Adams, Colorado, US',
	                     'Luzerne, Pennsylvania, US',
	                     'Bexar, Texas, US',
	                     'Hillsborough, Florida, US',
	                     'Orange, Florida, US',
	                     'Kern, California, US',
	                     'Hidalgo, Texas, US',
	                     'Travis, Texas, US',
	                     'Duval, Florida, US',
	                     'Mecklenburg, North Carolina, US',
	                     'Fresno, California, US',
	                     'Pima, Arizona, US',
	                     'Cameron, Texas, US',
	                     'El Paso, Texas, US',
	                     'Pinellas, Florida, US',
	                     'Nueces, Texas, US',
	                     'Lee, Florida, US',
	                     'Alameda, California, US',
	                     'Sacramento, California, US',
	                     'San Joaquin, California, US',
	                     'Santa Clara, California, US',
	                     'Polk, Florida, US',
	                     'Cobb, Georgia, US',
	                     'Jefferson, Alabama, US',
	                     'Wake, North Carolina, US',
	                     'Fort Bend, Texas, US',
	                     'Stanislaus, California, US',
	                     'Tulare, California, US',
	                     'Charleston, South Carolina, US',
	                     'Contra Costa, California, US',
	                     'Douglas, Nebraska, US',
	                     'Oklahoma, Oklahoma, US',
	                     'Tulsa, Oklahoma, US',
	                     'Yuma, Arizona, US',
	                     'Mobile, Alabama, US',
	                     'Greenville, South Carolina, US',
	                     'Jefferson, Kentucky, US']
	new_counties = ['IL-Cook',
	                'NY-Queens',
	                'NY-Kings',
	                'NY-Bronx',
	                'NY-Nassau',
	                'NY-Suffolk',
	                'CA-Los Angeles',
	                'NY-Westchester',
	                'NY-New York',
	                'MI-Wayne',
	                'PA-Philadelphia',
	                'MA-Middlesex',
	                'NJ-Hudson',
	                'NJ-Bergen',
	                'MA-Suffolk',
	                'NJ-Essex',
	                'FL-Miami-Dade',
	                'NJ-Passaic',
	                'NJ-Union',
	                'NJ-Middlesex',
	                'CT-Fairfield',
	                'NY-Richmond',
	                'NY-Rockland',
	                'MA-Essex',
	                'MD-Prince George\'s',
	                'CT-New Haven',
	                'NY-Orange',
	                'MI-Oakland',
	                'MA-Worcester',
	                'RI-Providence',
	                'TX-Harris',
	                'CT-Hartford',
	                'IN-Marion',
	                'NJ-Ocean',
	                'MD-Montgomery',
	                'MA-Norfolk',
	                'WA-King',
	                'NJ-Monmouth',
	                'MA-Plymouth',
	                'VA-Fairfax',
	                'TX-Dallas',
	                'LA-Jefferson',
	                'DC-District of Columbia',
	                'AZ-Maricopa',
	                'LA-Orleans',
	                'MI-Macomb',
	                'IL-Lake',
	                'FL-Broward',
	                'MA-Bristol',
	                'NJ-Morris',
	                'PA-Montgomery',
	                'NJ-Mercer',
	                'IL-DuPage',
	                'CA-Riverside',
	                'PA-Delaware',
	                'CA-San Diego',
	                'MA-Hampden',
	                'NJ-Camden',
	                'NV-Clark',
	                'NY-Erie',
	                'MN-Hennepin',
	                'WI-Milwaukee',
	                'CO-Denver',
	                'MD-Baltimore',
	                'FL-Palm Beach',
	                'OH-Franklin',
	                'PA-Bucks',
	                'MO-St. Louis',
	                'IL-Will',
	                'TX-Tarrant',
	                'NJ-Somerset',
	                'IL-Kane',
	                'CA-Orange',
	                'NJ-Burlington',
	                'TN-Davidson',
	                'UT-Salt Lake',
	                'GA-Fulton',
	                'MD-Baltimore City',
	                'TN-Shelby',
	                'PA-Berks',
	                'CO-Arapahoe',
	                'DE-Sussex',
	                'NY-Dutchess',
	                'VA-Prince William',
	                'PA-Lehigh',
	                'CA-San Bernardino',
	                'OH-Cuyahoga',
	                'SD-Minnehaha',
	                'LA-East Baton Rouge',
	                'MI-Kent',
	                'IA-Polk',
	                'WA-Snohomish',
	                'MD-Anne Arundel',
	                'GA-DeKalb',
	                'IN-Lake',
	                'DE-New Castle',
	                'PA-Northampton',
	                'GA-Gwinnett',
	                'CO-Adams',
	                'PA-Luzerne',
	                'TX-Bexar',
	                'FL-Hillsborough',
	                'FL-Orange',
	                'CA-Kern',
	                'TX-Hidalgo',
	                'TX-Travis',
	                'FL-Duval',
	                'NC-Mecklenburg',
	                'CA-Fresno',
	                'AZ-Pima',
	                'TX-Cameron',
	                'TX-El Paso',
	                'FL-Pinellas',
	                'TX-Nueces',
	                'FL-Lee',
	                'CA-Alameda',
	                'CA-Sacramento',
	                'CA-San Joaquin',
	                'CA-Santa Clara',
	                'FL-Polk',
	                'GA-Cobb',
	                'AL-Jefferson',
	                'NC-Wake',
	                'TX-Fort Bend',
	                'CA-Stanislaus',
	                'CA-Tulare',
	                'SC-Charleston',
	                'CA-Contra Costa',
	                'NE-Douglas',
	                'OK-Oklahoma',
	                'OK-Tulsa',
	                'AZ-Yuma',
	                'AL-Mobile',
	                'SC-Greenville',
	                'KY-Jefferson']
	df = pd.read_csv('JHU/pop_county.csv')
	out_df = pd.DataFrame(columns=['county', 'pop'])
	for i in range(len(new_counties_full)):
		county = new_counties_full[i]
		county_str = county.split(',')
		county_str = [st.strip() for st in county_str]
		county_key = f'{county_str[0]} County, {county_str[1]}'
		hit = df[df['county'] == county_key]
		if len(hit) > 0:
			out_df.loc[len(out_df)] = [new_counties[i], hit.iloc[0]['pop']]
		else:
			print(county_key, 'not found')
			out_df.loc[len(out_df)] = [new_counties[i], 0]
	# print(out_df)
	out_df.to_csv('JHU/new county pop.csv', index=False)
	return


def tmp2():
	counties = ['Cook, Illinois, US',
	            'Queens, New York, US',
	            'Kings, New York, US',
	            'Bronx, New York, US',
	            'Nassau, New York, US',
	            'Suffolk, New York, US',
	            'Los Angeles, California, US',
	            'Westchester, New York, US',
	            'New York, New York, US',
	            'Wayne, Michigan, US',
	            'Philadelphia, Pennsylvania, US',
	            'Middlesex, Massachusetts, US',
	            'Hudson, New Jersey, US',
	            'Bergen, New Jersey, US',
	            'Suffolk, Massachusetts, US',
	            'Essex, New Jersey, US',
	            'Miami-Dade, Florida, US',
	            'Passaic, New Jersey, US',
	            'Union, New Jersey, US',
	            'Middlesex, New Jersey, US',
	            'Fairfield, Connecticut, US',
	            'Richmond, New York, US',
	            'Rockland, New York, US',
	            'Essex, Massachusetts, US',
	            'Prince George\'s, Maryland, US',
	            'New Haven, Connecticut, US',
	            'Orange, New York, US',
	            'Oakland, Michigan, US',
	            'Worcester, Massachusetts, US',
	            'Providence, Rhode Island, US',
	            'Harris, Texas, US',
	            'Hartford, Connecticut, US',
	            'Marion, Indiana, US',
	            'Ocean, New Jersey, US',
	            'Montgomery, Maryland, US',
	            'Norfolk, Massachusetts, US',
	            'King, Washington, US',
	            'Monmouth, New Jersey, US',
	            'Plymouth, Massachusetts, US',
	            'Fairfax, Virginia, US',
	            'Dallas, Texas, US',
	            'Jefferson, Louisiana, US',
	            'District of Columbia,District of Columbia,US',
	            'Maricopa, Arizona, US',
	            'Orleans, Louisiana, US',
	            'Macomb, Michigan, US',
	            'Lake, Illinois, US',
	            'Broward, Florida, US',
	            'Bristol, Massachusetts, US',
	            'Morris, New Jersey, US',
	            'Montgomery, Pennsylvania, US',
	            'Mercer, New Jersey, US',
	            'DuPage, Illinois, US',
	            'Riverside, California, US',
	            'Delaware, Pennsylvania, US',
	            'San Diego, California, US',
	            'Hampden, Massachusetts, US',
	            'Camden, New Jersey, US',
	            'Clark, Nevada, US',
	            'Erie, New York, US',
	            'Hennepin, Minnesota, US',
	            'Milwaukee, Wisconsin, US',
	            'Denver, Colorado, US',
	            'Baltimore, Maryland, US',
	            'Palm Beach, Florida, US',
	            'Franklin, Ohio, US',
	            'Bucks, Pennsylvania, US',
	            'St. Louis, Missouri, US',
	            'Will, Illinois, US',
	            'Tarrant, Texas, US',
	            'Somerset, New Jersey, US',
	            'Kane, Illinois, US',
	            'Orange, California, US',
	            'Burlington, New Jersey, US',
	            'Davidson, Tennessee, US',
	            'Salt Lake, Utah, US',
	            'Fulton, Georgia, US',
	            'Baltimore City, Maryland, US',
	            'Shelby, Tennessee, US',
	            'Berks, Pennsylvania, US',
	            'Arapahoe, Colorado, US',
	            'Sussex, Delaware, US',
	            'Dutchess, New York, US',
	            'Prince William, Virginia, US',
	            'Lehigh, Pennsylvania, US',
	            'San Bernardino, California, US',
	            'Cuyahoga, Ohio, US',
	            'Minnehaha, South Dakota, US',
	            'East Baton Rouge, Louisiana, US',
	            'Kent, Michigan, US',
	            'Polk, Iowa, US',
	            'Snohomish, Washington, US',
	            'Anne Arundel, Maryland, US',
	            'DeKalb, Georgia, US',
	            'Lake, Indiana, US',
	            'New Castle, Delaware, US',
	            'Northampton, Pennsylvania, US',
	            'Gwinnett, Georgia, US',
	            'Adams, Colorado, US',
	            'Luzerne, Pennsylvania, US',
	            'Bexar, Texas, US',
	            'Hillsborough, Florida, US',
	            'Orange, Florida, US',
	            'Kern, California, US',
	            'Hidalgo, Texas, US',
	            'Travis, Texas, US',
	            'Duval, Florida, US',
	            'Mecklenburg, North Carolina, US',
	            'Fresno, California, US',
	            'Pima, Arizona, US',
	            'Cameron, Texas, US',
	            'El Paso, Texas, US',
	            'Pinellas, Florida, US',
	            'Nueces, Texas, US',
	            'Lee, Florida, US',
	            'Alameda, California, US',
	            'Sacramento, California, US',
	            'San Joaquin, California, US',
	            'Santa Clara, California, US',
	            'Polk, Florida, US',
	            'Cobb, Georgia, US',
	            'Jefferson, Alabama, US',
	            'Wake, North Carolina, US',
	            'Fort Bend, Texas, US',
	            'Stanislaus, California, US',
	            'Tulare, California, US',
	            'Charleston, South Carolina, US',
	            'Contra Costa, California, US',
	            'Douglas, Nebraska, US',
	            'Oklahoma, Oklahoma, US',
	            'Tulsa, Oklahoma, US',
	            'Yuma, Arizona, US',
	            'Mobile, Alabama, US',
	            'Greenville, South Carolina, US',
	            'Jefferson, Kentucky, US']
	# counties.sort()
	ct = []
	for i in range(len(counties)):
		county = counties[i]
		county_str = county.split(',')
		county_str = [st.strip() for st in county_str]
		county_key = f'{state_abbr[county_str[1]]}-{county_str[0]}'
		ct.append(county_key)

	# ct.sort()
	for c in ct:
		print(f'\'{c}\',')
	return


def main():
	# Parse1P3A_state_confirmed()

	# Parse1P3A_county_confirmed()
	# Parse1P3A_county_death()

	# Parse1P3A_city_death()
	# Parse1P3A_city_confirmed()

	# ParseJHU_county_confirmed()
	# ParseJHU_county_death()
	tmp()
	# tmp2()

	# Parse_mobility()
	# Parse_county_mobility()
	return 0


if __name__ == "__main__":
	main()
