import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
import time
import sys
import csv
import os
import concurrent.futures
import csaps
import datetime
from sklearn.linear_model import LinearRegression
import numpy.polynomial.polynomial as poly

color_dict = {'AL': 'lime', 'AZ': 'lime', 'CA': 'orange', 'CO': 'red', 'CT': 'red', 'DC': 'orange', 'DE': 'red',
              'FL': 'lime', 'GA': 'lime', 'IA': 'lime', 'IL': 'red', 'IN': 'lime', 'KY': 'orange', 'LA': 'lime',
              'MA': 'red', 'MD': 'lime', 'MI': 'orange', 'MN': 'lime', 'MO': 'lime', 'NC': 'lime', 'NE': 'lime',
              'NJ': 'red', 'NV': 'orange', 'NY': 'red', 'OH': 'lime', 'OK': 'lime', 'PA': 'red', 'RI': 'red',
              'SC': 'lime', 'SD': 'lime', 'TN': 'orange', 'TX': 'lime', 'UT': 'lime', 'VA': 'red', 'WA': 'orange',
              'WI': 'orange'}

FIPS_dict = {'01073': 'AL-Jefferson',
             '01097': 'AL-Mobile',
             '04013': 'AZ-Maricopa',
             '04019': 'AZ-Pima',
             '04027': 'AZ-Yuma',
             '06001': 'CA-Alameda',
             '06013': 'CA-Contra Costa',
             '06019': 'CA-Fresno',
             '06029': 'CA-Kern',
             '06037': 'CA-Los Angeles',
             '06059': 'CA-Orange',
             '06065': 'CA-Riverside',
             '06067': 'CA-Sacramento',
             '06071': 'CA-San Bernardino',
             '06073': 'CA-San Diego',
             '06077': 'CA-San Joaquin',
             '06085': 'CA-Santa Clara',
             '06099': 'CA-Stanislaus',
             '06107': 'CA-Tulare',
             '08001': 'CO-Adams',
             '08005': 'CO-Arapahoe',
             '08031': 'CO-Denver',
             '09001': 'CT-Fairfield',
             '09003': 'CT-Hartford',
             '09009': 'CT-New Haven',
             '10003': 'DE-New Castle',
             '10005': 'DE-Sussex',
             '11001': 'DC-District of Columbia',
             '12011': 'FL-Broward',
             '12031': 'FL-Duval',
             '12057': 'FL-Hillsborough',
             '12071': 'FL-Lee',
             '12086': 'FL-Miami-Dade',
             '12095': 'FL-Orange',
             '12099': 'FL-Palm Beach',
             '12103': 'FL-Pinellas',
             '12105': 'FL-Polk',
             '13067': 'GA-Cobb',
             '13089': 'GA-DeKalb',
             '13121': 'GA-Fulton',
             '13135': 'GA-Gwinnett',
             '17031': 'IL-Cook',
             '17043': 'IL-DuPage',
             '17089': 'IL-Kane',
             '17097': 'IL-Lake',
             '17197': 'IL-Will',
             '18089': 'IN-Lake',
             '18097': 'IN-Marion',
             '19153': 'IA-Polk',
             '21111': 'KY-Jefferson',
             '22033': 'LA-East Baton Rouge',
             '22051': 'LA-Jefferson',
             '22071': 'LA-Orleans',
             '24003': 'MD-Anne Arundel',
             '24005': 'MD-Baltimore',
             '24510': 'MD-Baltimore City',
             '24031': 'MD-Montgomery',
             '24033': 'MD-Prince George\'s',
             '25005': 'MA-Bristol',
             '25009': 'MA-Essex',
             '25013': 'MA-Hampden',
             '25017': 'MA-Middlesex',
             '25021': 'MA-Norfolk',
             '25023': 'MA-Plymouth',
             '25025': 'MA-Suffolk',
             '25027': 'MA-Worcester',
             '26081': 'MI-Kent',
             '26099': 'MI-Macomb',
             '26125': 'MI-Oakland',
             '26163': 'MI-Wayne',
             '27053': 'MN-Hennepin',
             '29189': 'MO-St. Louis',
             '31055': 'NE-Douglas',
             '32003': 'NV-Clark',
             '34003': 'NJ-Bergen',
             '34005': 'NJ-Burlington',
             '34007': 'NJ-Camden',
             '34013': 'NJ-Essex',
             '34017': 'NJ-Hudson',
             '34021': 'NJ-Mercer',
             '34023': 'NJ-Middlesex',
             '34025': 'NJ-Monmouth',
             '34027': 'NJ-Morris',
             '34029': 'NJ-Ocean',
             '34031': 'NJ-Passaic',
             '34035': 'NJ-Somerset',
             '34039': 'NJ-Union',
             '36005': 'NY-Bronx',
             '36027': 'NY-Dutchess',
             '36029': 'NY-Erie',
             '36047': 'NY-Kings',
             '36059': 'NY-Nassau',
             '36061': 'NY-New York',
             '36071': 'NY-Orange',
             '36081': 'NY-Queens',
             '36085': 'NY-Richmond',
             '36087': 'NY-Rockland',
             '36103': 'NY-Suffolk',
             '36119': 'NY-Westchester',
             '37119': 'NC-Mecklenburg',
             '37183': 'NC-Wake',
             '39035': 'OH-Cuyahoga',
             '39049': 'OH-Franklin',
             '40109': 'OK-Oklahoma',
             '40143': 'OK-Tulsa',
             '42011': 'PA-Berks',
             '42017': 'PA-Bucks',
             '42045': 'PA-Delaware',
             '42077': 'PA-Lehigh',
             '42079': 'PA-Luzerne',
             '42091': 'PA-Montgomery',
             '42095': 'PA-Northampton',
             '42101': 'PA-Philadelphia',
             '44007': 'RI-Providence',
             '45019': 'SC-Charleston',
             '45045': 'SC-Greenville',
             '46099': 'SD-Minnehaha',
             '47037': 'TN-Davidson',
             '47157': 'TN-Shelby',
             '48029': 'TX-Bexar',
             '48061': 'TX-Cameron',
             '48113': 'TX-Dallas',
             '48141': 'TX-El Paso',
             '48157': 'TX-Fort Bend',
             '48201': 'TX-Harris',
             '48215': 'TX-Hidalgo',
             '48355': 'TX-Nueces',
             '48439': 'TX-Tarrant',
             '48453': 'TX-Travis',
             '49035': 'UT-Salt Lake',
             '51059': 'VA-Fairfax',
             '51153': 'VA-Prince William',
             '53033': 'WA-King',
             '53061': 'WA-Snohomish',
             '55079': 'WI-Milwaukee'
             }

NAICS_dict = {'7211': 'Traveler Accommodation',
              '7213': 'Rooming and Boarding Houses, Dormitories, and Workers\' Camps',
              '7224': 'Drinking Places (Alcoholic Beverages)',
              '7225': 'Restaurants and Other Eating Places',
              '7139': 'Other Amusement and Recreation Industries',
              '7131': 'Amusement Parks and Arcades',
              '8131': 'Religious Organizations',
              '8121': 'Personal Care Services'}

counties = ['AL-Jefferson',
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

# counties = ['NY-New York', 'CA-Los Angeles', 'FL-Miami-Dade', 'IL-Cook', 'TX-Dallas', 'TX-Harris', 'AZ-Maricopa',
#             'GA-Fulton', 'NJ-Bergen', 'PA-Philadelphia', 'MD-Prince George\'s', 'NV-Clark', 'NC-Mecklenburg',
#             'LA-Jefferson', 'CA-Riverside', 'FL-Broward', 'NJ-Hudson', 'MA-Middlesex', 'OH-Franklin', 'VA-Fairfax',
#             'TN-Shelby', 'WI-Milwaukee', 'UT-Salt Lake', 'MN-Hennepin']

# color_dict = {'AZ-Maricopa': 'lime', 'CA-Los Angeles': 'orange', 'FL-Miami-Dade': 'lime', 'GA-Fulton': 'lime',
#               'IL-Cook': 'crimson', 'LA-Jefferson': 'lime', 'MD-Prince George\'s': 'crimson', 'MN-Hennepin': 'lime',
#               'NV-Clark': 'lime', 'NJ-Bergen': 'crimson', 'NY-New York': 'crimson', 'NC-Mecklenburg': 'lime',
#               'PA-Philadelphia': 'orange', 'TX-Harris': 'lime', 'CA-Riverside': 'orange', 'FL-Broward': 'lime',
#               'TX-Dallas': 'lime', 'NJ-Hudson': 'crimson', 'MA-Middlesex': 'crimson', 'OH-Franklin': 'lime',
#               'VA-Fairfax': 'crimson', 'TN-Shelby': 'lime', 'WI-Milwaukee': 'orange', 'UT-Salt Lake': 'orange'}
color_dict = {'AL': 'lime', 'AZ': 'lime', 'CA': 'orange', 'CO': 'red', 'CT': 'red', 'DC': 'orange', 'DE': 'red',
              'FL': 'lime', 'GA': 'lime', 'IA': 'lime', 'IL': 'red', 'IN': 'lime', 'KY': 'orange', 'LA': 'lime',
              'MA': 'red', 'MD': 'lime', 'MI': 'orange', 'MN': 'lime', 'MO': 'lime', 'NC': 'lime', 'NE': 'lime',
              'NJ': 'red', 'NV': 'orange', 'NY': 'red', 'OH': 'lime', 'OK': 'lime', 'PA': 'red', 'RI': 'red',
              'SC': 'lime', 'SD': 'lime', 'TN': 'orange', 'TX': 'lime', 'UT': 'lime', 'VA': 'red', 'WA': 'orange',
              'WI': 'orange'}


def read_population(county):
	df = pd.read_csv('JHU/CountyPopulation.csv')
	n_0 = df[df.iloc[:, 0] == county].iloc[0]['POP']
	return n_0


def parse_135_counties(in_folder, out_folder, filename):
	if not os.path.exists(f'{in_folder}{out_folder}'):
		os.makedirs(f'{in_folder}{out_folder}')
	counts = {}
	for i in FIPS_dict.keys():
		counts[i] = 0

	cols = ['safegraph_place_id', 'date_range_start', 'date_range_end', 'raw_visit_counts', 'visits_by_day', 'poi_cbg']

	out_df = pd.DataFrame(columns=cols)
	out_df.to_csv(f'{in_folder}{out_folder}{filename}', index=False)

	table = []
	total = 0

	for chunk in pd.read_csv(f'{in_folder}{filename}', usecols=cols, dtype={'poi_cbg': 'string'}, chunksize=1000):

		for row in chunk.iterrows():
			st = row[1].poi_cbg
			try:
				st = st[:5]

			except:
				pass

			if st in FIPS_dict:
				counts[st] += 1
				table.append(row[1])
				total += 1
				if total % 1000 == 0:
					with open(f'{in_folder}{out_folder}{filename}', 'a', encoding='utf-8', newline='') as f:
						writer = csv.writer(f)
						writer.writerows(table)
					del table
					table = []

	with open(f'{in_folder}{out_folder}{filename}', 'a', encoding='utf-8', newline='') as f:
		writer = csv.writer(f)
		writer.writerows(table)

	print('total=', total)

	return


def tmp3():
	# with concurrent.futures.ProcessPoolExecutor() as executor:
	# 	results = [executor.submit(tmp4, i) for i in range(5, 0, -1)]
	#
	# # for f in results:
	# for f in concurrent.futures.as_completed(results):
	# 	print(f.result())
	print(matplotlib.colors.to_rgb('lime'))
	print(matplotlib.colors.to_rgb('crimson'))
	print(matplotlib.colors.to_rgb('orange'))

	return


def tmp4():
	print(len(counties))
	# print(len(FIPS_dict2))
	return


def parse_month(in_folder):
	t1 = time.perf_counter()
	print(f'starting {in_folder}')
	out_folder = 'small/'
	file_names = ['patterns-part1.csv', 'patterns-part2.csv', 'patterns-part3.csv', 'patterns-part4.csv']
	with concurrent.futures.ProcessPoolExecutor() as executor:
		[executor.submit(parse_135_counties, in_folder, out_folder, file_name) for file_name in file_names]

	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds in {in_folder}')

	return


# select all months' records of the given counties
def parse_all_months():
	in_folders = ['D:/GIT/Safe Graph/Monthly Places Patterns/01/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/02/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/03/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/04/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/05/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/06/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/07/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/08/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/09/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/10/']
	# with concurrent.futures.ProcessPoolExecutor() as executor:
	# 	[executor.submit(parse_month, in_folder) for in_folder in in_folders[:1]]
	for in_folder in in_folders:
		parse_month(in_folder)
	return


def parse_core(in_folder, out_folder, filename):
	if not os.path.exists(f'{in_folder}{out_folder}'):
		os.makedirs(f'{in_folder}{out_folder}')
	counts = {}
	for i in NAICS_dict.keys():
		counts[i] = 0

	cols = ['safegraph_place_id', 'naics_code']

	out_df = pd.DataFrame(columns=cols)
	out_df.to_csv(f'{in_folder}{out_folder}{filename}', index=False)

	table = []
	total = 0

	for chunk in pd.read_csv(f'{in_folder}{filename}', usecols=cols, dtype={'naics_code': 'string'}, chunksize=1000):

		for row in chunk.iterrows():
			st = row[1].naics_code
			try:
				st = st[:4]

			except:
				pass

			if st in NAICS_dict:
				counts[st] += 1
				table.append(row[1])
				total += 1
				if total % 1000 == 0:
					with open(f'{in_folder}{out_folder}{filename}', 'a', encoding='utf-8', newline='') as f:
						writer = csv.writer(f)
						writer.writerows(table)
					del table
					table = []

	with open(f'{in_folder}{out_folder}{filename}', 'a', encoding='utf-8', newline='') as f:
		writer = csv.writer(f)
		writer.writerows(table)

	print('total=', total)

	return


# select places in the given categories
def parse_all_cores():
	t1 = time.perf_counter()
	in_folder = 'D:/GIT/Safe Graph/core places/'
	out_folder = 'small/'
	file_names = ['core_poi-part1.csv', 'core_poi-part2.csv', 'core_poi-part3.csv', 'core_poi-part4.csv',
	              'core_poi-part5.csv']

	with concurrent.futures.ProcessPoolExecutor() as executor:
		[executor.submit(parse_core, in_folder, out_folder, file_name) for file_name in file_names]

	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds for all core poi')

	return


# combine 5 poi csv files into 1
def combine_cores():
	folder = 'D:/GIT/Safe Graph/core places/small/'
	file_names = ['core_poi-part1.csv', 'core_poi-part2.csv', 'core_poi-part3.csv', 'core_poi-part4.csv',
	              'core_poi-part5.csv']

	df = pd.read_csv(f'{folder}{file_names[0]}', dtype={'naics_code': 'string'})
	for i in range(1, 5):
		df2 = pd.read_csv(f'{folder}{file_names[i]}', dtype={'naics_code': 'string'})
		df = df.append(df2, ignore_index=True)

	df.to_csv(f'{folder}combined_core_poi.csv', index=False)
	print(len(df))
	return


def combine_month(folder):
	file_names = ['patterns-part1.csv', 'patterns-part2.csv', 'patterns-part3.csv', 'patterns-part4.csv']

	df = pd.read_csv(f'{folder}{file_names[0]}', dtype={'poi_cbg': 'string'})
	for i in range(1, 4):
		df2 = pd.read_csv(f'{folder}{file_names[i]}', dtype={'poi_cbg': 'string'})
		df = df.append(df2, ignore_index=True)
	df.to_csv(f'{folder}combined_patterns.csv', index=False)
	return


# combine all months' 4 csv files into 1
def combine_all_months():
	t1 = time.perf_counter()
	folders = ['D:/GIT/Safe Graph/Monthly Places Patterns/01/small/',
	           'D:/GIT/Safe Graph/Monthly Places Patterns/02/small/',
	           'D:/GIT/Safe Graph/Monthly Places Patterns/03/small/',
	           'D:/GIT/Safe Graph/Monthly Places Patterns/04/small/',
	           'D:/GIT/Safe Graph/Monthly Places Patterns/05/small/',
	           'D:/GIT/Safe Graph/Monthly Places Patterns/06/small/',
	           'D:/GIT/Safe Graph/Monthly Places Patterns/07/small/',
	           'D:/GIT/Safe Graph/Monthly Places Patterns/08/small/',
	           'D:/GIT/Safe Graph/Monthly Places Patterns/09/small/',
	           'D:/GIT/Safe Graph/Monthly Places Patterns/10/small/']
	with concurrent.futures.ProcessPoolExecutor() as executor:
		[executor.submit(combine_month, folder) for folder in folders]

	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds combining all months')

	return


def monthly_NAICS_visits(in_folder, filename, s, sgid_to_naics):
	t1 = time.perf_counter()
	print('reading', f'{in_folder}{filename}')
	# pattern_df = pd.read_csv(f'{in_folder}{filename}', dtype={'poi_cbg': 'string'})

	visits = {}
	for F in FIPS_dict.keys():
		for N in NAICS_dict.keys():
			visits[(F, N)] = 0

	# len_df = len(pattern_df)
	days = 1
	for pattern_df in pd.read_csv(f'{in_folder}{filename}', dtype={'poi_cbg': 'string'}, chunksize=1000):
		for row in pattern_df.iterrows():
			s_id = row[1]['safegraph_place_id']
			if row[1]['safegraph_place_id'] in s:
				F = row[1]['poi_cbg'][:5]
				N = sgid_to_naics[s_id]
				V = row[1]['visits_by_day'][1: -1]
				V = V.split(',')
				V = [int(v) for v in V]
				if visits[(F, N)] == 0:
					visits[(F, N)] = V
					days = len(V)
				else:
					visits[(F, N)] = [visits[(F, N)][i] + V[i] for i in range(len(V))]
	# if row[0] % (len_df // 100) == 0:
	# 	print(f'{round(row[0] * 100 / len_df)}% complete')

	for F in FIPS_dict.keys():
		for N in NAICS_dict.keys():
			if visits[F, N] == 0:
				visits[F, N] = [0] * days
	table = []
	cols = ['county', 'NAICS']
	cols.extend(range(1, days + 1))
	for k in visits.keys():
		table.append([FIPS_dict[k[0]]])
		table[-1].append(k[1])
		table[-1].extend(visits[k])
	out_df = pd.DataFrame(table, columns=cols)
	out_df.to_csv(f'{in_folder}visits.csv', index=False)
	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds for {in_folder}')
	return


# count every month's visit numbers by county
def all_months_NAICS_visits():
	t1 = time.perf_counter()
	print('start building poi dictionary')
	poi_df = pd.read_csv('D:/GIT/Safe Graph/core places/small/combined_core_poi.csv', dtype={'naics_code': 'string'})
	s = set(poi_df['safegraph_place_id'])
	sgid_to_naics = {}
	for row in poi_df.iterrows():
		sgid_to_naics[row[1]['safegraph_place_id']] = row[1]['naics_code'][:4]

	t2 = time.perf_counter()
	print(f'{round(t2 - t1, 3)} seconds for poi dictionary')

	in_folders = ['D:/GIT/Safe Graph/Monthly Places Patterns/01/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/02/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/03/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/04/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/05/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/06/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/07/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/08/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/09/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/10/small/']
	filename = 'combined_patterns.csv'

	with concurrent.futures.ProcessPoolExecutor() as executor:
		[executor.submit(monthly_NAICS_visits, in_folder, filename, s, sgid_to_naics) for in_folder in in_folders]

	# for in_folder in in_folders:
	# 	monthly_NAICS_visits(in_folder, filename, s, sgid_to_naics)

	return


def all_months_total_visits():
	in_folders = ['D:/GIT/Safe Graph/Monthly Places Patterns/01/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/02/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/03/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/04/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/05/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/06/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/07/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/08/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/09/small/',
	              'D:/GIT/Safe Graph/Monthly Places Patterns/10/small/']
	filename = 'combined_patterns.csv'

	with concurrent.futures.ProcessPoolExecutor() as executor:
		results = [executor.submit(monthly_total_visits, in_folder, filename) for in_folder in in_folders]

	cols = ['county']
	visits = {}
	for county in counties:
		visits[county] = []
	days = 0
	for month in results:
		monthly_visits = month.result()
		for county in counties:
			county_visits = monthly_visits[county]
			print(county, county_visits)
			month_len = len(county_visits)
			visits[county].extend(county_visits)

		days += month_len
	start_dt = datetime.datetime(2020, 1, 1)
	dts = [start_dt + datetime.timedelta(i) for i in range(days)]
	dts = [dt.strftime('%Y-%m-%d') for dt in dts]
	cols.extend(dts)

	table = []
	for county in counties:
		row = [county]
		row.extend(visits[county])
		table.append(row)

	out_df = pd.DataFrame(table, columns=cols)
	out_df.to_csv('Safe Graph/total visits.csv', index=False)
	return


def monthly_total_visits(in_folder, filename):
	pattern_df = pd.read_csv(f'{in_folder}{filename}', dtype={'poi_cbg': 'string'})

	visits = {}
	for county in counties:
		visits[county] = 0

	days = 1
	for row in pattern_df.iterrows():
		F = row[1]['poi_cbg'][:5]
		county = FIPS_dict[F]
		# if county[:3] == 'NY-':
		# 	county = 'NY-New York'
		V = row[1]['visits_by_day'][1: -1]
		V = V.split(',')
		V = [int(v) for v in V]
		if visits[county] == 0:
			visits[county] = V
		else:
			visits[county] = [visits[county][i] + V[i] for i in range(len(V))]

	return visits


def display_visits():
	NAICSs = ['7211', '7224', '7225', '7139', '7131', '8131', '8121']
	# counties = list(color_dict.keys())
	r_num = 4
	c_num = 2
	fig, axes = plt.subplots(r_num, c_num)
	fig.set_figheight(20)
	fig.set_figwidth(10)

	sum_visits = [0] * len(counties)
	# print(sum_visits)

	start_dt = datetime.datetime(2020, 1, 1)
	dts = [start_dt + datetime.timedelta(i) for i in range(365)]
	restaurant_table = []
	bar_table = []
	for i in range(len(NAICSs)):
		ax = axes[i // c_num][i % c_num]
		NAICS = NAICSs[i]
		print(NAICS)
		ax.set_title(NAICS_dict[NAICS])
		for j in range(len(counties)):
			county = counties[j]
			n_0 = read_population(county)
			# df = pd.read_csv('JHU/CountyPopulation.csv')
			# n_0 = df[df.iloc[:, 0] == county].iloc[0]['POP']

			clr = color_dict[county[:2]]

			files = ['D:/GIT/Safe Graph/Monthly Places Patterns/01/small/visits.csv',
			         'D:/GIT/Safe Graph/Monthly Places Patterns/02/small/visits.csv',
			         'D:/GIT/Safe Graph/Monthly Places Patterns/03/small/visits.csv',
			         'D:/GIT/Safe Graph/Monthly Places Patterns/04/small/visits.csv',
			         'D:/GIT/Safe Graph/Monthly Places Patterns/05/small/visits.csv',
			         'D:/GIT/Safe Graph/Monthly Places Patterns/06/small/visits.csv',
			         'D:/GIT/Safe Graph/Monthly Places Patterns/07/small/visits.csv',
			         'D:/GIT/Safe Graph/Monthly Places Patterns/08/small/visits.csv',
			         'D:/GIT/Safe Graph/Monthly Places Patterns/09/small/visits.csv',
			         'D:/GIT/Safe Graph/Monthly Places Patterns/10/small/visits.csv']
			visits = []
			for file in files:
				df = pd.read_csv(file, dtype={'NAICS': 'string'})
				row = df[(df['county'] == county) & (df['NAICS'] == NAICS)].iloc[0, 2:]
				visits.extend(row)

			MA_visits = [v / n_0 * 1000 for v in visits]
			MA_visits = pd.Series(MA_visits).rolling(window=7).mean()
			if county != 'FL-Orange' or NAICS != '7131':
				ax.plot(dts[:len(MA_visits)], MA_visits, color=clr, alpha=0.5, linewidth=0.5)
			if NAICS == '7225':
				cols = dts[:len(MA_visits)]
				entry = [county]
				entry.extend(MA_visits)
				restaurant_table.append(entry)

			if NAICS == '7224':
				cols = dts[:len(MA_visits)]
				entry = [county]
				entry.extend(MA_visits)
				bar_table.append(entry)
			if i == 0:
				sum_visits[j] = MA_visits
			else:
				sum_visits[j] = [sum_visits[j][i] + MA_visits[i] for i in range(len(MA_visits))]
	cols.insert(0, 'county')
	restaurant_df = pd.DataFrame(restaurant_table, columns=cols)
	restaurant_df.to_csv('Safe Graph/restaurant.csv', index=False)

	bar_df = pd.DataFrame(bar_table, columns=cols)
	bar_df.to_csv('Safe Graph/bar.csv', index=False)

	ax = axes[r_num - 1][c_num - 1]
	ax.set_title('Summation of the 7 Categories')
	cols = ['county']
	cols.extend(list(range(1, len(sum_visits[0]) + 1)))
	table = []
	for i in range(len(counties)):
		row = [counties[i]]
		row.extend(sum_visits[i])
		table.append(row)
		clr = color_dict[counties[i][:2]]
		ax.plot(dts[:len(sum_visits[i])], sum_visits[i], color=clr, alpha=0.5, linewidth=0.5)

	fig.autofmt_xdate()

	fig.add_subplot(111, frameon=False)
	plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	plt.ylabel("Visits per 1000 People")

	fig.savefig('Safe Graph/CBG_grid.png', bbox_inches="tight")
	out_df = pd.DataFrame(table, columns=cols)
	out_df.to_csv('Safe Graph/total NAICS visits.csv', index=False)
	# plt.show()
	return


def combine_NY():
	months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
	in_files = [f'D:/GIT/Safe Graph/Monthly Places Patterns/{month}/small/visits.csv' for month in months]
	NYs = ['NY-New York', 'NY-Richmond', 'NY-Bronx', 'NY-Kings', 'NY-Queens']

	NAICSs = ['7211', '7224', '7225', '7139', '7131', '8131', '8121']

	for month in months:
		if not os.path.exists(f'Safe Graph/{month}/'):
			os.makedirs(f'Safe Graph/{month}/')
		in_file = f'D:/GIT/Safe Graph/Monthly Places Patterns/{month}/small/visits.csv'
		df = pd.read_csv(in_file, dtype={'NAICS': 'string'})
		cols = df.columns
		table = []
		for NAICS in NAICSs:
			NY_rows = df[(df['county'].isin(NYs)) & (df['NAICS'] == NAICS)].iloc[:, 2:]
			# print(NY_rows, len(NY_rows))
			row = ['NY-New York', NAICS]
			for i in range(len(cols) - 2):
				sum_visits = 0
				for j in range(len(NY_rows)):
					sum_visits += NY_rows.iloc[j, i]
				row.append(sum_visits)
			table.append(row)

		for county in counties:
			if county[:3] == 'NY-':
				continue
			for NAICS in NAICSs:
				row = df[(df['county'] == county) & (df['NAICS'] == NAICS)].iloc[0]
				table.append(row)

		out_df = pd.DataFrame(table, columns=cols)
		out_df.to_csv(f'Safe Graph/{month}/visits.csv', index=False)

	return


def compare_with_release():
	fig = plt.figure()
	ax = fig.add_subplot()
	for county in counties:
		n_0 = read_population(county)
		visits_df = pd.read_csv('Safe Graph/total visits.csv')
		row = visits_df[visits_df['county'] == county].iloc[0]
		para_df = pd.read_csv(f'ASYM/combined2W_2020-08-31/{county}/para.csv')
		eta = para_df.iloc[0]['eta']
		H_init = para_df.iloc[0]['Hiding_init']
		h = para_df.iloc[0]['h']
		daily_release = 1000 * eta * H_init * h
		v_150 = row['150']
		v_100 = row['100']
		ax.scatter(v_150 - v_100, daily_release, color=color_dict[county])
	ax.set_xlabel('increase in visits per thousand (day 150 - day 100)')
	ax.set_ylabel('daily release per thousand')
	plt.show()
	return


def compare_NAICS_and_total_visits():
	NAICS_file = 'Safe Graph/total NAICS visits.csv'
	total_file = 'Safe Graph/total visits.csv'
	df_NAICS = pd.read_csv(NAICS_file)
	df_total = pd.read_csv(total_file)
	fig = plt.figure()
	axes = fig.subplots(4, 6)
	days = range(len(df_NAICS.columns) - 1)
	for i in range(len(counties)):
		# fig, ax = plt.subplots()
		county = counties[i]
		n_0 = read_population(county)
		ax = axes[i // 6][i % 6]
		ax2 = ax.twinx()
		row_NAICS = df_NAICS[df_NAICS['county'] == county].iloc[0][1:]
		row_total = df_total[df_total['county'] == county].iloc[0][1:].rolling(window=7).mean()
		row_total = [v * 1000 / n_0 for v in row_total]
		# print(type(list(days)), type(row_total))
		xs = np.arange(days[0], days[-1] + 1, 1)
		smooth_total = csaps.csaps(days[7:], row_total[7:], xs, smooth=0.001)
		# print(smooth_total)
		# ax.plot(days, row_NAICS, color='red', label='NAICS')

		ax2.plot(days, row_total, color='green', label='total')
		ax2.plot(xs, smooth_total, color='red', label='smooth')
		# ax2.plot(days, row_total, color='blue', label='total')
		lines, labels = ax.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax2.legend(lines + lines2, labels + labels2, loc=0)

		ax.set_title(county)
	# plt.show()
	# plt.close(fig)
	# fig.tight_layout(pad=6.0)

	plt.subplots_adjust(hspace=0.3, wspace=0.3)
	plt.show()

	return


def slope_computer():
	in_file = 'Safe Graph/total NAICS visits.csv'
	# in_file = 'Safe Graph/restaurant.csv'
	# in_file = 'Safe Graph/bar.csv'
	df = pd.read_csv(in_file)
	table = []
	start_dt = datetime.datetime(2020, 1, 1)
	dts = [start_dt + datetime.timedelta(i) for i in range(365)]
	start_dt = datetime.datetime(2020, 4, 1)
	end_dt = datetime.datetime(2020, 7, 1)
	start_index = dts.index(start_dt)
	end_index = dts.index(end_dt)
	x = np.array(range(start_index, end_index + 1))
	x = x.reshape(-1, 1)
	# print(start_index, end_index)
	for county in counties:
		row = list(df[df['county'] == county].iloc[0, start_index:end_index + 1])
		regressor = LinearRegression()
		regressor.fit(x, row)
		table.append([county, regressor.coef_[0]])
	# fig, ax = plt.subplots()
	# ax.scatter(x, row)
	# ax.plot(x, linear(x, regressor.coef_[0], regressor.intercept_))
	# plt.show()
	# plt.close(fig)

	out_df = pd.DataFrame(table, columns=['county', 'slope'])
	out_df.to_csv('Safe Graph/slope.csv', index=False)
	return


def linear(x, coef, intercept):
	y = [i * coef + intercept for i in x]
	return y


def slope_VS_release():
	fig, ax = plt.subplots()
	slope_df = pd.read_csv('Safe Graph/slope.csv')
	table = []
	cols = ['county', 'daily release', 'daily release per 1000', 'slope']
	for county in counties:
		n_0 = read_population(county)
		para_df = pd.read_csv(f'JHU50/combined2W_2020-08-31/{county}/para.csv')
		row = para_df.iloc[0]
		# print(row)
		h = row['h']
		eta = row['eta']
		Hiding_init = row['Hiding_init']
		slope = slope_df[slope_df['county'] == county].iloc[0]['slope']
		# print(slope)
		# print(type(slope))
		table.append([county, n_0 * eta * Hiding_init * h, (Hiding_init * h * 1000), slope])
		ax.scatter(table[-1][3], table[-1][2], color=color_dict[county[:2]])
	# print(table[-1])
	out_df = pd.DataFrame(table, columns=cols)
	out_df.to_csv('Safe Graph/slope release.csv', index=False)

	# ax.scatter([table[i][2] for i in range(len(counties))], [table[i][3] for i in range(len(counties))])
	ax.set_ylabel('Release Rate per 1000 People')
	ax.set_xlabel('Visit Increase Rate per 1000 People')

	X = [table[i][3] for i in range(len(table))]
	Y = [table[i][2] for i in range(len(table))]
	rho = np.corrcoef(X, Y)
	C = poly.polyfit(X, Y, 1)
	X2 = np.arange(min(X), max(X), (max(X) - min(X)) / 100)
	Y2 = poly.polyval(X2, C)
	ax.plot(X2, Y2, color='grey', label=f'\u03C1={round(rho[0][1], 4)}')
	ax.legend()
	plt.show()
	fig.savefig('Safe Graph/slope.png', bbox_inches="tight")


def slope_VS_fitting():
	fig, ax = plt.subplots()
	ax.set_ylabel('Susceptible size per 1000')
	ax.set_xlabel('Log of Visit increase per 1000')
	slope_df = pd.read_csv('Safe Graph/slope.csv')
	slopes = []
	H = []
	for county in counties:
		n_0 = read_population(county)
		para_df = pd.read_csv(f'JHU50/combined2W_2020-08-31/{county}/para.csv')
		row = para_df.iloc[0]
		eta = row['eta']
		Hiding_init = row['Hiding_init']
		slope = slope_df[slope_df['county'] == county].iloc[0]['slope']
		slopes.append(math.log(slope))
		H.append((eta + eta * Hiding_init) * 1000)
		ax.scatter(slopes[-1], H[-1], c=color_dict[county[:2]])
	m, b = np.polyfit(slopes, H, 1)
	ax.plot(np.array(slopes), m * np.array(slopes) + b, label=f'corr={round(np.corrcoef(slopes, H)[0][1], 3)}')
	ax.legend()
	plt.show()
	return


def NAICS_each_categories():
	if not os.path.exists(f'Safe Graph/NAICS/'):
		os.makedirs(f'Safe Graph/NAICS/')
	NAICSs = ['7211', '7224', '7225', '7139', '7131', '8131', '8121']
	files = ['D:/GIT/Safe Graph/Monthly Places Patterns/01/small/visits.csv',
	         'D:/GIT/Safe Graph/Monthly Places Patterns/02/small/visits.csv',
	         'D:/GIT/Safe Graph/Monthly Places Patterns/03/small/visits.csv',
	         'D:/GIT/Safe Graph/Monthly Places Patterns/04/small/visits.csv',
	         'D:/GIT/Safe Graph/Monthly Places Patterns/05/small/visits.csv',
	         'D:/GIT/Safe Graph/Monthly Places Patterns/06/small/visits.csv',
	         'D:/GIT/Safe Graph/Monthly Places Patterns/07/small/visits.csv',
	         'D:/GIT/Safe Graph/Monthly Places Patterns/08/small/visits.csv',
	         'D:/GIT/Safe Graph/Monthly Places Patterns/09/small/visits.csv',
	         'D:/GIT/Safe Graph/Monthly Places Patterns/10/small/visits.csv']
	start_dt = datetime.datetime(2020, 1, 1)
	dts = [start_dt + datetime.timedelta(i) for i in range(365)]

	for NAICS in NAICSs:
		print('Saving', NAICS_dict[NAICS])
		table = []
		for county in counties:
			n_0 = read_population(county)
			visits = []
			for f in files:
				df = pd.read_csv(f, dtype={'NAICS': 'string'})
				row = df[(df['NAICS'] == NAICS) & (df['county'] == county)].iloc[0, 2:]
				visits.extend(row)
			MA_visits = [v / n_0 * 1000 for v in visits]
			MA_visits = pd.Series(MA_visits).rolling(window=7).mean()
			MA_visits = list(MA_visits)
			MA_visits.insert(0, county)
			table.append(MA_visits)
		cols = ['county']
		cols.extend([dt.strftime('%Y-%m-%d') for dt in dts[:len(table[0]) - 1]])
		out_df = pd.DataFrame(table, columns=cols)
		out_df.to_csv(f'Safe Graph/NAICS/{NAICS_dict[NAICS]} visits.csv', index=False)

	return


def main():
	# parse_all_months()
	# combine_all_months()

	# parse_all_cores()
	# combine_cores()

	# all_months_NAICS_visits()
	# all_months_total_visits()
	NAICS_each_categories()

	# combine_NY()
	# display_visits()

	# compare_with_release()
	# compare_NAICS_and_total_visits()
	# tmp3()
	# slope_computer()
	# slope_VS_release()

	# slope_VS_fitting()
	# tmp4()
	return


if __name__ == "__main__":
	main()
