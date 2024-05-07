import warnings

import pandas as pd
import numpy as np
import json
import requests
import datetime
import matplotlib.pyplot as plt
import os
import scipy
from scipy.optimize import minimize
from sklearn import linear_model

states = ['ap', 'ar', 'as', 'br', 'ct', 'dl', 'ga', 'gj', 'hr', 'hp', 'jh', 'ka', 'kl', 'mp', 'mh', 'mn', 'ml', 'nl',
		  'or', 'pb', 'rj', 'tn', 'tg', 'tr', 'up', 'ut', 'wb']

state_dict = {'ap': 'Andhra Pradesh',
			  'ar': 'Arunachal Pradesh',
			  'as': 'Assam',
			  'br': 'Bihar',
			  'ct': 'Chhattisgarh',
			  'dl': 'Delhi',
			  'ga': 'Goa',
			  'gj': 'Gujarat',
			  'hr': 'Haryana',
			  'hp': 'Himachal Pradesh',
			  'jh': 'Jharkhand',
			  'ka': 'Karnataka',
			  'kl': 'Kerala',
			  'mp': 'Madhya Pradesh',
			  'mh': 'Maharashtra',
			  'mn': 'Manipur',
			  'ml': 'Meghalaya',
			  'nl': 'Nagaland',
			  'or': 'Odisha',
			  'pb': 'Punjab',
			  'rj': 'Rajasthan',
			  'tn': 'Tamil Nadu',
			  'tg': 'Telangana',
			  'tr': 'Tripura',
			  'up': 'Uttar Pradesh',
			  'ut': 'Uttarakhand',
			  'wb': 'West Bengal'}
startWindow = 10


def linearRegressState(state):
	df = pd.read_csv(f'india/fitting_split2_2021-07-16_2021-03-15_2021-06-10/{state}/sim.csv')
	release = df[df['series'] == 'H'].iloc[0].iloc[1:]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/mask.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	mask = pd.Series(df['mean'])
	mask.index = [st[:10] for st in df['date_reported']]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/sd.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	sd = pd.Series(df['mean'])
	sd.index = [st[:10] for st in df['date_reported']]
	for i in sd.keys():
		sd[i] = - sd[i]
	best_rho = 0
	releaseStartDate = release.index[0]
	for lag in range(30):
		maskStartDate = mask.index[mask.index.get_loc(releaseStartDate) - lag]
		sdStartDate = sd.index[sd.index.get_loc(releaseStartDate) - lag]
		tmpEndDate = release.index[-1 - lag]
		if tmpEndDate > mask.index[-1]:
			tmpEndDate = mask.index[-1]
		if tmpEndDate > sd.index[-1]:
			tmpEndDate = sd.index[-1]
		releaseEndDate = release.index[release.index.get_loc(tmpEndDate) + lag]
		maskEndDate = tmpEndDate
		sdEndDate = tmpEndDate
		releaseData = release[releaseStartDate:releaseEndDate]
		maskData = mask[maskStartDate:maskEndDate]
		sdData = sd[sdStartDate:sdEndDate]

		maskData.name = 'mask'
		sdData.name = 'sd'

		X = pd.concat([maskData, sdData], axis=1)
		print(X)
		regr = linear_model.LinearRegression(positive=True)
		regr.fit(X, releaseData)
		Y = regr.predict(X)
		# print(Y)
		intercept = regr.intercept_
		coef = regr.coef_
		print(coef)
		rho = scipy.stats.pearsonr(releaseData, Y)[0]
		if rho > best_rho:
			bestLag = lag
			best_rho = rho
			bestRelease = releaseData
			bestMask = maskData
			bestSd = sdData
			bestX = X
			bestRegr = regr
			bestY = Y

	size = len(bestY)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	box = ax1.get_position()
	ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	ax2 = ax1.twinx()
	box = ax2.get_position()
	ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	lns1 = ax1.plot(range(size), bestRelease, label='release')
	coefficient = bestRegr.coef_
	lns2 = ax1.plot(range(size), bestY, label='regression')
	fig.suptitle(
		f'{state_dict[state]} lag={bestLag} days\ncoefficient={[round(i, 2) for i in coefficient]} cor={round(scipy.stats.pearsonr(bestRelease, bestY)[0], 4)}')
	lns3 = ax2.plot(range(size), bestMask, label='mask', color='green')
	lns4 = ax2.plot(range(size), bestSd, label='SD', color='red')
	lns = lns1 + lns2 + lns3 + lns4
	lbls = [l.get_label() for l in lns]
	ax1.legend(bbox_to_anchor=(0, -0.05), loc='upper left')
	ax2.legend(bbox_to_anchor=(1, -0.05), loc='upper right')
	plt.show()
	plt.close(fig)

	return


def linearRegressTimeWindowState(state):
	df = pd.read_csv(f'india/fitting_split2_2021-07-16_2021-03-15_2021-06-10/{state}/sim.csv')
	release = df[df['series'] == 'H'].iloc[0].iloc[1:]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/mask.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	mask = pd.Series(df['mean'])
	mask.index = [st[:10] for st in df['date_reported']]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/sd.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	sd = pd.Series(df['mean'])
	sd.index = [st[:10] for st in df['date_reported']]
	for i in sd.keys():
		sd[i] = - sd[i]
	best_rho = 0
	releaseStartDate = release.index[0]
	for lag in range(30):
		maskStartDate = mask.index[mask.index.get_loc(releaseStartDate) - lag]
		sdStartDate = sd.index[sd.index.get_loc(releaseStartDate) - lag]
		tmpEndDate = release.index[-1 - lag]
		if tmpEndDate > mask.index[-1]:
			tmpEndDate = mask.index[-1]
		if tmpEndDate > sd.index[-1]:
			tmpEndDate = sd.index[-1]
		releaseEndDate = release.index[release.index.get_loc(tmpEndDate) + lag]
		maskEndDate = tmpEndDate
		sdEndDate = tmpEndDate
		releaseData = release[releaseStartDate:releaseEndDate].copy()
		maskData = mask[maskStartDate:maskEndDate].copy()
		sdData = sd[sdStartDate:sdEndDate].copy()
		releaseStarted = False
		for day in range(len(releaseData)):
			if releaseStarted:
				if releaseData.iloc[day] <= releaseData.iloc[day + 1] + 1:
					releaseEndIndex = day
					break
			if (not releaseStarted) and releaseData.iloc[day] > releaseData.iloc[day + 1] + 1:
				# print(releaseData.iloc[day], releaseData.iloc[day + 1])
				releaseStartIndex = day
				releaseStarted = True
		releaseData = releaseData.iloc[releaseStartIndex - 15:releaseEndIndex]
		maskData = maskData.iloc[releaseStartIndex - 15:releaseEndIndex]
		sdData = sdData.iloc[releaseStartIndex - 15:releaseEndIndex]
		X = pd.concat([maskData, sdData], axis=1)
		# print(X)
		regr = linear_model.LinearRegression(positive=True)
		regr.fit(X, releaseData)
		Y = regr.predict(X)
		# print(Y)
		intercept = regr.intercept_
		coef = regr.coef_
		# print(releaseStartIndex, releaseEndIndex)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			rho = scipy.stats.pearsonr(releaseData, Y)[0]
		if rho > best_rho:
			bestLag = lag
			best_rho = rho
			bestRelease = releaseData
			bestMask = maskData
			bestSd = sdData
			bestX = X
			bestRegr = regr
			bestY = Y
	if best_rho == 0:
		print('best rho=0 for', state_dict[state])
		return
	size = len(bestY)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	box = ax1.get_position()
	ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	ax2 = ax1.twinx()
	box = ax2.get_position()
	ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	lns1 = ax1.plot(range(size), bestRelease, label='release')
	coefficient = bestRegr.coef_
	lns2 = ax1.plot(range(size), bestY, label='regression')
	fig.suptitle(
		f'{state_dict[state]} lag={bestLag} days\ncoefficient={[round(i, 2) for i in coefficient]} cor={round(scipy.stats.pearsonr(bestRelease, bestY)[0], 4)}')
	lns3 = ax2.plot(range(size), bestMask, label='mask', color='green')
	lns4 = ax2.plot(range(size), bestSd, label='SD', color='red')
	lns = lns1 + lns2 + lns3 + lns4
	lbls = [l.get_label() for l in lns]
	ax1.legend(bbox_to_anchor=(0, -0.05), loc='upper left')
	ax2.legend(bbox_to_anchor=(1, -0.05), loc='upper right')
	plt.show()
	plt.close(fig)

	return


def linearRegressDailyTimeWindowState(state):
	df = pd.read_csv(f'india/fitting_split2_2021-07-16_2021-03-15_2021-06-10/{state}/sim.csv')
	release = df[df['series'] == 'H'].iloc[0].iloc[1:]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/mask.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	mask = pd.Series(df['mean'])
	mask.index = [st[:10] for st in df['date_reported']]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/sd.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	sd = pd.Series(df['mean'])
	sd.index = [st[:10] for st in df['date_reported']]
	for i in sd.keys():
		sd[i] = - sd[i]
	best_rho = 0
	releaseStartDate = release.index[0]
	for lag in range(30):
		maskStartDate = mask.index[mask.index.get_loc(releaseStartDate) - lag]
		sdStartDate = sd.index[sd.index.get_loc(releaseStartDate) - lag]
		tmpEndDate = release.index[-1 - lag]
		if tmpEndDate > mask.index[-1]:
			tmpEndDate = mask.index[-1]
		if tmpEndDate > sd.index[-1]:
			tmpEndDate = sd.index[-1]
		releaseEndDate = release.index[release.index.get_loc(tmpEndDate) + lag]
		maskEndDate = tmpEndDate
		sdEndDate = tmpEndDate
		releaseData = release[releaseStartDate:releaseEndDate].copy()
		maskData = mask[maskStartDate:maskEndDate].copy()
		sdData = sd[sdStartDate:sdEndDate].copy()
		releaseStarted = False
		for day in range(len(releaseData)):
			if releaseStarted:
				if releaseData.iloc[day] <= releaseData.iloc[day + 1] + 1:
					releaseEndIndex = day
					break
			if (not releaseStarted) and releaseData.iloc[day] > releaseData.iloc[day + 1] + 1:
				# print(releaseData.iloc[day], releaseData.iloc[day + 1])
				releaseStartIndex = day
				releaseStarted = True
		releaseData = releaseData.iloc[releaseStartIndex - 15:releaseEndIndex]
		maskData = maskData.iloc[releaseStartIndex - 15:releaseEndIndex]
		sdData = sdData.iloc[releaseStartIndex - 15:releaseEndIndex]
		X = pd.concat([maskData, sdData], axis=1)
		# print(X)
		regr = linear_model.LinearRegression(positive=True)
		regr.fit(X, releaseData)
		Y = regr.predict(X)
		# print(Y)
		intercept = regr.intercept_
		coef = regr.coef_
		# print(releaseStartIndex, releaseEndIndex)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			rho = scipy.stats.pearsonr(releaseData, Y)[0]
		if rho > best_rho:
			bestLag = lag
			best_rho = rho
			bestRelease = releaseData
			bestMask = maskData
			bestSd = sdData
			bestX = X
			bestRegr = regr
			bestY = Y
	if best_rho == 0:
		print('best rho=0 for', state_dict[state])
		return
	size = len(bestY)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	box = ax1.get_position()
	ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	ax2 = ax1.twinx()
	box = ax2.get_position()
	ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	lns1 = ax1.plot(range(size), bestRelease, label='release')
	coefficient = bestRegr.coef_
	lns2 = ax1.plot(range(size), bestY, label='regression')
	fig.suptitle(
		f'{state_dict[state]} lag={bestLag} days\ncoefficient={[round(i, 2) for i in coefficient]} cor={round(scipy.stats.pearsonr(bestRelease, bestY)[0], 4)}')
	lns3 = ax2.plot(range(size), bestMask, label='mask', color='green')
	lns4 = ax2.plot(range(size), bestSd, label='SD', color='red')
	lns = lns1 + lns2 + lns3 + lns4
	lbls = [l.get_label() for l in lns]
	ax1.legend(bbox_to_anchor=(0, -0.05), loc='upper left')
	ax2.legend(bbox_to_anchor=(1, -0.05), loc='upper right')
	plt.show()
	plt.close(fig)

	return


def interactingLinearRegressTimeWindowState(state):
	df = pd.read_csv(f'india/fitting_split2_2021-07-16_2021-03-15_2021-06-10/{state}/sim.csv')
	release = df[df['series'] == 'H'].iloc[0].iloc[1:]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/mask.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	mask = pd.Series(df['mean'])
	mask.index = [st[:10] for st in df['date_reported']]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/sd.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	sd = pd.Series(df['mean'])
	sd.index = [st[:10] for st in df['date_reported']]
	for i in sd.keys():
		sd[i] = - sd[i]
	best_rho = 0
	releaseStartDate = release.index[0]
	for lag in range(45):
		maskStartDate = mask.index[mask.index.get_loc(releaseStartDate) - lag]
		sdStartDate = sd.index[sd.index.get_loc(releaseStartDate) - lag]
		tmpEndDate = release.index[-1 - lag]
		if tmpEndDate > mask.index[-1]:
			tmpEndDate = mask.index[-1]
		if tmpEndDate > sd.index[-1]:
			tmpEndDate = sd.index[-1]
		releaseEndDate = release.index[release.index.get_loc(tmpEndDate) + lag]
		maskEndDate = tmpEndDate
		sdEndDate = tmpEndDate
		releaseData = release[releaseStartDate:releaseEndDate].copy()
		maskData = mask[maskStartDate:maskEndDate].copy()
		sdData = sd[sdStartDate:sdEndDate].copy()
		releaseStarted = False
		for day in range(len(releaseData)):
			if releaseStarted:
				if releaseData.iloc[day] <= releaseData.iloc[day + 1] + 1:
					releaseEndIndex = day
					break
			if (not releaseStarted) and releaseData.iloc[day] > releaseData.iloc[day + 1] + 1:
				# print(releaseData.iloc[day], releaseData.iloc[day + 1])
				releaseStartIndex = day
				releaseStarted = True
		releaseData = releaseData.iloc[releaseStartIndex - 15:releaseEndIndex]
		maskData = maskData.iloc[releaseStartIndex - 15:releaseEndIndex]
		sdData = sdData.iloc[releaseStartIndex - 15:releaseEndIndex]
		interData = pd.Series([maskData.iloc[i] * sdData.iloc[i] for i in range(len(maskData))], maskData.keys())
		X = pd.concat([maskData, sdData, interData], axis=1)
		# print(X)
		regr = linear_model.LinearRegression(positive=True)
		regr.fit(X, releaseData)
		Y = regr.predict(X)
		# print(Y)
		intercept = regr.intercept_
		coef = regr.coef_
		# print(releaseStartIndex, releaseEndIndex)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			rho = scipy.stats.pearsonr(releaseData, Y)[0]
		if rho > best_rho:
			bestLag = lag
			best_rho = rho
			bestRelease = releaseData
			bestMask = maskData
			bestSd = sdData
			bestX = X
			bestRegr = regr
			bestY = Y
	if best_rho == 0:
		print('best rho=0 for', state_dict[state])
		return
	size = len(bestY)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	box = ax1.get_position()
	ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	ax2 = ax1.twinx()
	box = ax2.get_position()
	ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	lns1 = ax1.plot(range(size), bestRelease, label='release')
	coefficient = bestRegr.coef_
	lns2 = ax1.plot(range(size), bestY, label='regression')
	fig.suptitle(
		f'{state_dict[state]} lag={bestLag} days\ncoefficient={[round(i, 2) for i in coefficient]} cor={round(scipy.stats.pearsonr(bestRelease, bestY)[0], 4)}')
	lns3 = ax2.plot(range(size), bestMask, label='mask', color='green')
	lns4 = ax2.plot(range(size), bestSd, label='SD', color='red')
	lns = lns1 + lns2 + lns3 + lns4
	lbls = [l.get_label() for l in lns]
	ax1.legend(bbox_to_anchor=(0, -0.05), loc='upper left')
	ax2.legend(bbox_to_anchor=(1, -0.05), loc='upper right')
	plt.show()
	plt.close(fig)

	return


def interactingLogLinearRegressTimeWindowState(state):
	df = pd.read_csv(f'india/fitting_split2_2021-07-16_2021-03-15_2021-06-10/{state}/sim.csv')
	release = df[df['series'] == 'H'].iloc[0].iloc[1:]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/mask.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	mask = pd.Series(df['mean'])
	mask.index = [st[:10] for st in df['date_reported']]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/sd.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	sd = pd.Series(df['mean'])
	sd.index = [st[:10] for st in df['date_reported']]
	for i in sd.keys():
		sd[i] = - sd[i]
	best_rho = 0
	releaseStartDate = release.index[0]
	for lag in range(45):
		maskStartDate = mask.index[mask.index.get_loc(releaseStartDate) - lag]
		sdStartDate = sd.index[sd.index.get_loc(releaseStartDate) - lag]
		tmpEndDate = release.index[-1 - lag]
		if tmpEndDate > mask.index[-1]:
			tmpEndDate = mask.index[-1]
		if tmpEndDate > sd.index[-1]:
			tmpEndDate = sd.index[-1]
		releaseEndDate = release.index[release.index.get_loc(tmpEndDate) + lag]
		maskEndDate = tmpEndDate
		sdEndDate = tmpEndDate
		releaseData = release[releaseStartDate:releaseEndDate].copy()
		maskData = mask[maskStartDate:maskEndDate].copy()
		sdData = sd[sdStartDate:sdEndDate].copy()
		releaseStarted = False
		for day in range(len(releaseData)):
			if releaseStarted:
				if releaseData.iloc[day] <= releaseData.iloc[day + 1] + 1:
					releaseEndIndex = day
					break
			if (not releaseStarted) and releaseData.iloc[day] > releaseData.iloc[day + 1] + 1:
				# print(releaseData.iloc[day], releaseData.iloc[day + 1])
				releaseStartIndex = day
				releaseStarted = True
		releaseData = releaseData.iloc[releaseStartIndex - 15:releaseEndIndex]
		maskData = maskData.iloc[releaseStartIndex - 15:releaseEndIndex]
		sdData = sdData.iloc[releaseStartIndex - 15:releaseEndIndex]
		interData = pd.Series([maskData.iloc[i] * sdData.iloc[i] for i in range(len(maskData))], maskData.keys())
		for i in releaseData.keys():
			releaseData[i] = np.log(releaseData[i])
		for i in maskData.keys():
			maskData[i] = np.log(maskData[i])
		for i in sdData.keys():
			sdData[i] = np.log(sdData[i])
		for i in interData.keys():
			interData[i] = np.log(interData[i])

		X = pd.concat([maskData, sdData, interData], axis=1)
		# print(X)
		regr = linear_model.LinearRegression()
		regr.fit(X, releaseData)
		Y = regr.predict(X)
		# print(Y)
		intercept = regr.intercept_
		coef = regr.coef_
		# print(releaseStartIndex, releaseEndIndex)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			rho = scipy.stats.pearsonr(releaseData, Y)[0]
		if rho > best_rho:
			bestLag = lag
			best_rho = rho
			bestRelease = releaseData
			bestMask = maskData
			bestSd = sdData
			bestX = X
			bestRegr = regr
			bestY = Y
	if best_rho == 0:
		print('best rho=0 for', state_dict[state])
		return
	size = len(bestY)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	box = ax1.get_position()
	ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	ax2 = ax1.twinx()
	box = ax2.get_position()
	ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	lns1 = ax1.plot(range(size), bestRelease, label='release')
	coefficient = bestRegr.coef_
	lns2 = ax1.plot(range(size), bestY, label='regression')
	fig.suptitle(
		f'{state_dict[state]} lag={bestLag} days\ncoefficient={[round(i, 2) for i in coefficient]} cor={round(scipy.stats.pearsonr(bestRelease, bestY)[0], 4)}')
	lns3 = ax2.plot(range(size), bestMask, label='mask', color='green')
	lns4 = ax2.plot(range(size), bestSd, label='SD', color='red')
	lns = lns1 + lns2 + lns3 + lns4
	lbls = [l.get_label() for l in lns]
	ax1.legend(bbox_to_anchor=(0, -0.05), loc='upper left')
	ax2.legend(bbox_to_anchor=(1, -0.05), loc='upper right')
	plt.show()
	plt.close(fig)

	return


def logLinearRegressTimeWindowState(state):
	print(state_dict[state])
	df = pd.read_csv(f'india/fitting_split2_2021-07-16_2021-03-15_2021-06-10/{state}/sim.csv')
	release = df[df['series'] == 'H'].iloc[0].iloc[1:]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/mask.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	mask = pd.Series(df['mean'])
	mask.index = [st[:10] for st in df['date_reported']]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/sd.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	sd = pd.Series(df['mean'])
	sd.index = [st[:10] for st in df['date_reported']]

	for i in sd.keys():
		sd[i] = sd[i] + 100

	best_rho = 0
	releaseStartDate = release.index[0]
	for lag in range(50):
		maskStartDate = mask.index[mask.index.get_loc(releaseStartDate) - lag]
		sdStartDate = sd.index[sd.index.get_loc(releaseStartDate) - lag]
		tmpEndDate = release.index[-1 - lag]
		if tmpEndDate > mask.index[-1]:
			tmpEndDate = mask.index[-1]
		if tmpEndDate > sd.index[-1]:
			tmpEndDate = sd.index[-1]
		releaseEndDate = release.index[release.index.get_loc(tmpEndDate) + lag]
		maskEndDate = tmpEndDate
		sdEndDate = tmpEndDate
		releaseData = release[releaseStartDate:releaseEndDate].copy()
		maskData = mask[maskStartDate:maskEndDate].copy()
		sdData = sd[sdStartDate:sdEndDate].copy()
		releaseStarted = False
		for day in range(len(releaseData)):
			if releaseStarted:
				if releaseData.iloc[day] <= releaseData.iloc[day + 1] + 1:
					releaseEndIndex = day
					break
			if (not releaseStarted) and releaseData.iloc[day] > releaseData.iloc[day + 1] + 1:
				# print(releaseData.iloc[day], releaseData.iloc[day + 1])
				releaseStartIndex = day
				releaseStarted = True
		releaseData = releaseData.iloc[releaseStartIndex - startWindow:releaseEndIndex]
		maskData = maskData.iloc[releaseStartIndex - startWindow:releaseEndIndex]
		sdData = sdData.iloc[releaseStartIndex - startWindow:releaseEndIndex]

		for i in releaseData.keys():
			releaseData[i] = np.log(releaseData[i])
		for i in maskData.keys():
			maskData[i] = np.log(maskData[i])
		for i in sdData.keys():
			sdData[i] = np.log(sdData[i])
		# print(sdData)
		X = pd.concat([maskData, sdData], axis=1)
		# print(lag)
		# print(X)
		regr = linear_model.LinearRegression(positive=True)
		regr.fit(X, releaseData)
		Y = regr.predict(X)
		# print(Y)
		# intercept = regr.intercept_
		# coef = regr.coef_
		# print(releaseStartIndex, releaseEndIndex)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			rho = scipy.stats.pearsonr(releaseData, Y)[0]
		if rho > best_rho:
			bestLag = lag
			best_rho = rho
			bestRelease = releaseData
			bestMask = maskData
			bestSd = sdData
			bestX = X
			bestRegr = regr
			bestY = Y
	if best_rho == 0:
		print('best rho=0 for', state_dict[state])
		return [[0, 0], 0, 0, 0]

	coefficient = bestRegr.coef_
	intercept = bestRegr.intercept_
	# print(intercept)

	# plot of the time window

	# size = len(bestY)
	# fig = plt.figure()
	# ax1 = fig.add_subplot()
	# box = ax1.get_position()
	# ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	# ax2 = ax1.twinx()
	# box = ax2.get_position()
	# ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	# lns1 = ax1.plot(range(size), bestRelease, label='release')
	# lns2 = ax1.plot(range(size), bestY, label='regression')
	# fig.suptitle(
	# 	f'{state_dict[state]} lag={bestLag} days\ncoefficient={[round(i, 2) for i in coefficient]} cor={round(scipy.stats.pearsonr(bestRelease, bestY)[0], 4)}')
	# lns3 = ax2.plot(range(size), bestMask, label='mask', color='green')
	# lns4 = ax2.plot(range(size), bestSd, label='SD', color='red')
	# lns = lns1 + lns2 + lns3 + lns4
	# lbls = [l.get_label() for l in lns]
	# ax1.legend(bbox_to_anchor=(0, -0.05), loc='upper left')
	# ax2.legend(bbox_to_anchor=(1, -0.05), loc='upper right')
	# plt.show()
	# plt.close(fig)

	# plot of entire time
	fig = plt.figure()
	ax1 = fig.add_subplot()

	releaseStart = release.keys()[0]
	releaseEnd = release.keys()[-1]
	release_ext = release.loc[releaseStart:releaseEnd]
	indicatorEnd = release.keys()[-bestLag - 1]
	if indicatorEnd > mask.keys()[-1]:
		indicatorEnd = mask.keys()[-1]
		releaseEnd = release.keys()[release.keys().get_loc(indicatorEnd) + bestLag]
		release_ext = release.loc[releaseStart:releaseEnd]
	indicatorStart = mask.keys()[mask.keys().get_loc(indicatorEnd) - len(release_ext) + 1]

	indicatorEnd = mask.keys()[-1] #SK change
	mask_ext = mask.loc[indicatorStart:indicatorEnd]
	sd_ext = sd.loc[indicatorStart:indicatorEnd]
	for i in release_ext.keys():
		release_ext[i] = np.log(release_ext[i])
	for i in mask_ext.keys():
		mask_ext[i] = np.log(mask_ext[i])
	for i in sd_ext.keys():
		sd_ext[i] = np.log(sd_ext[i])

	X_ext = pd.concat([mask_ext, sd_ext], axis=1)
	Y_ext = bestRegr.predict(X_ext)
	dates_ext = [datetime.datetime.strptime(s, '%Y-%m-%d') for s in release_ext.keys()]
	dates_m = [datetime.datetime.strptime(s, '%Y-%m-%d') for s in mask_ext.keys()]
	Yinv = np.exp(Y_ext)
	Yinv = Yinv[-180:]
	time_frame = -180

	ax1.plot(dates_ext[time_frame:], release_ext[time_frame:], label='release')
	ax1.plot(dates_m[time_frame:], Y_ext[time_frame:], label='regression')
	ax1.plot(dates_m[time_frame:], np.exp(mask_ext[time_frame:]), label="MASKS")
	ax1.plot(dates_m[time_frame:], np.exp(sd_ext[time_frame:]), label="SD")

	ax1.plot()
	fig.suptitle(
		f'{state_dict[state]} lag={bestLag} days\ncoefficient={[round(i, 2) for i in coefficient]} cor={round(scipy.stats.pearsonr(bestRelease, bestY)[0], 4)}')
	ax1.legend()
	plt.show()
	plt.close(fig)
	return [coefficient, intercept, best_rho, bestLag]


def productRegressState(state):
	df = pd.read_csv(f'india/fitting_split2_2021-07-16_2021-03-15_2021-06-10/{state}/sim.csv')
	release = df[df['series'] == 'H'].iloc[0].iloc[1:]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/mask.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	mask = pd.Series(df['mean'])
	mask.index = [st[:10] for st in df['date_reported']]

	df = pd.read_csv(f'india/indicator/IHME/{state_dict[state]}/sd.csv',
					 usecols=['date_reported', 'data_type_name', 'mean'])
	df = df[df['data_type_name'] == 'observed']
	sd = pd.Series(df['mean'])
	sd.index = [st[:10] for st in df['date_reported']]
	min_loss = 10000000000000000000
	min_c1 = 0
	min_c2 = 0
	best_lag = 0
	releaseStartDate = release.index[0]

	for lag in range(30):
		maskStartDate = mask.index[mask.index.get_loc(releaseStartDate) - lag]
		sdStartDate = sd.index[sd.index.get_loc(releaseStartDate) - lag]
		tmpEndDate = release.index[-1 - lag]
		if tmpEndDate > mask.index[-1]:
			tmpEndDate = mask.index[-1]
		if tmpEndDate > sd.index[-1]:
			tmpEndDate = sd.index[-1]
		releaseEndDate = release.index[release.index.get_loc(tmpEndDate) + lag]
		maskEndDate = tmpEndDate
		sdEndDate = tmpEndDate
		releaseData = release[releaseStartDate:releaseEndDate]
		maskData = mask[maskStartDate:maskEndDate]
		sdData = sd[sdStartDate:sdEndDate]
		releaseData = releaseData.iloc[:100]
		maskData = maskData.iloc[:100]
		sdData = sdData.iloc[:100]
		print(sdData[1])
		optimal = minimize(lossProduct, [100, 10000],
						   args=(maskData, sdData, releaseData), method='L-BFGS-B',
						   bounds=[(-1000, 1000), (-1000000, 1000000)])
		# print(optimal.fun)
		current_loss = optimal.fun
		if current_loss < min_loss:
			print(current_loss)
			min_c1 = optimal.x[0]
			min_c2 = optimal.x[1]
			min_loss = current_loss
			best_lag = lag
			bestRelease = releaseData
			bestMask = maskData
			bestSd = sdData
	# size = len(releaseData)
	# Y = [min_c1 * (maskData[i]) * (- sdData[i]) + min_c2 for i in range(len(releaseData))]
	# fig = plt.figure()
	# ax1 = fig.add_subplot()
	# fig.suptitle(
	# 	f'{state_dict[state]} lag={best_lag} days\nc1={round(min_c1, 4)} c2={round(min_c2, 4)} cor={round(-min_loss, 4)}')
	# ax1.plot(range(size), releaseData, label='release')
	# ax1.plot(range(size), Y, label='regression')
	# ax1.legend()
	# plt.show()
	# plt.close(fig)

	Y = [min_c1 * (bestMask[i]) * (- bestSd[i]) + min_c2 for i in range(len(bestRelease))]
	print(Y)
	size = len(bestRelease)
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	fig.suptitle(
		f'{state_dict[state]} lag={best_lag} days\nc1={round(min_c1, 4)} c2={round(min_c2, 4)} cor={round(-min_loss, 4)}')
	ax1.plot(range(size), bestRelease, label='release')
	ax1.plot(range(size), Y, label='regression')
	ax2.plot(range(size), bestMask, label='mask')
	ax2.plot(range(size), bestSd, label='sd')
	ax1.legend()
	ax2.legend()
	plt.show()
	plt.close(fig)
	return


def lossProduct(point, mask, sd, release):
	c1 = point[0]
	c2 = point[1]
	print(point)
	# a1 = point[1]
	# a2 = point[2]
	Y = [c1 * (mask[i]) * (- sd[i]) + c2 for i in range(len(release))]
	# print(point, Y)
	loss = sum([(release[i] - Y[i]) ** 2 for i in range(len(Y))])
	print(loss)
	return loss


def main():
	# states = ['dl', 'kl']
	cols = ['state', 'mask', 'social distancing', 'intercept', 'correlation', 'lag']
	table = []
	for state in states:
		# linearRegressState(state)
		# linearRegressTimeWindowState(state)
		# interactingLinearRegressTimeWindowState(state)
		# interactingLogLinearRegressTimeWindowState(state)
		[coeff, intercept, rho, lag] = logLinearRegressTimeWindowState(state)
		table.append([state_dict[state], coeff[0], coeff[1], intercept, rho, lag])
	# productRegressState(state)
	df = pd.DataFrame(table, columns=cols)
	df.to_csv('india/indicator/IHME/correlation.csv', index=False)
	return


def tmp():
	a = pd.Series([1, 2, 3])
	print(a[:-1] - a[1:])
	return


if __name__ == '__main__':
	# tmp()
	main()
