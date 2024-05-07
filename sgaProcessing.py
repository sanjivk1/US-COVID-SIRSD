import pandas as pd
from datetime import datetime
import datefinder
import numpy as np
import re
from calendar import month_name
def filterWords(rowString):
	# bagOfWords = ['reopen', 're-open', 're-opening', 'reopening', 'restaurant', 'closed', 'closures', 'closing', 'restaurants', 'school', 'schools', 'bars', 'day-care', 'daycare', 'factory', 'factories', 'office', 'offices', 'lockdown', 'lock-down', 'lockdowns', 'businesses', 'workplaces']
	# bagOfWords = ['bars', 'restaurants', 'schools', 'day-care', 'daycare']
	# bagOfWords = ['reopen', 're-open', 're-opening', 'reopening', 'face', 'face-covering', 'face-coverings', 'mask', 'masks']
	bagOfWords = ['reopen', 're-open', 're-opening', 'reopening']
	BagOfWords = [eachWord.capitalize() for eachWord in bagOfWords]
	bagOfWords = bagOfWords + BagOfWords
	if any(word in rowString for word in bagOfWords):
		return True
	return False
def checkIfMonthLine(dateVal):
	pattern = '|'.join(month_name[1:])
	theMonth = re.search(pattern, dateVal, re.IGNORECASE)
	if not theMonth:
		return 0
	else:
		return 1
def extractDate(rowString):
	matches = list(datefinder.find_dates(rowString))
	dateVal = ""
	isFirstWordMonth = 0
	if len(matches) > 0:
		dateVal = matches[0]
		isFirstWordMonth = checkIfMonthLine(rowString)
	return dateVal, rowString, isFirstWordMonth

def processFile(df):
	startDate = datetime.strptime("01-02-2020", "%d-%m-%Y")
	endDate = datetime.strptime("31-07-2020", "%d-%m-%Y")
	dateList = pd.date_range(start=startDate, end=endDate, normalize=True)
	columnNames = ['State'] + dateList.tolist()
	# print(f'columnNames = {columnNames}')
	newdf = pd.DataFrame(columns=columnNames)
	newdfIndex: int = 1
	prevDate = startDate
	for index, row in df.iterrows():
		if not pd.isnull(row[0]):
			newdfIndex = newdfIndex + 1
			newdf.append(pd.Series(dtype=str), ignore_index=True)
			newdf.at[newdfIndex, 'State'] = row[0]
		elif not pd.isnull(row[1]):
			if filterWords(row[1]):
				atCol, eventString, isFirstWordMonth = extractDate(row[1])
				if isFirstWordMonth:
					prevDate = atCol
					newdf.at[newdfIndex, atCol] = eventString
				else:
					atCol = prevDate
					newdf.at[newdfIndex, atCol] = str(newdf.at[newdfIndex, atCol]) + eventString
	# print(f'newdf = {newdf}')
	newdf.to_excel("data/sgaProcessed.xlsx")
	return newdf
def listStatesOpenings(newdf):
	# list states and their opening events with dates
	columnNames = ['State', 'EarliestOpening', 'LatestOpening']
	newdf1 = pd.DataFrame(columns=columnNames)
	for index, row in newdf.iterrows():
		toAdd = [row[0], row[row.first_valid_index()],row[row.last_valid_index()]]
		# print(toAdd)
		# newdf1 = newdf1.append(toAdd, ignore_index=True, columns = newdf1.columns)
		newdf1.loc[len(newdf1)] = toAdd
	# print(f'newdf1 = {newdf1}')
	newdf1.to_excel("data/sgaReadFilter.xlsx")
	return newdf1
def printReadFilter(newdf):
	# FOR each state	FOR each date	PRINT event
	print(f'Read Filter')
	columnNames = ['State', 'Events']
	newdf1 = pd.DataFrame(columns=columnNames)
	for index, row in newdf.iterrows():
		toAdd = [row[0], np.nan]
		newdf1.loc[len(newdf1)] = toAdd
		for r in range(1,len(row)):
			if not pd.isnull(row[r]):
				toAdd = [np.nan, row[r]]
				newdf1.loc[len(newdf1)] = toAdd
	newdf1.to_excel("data/sgaReadability.xlsx")
def main():
	df = pd.read_csv('data/sgaExpandAll.csv')
	df = df.dropna(how='all')
	# print(df)
	newdf = processFile(df)
	# printForReadability(df)
	listStatesOpenings(newdf)

if __name__ == "__main__":
	main()