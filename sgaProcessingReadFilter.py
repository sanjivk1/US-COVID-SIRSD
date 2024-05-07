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
def check15States(stateName):
	listOfStates = ['California', 'Florida', 'New York', 'Texas', 'New Jersey', 'Illinois', 'Arizona', 'Georgia', 'Pennsylvania', 'Maryland', 'Nevada', 'North Carolina', 'Louisiana', 'Minnesota']
	if stateName in listOfStates:
		return 1
	else:
		return 0
def printReadFilter(df):
	# FOR each state	FOR each date	PRINT event
	columnNames = ['State', 'Events']
	newdf = pd.DataFrame(columns=columnNames)
	includeThisState = 0
	for index, row in df.iterrows():
		if not pd.isnull(row[0]):
			includeThisState = check15States(row[0])
		if includeThisState:
			if not pd.isnull(row[0]):
				toAdd = [row[0], row[1]]
				newdf.loc[len(newdf)] = toAdd
			elif not pd.isnull(row[1]):
				toAdd = [row[0], row[1]]
				# if filterWords(row[1]):
				newdf.loc[len(newdf)] = toAdd
	newdf.to_excel("data/sga15States.xlsx")
def main():
	df = pd.read_csv('data/sgaExpandAll.csv', header=None)
	df = df.dropna(how='all')
	newdf = printReadFilter(df)
if __name__ == "__main__":
	main()