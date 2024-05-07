import pandas as pd
import numpy as np
import json
import requests
import datetime
import matplotlib.pyplot as plt
import os


def saveIndicator_UM(RequestString, CSVName, DataType):
	print('downloading', DataType, CSVName)
	response = requests.get(RequestString).text
	if response:
		try:
			jsonData = json.loads(response)

			df = pd.DataFrame.from_dict(jsonData['data'])
			df.to_csv(f'india/indicator/{DataType}/{CSVName}.csv', index=False)
		except Exception as inst:
			print('error downloading', CSVName)
			print(inst)
			print(response)

	return


def downloadIndicators_UM(DataType):
	Indicators = ['mask', 'contact', 'work_outside_home_1d', 'shop_1d', 'restaurant_1d', 'spent_time_1d',
	              'large_event_1d', 'public_transit_1d']
	for Indicator in Indicators[:2]:
		RequestString = f'https://covidmap.umd.edu/api/resources?indicator={Indicator}&type={DataType}&country=India&region=all&daterange=20200715-20210719'
		saveIndicator_UM(RequestString, Indicator, DataType)
	return


# [
#     indicator: 'covid' or 'flu' or 'cli_w11' or 'ili_W11' or 'mask' or 'contact' or 'finance' or 'anosmia' or
#     'vaccine_acpt' or
#                'access_wash' or 'covid_vaccine' or 'trust_fam' or 'trust_healthcare' or 'trust_who' or 'trust_govt' or
#                'trust_politicians' or 'twodoses' or 'concerned_sideeffects' or 'wash_hands_24h_3to6' or
#                'wash_hands_24h_7orMore'
#                or 'hesitant_sideeffects' or 'modified_acceptance' or 'cmty_covid' or 'barrier_reason_side_effects' or
#                'barrier_reason_wontwork' or 'barrier_reason_dontbelieve' or 'barrier_reason_dontlike' or
#                'barrier_reason_waitlater'
#                or 'barrier_reason_otherpeople' or 'barrier_reason_cost' or 'barrier_reason_religious' or
#                'barrier_reason_other' or
#                'trust_doctors' or 'barrier_reason_dontneed_alreadyhad' or 'barrier_reason_dontneed_dontspendtime' or
#                'barrier_reason_dontneed_nothighrisk' or 'barrier_reason_dontneed_takeprecautions' or
#                'barrier_reason_dontneed_notserious' or 'barrier_reason_dontneed_notbeneficial' or
#                'barrier_reason_dontneed_other' or 'informed_access' or 'appointment_have' or 'appointment_tried' or
#                'barrier_reason_government' or 'activity_work_outside_home' or 'activity_shop' or
#                'activity_restaurant_bar' or
#                'activity_spent_time' or 'activity_large_event' or 'activity_public_transit' or 'food_security' or
#                'anxious_7d' or 'depressed_7d' or 'worried_become_ill' or 'symp_fever' or 'symp_cough' or
#                'symp_diff_breathing' or 'symp_fatigue' or 'symp_stuffy_nose' or 'symp_aches' or 'symp_sore_throat' or
#                'symp_chest_pain' or 'symp_nausea' or 'symp_eye_pain' or 'symp_headache' or 'sick_spend_time_7d' or
#                'ever_tested' or 'pay_test' or 'reduce_spending' or 'symp_chills' or 'symp_changes' or
#                'testing_rate' or
#                'tested_positive_14d' or 'tested_positive_recent' or 'flu_vaccine_thisyr' or 'flu_vaccine_lastyr' or
#                'avoid_contact' or 'vaccinated_appointment_or_accept' or 'appointment_or_accept_covid_vaccine' or
#                'accept_covid_vaccine_no_appointment' or 'appointment_not_vaccinated' or 'vaccine_tried' or
#                'had_covid_ever' or 'worried_catch_covid' or 'belief_distancing_effective' or
#                'belief_masking_effective' or 'others_distanced_public' or 'others_masked_public' or
#                'covid_vaccinated_friends' or 'belief_vaccinated_mask_unnecessary' or 'belief_children_immune' or
#                'belief_no_spread_hot_humid' or 'received_news_local_health' or 'received_news_experts' or
#                'received_news_who' or 'received_news_govt_health' or 'received_news_politicians' or
#                'received_news_journalists' or 'received_news_friends' or 'received_news_religious' or
#                'received_news_none' or 'trust_covid_info_local_health' or 'trust_covid_info_experts' or
#                'trust_covid_info_who' or 'trust_covid_info_govt_health' or 'trust_covid_info_politicians' or
#                'trust_covid_info_journalists' or 'trust_covid_info_friends' or 'trust_covid_info_religious' or
#                'want_info_covid_treatment' or 'want_info_vaccine_access' or 'want_info_covid_variants' or
#                'want_info_children_education' or 'want_info_economic_impact' or 'want_info_mental_health' or
#                'want_info_relationships' or 'want_info_employment' or 'want_info_none' or 'news_online' or
#                'news_messaging' or 'news_newspaper' or 'news_television' or 'news_radio' or 'news_none' or
#                'trust_news_online' or 'trust_news_messaging' or 'trust_news_newspaper' or 'trust_news_television' or
#                'trust_news_radio' or 'vaccinate_children' or 'delayed_care_cost' or 'vaccine_barrier_eligible' or
#                'vaccine_barrier_no_appointments' or 'vaccine_barrier_appointment_time' or
#                'vaccine_barrier_technical_difficulties' or 'vaccine_barrier_document' or
#                'vaccine_barrier_technology_access' or 'vaccine_barrier_travel' or 'vaccine_barrier_language' or
#                'vaccine_barrier_childcare' or 'vaccine_barrier_time' or 'vaccine_barrier_type' or
#                'vaccine_barrier_none' or 'try_vaccinate_1m' or 'vaccine_barrier_eligible_has' or
#                'vaccine_barrier_no_appointments_has' or 'vaccine_barrier_appointment_time_has' or
#                'vaccine_barrier_technical_difficulties_has' or
#                'vaccine_barrier_document_has' or 'vaccine_barrier_technology_access_has' or
#                'vaccine_barrier_travel_has' or 'vaccine_barrier_language_has' or 'vaccine_barrier_childcare_has' or
#                'vaccine_barrier_time_has' or 'vaccine_barrier_type_has' or 'vaccine_barrier_none_has' or
#                'vaccine_barrier_eligible_tried' or 'vaccine_barrier_no_appointments_tried' or
#                'vaccine_barrier_appointment_time_tried' or 'vaccine_barrier_technical_difficulties_tried' or
#                'vaccine_barrier_document_tried' or 'vaccine_barrier_technology_access_tried' or
#                'vaccine_barrier_travel_tried' or 'vaccine_barrier_language_tried' or
#                'vaccine_barrier_childcare_tried' or
#                'vaccine_barrier_time_tried' or 'vaccine_barrier_type_tried' or 'vaccine_barrier_none_tried']"}


def plotIndicator_UM(DataType):
	print(DataType, 'mask')
	df = pd.read_csv(f'india/indicator/{DataType}/mask.csv',
	                 usecols=['smoothed_mc' if DataType == 'smoothed' else 'percent_mc', 'region', 'survey_date'],
	                 dtype={'survey_date': str})
	Regions = df['region']
	RegionSet = set()
	for r in Regions:
		if r not in RegionSet:
			RegionSet.add(r)
	print(RegionSet)
	for Region in RegionSet:
		RegionData = df[df['region'] == Region]
		# print(RegionData)
		fig = plt.figure()
		fig.suptitle(f'{Region} {DataType}')
		ax = fig.add_subplot()
		Dates = [datetime.datetime.strptime(date, '%Y%m%d') for date in RegionData['survey_date']]
		Percentages = RegionData['smoothed_mc' if DataType == 'smoothed' else 'percent_mc']
		if DataType == 'daily':
			Percentages = Percentages.rolling(7, min_periods=1).mean()
		ax.plot(Dates, Percentages)
		fig.autofmt_xdate()
		fig.savefig(f'india/indicator/{DataType}/mask/{Region}.png')
		plt.close(fig)

	print(DataType, 'contact')
	df = pd.read_csv(f'india/indicator/{DataType}/contact.csv',
	                 usecols=['smoothed_dc' if DataType == 'smoothed' else 'percent_dc', 'region', 'survey_date'],
	                 dtype={'survey_date': str})
	Regions = df['region']
	RegionSet = set()
	for r in Regions:
		if r not in RegionSet:
			RegionSet.add(r)
	print(RegionSet)
	for Region in RegionSet:
		RegionData = df[df['region'] == Region]
		# print(RegionData)
		fig = plt.figure()
		fig.suptitle(f'{Region} {DataType}')
		ax = fig.add_subplot()
		Dates = [datetime.datetime.strptime(date, '%Y%m%d') for date in RegionData['survey_date']]
		Percentages = RegionData['smoothed_dc' if DataType == 'smoothed' else 'percent_dc']
		if DataType == 'daily':
			Percentages = Percentages.rolling(7, min_periods=1).mean()
		ax.plot(Dates, Percentages)
		fig.autofmt_xdate()
		fig.savefig(f'india/indicator/{DataType}/contact/{Region}.png')
		plt.close(fig)

	return


def downloadIndicators_IHME():
	locDict = {'41': 'Andhra Pradesh',
	           '42': 'Arunachal Pradesh',
	           '43': 'Assam',
	           '44': 'Bihar',
	           '46': 'Chhattisgarh',
	           '49': 'Delhi',
	           '50': 'Goa',
	           '51': 'Gujarat',
	           '52': 'Haryana',
	           '53': 'Himachal Pradesh',
	           '54': 'Jammu & Kashmir and Ladakh',
	           '55': 'Jharkhand',
	           '56': 'Karnataka',
	           '57': 'Kerala',

	           '59': 'Madhya Pradesh',
	           '60': 'Maharashtra',
	           '61': 'Manipur',
	           '62': 'Meghalaya',

	           '64': 'Nagaland',
	           '65': 'Odisha',

	           '67': 'Punjab',
	           '68': 'Rajasthan',

	           '70': 'Tamil Nadu',
	           '71': 'Telangana',
	           '72': 'Tripura',
	           '73': 'Uttar Pradesh',
	           '74': 'Uttarakhand',
	           '75': 'West Bengal'}
	for loc_id in locDict.keys():
		loc_name = locDict[loc_id]
		folder = f'india/indicator/IHME/{loc_name}'
		if not os.path.exists(folder):
			os.makedirs(folder)

		# social distancing
		print('downloading', loc_name, 'social distancing')
		RequestString = f'https://covid19.healthdata.org/api/data/hospitalization?location=48{loc_id}&measure=22&scenario%5B%5D=6&scenario%5B%5D=1&scenario%5B%5D=2&scenario%5B%5D=3'
		response = requests.get(RequestString).text
		jsonData = json.loads(response)
		print(jsonData['keys'])
		df = pd.DataFrame.from_dict(jsonData['values'])
		df.columns = jsonData['keys']
		df.to_csv(f'{folder}/sd.csv', index=False)

		# mask
		print('downloading', loc_name, 'mask')
		RequestString2 = f'https://covid19.healthdata.org/api/data/hospitalization?location=48{loc_id}&measure=25&scenario%5B%5D=6&scenario%5B%5D=1&scenario%5B%5D=3'
		response = requests.get(RequestString2).text
		jsonData = json.loads(response)
		print(jsonData['keys'])
		df = pd.DataFrame.from_dict(jsonData['values'])
		df.columns = jsonData['keys']
		df.to_csv(f'{folder}/mask.csv', index=False)

	return


def main():
	# downloadIndicators_UM('smoothed')
	# downloadIndicators_UM('daily')
	# plotIndicator_UM('smoothed')
	# plotIndicator_UM('daily')
	downloadIndicators_IHME()
	return


def indiaMask():
	df = json
	return


if __name__ == '__main__':
	main()
