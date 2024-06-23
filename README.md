Fitting India
This is a python package to fit and forecast the COVID-19 infection dynamics based on the paper Early guidance for Sars-Cov-2 health policies in India: Social Distancing amidst Vaccination and Virus Variants*. It is a multiprocessing program where the number of processes can be specified. We recommend running it using multi-core machine. The model fitting currently available was done using a 96 core machine.
Fitting
Fitting is done by the python package scipy.optimize with randomized initial parameters and their ranges with multiprocessing.
Forecast
The forecast uses the best parameters acquired by the fitting, with assumptions on lockdown, reopening, and vaccination.
How to use
Run fit_all_split() in main() of Fitting_India_split.py to fit the model for every state in India. It is very computation heavy and takes a very long time to finish.
The current data contain confirmed numbers up to 2022-02-28. If you want to run the fitting with updated data, you need to update 'india/indian_cases_confirmed_cases.csv' and ' india/indian_cases_confirmed_deaths.csv' with complete and consecutive case numbers for each state in India, and make sure the dates in the first row follows the format "YYYY-mm-dd".
Please read the comments in the code to learn how to change the ending date of the fitting. The current fitting ending date is set to be 2021-12-15.
Ending date of the fitting
fitting_enddate = '2021-12-15'
The new reopening date in the forecast. this should not be earlier than the fitting ending date
reopen_date3 = '2021-12-16'
Starting date of the Omicron variant or any other variant
Om_start_date = '2021-12-15'
How to get forecast projects after fitting
Run compareVacRatesExtended() in main() of Fitting_India_split.py to get the forecast based on the model fitting for every state in India.
How to get accuracy results
Run RMSE_ALL() from main gives the errors in a csv file with columns ['state', 'RMSE_confirmed', 'RMSE_confirmed_phase1', 'RMSE_confirmed_phase2', 'RMSE_death', 'RMSE_death_phase1', 'RMSE_death_phase2', 'R2_confirmed', 'R2_confirmed_phase1', 'R2_confirmed_phase2', 'R2_death', 'R2_death_phase1', 'R2_death_phase2']
More details on different parameters can be found in the code.