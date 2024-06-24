# Python package to fit and forecast the COVID-19 infection dynamics in US Counties.
Based on the paper "Hidden ParametersHidden Parameters Impacting Resurgence of SARS-CoV-2 Pandemic."
$SIR$

The equations used are:
$$\frac{\mathrm{d}S}{\mathrm{d}t} = - \beta SI/N + \kappa_1;$$

$$\frac{\mathrm{d}I}{\mathrm{d}t} = \beta SI/N -(\gamma+\gamma_2) I;$$

$$\frac{\mathrm{d}I_H}{\mathrm{d}t} =   \gamma I - (\alpha_1 + \alpha_2) I_H;$$

$$\frac{\mathrm{d}I_N}{\mathrm{d}t} =   \gamma_2 I - \alpha_3 I_N; $$

$$\frac{\mathrm{d}D}{\mathrm{d}t} = \alpha_2 I_H;$$

$$\frac{\mathrm{d}R}{\mathrm{d}t}=  \alpha_1 I_H + \alpha_3 I_N; $$

$$\frac{\mathrm{d}G}{\mathrm{d}t} =  \beta SI/N ; &\frac{\mathrm{d} L}{\mathrm{d}t} =  - \kappa_1  ;
$$

NOTE: The variable L is replaced by H in the code.

## Fitting:
Uses python package scipy.optimize with randomized initial parameters and their ranges with multiple threads for
multiple starting points.

The fitting is in two phases: 
The first phase starts with the date at which the county has 5 confirmed cases and ends on 15th May, 2020.
The second phase starts at the time of reopening of the county after lifting restrictions. This reopen date varied for each county.
In the second phase a linear release of population was fitted.

## Forecast: 
The forecast uses the best parameters acquired by the fitting.

## How to use
The main code to be used is : MultiPhase_JHU50.py 

All other files are for testing, statistics, plots, library functions

The key subroutine used is the minimize from scipy.optimize

The following calls in the main code can be used:

1. fit_all_init() will fit the intiial phase only. 

2. fit_all_combined() will fit  the first and second  phase . It minimizes a loss function obtained from R^2 values.

More details on different parameters can be found in the code.