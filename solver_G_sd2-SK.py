#!/usr/bin/python
import numpy as np
import pandas as pd
# import math as math
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
# from scipy.special import erf
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import sys
# import json
# import ssl
# import urllib.request
from sklearn.metrics import mean_squared_error, r2_score
from SIRfunctions import SIRG 
# from SIRfunctions import computeBeta

np.set_printoptions(threshold=sys.maxsize)
Geo = 0.8

class Learner(object):
    def __init__(self, country, loss, start_date, predict_range, s_0, i_0, r_0, n_0, end_date):
        self.country = country
        self.loss = loss
        self.start_date = start_date
        self.predict_range = predict_range
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.n_0 = n_0
        self.end_date = end_date

    def load_confirmed(self, country):
        df = pd.read_csv('data/time_series_19-covid-Confirmed-US.csv')
        country_df = df[df['Country/Region'] == country]
        # self.i_0 = country_df.iloc[0].loc[self.start_date]
        # return country_df.iloc[0].loc[self.start_date: self.end_date]
        return country_df

    def load_recovered(self, country):
        df = pd.read_csv('data/time_series_19-covid-Recovered-US.csv')
        country_df = df[df['Country/Region'] == country]
        # self.r_0 = country_df.iloc[0].loc[self.start_date]
        # return country_df.iloc[0].loc[self.start_date: self.end_date]
        return country_df

    def load_dead(self, country):
        df = pd.read_csv('data/time_series_19-covid-Deaths-US.csv')
        country_df = df[df['Country/Region'] == country]
        # return country_df.iloc[0].loc[self.start_date: self.end_date]
        return country_df

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def train(self):
        recovered = self.load_recovered(self.country)
        death = self.load_dead(self.country)
        confirmed = self.load_confirmed(self.country)
        # for e in recovered.columns():
        #     e = str(e)
        # data = (confirmed - recovered - death)
        regEta = -1.0
        regGamma = -1.0
        regLoss = 100000000000
        regBeta = -1.0
        regStartDate = self.start_date
        # regData = ''
        # regRecovered = ''
        reg_i_0 = -1
        reg_r_0 = -1
        dates = confirmed.columns[30:50]    # will iterate over these starting dates

        # loop over dates
        for d in dates:
            print(f'on date {d}')
            recoveredLoop = recovered.iloc[0].loc[d: self.end_date]
            deathLoop = death.iloc[0].loc[d: self.end_date]
            confirmedLoop = confirmed.iloc[0].loc[d: self.end_date]
            dataLoop = (confirmedLoop - recoveredLoop - deathLoop)
            recoveredLoop = recoveredLoop + deathLoop
            self.i_0 = dataLoop.iloc[0]
            self.r_0 = recoveredLoop.iloc[0]
            if self.i_0 == 0:
                self.i_0 = 0.001
            optimal = minimize(loss, [25, 0.01, 0.05], args=(confirmedLoop, self.i_0, self.r_0, self.n_0),
                               method='L-BFGS-B', bounds=[(0.1, 500), (0.0001, 0.05), (0.001, 0.04)])

            current_loss = loss(optimal.x, confirmedLoop, self.i_0, self.r_0, self.n_0)
            if current_loss < regLoss:
                regLoss = current_loss
                regBeta = optimal.x[0]
                regEta = optimal.x[2]
                regGamma = optimal.x[1]
                regStartDate = d
                regConfirmed = confirmedLoop
                # regRecovered = recoveredLoop
                # regData = dataLoop
                reg_i_0 = self.i_0
                reg_r_0 = self.r_0
                print(regLoss, regBeta, regGamma, regEta, d, regGamma / regEta)

        print(f'beta = {round(regBeta, 8)} , gamma = {round(regGamma, 8)} , eta = {round(regEta, 8)} , regLoss = {regLoss}, regStartDate = {regStartDate} ')

        n_0 = self.n_0
        r_0 = reg_r_0
        i_0 = reg_i_0
        beta = regBeta
        gamma = regGamma
        eta = regEta

        size = len(regConfirmed) 
        
        solution = solve_ivp(SIRG, [0, size], [n_0 * eta, i_0, r_0, regConfirmed[0], beta, gamma, eta, n_0], t_eval=np.arange(0, size, 1), vectorized=True)

        # idx = range(len(solution.y[1]))
        # print(idx)
        plt.plot(regConfirmed, label="G")
        plt.plot(solution.y[3], label="G'")
        # plt.plot(regRecovered, label="R")
        # plt.plot(solution.y[2], label="R'")
        # plt.plot(solution.y[0], label="S'")
        # plt.plot(solution.y[1], label="I'")
        # plt.plot(regData, label="I")

        # plt.xticks([], [])
        plt.xticks([])
        plt.legend()
        plt.title(self.country)

        confirmed_derivative = np.diff(regConfirmed)
        G_derivative = np.diff(solution.y[3])
        weights = [Geo ** (n - 1) for n in range(1, size)]
        weights.reverse()
        confirmed_derivative *= weights
        G_derivative *= weights

        # alpha = 0.8
        metric0 = r2_score(regConfirmed, solution.y[3])
        metric1 = r2_score(confirmed_derivative, G_derivative)

        print(f'errors: {metric0} and {metric1}')
        #
        plt.show()


def loss(point, confirmed, i_0, r_0, n_0):
    size = len(confirmed)
    beta = point[0]
    gamma = point[1]
    eta = point[2]
 
    solution = solve_ivp(SIRG, [0, size], [n_0 * eta, i_0, r_0, confirmed[0], beta, gamma, eta, n_0], t_eval=np.arange(0, size, 1), vectorized=True)

    confirmed_derivative = np.diff(confirmed)
    G_derivative = np.diff(solution.y[3])
    weights = [Geo ** (n - 1) for n in range(1, size)]
    weights.reverse()
    confirmed_derivative *= weights
    G_derivative *= weights

    alpha = 0.7
    metric0 = r2_score(confirmed, solution.y[3])
    metric1 = r2_score(confirmed_derivative, G_derivative)
    return - (alpha * metric0 + (1 - alpha) * metric1)

def main():
    # country = 'Illinois'
    # n_0 = 12000000
    country = 'New York'
    n_0 = 23000000
    i_0 = 2
    r_0 = 0
    startdate = '01/22/20'
    predict_range = 150
    s_0 = 100000
    enddate = '05/16/20'
    learner = Learner(country, loss, startdate, predict_range, s_0, i_0, r_0, n_0, enddate)
    learner.train()

if __name__ == '__main__':
    main()