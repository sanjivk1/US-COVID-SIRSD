import numpy as np
import random as rd
# import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import math
from SIRfunctions import SIRG_sd, SIRG, computeBeta


nodesObjArr = []

# portion of traffic to shutdown
mu = 0

numNodes = 54

beta = 100

# proportion of infected population showing no symptoms
beta_asx = 1

# gamma = 0.05

c1 = 0.98

# fraction of initial susceptible population going hiding
alpha = 0

# minimum simulation duration
duration = 100

# threshold of total I to terminate
cut_off = 5000

# threshold of inf/sus for a node to shutdown
TH_shutdown = 0

# reopen method
# 1: inf drops by a fraction TH_reopen of the peak
# 2: derivative of inf drops to a certain level TH_reopen_del
reopen_method = 1
TH_reopen_del = 0
TH_reopen = 1

shutdown = []
reopen = []
delay = 0

# days to recover
recovery_period = 1

# first city to infect
infect_first = 34

# percentage of susceptible population to infect on day 0
infection_rate = 0.0086

# multiplier of the travel beta versus regular beta
travel_multiplier = 10

# susceptible population in percentage of actual population

# eta = 0.0049

etas = []

# population_scale = 0

# fraction of hiding population to be released everyday
release_rate = 0.005

global cities
global population
global travel_cap
global betas
global gammas

# gammas =        [0.05,          0.05,       0.05,       0.05,           0.05,       0.05,       0.05,       0.05,       0.05,           0.05,       0.05]
# betas =         [65,            65,         65,         65,             65,         65,         65,         65,         65,             65,         65]
# cities =        ['Chicago',     'Delhi',    'London',   'Los Angeles',  'Madrid',   'New York', 'Paris',    'Rome',     'Shanghai',     'Tokyo',    'Wuhan']
# population =    [2695598,        16787941,   8908081,    3792621,        3223334,    19304000,   2148271,    2879728,    24281400,       13929286,   11081000]
susArr =        []
# capacities of flights between cities
# travel_cap =    [[0,            340.0,      3432.0,     1340.0,         1135.5,     1845.0,     2484.0,     1785.0,     2141.0,         2157.0,     0],
#                 [340.0,         0,          1676.0,     0,              0,          340.0,      474.0,      582.0,      474.0,          827.0,      0],
#                 [3432.0,        1641.0,     0,          5615.0,         1433.0,     4753.0,     1732.0,     1754.0,     1221.0,         2581.0,     0],
#                 [1622.0,        0,          5615.0,     0,              1267.25,    2309.0,     2836.0,     750.0,      2137.0,         5452.0,     0],
#                 [1135.5,        0,          1435.0,     1267.25,        0,          2444.25,    1158.0,     816.0,      0,              0,          0],
#                 [1861.0,        340.0,      5281.0,     2309.0,         2444.25,    0,          5155.0,     2010.0,     652.0,          1739.0,     0],
#                 [2484.0,        474.0,      1732.0,     2358.0,         1158.0,     5124.0,     0,          1319.25,    1349.5,         2382.0,     234.0],
#                 [1785.0,        582.0,      1754.0,     750.0,          816.0,      2010.0,     1319.25,    0,          468.0,          375.0,      0],
#                 [2141.0,        474.0,      1221.0,     1802.0,         0,          652.0,      1349.5,     468.0,      0,              2862.0,     1780.0],
#                 [2157.0,        827.0,      2581.0,     5208.0,         0,          1739.0,     2382.0,     375.0,      2862.0,         0,          0],
#                 [0,             0,          0,          0,              0,          0,          234.0,      0,          1780.0,         0,          0]]


class Node:

    def __init__(self, i, beta, beta_asx, gamma, mu):
        self.nodeNum = i
        self.sus = []                   # exposed susceptible population
        self.cum = []
        self.inf = []                   # symptomatic infected population
        self.rec = []                   # recovered population
        self.hid = []                   # hiding susceptible population
        self.asx = []                   # asymptomatic susceptible population
        self.susLeav = []               # 2D array [time][neighbor]
        self.susLeavTot = []            # total leaving susceptible
        self.asxLeav = []
        self.asxLeavTot = []
        self.infLeav = []               # 2D array [time][neighbor]
        self.beta = beta
        self.beta_asx = beta_asx
        self.gamma = gamma
        self.c1 = c1
        self.mu = mu
        self.inf_peak = 0
        # derivative of symptomatic infection
        self.inf_del = 0
        self.asx_peak = 0
        self.asx_del = 0

    def release_hiding(self, currentDay):
        self.sus[currentDay] += self.hid[currentDay] * release_rate
        self.hid[currentDay] -= self.hid[currentDay] * release_rate

    def infect(self, day, rate):
        self.asx[day] += self.sus[day] * rate * self.beta_asx
        self.inf[day] += self.sus[day] * rate * (1 - self.beta_asx)
        self.cum[day] += self.sus[day] * rate
        self.sus[day] -= self.sus[day] * rate
        print(cities[self.nodeNum], ' has been infected')

    # compute people travelling out of the node with actual counts
    def travel_out(self, currentDay, cap):

        # recalculate the capacity of edges based on travel control
        for i in range(numNodes):
            cap[i] *= (1 - edge_mu(self.nodeNum, i))

        # the unit of travelling out is actual count
        self.susLeavTot[currentDay] = 0
        self.asxLeavTot[currentDay] = 0

        # calculate LS_v and LA_v on each edge
        for i in range(numNodes):
            self.susLeav[currentDay][i] = self.sus[currentDay] * cap[i] / population[self.nodeNum]
            self.susLeavTot[currentDay] += self.susLeav[currentDay][i]
            self.asxLeav[currentDay][i] = self.asx[currentDay] * cap[i] / population[self.nodeNum]
            self.asxLeavTot[currentDay] += self.asxLeav[currentDay][i]

        # if cities[self.nodeNum] == 'Illinois':
        #     print(self.susLeavTot[currentDay] / self.sus[currentDay])
        # S = S - LS_v, unit in fraction
        self.sus[currentDay] -= self.susLeavTot[currentDay]

        # IA = IA - LA_v, unit in fraction
        self.asx[currentDay] -= self.asxLeavTot[currentDay]

    # compute infection during travelling
    def travel_infection(self, currentDay, cap):

        # recalculate the capacity of edges based on travel control
        for i in range(numNodes):
            cap[i] *= (1 - edge_mu(self.nodeNum, i))
        for i in range(numNodes):
            if cap[i] == 0:
                continue
            tot = self.susLeav[currentDay][i] + self.asxLeav[currentDay][i]

            # Delta_travel = beta * travel_multiplier * LS_v * L, unit in fraction
            new_infected = self.susLeav[currentDay][i] * self.asxLeav[currentDay][i] / cap[i] * self.beta * travel_multiplier
            if new_infected > self.susLeav[currentDay][i]:
                new_infected = self.susLeav[currentDay][i]
                # print("travel change too large on day", currentDay, 'from', cities[self.nodeNum], 'to', cities[i], '!!!!!!!!!!')
                # print(new_infected)

            # LA_v = LA_v + beta_asx * Delta_travel
            self.asxLeav[currentDay][i] += new_infected * beta_asx
            # LI_v = LI_v + (1 - beta_asx) * Delta_travel
            self.infLeav[currentDay][i] += new_infected * (1 - beta_asx)
            # LS_v = LS_v - Delta_travel
            self.susLeav[currentDay][i] -= new_infected

            if abs(tot - self.asxLeav[currentDay][i] - self.infLeav[currentDay][i] - self.susLeav[currentDay][i]) > 1:
                print('travel infection error from', cities[self.nodeNum], 'to', cities[i], 'on day', currentDay)

    # returns total incoming susceptible population
    def NSv(self, currentDay):
        ret = 0
        for i in range(numNodes):
            ret += nodesObjArr[i].getLSv(self.nodeNum, currentDay)
        # print(cities[self.nodeNum], 'has', round(ret, 2), 'new sus on day', currentDay)
        return ret

    # returns total incoming asymptomatic infected population
    def NAv(self, currentDay):
        ret = 0
        for i in range(numNodes):
            ret += nodesObjArr[i].getLAv(self.nodeNum, currentDay)
        # print(cities[self.nodeNum], 'has', round(ret, 2), 'new asx on day', currentDay)
        return ret

    # returns total incoming symptomatic infected population
    def NIv(self, currentDay):
        ret = 0
        for i in range(numNodes):
            ret += nodesObjArr[i].getLIv(self.nodeNum, currentDay)
        # print(cities[self.nodeNum], 'has', round(ret, 2), 'new inf on day', currentDay)
        return ret

    def getLSv(self, nbr, currentDay):
        return self.susLeav[currentDay][nbr]

    def getLAv(self, nbr, currentDay):
        return self.asxLeav[currentDay][nbr]

    def getLIv(self, nbr, currentDay):
        return self.infLeav[currentDay][nbr]

    # calculate SIR
    def updateNodeSIR(self, currentDay):
        delta = SIRG_sd(currentDay, [self.sus[currentDay], self.asx[currentDay], self.rec[currentDay], 0, self.beta, self.gamma, etas[self.nodeNum], population[self.nodeNum], self.c1])
        # Delta_S = beta * S * IA
        delta_S = delta[0]
        # print()
        # print(delta)
        delta_asx = delta[1]
        delta_rec = delta[2]

        if (- delta_S) > self.sus[currentDay]:
            delta_asx = delta_asx + self.sus[currentDay] + delta_S
            delta_S = - self.sus[currentDay]
            print(cities[self.nodeNum], 'S is ', self.sus[currentDay])
            print('******************S becomes too small at ', cities[self.nodeNum], 'on day', currentDay, '!****************')

        self.sus[currentDay] = self.sus[currentDay] + self.NSv(currentDay) + delta_S

        self.cum[currentDay] -= delta_S

        self.asx[currentDay] = self.asx[currentDay] + self.NAv(currentDay) + delta_asx

        self.rec[currentDay] = self.rec[currentDay] + delta_rec

    # calculate S,IA,IS,R
    def updateNodeSIR2(self, currentDay):
        # Delta_S = beta * S * IA
        delta_S = self.sus[currentDay] * self.asx[currentDay] * self.beta

        if delta_S > self.sus[currentDay]:
            delta_S = self.sus[currentDay]
            print(cities[self.nodeNum], 'S is ', self.sus[currentDay])
            print('******************S becomes too small at ', cities[self.nodeNum], 'on day', currentDay, '!****************')

        # Delta_IA = gamma * IA
        delta_asx = self.asx[currentDay] * self.gamma
        # Delta_IS = gamma * IS
        delta_inf = self.inf[currentDay] * self.gamma

        # S = S - Delta_S + NS_v
        self.sus[currentDay] = self.sus[currentDay] + self.NSv(currentDay) / population[self.nodeNum] - delta_S
        # IA = IA + Delta_S + NA_v
        self.asx[currentDay] = self.asx[currentDay] + self.NAv(currentDay) / population[self.nodeNum] + delta_S * beta_asx
        # IS = IS + Delta_S + NI_v
        self.inf[currentDay] = self.inf[currentDay] + self.NIv(currentDay) / population[self.nodeNum] + delta_S * (1 - beta_asx)
        # IA = IA - Delta_IA
        self.asx[currentDay] = self.asx[currentDay] - delta_asx
        # IS = IS - Delta_IS
        self.inf[currentDay] = self.inf[currentDay] - delta_inf
        # R = R + Delta_IA + Delta_IS
        self.rec[currentDay] = self.rec[currentDay] + delta_inf
        # self.rec[currentDay] = self.rec[currentDay] + delta_asx + delta_inf

    # check for shutdown and reopen
    def travel_control(self, currentDay):
        if self.asx_peak < self.asx[currentDay]:
            self.asx_peak = self.asx[currentDay]
        if self.asx_del < (self.asx[currentDay] - self.asx[currentDay - 1]):
            self.asx_del = (self.asx[currentDay] - self.asx[currentDay - 1])
        if not reopen[self.nodeNum]:
            if shutdown[self.nodeNum]:
                if reopen_method == 1:
                    # if self.inf[currentDay] / susArr[self.nodeNum] < TH_reopen:
                    # if self.inf[currentDay] < self.inf[currentDay - 1]:
                    if self.asx[currentDay] < self.asx[currentDay - 1] and self.asx[currentDay] < self.asx_peak * (1 - TH_reopen):
                        reopen[self.nodeNum] = True
                        print('**********', cities[self.nodeNum], 'reopen on day', currentDay, '**********')
                        # print(self.sus[currentDay], 'sus at', cities[self.nodeNum], 'on day', currentDay)
                        # print(self.hid[currentDay], 'hid at', cities[self.nodeNum], 'on day', currentDay)
                elif reopen_method == 2:
                    if self.asx_del > self.asx[currentDay] - self.asx[currentDay - 1] and (self.asx[currentDay] - self.asx[currentDay - 1]) < TH_reopen_del:
                        reopen[self.nodeNum] = True
                        print('**********', cities[self.nodeNum], 'reopen on day', currentDay, '**********')
            elif self.asx[currentDay] / susArr[self.nodeNum] >= TH_shutdown:
                shutdown[self.nodeNum] = True
                print('**********', cities[self.nodeNum], 'shutdown by', self.mu, 'on day', currentDay, '**********')
                # print('asymptomatic infected population:', round(self.asx[currentDay] * population[self.nodeNum], 2))


# returns 1 if the city is under travel control
def check_city(i):
    ret = 0
    if shutdown[i] and (not reopen[i]):
        ret = 1
    return ret


# returns the traffic control on an edge
def edge_mu(i, j):
    ret = max(check_city(i) * nodesObjArr[i].mu, check_city(j) * nodesObjArr[j].mu)
    return ret


# plot overall curve
def plot_results(currentDay, out_file):
    writer = ExcelWriter(out_file)

    df_sus = pd.DataFrame()
    for i in range(numNodes):
        # for j in range(currentDay):
            # nodesObjArr[i].sus[j] *= population[i]
        df_sus[cities[i]] = nodesObjArr[i].sus
    tot = []
    for j in range(currentDay):
        tot.append(0)
        for i in range(numNodes):
            tot[j] += nodesObjArr[i].sus[j]
    # adding the total number
    df_sus['Total'] = tot
    df_sus.to_excel(writer, sheet_name='Susceptible', index=False)

    df_hid = pd.DataFrame()
    for i in range(numNodes):
        # for j in range(currentDay):
        #     nodesObjArr[i].hid[j] *= population[i]
        df_hid[cities[i]] = nodesObjArr[i].hid
    tot = []
    for j in range(currentDay):
        tot.append(0)
        for i in range(numNodes):
            tot[j] += nodesObjArr[i].hid[j]
    # adding the total number
    df_hid['Total'] = tot
    df_hid.to_excel(writer, sheet_name='Hiding', index=False)

    df_asx = pd.DataFrame()
    for i in range(numNodes):
        # for j in range(currentDay):
        #     nodesObjArr[i].asx[j] *= population[i]
        df_asx[cities[i]] = nodesObjArr[i].asx
    tot = []
    for j in range(currentDay):
        tot.append(0)
        for i in range(numNodes):
            tot[j] += nodesObjArr[i].asx[j]
    # adding the total number
    df_asx['Total'] = tot
    df_asx.to_excel(writer, sheet_name='Asymptomatic', index=False)

    df_inf = pd.DataFrame()
    for i in range(numNodes):
        # for j in range(currentDay):
        #     nodesObjArr[i].inf[j] *= population[i]
        df_inf[cities[i]] = nodesObjArr[i].inf
    tot = []
    for j in range(currentDay):
        tot.append(0)
        for i in range(numNodes):
            tot[j] += nodesObjArr[i].inf[j]
    # adding the total number
    df_inf['Total'] = tot
    df_inf.to_excel(writer, sheet_name='Symptomatic', index=False)

    df_rec = pd.DataFrame()
    for i in range(numNodes):
        # for j in range(currentDay):
        #     nodesObjArr[i].rec[j] *= population[i]
        df_rec[cities[i]] = nodesObjArr[i].rec
    tot = []
    for j in range(currentDay):
        tot.append(0)
        for i in range(numNodes):
            tot[j] += nodesObjArr[i].rec[j]
    # adding the total number
    df_rec['Total'] = tot
    df_rec.to_excel(writer, sheet_name='Recovered', index=False)

    df_cum = pd.DataFrame()
    for i in range(numNodes):
        # for j in range(currentDay):
        # nodesObjArr[i].sus[j] *= population[i]
        df_cum[cities[i]] = nodesObjArr[i].cum
    tot = []
    for j in range(currentDay):
        tot.append(0)
        for i in range(numNodes):
            tot[j] += nodesObjArr[i].cum[j]
    # adding the total number
    df_cum['Total'] = tot
    df_cum.to_excel(writer, sheet_name='Cumulative', index=False)

    writer.save()
    plt.plot(df_asx)
    plt.show()
    plot_states(df_sus, df_hid, df_asx, df_rec, df_cum)


# plot SIR of selected states
def plot_states(df_sus, df_hid, df_asx, df_rec, df_cum):
    states = ['New York', 'Illinois']
    for state in states:
        plt.plot(df_sus[state], label='S')
        plt.plot(df_asx[state], label='I')
        plt.plot(df_rec[state], label='R')
        plt.plot(df_cum[state], label='G')
        plt.plot(df_hid[state], label='H')
        plt.title(state)
        plt.legend()
        plt.show()


def read_input(in_file):
    # df = pd.read_excel(in_file, sheet_name='parameters', index_col=0, header=None).T
    # global mu
    # mu = df['mu'][1]
    # global numNodes
    # numNodes = int(df['size'][1])
    # global beta_asx
    # beta_asx = df['beta_asx'][1]
    # global alpha
    # alpha = df['alpha'][1]
    # global duration
    # duration = int(df['duration'][1])
    # global cut_off
    # cut_off = df['cutoff'][1]
    # global TH_shutdown
    # TH_shutdown = df['TH_shutdown'][1]
    # global reopen_method
    # reopen_method = int(df['reopen_method'][1])
    # global TH_reopen_del
    # TH_reopen_del = df['TH_reopen_del'][1]
    # global TH_reopen
    # TH_reopen = df['TH_reopen'][1]
    # global infect_first
    # infect_first = int(df['infect_first'][1])
    # global infection_rate
    # infection_rate = df['infection_rate'][1]
    # global travel_multiplier
    # travel_multiplier = df['travel_multiplier'][1]
    # global eta
    # eta = df['eta'][1]
    # global release_rate
    # release_rate = df['release_rate'][1]

    df = pd.read_excel(in_file, sheet_name='nodes', usecols=['State', 'Population', 'Beta', 'Gamma', 'Eta'])
    pop = []
    names = []
    b = []
    g = []
    global etas
    for i in df.index:
        names.append(df['State'][i])
        pop.append(df['Population'][i])
        b.append(df['Beta'][i])
        g.append(df['Gamma'][i])
        etas.append(df['Eta'][i])

    df = pd.read_excel(in_file, sheet_name='traffic')
    traffic = []
    for i in range(numNodes):
        traffic.append([])
        for j in range(numNodes):
            traffic[i].append(df.loc[i][j] / 30)
        # print(traffic[i])
    global cities
    cities = names
    global population
    population = pop
    global travel_cap
    travel_cap = traffic
    global betas
    betas = b
    global gammas
    gammas = g


def main():
    read_input('data/interstate_traffic.xlsx')
    # global population_scale
    # population_scale = eta

    # calculate the susceptible population size
    for i in range(numNodes):
        susArr.append(etas[i] * population[i])

    # create an object for each node
    for i in range(numNodes):
        nodesObjArr.append(Node(i, betas[i], beta_asx, gammas[i], mu))
        shutdown.append(False)
        reopen.append(False)

    # calculate the outgoing travel capacity for each city
    city_cap = []
    for i in range(numNodes):
        city_cap.append(0)
        for j in range(numNodes):
            # travel_cap[i][j] *= population_scale
            city_cap[i] += travel_cap[i][j]

    # start simulating
    max_total_asx = 0
    avg_total_asx = 0
    peakDay = 0
    currentDay = 0
    while True:
        for i in nodesObjArr:
            if currentDay == 0:
                i.sus.append(susArr[i.nodeNum])
                i.hid.append(susArr[i.nodeNum] * alpha / (1 - alpha))
                susArr[i.nodeNum] *= (1 - alpha)
                i.cum.append(0)
                i.inf.append(0)
                i.rec.append(0)
                i.asx.append(0)
            else:
                i.sus.append(i.sus[currentDay - 1])
                i.cum.append(i.cum[currentDay - 1])
                i.hid.append(i.hid[currentDay - 1])
                i.inf.append(i.inf[currentDay - 1])
                i.rec.append(i.rec[currentDay - 1])
                i.asx.append(i.asx[currentDay - 1])

            i.susLeav.append([])
            i.infLeav.append([])
            i.asxLeav.append([])
            for j in range(numNodes):
                i.susLeav[currentDay].append(0)
                i.infLeav[currentDay].append(0)
                i.asxLeav[currentDay].append(0)

            i.susLeavTot.append(0)
            i.asxLeavTot.append(0)

        if currentDay == 0:
            # print('node ', infectedFirst, ' is infected first')
            nodesObjArr[infect_first].infect(currentDay, infection_rate)

        # update stuff Day 1 onwards
        if currentDay > 0:

            for i in range(numNodes):
                # update travel control
                nodesObjArr[i].travel_control(currentDay)

            for i in range(numNodes):
                if reopen[i]:
                    nodesObjArr[i].release_hiding(currentDay)

            for i in range(numNodes):

                # update Leaving number
                nodesObjArr[i].travel_out(currentDay, travel_cap[i])

                # adjust leaving and entering number due to infection on the way
                # nodesObjArr[i].travel_infection(currentDay, travel_cap[i])

            for i in range(numNodes):
                # update SIR
                nodesObjArr[i].updateNodeSIR(currentDay)

        # for i in range(numNodes):
        #     # update travel control
        #     nodesObjArr[i].travel_control(currentDay)

        # print('\nDay', currentDay)
        totalasx = 0
        totalsus = 0
        for i in range(numNodes):
            totalasx += nodesObjArr[i].asx[currentDay]
            totalsus += nodesObjArr[i].sus[currentDay]
        # print("total inf ",round(totalinf,2))
        # print("total sus ", round(totalsus, 2))
        avg_total_asx += totalasx
        if max_total_asx < totalasx:
            peakDay = currentDay
            max_total_asx = totalasx

        popu = 0
        for i in range(numNodes):
            popu += (nodesObjArr[i].sus[currentDay] + nodesObjArr[i].hid[currentDay] + nodesObjArr[i].asx[currentDay] + nodesObjArr[i].inf[currentDay] + nodesObjArr[i].rec[currentDay])
        # print('total population :', round(popu, 4))
        # for i in range(numNodes):
        #     print(cities[i], ':', round((nodesObjArr[i].sus[currentDay] + nodesObjArr[i].inf[currentDay]
        #                            + nodesObjArr[i].rec[currentDay]) / popu * 100, 2), '% ', end='')
        # print()
        # print('total asx :', totalasx)
        currentDay += 1
        if totalasx < cut_off and currentDay > duration:
        # if currentDay > 100:
            break

    print()
    print()
    print('peak asx is ', round(max_total_asx, 4), ' at day ', peakDay)
    print('total population ', round(popu, 4))
    total_S = 0
    for i in range(numNodes):
        total_S += nodesObjArr[i].sus[currentDay - 1]
    print(round(total_S / popu * 100, 4), '% susceptible left')
    print('avg asx is ', round(avg_total_asx / currentDay, 4))
    print('total day =', currentDay)

    # output simulation results
    plot_results(currentDay, 'data/fractional simulation results.xlsx')


if __name__ == "__main__":
    main()
