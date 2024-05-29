import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def AIC_compare():
    # folder = '50Counties/comparison'
    df = pd.read_csv('50Counties/comparison/RMSE.csv')[['state', 'RMSE_G', 'RMSE_G(SIR)']]
    para_SIR, para_SD = 3, 8
    out_df = pd.DataFrame(columns=['state', 'AIC_SD', 'AIC_SIR', 'MSE_SD', 'MSE_SIR', 'sample size'])
    for index, row in df.iterrows():
        state, RMSE_SD, RMSE_SIR = row
        n = pd.read_csv(f'50Counties/SIR/{state}/sim.csv').shape[1] - 1
        # print(n, state, RMSE_SD, RMSE_SIR)
        MSE_SD = RMSE_SD ** 2
        MSE_SIR = RMSE_SIR ** 2
        AIC_SIR = n * np.log(MSE_SIR) + 2 * para_SIR
        AIC_SD = n * np.log(MSE_SD) + 2 * para_SD
        out_df.loc[len(out_df)] = [state, AIC_SD, AIC_SIR, MSE_SD, MSE_SIR, n]
    # print(out_df.columns)
    out_df.to_csv('50Counties/comparison/AIC.csv', index=False)


def AIC_compare_SEIR_SD():
    states = ['AZ-Maricopa',
              'CA-Los Angeles',
              'CA-Orange',
              'CA-Riverside',
              'CA-San Bernardino',
              'CA-San Diego',
              'CO-Adams',
              'CO-Arapahoe',
              'CO-Denver',
              'CT-Fairfield',
              'CT-Hartford',
              'CT-New Haven',
              'DC-District of Columbia',
              'DE-New Castle',
              'DE-Sussex',
              'FL-Broward',
              'FL-Miami-Dade',
              'FL-Palm Beach',
              'GA-DeKalb',
              'GA-Fulton',
              'GA-Gwinnett',
              'IA-Polk',
              'IL-Cook',
              'IL-DuPage',
              'IL-Kane',
              'IL-Lake',
              'IL-Will',
              'IN-Lake',
              'IN-Marion',
              'LA-East Baton Rouge',
              'LA-Jefferson',
              'LA-Orleans',
              'MA-Bristol',
              'MA-Essex',
              'MA-Hampden',
              'MA-Middlesex',
              'MA-Norfolk',
              'MA-Plymouth',
              'MA-Suffolk',
              'MA-Worcester',
              'MD-Anne Arundel',
              'MD-Baltimore',
              'MD-Baltimore City',
              'MD-Montgomery',
              'MD-Prince George\'s',
              'MI-Kent',
              'MI-Macomb',
              'MI-Oakland',
              'MI-Wayne',
              'MN-Hennepin',
              'MO-St. Louis',
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
              'NV-Clark',
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
              'OH-Cuyahoga',
              'OH-Franklin',
              'PA-Berks',
              'PA-Bucks',
              'PA-Delaware',
              'PA-Lehigh',
              'PA-Luzerne',
              'PA-Montgomery',
              'PA-Northampton',
              'PA-Philadelphia',
              'RI-Providence',
              'SD-Minnehaha',
              'TN-Davidson',
              'TN-Shelby',
              'TX-Dallas',
              'TX-Harris',
              'TX-Tarrant',
              'UT-Salt Lake',
              'VA-Fairfax',
              'VA-Prince William',
              'WA-King',
              'WA-Snohomish',
              'WI-Milwaukee']
    # folder = '50Counties/comparison'
    df_SD = pd.read_csv('50Counties/comparison/RMSE.csv')[['state', 'RMSE_G']]
    para_SEIR_SD, para_SD = 5, 8
    out_df = pd.DataFrame(columns=['state', 'AIC_SD', 'AIC_SEIR_SD', 'MSE_SD', 'MSE_SEIR_SD', 'sample size'])
    for index, row in df_SD.iterrows():
        state, RMSE_SD = row
        n = pd.read_csv(f'50Counties/SEIR_SD_2020-05-15/{state}/sim.csv').shape[1] - 1
        # print(n, state, RMSE_SD, RMSE_SIR)
        MSE_SD = RMSE_SD ** 2
        RMSE_SEIR_SD = pd.read_csv(f'50Counties/SEIR_SD_2020-05-15/{state}/para.csv')[['RMSE']].iloc[0, 0]
        MSE_SEIR_SD = RMSE_SEIR_SD ** 2
        AIC_SEIR_SD = n * np.log(MSE_SEIR_SD) + 2 * para_SEIR_SD
        AIC_SD = n * np.log(MSE_SD) + 2 * para_SD
        out_df.loc[len(out_df)] = [state, AIC_SD, AIC_SEIR_SD, MSE_SD, MSE_SEIR_SD, n]
    # print(out_df.columns)
    out_df.to_csv('50Counties/comparison/AIC_SEIR_SD.csv', index=False)


def AIC_compare_SEIR():
    states = ['AZ-Maricopa',
              'CA-Los Angeles',
              'CA-Orange',
              'CA-Riverside',
              'CA-San Bernardino',
              'CA-San Diego',
              'CO-Adams',
              'CO-Arapahoe',
              'CO-Denver',
              'CT-Fairfield',
              'CT-Hartford',
              'CT-New Haven',
              'DC-District of Columbia',
              'DE-New Castle',
              'DE-Sussex',
              'FL-Broward',
              'FL-Miami-Dade',
              'FL-Palm Beach',
              'GA-DeKalb',
              'GA-Fulton',
              'GA-Gwinnett',
              'IA-Polk',
              'IL-Cook',
              'IL-DuPage',
              'IL-Kane',
              'IL-Lake',
              'IL-Will',
              'IN-Lake',
              'IN-Marion',
              'LA-East Baton Rouge',
              'LA-Jefferson',
              'LA-Orleans',
              'MA-Bristol',
              'MA-Essex',
              'MA-Hampden',
              'MA-Middlesex',
              'MA-Norfolk',
              'MA-Plymouth',
              'MA-Suffolk',
              'MA-Worcester',
              'MD-Anne Arundel',
              'MD-Baltimore',
              'MD-Baltimore City',
              'MD-Montgomery',
              'MD-Prince George\'s',
              'MI-Kent',
              'MI-Macomb',
              'MI-Oakland',
              'MI-Wayne',
              'MN-Hennepin',
              'MO-St. Louis',
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
              'NV-Clark',
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
              'OH-Cuyahoga',
              'OH-Franklin',
              'PA-Berks',
              'PA-Bucks',
              'PA-Delaware',
              'PA-Lehigh',
              'PA-Luzerne',
              'PA-Montgomery',
              'PA-Northampton',
              'PA-Philadelphia',
              'RI-Providence',
              'SD-Minnehaha',
              'TN-Davidson',
              'TN-Shelby',
              'TX-Dallas',
              'TX-Harris',
              'TX-Tarrant',
              'UT-Salt Lake',
              'VA-Fairfax',
              'VA-Prince William',
              'WA-King',
              'WA-Snohomish',
              'WI-Milwaukee']
    # folder = '50Counties/comparison'
    df_SD = pd.read_csv('50Counties/comparison/RMSE.csv')[['state', 'RMSE_G']]
    para_SEIR, para_SD = 4, 8
    out_df = pd.DataFrame(columns=['state', 'AIC_SD', 'AIC_SEIR', 'MSE_SD', 'MSE_SEIR', 'sample size'])
    for index, row in df_SD.iterrows():
        state, RMSE_SD = row
        n = pd.read_csv(f'50Counties/SEIR_2020-05-15/{state}/sim.csv').shape[1] - 1
        # print(n, state, RMSE_SD, RMSE_SIR)
        MSE_SD = RMSE_SD ** 2
        RMSE_SEIR = pd.read_csv(f'50Counties/SEIR_2020-05-15/{state}/para.csv')[['RMSE']].iloc[0, 0]
        MSE_SEIR = RMSE_SEIR ** 2
        AIC_SEIR = n * np.log(MSE_SEIR) + 2 * para_SEIR
        AIC_SD = n * np.log(MSE_SD) + 2 * para_SD
        out_df.loc[len(out_df)] = [state, AIC_SD, AIC_SEIR, MSE_SD, MSE_SEIR, n]
    # print(out_df.columns)
    out_df.to_csv('50Counties/comparison/AIC_SEIR.csv', index=False)


def AIC_boxplot():
    df = pd.read_csv('50Counties/comparison/AIC.csv')
    cmap = ['red' if AIC_SD >= AIC_SIR else 'green' for AIC_SD, AIC_SIR in zip(df['AIC_SD'], df['AIC_SIR'])]
    AIC_ratio = list(df['AIC_SD'] / df['AIC_SIR'])
    fig_box = plt.figure(figsize=(3, 6))
    fig_box.suptitle('Ratios of AIC(SIR-SD/SIR)')
    ax = fig_box.add_subplot()
    ax.boxplot(AIC_ratio,
               showfliers=False
               )
    ax.scatter(np.random.normal(1, 0.04, len(AIC_ratio)), AIC_ratio, color=cmap)
    ax.set_xticklabels([])
    # fig_box.savefig('50Counties/comparison/AIC_box.png', bbox_inches="tight")
    plt.show()


def AIC_scatter():
    df = pd.read_csv('50Counties/comparison/AIC.csv')
    cmap = ['red' if AIC_SD >= AIC_SIR else 'green' for AIC_SD, AIC_SIR in zip(df['AIC_SD'], df['AIC_SIR'])]
    AIC_ratio = list(df['AIC_SD'] / df['AIC_SIR'])
    fig = plt.figure()
    fig.suptitle('AIC')
    ax = fig.add_subplot()
    ax.scatter(df['AIC_SD'], df['AIC_SIR'], color=cmap)
    ax.set_xlabel('SIR-SD')
    ax.set_ylabel('SIR')
    line_range = [max(min(df['AIC_SD']), min(df['AIC_SIR'])), min(max(df['AIC_SD']), max(df['AIC_SIR']))]
    ax.plot(line_range, line_range, linestyle=':', color='grey')
    fig.savefig('50Counties/comparison/AIC_scatter.png', bbox_inches="tight")


def AIC_hist():
    df = pd.read_csv('50Counties/comparison/AIC.csv')
    cmap = ['red' if AIC_SD >= AIC_SIR else 'green' for AIC_SD, AIC_SIR in zip(df['AIC_SD'], df['AIC_SIR'])]
    AIC_ratio = list(df['AIC_SD'] / df['AIC_SIR'])
    fig = plt.figure()
    fig.suptitle('Ratio of AIC(SIR-SD/SIR)')
    ax = fig.add_subplot()
    ax.hist(AIC_ratio, edgecolor='black')
    ax.axvline(np.mean(AIC_ratio), linestyle='dashed', color='tab:grey',
               label=f'AVG={round(np.mean(AIC_ratio), 3)}')
    ax.axvline(1, color='red')
    ax.legend()
    fig.savefig('50Counties/comparison/AIC_hist.png', bbox_inches="tight")


if __name__ == '__main__':
    # AIC_compare()
    # AIC_boxplot()
    # AIC_scatter()
    # AIC_hist()
    AIC_compare_SEIR_SD()
    # AIC_compare_SEIR()
