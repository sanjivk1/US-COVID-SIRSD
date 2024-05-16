import pandas as pd
import os


def report_para():
    sd_init_dir = '50Counties/init_only_2020-08-31'
    sir_dir = '50Counties/SIR'
    sd_full_dir = 'JHU50/combined2W_2020-08-31'
    states_init = [
        'AZ-Maricopa', 'CA-Los Angeles', 'CA-Orange', 'CA-Riverside', 'CA-San Bernardino', 'CA-San Diego', 'CO-Adams',
        'CO-Arapahoe', 'CO-Denver', 'CT-Fairfield', 'CT-Hartford', 'CT-New Haven', 'DC-District of Columbia',
        'DE-New Castle', 'DE-Sussex', 'FL-Broward', 'FL-Miami-Dade', 'FL-Palm Beach', 'GA-DeKalb', 'GA-Fulton',
        'GA-Gwinnett', 'IA-Polk', 'IL-Cook', 'IL-DuPage', 'IL-Kane', 'IL-Lake', 'IL-Will', 'IN-Lake', 'IN-Marion',
        'LA-East Baton Rouge', 'LA-Jefferson', 'LA-Orleans', 'MA-Bristol', 'MA-Essex', 'MA-Hampden', 'MA-Middlesex',
        'MA-Norfolk', 'MA-Plymouth', 'MA-Suffolk', 'MA-Worcester', 'MD-Anne Arundel', 'MD-Baltimore',
        'MD-Baltimore City', 'MD-Montgomery', 'MD-Prince George\'s', 'MI-Kent', 'MI-Macomb', 'MI-Oakland', 'MI-Wayne',
        'MN-Hennepin', 'MO-St. Louis', 'NJ-Bergen', 'NJ-Burlington', 'NJ-Camden', 'NJ-Essex', 'NJ-Hudson', 'NJ-Mercer',
        'NJ-Middlesex', 'NJ-Monmouth', 'NJ-Morris', 'NJ-Ocean', 'NJ-Passaic', 'NJ-Somerset', 'NJ-Union', 'NV-Clark',
        'NY-Bronx', 'NY-Dutchess', 'NY-Erie', 'NY-Kings', 'NY-Nassau', 'NY-New York', 'NY-Orange', 'NY-Queens',
        'NY-Richmond', 'NY-Rockland', 'NY-Suffolk', 'NY-Westchester', 'OH-Cuyahoga', 'OH-Franklin', 'PA-Berks',
        'PA-Bucks', 'PA-Delaware', 'PA-Lehigh', 'PA-Luzerne', 'PA-Montgomery', 'PA-Northampton', 'PA-Philadelphia',
        'RI-Providence', 'SD-Minnehaha', 'TN-Davidson', 'TN-Shelby', 'TX-Dallas', 'TX-Harris', 'TX-Tarrant',
        'UT-Salt Lake', 'VA-Fairfax', 'VA-Prince William', 'WA-King', 'WA-Snohomish', 'WI-Milwaukee'
    ]
    states_full = [
        'AL-Jefferson', 'AL-Mobile', 'AZ-Maricopa', 'AZ-Pima', 'AZ-Yuma', 'CA-Alameda', 'CA-Contra Costa', 'CA-Fresno',
        'CA-Kern', 'CA-Los Angeles', 'CA-Orange', 'CA-Riverside', 'CA-Sacramento', 'CA-San Bernardino', 'CA-San Diego',
        'CA-San Joaquin', 'CA-Santa Clara', 'CA-Stanislaus', 'CA-Tulare', 'CO-Adams', 'CO-Arapahoe', 'CO-Denver',
        'CT-Fairfield', 'CT-Hartford', 'CT-New Haven', 'DE-New Castle', 'DE-Sussex', 'DC-District of Columbia',
        'FL-Broward', 'FL-Duval', 'FL-Hillsborough', 'FL-Lee', 'FL-Miami-Dade', 'FL-Orange', 'FL-Palm Beach',
        'FL-Pinellas', 'FL-Polk', 'GA-Cobb', 'GA-DeKalb', 'GA-Fulton', 'GA-Gwinnett', 'IL-Cook', 'IL-DuPage', 'IL-Kane',
        'IL-Lake', 'IL-Will', 'IN-Lake', 'IN-Marion', 'IA-Polk', 'KY-Jefferson', 'LA-East Baton Rouge', 'LA-Jefferson',
        'LA-Orleans', 'MD-Anne Arundel', 'MD-Baltimore', 'MD-Baltimore City', 'MD-Montgomery', 'MD-Prince George\'s',
        'MA-Bristol', 'MA-Essex', 'MA-Hampden', 'MA-Middlesex', 'MA-Norfolk', 'MA-Plymouth', 'MA-Suffolk',
        'MA-Worcester', 'MI-Kent', 'MI-Macomb', 'MI-Oakland', 'MI-Wayne', 'MN-Hennepin', 'MO-St. Louis', 'NE-Douglas',
        'NV-Clark', 'NJ-Bergen', 'NJ-Burlington', 'NJ-Camden', 'NJ-Essex', 'NJ-Hudson', 'NJ-Mercer', 'NJ-Middlesex',
        'NJ-Monmouth', 'NJ-Morris', 'NJ-Ocean', 'NJ-Passaic', 'NJ-Somerset', 'NJ-Union', 'NY-Bronx', 'NY-Dutchess',
        'NY-Erie', 'NY-Kings', 'NY-Nassau', 'NY-New York', 'NY-Orange', 'NY-Queens', 'NY-Richmond', 'NY-Rockland',
        'NY-Suffolk', 'NY-Westchester', 'NC-Mecklenburg', 'NC-Wake', 'OH-Cuyahoga', 'OH-Franklin', 'OK-Oklahoma',
        'OK-Tulsa', 'PA-Berks', 'PA-Bucks', 'PA-Delaware', 'PA-Lehigh', 'PA-Luzerne', 'PA-Montgomery', 'PA-Northampton',
        'PA-Philadelphia', 'RI-Providence', 'SC-Charleston', 'SC-Greenville', 'SD-Minnehaha', 'TN-Davidson',
        'TN-Shelby', 'TX-Bexar', 'TX-Cameron', 'TX-Dallas', 'TX-El Paso', 'TX-Fort Bend', 'TX-Harris', 'TX-Hidalgo',
        'TX-Nueces', 'TX-Tarrant', 'TX-Travis', 'UT-Salt Lake', 'VA-Fairfax', 'VA-Prince William', 'WA-King',
        'WA-Snohomish', 'WI-Milwaukee', ]
    print('length of states_init:', len(states_init))
    print('length of states_full:', len(states_full))
    if not os.path.exists('para report'):
        os.makedirs('para report')

    df_sir = pd.DataFrame(columns=['County', 'beta', 'gamma', 'eta'])
    df_sd_init = pd.DataFrame(columns=['County', 'beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'eta', 'c1'])
    df_sd_full = pd.DataFrame()
    for state in states_init:
        para_sir = pd.read_csv(f'{sir_dir}/{state}/para.csv')
        para_sir = [state] + list(para_sir[['beta', 'gamma', 'eta']].iloc[0])
        df_sir.loc[len(df_sir)] = para_sir

        para_sd_init = pd.read_csv(f'{sd_init_dir}/{state}/para.csv')
        para_sd_init = [state] + list(para_sd_init[['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'eta', 'c1']].iloc[0])
        df_sd_init.loc[len(df_sir)] = para_sd_init
    print(df_sir.head(20))
    print(df_sd_init.head(20))
    df_sir.to_csv('para report/SIR.csv', index=False)
    df_sd_init.to_csv('para report/SD_init.csv', index=False)
    return


def main():
    report_para()


if __name__ == '__main__':
    main()
