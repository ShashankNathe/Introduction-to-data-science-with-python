#Question 1

import pandas as pd
import numpy as np
import scipy.stats as stats
import re

cities = pd.read_html("assets/wikipedia_data.html")[1]
cities = cities.iloc[:-1, [0, 3, 5, 6, 7, 8]]
cities.rename(columns={"Population (2016 est.)[8]": "Population"},inplace=True)
cities['NFL'] = cities['NFL'].str.replace(r"\[.*\]", "")
cities['MLB'] = cities['MLB'].str.replace(r"\[.*\]", "")
cities['NBA'] = cities['NBA'].str.replace(r"\[.*\]", "")
cities['NHL'] = cities['NHL'].str.replace(r"\[.*\]", "")

Big4 = 'NHL'


def nhl_correlation():
    # YOUR CODE HERE
    # raise NotImplementedError()

    team = cities[Big4].str.extract('([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team['Metropolitan area'] = cities['Metropolitan area']
    team = pd.melt(team, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("", np.nan).replace("—",np.nan).dropna().reset_index().rename(columns={"value": "team"})
    team = pd.merge(team, cities, how='left', on='Metropolitan area').iloc[:, 1:4]
    team = team.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team['team'] = team['team'].str.replace('[\w.]*\ ', '')

    _df = pd.read_csv("assets/" + str.lower(Big4) + ".csv")
    _df = _df[_df['year'] == 2018]
    _df['team'] = _df['team'].str.replace(r'\*', "")
    _df = _df[['team', 'W', 'L']]

    dropList = []
    for i in range(_df.shape[0]):
        row = _df.iloc[i]
        if row['team'] == row['W'] and row['L'] == row['W']:
            dropList.append(i)
    _df = _df.drop(dropList)

    _df['team'] = _df['team'].str.replace('[\w.]* ', '')
    _df = _df.astype({'team': str, 'W': int, 'L': int})
    _df['W/L%'] = _df['W'] / (_df['W'] + _df['L'])

    merge = pd.merge(team, _df, 'outer', on='team')
    merge = merge.groupby('Metropolitan area').agg({'W/L%': np.nanmean, 'Population': np.nanmean})

    population_by_region = merge['Population']
    win_loss_by_region = merge['W/L%']

    assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]








#Question 2

import pandas as pd
import numpy as np
import scipy.stats as stats
import re

cities = pd.read_html("assets/wikipedia_data.html")[1]
cities = cities.iloc[:-1, [0, 3, 5, 6, 7, 8]]
cities.rename(columns={"Population (2016 est.)[8]": "Population"},inplace=True)
cities['NFL'] = cities['NFL'].str.replace(r"\[.*\]", "")
cities['MLB'] = cities['MLB'].str.replace(r"\[.*\]", "")
cities['NBA'] = cities['NBA'].str.replace(r"\[.*\]", "")
cities['NHL'] = cities['NHL'].str.replace(r"\[.*\]", "")

Big4 = 'NBA'


def nba_correlation():
    team = cities[Big4].str.extract('([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team['Metropolitan area'] = cities['Metropolitan area']
    team = pd.melt(team, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("", np.nan).replace("—",np.nan).dropna().reset_index().rename(columns={"value": "team"})
    team = pd.merge(team, cities, how='left', on='Metropolitan area').iloc[:, 1:4]
    team = team.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team['team'] = team['team'].str.replace('[\w.]*\ ', '')

    _df = pd.read_csv("assets/" + str.lower(Big4) + ".csv")
    _df = _df[_df['year'] == 2018]
    _df['team'] = _df['team'].str.replace(r'[\*]', "")
    _df['team'] = _df['team'].str.replace(r'\(\d*\)', "")
    _df['team'] = _df['team'].str.replace(r'[\xa0]', "")
    _df = _df[['team', 'W/L%']]
    _df['team'] = _df['team'].str.replace('[\w.]* ', '')
    _df = _df.astype({'team': str, 'W/L%': float})

    merge = pd.merge(team, _df, 'outer', on='team')
    merge = merge.groupby('Metropolitan area').agg({'W/L%': np.nanmean, 'Population': np.nanmean})

    population_by_region = merge['Population']
    win_loss_by_region = merge['W/L%']

    assert len(population_by_region) == len(win_loss_by_region), "Q2: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q2: There should be 28 teams being analysed for NBA"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]







#Question 3



import pandas as pd
import numpy as np
import scipy.stats as stats
import re

cities = pd.read_html("assets/wikipedia_data.html")[1]
cities = cities.iloc[:-1, [0, 3, 5, 6, 7, 8]]
cities.rename(columns={"Population (2016 est.)[8]": "Population"},inplace=True)
cities['NFL'] = cities['NFL'].str.replace(r"\[.*\]", "")
cities['MLB'] = cities['MLB'].str.replace(r"\[.*\]", "")
cities['NBA'] = cities['NBA'].str.replace(r"\[.*\]", "")
cities['NHL'] = cities['NHL'].str.replace(r"\[.*\]", "")

Big4 = 'MLB'


def mlb_correlation():
    team = cities[Big4].str.extract('([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team['Metropolitan area'] = cities['Metropolitan area']
    team = pd.melt(team, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("", np.nan).replace("—",np.nan).dropna().reset_index().rename(columns={"value": "team"})
    team = pd.merge(team, cities, how='left', on='Metropolitan area').iloc[:, 1:4]
    team = team.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team['team'] = team['team'].str.replace('\ Sox', 'Sox')
    team['team'] = team['team'].str.replace('[\w.]*\ ', '')

    _df = pd.read_csv("assets/" + str.lower(Big4) + ".csv")
    _df = _df[_df['year'] == 2018]
    _df['team'] = _df['team'].str.replace(r'[\*]', "")
    _df['team'] = _df['team'].str.replace(r'\(\d*\)', "")
    _df['team'] = _df['team'].str.replace(r'[\xa0]', "")
    _df = _df[['team', 'W-L%']]
    _df.rename(columns={"W-L%": "W/L%"}, inplace=True)
    _df['team'] = _df['team'].str.replace('\ Sox', 'Sox')
    _df['team'] = _df['team'].str.replace('[\w.]* ', '')
    _df = _df.astype({'team': str, 'W/L%': float})

    merge = pd.merge(team, _df, 'outer', on='team')
    merge = merge.groupby('Metropolitan area').agg({'W/L%': np.nanmean, 'Population': np.nanmean})

    population_by_region = merge['Population']
    win_loss_by_region = merge['W/L%']

    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 26, "Q3: There should be 26 teams being analysed for MLB"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]






#Question 4


import pandas as pd
import numpy as np
import scipy.stats as stats
import re

cities = pd.read_html("assets/wikipedia_data.html")[1]
cities = cities.iloc[:-1, [0, 3, 5, 6, 7, 8]]
cities.rename(columns={"Population (2016 est.)[8]": "Population"},inplace=True)
cities['NFL'] = cities['NFL'].str.replace(r"\[.*\]", "")
cities['MLB'] = cities['MLB'].str.replace(r"\[.*\]", "")
cities['NBA'] = cities['NBA'].str.replace(r"\[.*\]", "")
cities['NHL'] = cities['NHL'].str.replace(r"\[.*\]", "")

Big4 = 'NFL'


def nfl_correlation():
    team = cities[Big4].str.extract('([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team['Metropolitan area'] = cities['Metropolitan area']
    team = pd.melt(team, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("", np.nan).replace("—",np.nan).dropna().reset_index().rename(columns={"value": "team"})
    team = pd.merge(team, cities, how='left', on='Metropolitan area').iloc[:, 1:4]
    team = team.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team['team'] = team['team'].str.replace('[\w.]*\ ', '')

    _df = pd.read_csv("assets/" + str.lower(Big4) + ".csv")
    _df = _df[_df['year'] == 2018]
    _df['team'] = _df['team'].str.replace(r'[\*]', "")
    _df['team'] = _df['team'].str.replace(r'\(\d*\)', "")
    _df['team'] = _df['team'].str.replace(r'[\xa0]', "")
    _df = _df[['team', 'W-L%']]
    _df.rename(columns={"W-L%": "W/L%"}, inplace=True)
    dropList = []
    for i in range(_df.shape[0]):
        row = _df.iloc[i]
        if row['team'] == row['W/L%']:
            dropList.append(i)
    _df = _df.drop(dropList)

    _df['team'] = _df['team'].str.replace('[\w.]* ', '')
    _df['team'] = _df['team'].str.replace('+', '')
    _df = _df.astype({'team': str, 'W/L%': float})

    merge = pd.merge(team, _df, 'outer', on='team')
    merge = merge.groupby('Metropolitan area').agg({'W/L%': np.nanmean, 'Population': np.nanmean})

    population_by_region = merge['Population']
    win_loss_by_region = merge['W/L%']

    assert len(population_by_region) == len(win_loss_by_region), "Q4: Your lists must be the same length"
    assert len(population_by_region) == 29, "Q4: There should be 29 teams being analysed for NFL"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]






#Question 5


import pandas as pd
import numpy as np
import scipy.stats as stats
import re

cities = pd.read_html("assets/wikipedia_data.html")[1]
cities = cities.iloc[:-1, [0, 3, 5, 6, 7, 8]]
cities.rename(columns={"Population (2016 est.)[8]": "Population"},inplace=True)
cities['NFL'] = cities['NFL'].str.replace(r"\[.*\]", "")
cities['MLB'] = cities['MLB'].str.replace(r"\[.*\]", "")
cities['NBA'] = cities['NBA'].str.replace(r"\[.*\]", "")
cities['NHL'] = cities['NHL'].str.replace(r"\[.*\]", "")


def nhl_df():
    Big4 = 'NHL'
    team = cities[Big4].str.extract('([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team['Metropolitan area'] = cities['Metropolitan area']
    team = pd.melt(team, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("", np.nan).replace("—",np.nan).dropna().reset_index().rename(columns={"value": "team"})
    team = pd.merge(team, cities, how='left', on='Metropolitan area').iloc[:, 1:4]
    team = team.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team['team'] = team['team'].str.replace('[\w.]*\ ', '')

    _df = pd.read_csv("assets/" + str.lower(Big4) + ".csv")
    _df = _df[_df['year'] == 2018]
    _df['team'] = _df['team'].str.replace(r'\*', "")
    _df = _df[['team', 'W', 'L']]

    dropList = []
    for i in range(_df.shape[0]):
        row = _df.iloc[i]
        if row['team'] == row['W'] and row['L'] == row['W']:
            dropList.append(i)
    _df = _df.drop(dropList)

    _df['team'] = _df['team'].str.replace('[\w.]* ', '')
    _df = _df.astype({'team': str, 'W': int, 'L': int})
    _df['W/L%'] = _df['W'] / (_df['W'] + _df['L'])

    merge = pd.merge(team, _df, 'inner', on='team')
    merge = merge.groupby('Metropolitan area').agg({'W/L%': np.nanmean, 'Population': np.nanmean})

    return merge[['W/L%']]


def nba_df():
    Big4 = 'NBA'
    team = cities[Big4].str.extract('([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team['Metropolitan area'] = cities['Metropolitan area']
    team = pd.melt(team, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("", np.nan).replace("—",np.nan).dropna().reset_index().rename(columns={"value": "team"})
    team = pd.merge(team, cities, how='left', on='Metropolitan area').iloc[:, 1:4]
    team = team.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team['team'] = team['team'].str.replace('[\w.]*\ ', '')

    _df = pd.read_csv("assets/" + str.lower(Big4) + ".csv")
    _df = _df[_df['year'] == 2018]
    _df['team'] = _df['team'].str.replace(r'[\*]', "")
    _df['team'] = _df['team'].str.replace(r'\(\d*\)', "")
    _df['team'] = _df['team'].str.replace(r'[\xa0]', "")
    _df = _df[['team', 'W/L%']]
    _df['team'] = _df['team'].str.replace('[\w.]* ', '')
    _df = _df.astype({'team': str, 'W/L%': float})

    merge = pd.merge(team, _df, 'outer', on='team')
    merge = merge.groupby('Metropolitan area').agg({'W/L%': np.nanmean, 'Population': np.nanmean})
    return merge[['W/L%']]


def mlb_df():
    Big4 = 'MLB'
    team = cities[Big4].str.extract('([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team['Metropolitan area'] = cities['Metropolitan area']
    team = pd.melt(team, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("", np.nan).replace("—",np.nan).dropna().reset_index().rename(columns={"value": "team"})
    team = pd.merge(team, cities, how='left', on='Metropolitan area').iloc[:, 1:4]
    team = team.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team['team'] = team['team'].str.replace('\ Sox', 'Sox')
    team['team'] = team['team'].str.replace('[\w.]*\ ', '')

    _df = pd.read_csv("assets/" + str.lower(Big4) + ".csv")
    _df = _df[_df['year'] == 2018]
    _df['team'] = _df['team'].str.replace(r'[\*]', "")
    _df['team'] = _df['team'].str.replace(r'\(\d*\)', "")
    _df['team'] = _df['team'].str.replace(r'[\xa0]', "")
    _df = _df[['team', 'W-L%']]
    _df.rename(columns={"W-L%": "W/L%"}, inplace=True)
    _df['team'] = _df['team'].str.replace('\ Sox', 'Sox')
    _df['team'] = _df['team'].str.replace('[\w.]* ', '')
    _df = _df.astype({'team': str, 'W/L%': float})

    merge = pd.merge(team, _df, 'outer', on='team')
    merge = merge.groupby('Metropolitan area').agg({'W/L%': np.nanmean, 'Population': np.nanmean})

    return merge[['W/L%']]


def nfl_df():
    Big4 = 'NFL'
    team = cities[Big4].str.extract('([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team['Metropolitan area'] = cities['Metropolitan area']
    team = pd.melt(team, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("", np.nan).replace("—",np.nan).dropna().reset_index().rename(columns={"value": "team"})
    team = pd.merge(team, cities, how='left', on='Metropolitan area').iloc[:, 1:4]
    team = team.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team['team'] = team['team'].str.replace('[\w.]*\ ', '')

    _df = pd.read_csv("assets/" + str.lower(Big4) + ".csv")
    _df = _df[_df['year'] == 2018]
    _df['team'] = _df['team'].str.replace(r'[\*]', "")
    _df['team'] = _df['team'].str.replace(r'\(\d*\)', "")
    _df['team'] = _df['team'].str.replace(r'[\xa0]', "")
    _df = _df[['team', 'W-L%']]
    _df.rename(columns={"W-L%": "W/L%"}, inplace=True)
    dropList = []
    for i in range(_df.shape[0]):
        row = _df.iloc[i]
        if row['team'] == row['W/L%']:
            dropList.append(i)
    _df = _df.drop(dropList)

    _df['team'] = _df['team'].str.replace('[\w.]* ', '')
    _df['team'] = _df['team'].str.replace('+', '')
    _df = _df.astype({'team': str, 'W/L%': float})

    merge = pd.merge(team, _df, 'outer', on='team')
    merge = merge.groupby('Metropolitan area').agg({'W/L%': np.nanmean, 'Population': np.nanmean})

    return merge[['W/L%']]


def create_df(sport):
    if sport == 'NFL':
        return nfl_df()
    elif sport == 'NBA':
        return nba_df()
    elif sport == 'NHL':
        return nhl_df()
    elif sport == 'MLB':
        return mlb_df()
    else:
        print("ERROR with intput!")


def sports_team_performance():
    sports = ['NFL', 'NBA', 'NHL', 'MLB']
    p_values = pd.DataFrame({k: np.nan for k in sports}, index=sports)

    for i in sports:
        for j in sports:
            if i != j:
                merge = pd.merge(create_df(i), create_df(j), 'inner', on=['Metropolitan area'])
                p_values.loc[i, j] = stats.ttest_rel(merge['W/L%_x'], merge['W/L%_y'])[1]

    assert abs(p_values.loc["NBA", "NHL"] - 0.02) <= 1e-2, "The NBA-NHL p-value should be around 0.02"
    assert abs(p_values.loc["MLB", "NFL"] - 0.80) <= 1e-2, "The MLB-NFL p-value should be around 0.80"
    return p_values