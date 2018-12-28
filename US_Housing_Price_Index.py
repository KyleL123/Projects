import quandl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from statistics import mean
from sklearn import svm, preprocessing, cross_validation, model_selection

style.use('fivethirtyeight')

pd.set_option('display.max_columns',100)

quandl.ApiConfig.api_key = '****************'

def state_list():
    fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fifty_states[0][1][1:]

def grab_initial_state_data():
    states = state_list()
    main_df = pd.DataFrame()

    for abbv in states:
        query = ("FMAC/HPI_" + str(abbv))
        df = quandl.get(query)
        df.rename(columns={'NSA Value': abbv, 'SA Value': abbv}, inplace=True)
        df=df.iloc[:,[0]] #only including NSA value
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0


        if main_df.empty:
            main_df = df.iloc[:,[0]]
        else:
            main_df = main_df.join(df.iloc[:,[0]])

        print(main_df)


    pickle_out = open('fifty_states3.pickle','wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

def HPI_Benchmark():
    df = quandl.get("FMAC/HPI_USA")
    df.rename(columns={'NSA Value': 'United States HPI', 'SA Value': 'United States2'}, inplace=True)
    df = df.iloc[:, [0]]
    df['United States HPI'] = (df['United States HPI'] - df['United States HPI'][0]) / df['United States HPI'][0] * 100.0
    return df

def mortage_30y():
    df = quandl.get("FMAC/MORTG", trim_start="1975-01-01")
    df['Value'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100.0
    df = df.resample('D').mean()
    df = df.resample('M').mean()
    df.columns = ['M30']
    return df

def sp500_data(): #S&P historical adjusted close data, another economic indicator
    df = pd.read_csv('SP500.csv.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
    df['Adj Close']=(df['Adj Close']-df['Adj Close'][0]) / df['Adj Close'][0] * 100.0
    df = df.resample('M').mean()
    df.rename(columns={'Adj Close':'SP500'}, inplace=True)
    df=df['SP500']
    return df

def gdp_data(): #economic indicator
    df = quandl.get("BCB/4385", trim_start = "1975-01-01")
    df['Value'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100.0
    df = df.resample('M').mean()
    df.rename(columns={'Value':'GDP'}, inplace=True)
    df= df['GDP']
    return df

def unemployment_rate(): #economic indicator
    df = quandl.get("USMISERY/INDEX", trim_start = "1976-01-01")
    df = df.iloc[:, [0]]
    df['Unemployment Rate'] = (df['Unemployment Rate']- df['Unemployment Rate'][0]) / df['Unemployment Rate'][0] * 100.0
    df=df.resample('M').mean()
    return df

SP500 = sp500_data()
US_GDP = gdp_data()
US_unemployment = unemployment_rate()
m30 = mortage_30y()
HPI_data = pd.read_pickle('fifty_states3.pickle')
HPI_bench = HPI_Benchmark()

HPI = HPI_data.join([HPI_bench,m30,US_unemployment,US_GDP,SP500])
HPI.dropna(inplace=True)
print(HPI)
print(HPI.corr())

HPI.to_pickle('HPI.pickle')

def create_labels(cur_hpi, fut_hpi):
    if fut_hpi > cur_hpi:
        return 1
    else:
        return 0

def moving_average(values):
    return mean(values)

housing_data = pd.read_pickle('HPI.pickle')

housing_data = housing_data.pct_change()


housing_data.replace([np.inf, -np.inf], np.nan, inplace=True)
housing_data['US_HPI_future'] = housing_data['United States HPI'].shift(-1)

housing_data.dropna(inplace=True)
#print(housing_data[['US_HPI_future','United States HPI']].head())
housing_data['label'] = list(map(create_labels, housing_data['United States HPI'], housing_data['US_HPI_future']))

print(housing_data.head())

X = np.array(housing_data.drop(['US_HPI_future','label'],1))
X = preprocessing.scale(X)
y = np.array(housing_data['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2) #train on 80% of data and test on 20%

clf = svm.SVC(kernel='linear')
clf.fit(X_train,y_train)

print(clf.score(X_test,y_test)) #accuracy of prediction: 0 or 1 based on HPI lower or more than previous HPI
#print(housing_data.drop(['US_HPI_future','label'],1))
