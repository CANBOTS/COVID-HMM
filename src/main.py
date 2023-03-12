import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timedelta

class GaussianMixtureHMM:
    def __init__(self, num_states):
        self.num_states = num_states
        self.model = GaussianHMM(n_components=num_states,covariance_type="full")
    
    def fit_bic_score(self, data):
        best_hmm = None
        best_bic = np.inf

        for n_states in range(1,11):
            model = GaussianHMM(n_components=n_states, covariance_type="full")
            model.fit(data)

            # calculate BIC score
            log_likelihood = model.score(data)
            n_params = n_states ** 2 + 2 * self.num_states * n_states - 1
            bic = -2 * log_likelihood + n_params * np.log(data.shape[0])

            if bic < best_bic:
                best_bic = bic
                best_model = model

        self.num_states = best_model.n_components
        self.model = best_model


    def forecast(self, last_obs, forecast_steps):
        means, covs = self.get_params()
        forecast = []
        for i in range(forecast_steps):
            # calculate probabilities of next state given the current state
            last_obs_array = np.array([last_obs]).reshape(-1,1)
            state_probs = self.predict_proba(last_obs_array.reshape(-1,1))[0]

            # get mean and covariance of most likely next state
            next_state = np.argmax(state_probs)
            next_mean = means[next_state][0]
            next_cov = covs[next_state][0][0]

            # generate next observation as mean of most likely next state
            next_obs = np.random.normal(next_mean, next_cov)
            forecast.append(next_obs)
            last_obs = next_obs

        return np.array(forecast)

    def fit(self, data):
        self.model.fit(data)

    def get_params(self):
        return self.model.means_, self.model.covars_

    def predict_proba(self, data):
        return self.model.predict_proba(data)

    def sample(self, n_samples):
        return self.model.sample(n_samples)

    def decode(self, data):
        return self.model.decode(data)

    def predict(self, data):
        return self.model.predict(data)

    def score(self, data):
        return self.model.score(data)

def exponential_smoothing(series, alpha, start_date=None, end_date=None):
    if start_date is not None:
        series = series.loc[start_date:]

    if end_date is not None:
        series = series.loc[:end_date]

    result = [series[0]] # first value is same as series

    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])

    return pd.Series(result, index=series.index)

def read_from_dir(directory,mimetype=".csv"):
    df_list = []
    df_dict = {}
    for file in os.listdir(directory):
        if file.endswith(mimetype):
            filepath = os.path.join(directory,file)
            df = pd.read_csv(filepath)
            df_dict[file] = df
    return df_dict

def get_location_data(df,country,province=None,agg='7D',start_col=4):
    
    location_df = df[df['Country/Region'] == country]

    if province == None:
        location_df = df.iloc[:,start_col:].sum(axis=0)
    else:
        location_df = df.loc[df['Province/State'] == province].iloc[:,start_col:].sum()

    
    location_df.index = pd.to_datetime(location_df.index)
    location_df = location_df.sort_index()

    new_df = location_df.resample('D').sum().fillna(0)

    return new_df.resample(agg).sum()

def parse_args():
    parser = argparse.ArgumentParser(description='Model & forecast COVID-19 data')
    parser.add_argument('-d', '--directory', type=str, help='Directory of CSV files')
    parser.add_argument('-c', '--country', type=str, help='Country/Region')
    parser.add_argument('-p', '--province', type=str, default=None, help='Province/State')
    parser.add_argument('-t', '--date_range', nargs=2, metavar=("start_date,end_date"),help='Date range in YYYY-MM-DD format')
    parser.add_argument('-a', '--agg_days', type=int, default=7, help='Number of days to aggregate')
    args = parser.parse_args()
    
    if args.date_range is not None:
        start_date, end_date = args.date_range
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            args.date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        except ValueError:
            print("Invalid date format. Date range should be in the format 'YYYY-MM-DD,YYYY-MM-DD'")
            exit()
    
    return args

def main():

    # get command line arguments
    args = parse_args()

    dir_name = args.directory
    country = args.country
    province = args.province
    date_range = args.date_range
    agg = args.agg_days

    # get the data from directory
    data_dir = os.getcwd() + dir_name
    data_df = read_from_dir(data_dir)
    labels = list(data_df.keys())

    # get the location specific data
    global_cases = data_df[labels[0]]
    country_cases = get_location_data(global_cases,country,province)

    # smooth the data before using in HMM
    alpha = 2
    smoothed_data = exponential_smoothing(country_cases,2,date_range[0],date_range[-1])

    # train HMM model
    model = GaussianMixtureHMM(2)
    model.fit(np.array(smoothed_data).reshape(-1,1))
    #model.fit_bic_score(np.array(smoothed_data).reshape(-1, 1))
   
    means,covars = model.get_params()
    
    x = np.linspace(-5, 15, 1000).reshape(-1, 1)
    colors = ['r', 'g', 'b']
    for i in range(model.num_states):
        mu = means[i][0]
        cov = covars[i][0][0]
    
        y = np.exp(-(x - mu)**2 / (2 * cov)) / np.sqrt(2 * np.pi * cov)
        plt.plot(x, y, colors[i], linewidth=2, label=f'Component {i+1}')
    
    plt.legend()
    plt.show()

    # forecast next 100 time steps
    #last_obs = np.array([smoothed_data[-1]])
    #forecast_steps = 4
    #forecast = model.forecast(last_obs, forecast_steps)

    # plot actual and forecasted data
    #plt.plot(np.arange(len(smoothed_data)), smoothed_data, label='Actual')
    #plt.plot(np.arange(len(smoothed_data), len(smoothed_data)+forecast_steps), forecast, label='Forecast')
    #plt.legend()
    #plt.show()

if __name__ == "__main__":
    main()

