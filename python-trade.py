import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfolio as pf

# Set the start and end dates for the data
start_date = '2017-01-01'
end_date = '2023-04-11'

# Download the Bitcoin price data from Yahoo Finance
data = yf.download('BTC-USD', start=start_date, end=end_date)

# Define the RSI function
def calculate_rsi(data, time_window):
    diff = data.diff(1).dropna()
    up_chg = 0 * diff
    down_chg = 0 * diff
    up_chg[diff > 0] = diff[diff > 0]
    down_chg[diff < 0] = diff[diff < 0]
    up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi

# Calculate the RSI
rsi = calculate_rsi(data['Close'], 14)

# Define the Q-learning parameters
q_table = np.zeros((2, 2))
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1

# Initialize the variables for the first iteration
prev_state = 0
prev_close_price = data['Close'][0]
dates = []
profits = []
actions = []

# Loop through the data and trade based on the Q-learning algorithm
for date, close_price, rsi_value in zip(data.index, data['Close'], rsi):
    # Choose the action based on the epsilon-greedy policy
    if np.random.uniform() < epsilon:
        action = np.random.choice([0, 1])
    else:
        action = np.argmax(q_table[prev_state])

    # Buy if the RSI is below 30 and the previous action was a sell
    if rsi_value < 30 and prev_state == 0 and action == 0:
        action = 1
    # Sell if the RSI is above 70 and the previous action was a buy
    elif rsi_value > 70 and prev_state == 1 and action == 1:
        action = 0
    
    # Execute the action and calculate the profit
    if action == 0:
        profit = prev_close_price - close_price
    else:
        profit = close_price - prev_close_price
    
    # Append the results to the lists
    dates.append(date)
    profits.append(profit)
    actions.append(action)
    
    # Get the next state based on the action
    if action == 0:
        state = 0
    else:
        state = 1
    
    # Calculate the reward
    if state == 1:
        reward = close_price - prev_close_price
    else:
        reward = prev_close_price - close_price
    
    # Update the Q table
    q_table[prev_state][action] += learning_rate * (reward + discount_factor * np.max(q_table[state]) - q_table[prev_state][action])
    
    # Update the variables for the next iteration
    prev_state = state
    prev_close_price = close_price

# Calculate the cumulative profits and plot the results
cumulative_profits = np.cumsum(profits)
plt.plot(dates, cumulative_profits)
plt.title('Cumulative Profits')
plt.xlabel('Date')
plt.ylabel('Profit')
plt.show()

# Calculate and print the performance metrics
returns = pd.DataFrame(profits, columns=['Returns'], index=dates)
returns['Returns'] = returns['Returns'].astype(float)
returns.index = pd.to_datetime(returns.index)
returns.index.name = 'date'

perf_stats = pf.timeseries.perf_stats(returns)
perf_stats_all = pf.timeseries.perf_stats(returns, benchmark_rets=None)

print('Performance Metrics:')
print('---------------------')
print('Sharpe Ratio: {:.2f}'.format(perf_stats['Sharpe ratio']))
print('Max Drawdown: {:.2f}%'.format(perf_stats['Max drawdown'] * 100))
print('Sortino Ratio: {:.2f}'.format(perf_stats['Sortino ratio']))
print('Beta: {:.2f}'.format(perf_stats_all['Beta']))
print('Alpha: {:.2f}'.format(perf_stats_all['Alpha']))
print('Win Rate: {:.2f}%'.format(perf_stats_all['Winning Days']*100))
