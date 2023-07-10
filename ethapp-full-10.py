# conda install -c conda-forge yfinance

# Initializing yfinance

import yfinance as yf

# Initializing library

import numpy as np
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime
from datetime import timedelta
import time
import logging
from binance import ThreadedWebsocketManager

# Defining global variables

window_size = 20
skip = 1
layer_size = 500

# Defining global variables
output_size = 3

global counter
counter = 0

logging.basicConfig(
    filename="../logs/ethapp-full-10.log",
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.WARNING,
)

logging.warning("Starting")

# Defining Functions


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


def get_state(parameters, t, window_size=20):
    outside = []
    d = t - window_size + 1
    for parameter in parameters:
        block = (
            parameter[d : t + 1]
            if d >= 0
            else -d * [parameter[0]] + parameter[0 : t + 1]
        )
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        for i in range(1, window_size, 1):
            res.append(block[i] - block[0])
        outside.append(res)
    return np.array(outside).reshape((1, -1))


# Initializing model

logging.warning("Defining Neron")


class Deep_Evolution_Strategy:
    inputs = None

    def __init__(self, weights, reward_function, population_size, sigma, learning_rate):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights

    def train(self, epoch=100, print_every=1):
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.weights, population[k]
                )
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T
                )
            if (i + 1) % print_every == 0:
                logging.warning(
                    "iter %d. reward: %f" % (i + 1, self.reward_function(self.weights))
                )
        logging.warning("time taken to train:", time.time() - lasttime, "seconds")


logging.warning("Defining Model")


class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.rand(input_size, layer_size)
            * np.sqrt(1 / (input_size + layer_size)),
            np.random.rand(layer_size, output_size)
            * np.sqrt(1 / (layer_size + output_size)),
            np.zeros((1, layer_size)),
            np.zeros((1, output_size)),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-2]
        decision = np.dot(feed, self.weights[1]) + self.weights[-1]
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


logging.warning("Defining Agent")


class Agent:
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, timeseries, skip, initial_money, real_trend, minmax):
        self.model = model
        self.timeseries = timeseries
        self.skip = skip
        self.real_trend = real_trend
        self.initial_money = initial_money
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )
        self.minmax = minmax
        self._initiate()

    def _initiate(self):
        # i assume first index is the close value
        self.trend = self.timeseries[0]
        self._mean = np.mean(self.trend)
        self._std = np.std(self.trend)
        self._inventory = []
        self.real_inventory = []
        self._capital = self.initial_money
        self._queue = []
        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]

    def reset_capital(self, capital):
        if capital:
            self._capital = capital
        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]
        self._queue = []
        self._inventory = []
        self.real_inventory = []

    def trade(self, data):
        """
        you need to make sure the data is [close, volume]
        """
        scaled_data = self.minmax.transform([data])[0]
        real_close = data[0]
        close = scaled_data[0]
        if len(self._queue) >= window_size:
            self._queue.pop(0)
        self._queue.append(scaled_data)

        if len(self._queue) < window_size:
            return {
                "status": "data not enough to trade",
                "action": "fail",
                "balance": self._capital,
                "timestamp": str(datetime.now()),
            }

        state = self.get_state(
            window_size - 1,
            self._inventory,
            self._scaled_capital,
            timeseries=np.array(self._queue).T.tolist(),
        )
        action, prob = self.act_softmax(state)

        logging.warning(prob)

        if action == 1 and self._scaled_capital >= close:
            self._inventory.append(close)
            self.real_inventory.append(real_close)
            self._scaled_capital -= close
            self._capital -= real_close
            return {
                "status": "buy 1 unit, cost %f, inventory %d"
                % (real_close, len(self._inventory)),
                "action": "buy",
                "balance": self._capital,
                "timestamp": str(datetime.now()),
            }
        elif action == 2 and len(self._inventory):
            bought_price = self._inventory.pop(0)
            self.real_inventory.pop(0)
            self._scaled_capital += close
            self._capital += real_close
            scaled_bought_price = self.minmax.inverse_transform([[bought_price, 2]])[
                0, 0
            ]
            try:
                invest = (
                    ((real_close) - scaled_bought_price) / scaled_bought_price
                ) * 100
            except Exception as e:
                invest = 0
            return {
                "status": "sell 1 unit, price %f, inventory %d"
                % (real_close, len(self._inventory)),
                "investment": invest,
                "gain": real_close - scaled_bought_price,
                "balance": self._capital,
                "action": "sell",
                "timestamp": str(datetime.now()),
            }
        else:
            return {
                "status": "do nothing",
                "action": "nothing",
                "balance": self._capital,
                "real_inventory": self.real_inventory,
                "timestamp": str(datetime.now()),
            }

    def change_data(self, timeseries, skip, initial_money, real_trend, minmax):
        self.timeseries = timeseries
        self.skip = skip
        self.initial_money = initial_money
        self.real_trend = real_trend
        self.minmax = minmax
        self._initiate()

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))

        return np.argmax(decision[0])

    def act_softmax(self, sequence):
        decision = self.model.predict(np.array(sequence))

        return np.argmax(decision[0]), softmax(decision)[0]

    def get_state(self, t, inventory, capital, timeseries):
        state = get_state(timeseries, t)
        len_inventory = len(inventory)
        if len_inventory:
            mean_inventory = np.mean(inventory)
        else:
            mean_inventory = 0
        z_inventory = (mean_inventory - self._mean) / self._std
        z_capital = (capital - self._mean) / self._std
        concat_parameters = np.concatenate(
            [state, [[len_inventory, z_inventory, z_capital]]], axis=1
        )
        return concat_parameters

    def get_reward(self, weights):
        initial_money = self._scaled_capital
        starting_money = initial_money
        invests = []
        self.model.weights = weights
        inventory = []
        state = self.get_state(0, inventory, starting_money, self.timeseries)

        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t])
                starting_money -= self.trend[t]

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                starting_money += self.trend[t]
                invest = ((self.trend[t] - bought_price) / bought_price) * 100
                invests.append(invest)

            state = self.get_state(t + 1, inventory, starting_money, self.timeseries)
        invests = np.mean(invests)
        if np.isnan(invests):
            invests = 0
        score = (starting_money - initial_money) / initial_money * 100
        return invests * 0.7 + score * 0.3

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every=checkpoint)

    def buy(self):
        initial_money = self._scaled_capital
        starting_money = initial_money

        real_initial_money = self.initial_money
        real_starting_money = self.initial_money
        inventory = []
        real_inventory = []
        state = self.get_state(0, inventory, starting_money, self.timeseries)
        states_sell = []
        states_buy = []

        for t in range(0, len(self.trend) - 1, self.skip):
            action, prob = self.act_softmax(state)
            # logging.warning(t, prob)

            if (
                action == 1
                and starting_money >= self.trend[t]
                and t < (len(self.trend) - 1 - window_size)
            ):
                inventory.append(self.trend[t])
                real_inventory.append(self.real_trend[t])
                real_starting_money -= self.real_trend[t]
                starting_money -= self.trend[t]
                states_buy.append(t)
                logging.warning(
                    "day %d: buy 1 unit at price %f, total balance %f"
                    % (t, self.real_trend[t], real_starting_money)
                )

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                real_bought_price = real_inventory.pop(0)
                starting_money += self.trend[t]
                real_starting_money += self.real_trend[t]
                states_sell.append(t)
                try:
                    invest = (
                        (self.real_trend[t] - real_bought_price) / real_bought_price
                    ) * 100
                except Exception as e:
                    invest = 0
                logging.warning(
                    "day %d, sell 1 unit at price %f, investment %f %%, total balance %f,"
                    % (t, self.real_trend[t], invest, real_starting_money)
                )
            state = self.get_state(t + 1, inventory, starting_money, self.timeseries)

        invest = ((real_starting_money - real_initial_money) / real_initial_money) * 100
        total_gains = real_starting_money - real_initial_money
        return states_buy, states_sell, total_gains, invest


# #Downloading intial data

# #!mkdir dataset

# try:
#     data = yf.download(
#         tickers="BTC-USD",
#         interval="1h",  # trading interval
#         period="2y",  # time period
#         prepost=True,  # download pre/post market hours data?
#         repair=True,
#     )
#     data.to_csv(f"trend/BTC-USD.csv")
# except Exception as e:
#     logging.warning(f"An exception occurred while downloading trend data {e}")

# Loading initialdata

logging.warning("Reading Initial Dataset")

df = pd.read_csv(
    "../trend/ETHUSDT.csv", skiprows=[i for i in range(0, 525000) if i % 10 != 0]
)
df = df.dropna()

real_trend = df["close"].tolist()
parameters = [df["close"].tolist(), df["volume"].tolist()]
minmax = MinMaxScaler(feature_range=(100, 200)).fit(np.array(parameters).T)
scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
initial_money = np.max(parameters[0]) * 2

logging.warning(f"Initial Money: {initial_money}")

# Loading Model

with open("../models/model-stock-f.pkl", "rb") as fopen:
    model = pickle.load(fopen)

agent = Agent(
    model=model,
    timeseries=scaled_parameters,
    skip=skip,
    initial_money=initial_money,
    real_trend=real_trend,
    minmax=minmax,
)

# COLUMNS = [
#     "open_time",
#     "open",
#     "high",
#     "low",
#     "close",
#     "volume",
#     "close_time",
#     "quote_volume",
#     "count",
#     "taker_buy_volume",
#     "taker_buy_quote_volume",
#     "ignore",
# ]


# data = pd.read_csv(
#     "../timeseries/ETHUSDT-1m.csv",
#     skiprows=[i for i in range(0, 180) if i % 10 != 0],
#     names=COLUMNS,
#     header=None,
# )
# data = data.dropna()

# logging.warning("Starting with 3 hours data")

# real_trend = data["close"].tolist()
# volume = data["volume"].tolist()

# if len(real_trend) > 0 and len(volume):
#     for i in range(len(real_trend)):
#         logging.warning(agent.trade([real_trend[i], volume[i]]))
# else:
#     logging.warning("data not found")

# time.sleep(60 * 10)

logging.warning("Starting threadpool")


def trade(price):
    global counter
    if counter % (60 * 10) == 0:
        real_trend = float(price["c"])
        volume = float(price["v"])

        if real_trend and volume:
            logging.warning(agent.trade([real_trend, volume]))
            logging.warning([real_trend, volume])
        else:
            logging.warning("data not found in timeseries")

        counter = 0
        logging.warning("After 10 mins")

    counter += 1


symboltoInvest = "ETHUSDT"
bsm = ThreadedWebsocketManager()
bsm.start()
bsm.start_symbol_ticker_socket(callback=trade, symbol=symboltoInvest)
bsm.join()

# while 1:
# try:
# try:
#     data = yf.download(
#         tickers="BTC-USD",
#         interval="1m",  # trading interval
#         period="1d",    # time period
#         prepost=False,  # download pre/post market hours data?
#         repair=True,
#     )
#     data = data.dropna()
#     data.to_csv(f"timeseries/BTC-USD.csv")
# except Exception as e:

#     logging.warning(f"Error Downloading last 1 day data {e}")

#     time.sleep(60 * 5)
#     logging.warning("After 5 mins")

#     continue

# real_trend = data["Close"].tolist()
# volume = data["Volume"].tolist()

# if len(real_trend) > 0 and len(volume) > 0:
#     logging.warning(
#         agent.trade(
#             [real_trend[len(real_trend) - 1], volume[len(volume) - 1]]
#         )
#     )
#     logging.warning(
#         [real_trend[len(real_trend) - 1], volume[len(volume) - 1]]
#     )
# else:
#     logging.warning("data not found in timeseries cont")

# time.sleep(60 * 5)
# logging.warning("After 5 mins")

# except Exception as e:
#     logging.warning(f"Error in while loop {e}")
