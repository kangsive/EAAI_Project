from math import sqrt
import pandas as pd
import numpy as np
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

class BakeryModel:

    def __init__(self, timestep, predict_len, features) -> None:
        self.timestep = timestep
        self.predict_len = predict_len
        self.features = features
        self.train_scaler = MinMaxScaler(feature_range=(0, 1))
        self.predict_scaler = MinMaxScaler(feature_range=(0, 1))

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.timesteps, self.features)))
        self.model.add(Dense(1))
        self.model.compile(loss="mae", optimizer="adam")

    def train(self, train_X, train_y, n_epoch, validation_data=None):
        history = self.model.fit(train_X, train_y, epochs=n_epoch, batch_size=72, validation_data=validation_data, verbose=2, shuffle=True)

    def load_model(self):
        pass
    
    def predict(self, data: np.ndarray):
        self.predict_scaler.fit_transform(data)
        # TODO - transfer data format as train data
        y = self.model.predict(data)
        # TODO - reverse and get final predict value
        predict_value = None
        return predict_value

    def load_data(self):
        pass

    def train_model(self):
        pass

