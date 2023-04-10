import os
import joblib
from math import sqrt
import pandas as pd
import numpy as np
from numpy import concatenate
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

class BakeryModel:

    def __init__(self, features, timestep=5, predict_len=1) -> None:
        self.timestep = timestep
        self.predict_len = predict_len
        self.features = features
        self.train_scaler = MinMaxScaler(feature_range=(0, 1))
        if os.path.exists(os.path.join(os.getcwd(), "pkl/scaler.pkl")):
            self.predict_scaler = joblib.load("pkl/scaler.pkl")
        else:
            os.makedirs(os.path.join(os.getcwd(), "pkl"))
            self.predict_scaler = None
        
        # set params
        self.n_epoch = 150
        self.split_rate = 0.8

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.timestep, self.features)))
        self.model.add(Dense(self.predict_len))
        self.model.compile(loss="mae", optimizer="adam")

    def save_model(self):
        joblib.dump(self.model, "pkl/model.pkl")

    def load_model(self):
        try:
            self.model = joblib.load("pkl/model.pkl")
        except:
            self.model = None
    
    def predict(self, data: dict, predict_variable: str):
        df = pd.DataFrame(data=data)
        # move predict var to the last column
        order = df.columns.to_list()
        order.remove(predict_variable)
        order.append(predict_variable)
        df = df[order]
        # normalization
        X = self.predict_scaler.transform(df.values)
        X = X.reshape(-1, X.shape[0], X.shape[1])
        y = self.model.predict([X])
        # invert scaling for prediction
        y = y.reshape(-1, 1)
        y = np.repeat(y, repeats=self.features, axis=1)
        inv_y = self.predict_scaler.inverse_transform(y)
        inv_y = inv_y[:,-1]
        predict_value = inv_y
        return predict_value

    def load_data(self, data_path, features: list):
        data = pd.read_csv(data_path)
        data = data[features]
        item_names = data['item'].unique()
        dfs = []
        for index, item in enumerate(item_names):
            item_data = data[data["item"]==item]
            item_data["item"] = [index] * item_data.shape[0]
            dfs.append(item_data)
        return dfs
    
    def prepare_data(self, data_path, features: list):
        dfs = self.load_data(data_path, features)
        # split train and test sets
        trains, tests = [], []
        for item_data in dfs:
            n_train = int(item_data.shape[0] * self.split_rate)
            trains.append(item_data.values[:n_train,:])
            tests.append(item_data.values[n_train:,:])

        train = concatenate(trains, axis=0)
        test = concatenate(tests, axis=0)

        # normalization
        self.train_scaler = MinMaxScaler(feature_range=(0, 1))
        train = self.train_scaler.fit_transform(train)
        test = self.train_scaler.transform(test)
        joblib.dump(self.train_scaler, "pkl/scaler.pkl")

        # shift to create data clips with 
        train = self.__to_reframed_data(train, self.timestep, self.predict_len)
        test = self.__to_reframed_data(test, self.timestep, self.predict_len)

        # split into input and outputs
        n_obs = self.timestep*self.features
        train_X, train_y = train[:, :n_obs], train[:, -self.predict_len:]
        test_X, test_y = test[:, :n_obs], test[:, -self.predict_len:]

        # shuffle data
        train_X, train_y = shuffle(train_X, train_y, random_state=12580)

        # reshape input to 3D [samples, timesteps, feactures]
        train_X = train_X.reshape((train_X.shape[0], self.timestep, self.features))
        test_X = test_X.reshape((test_X.shape[0], self.timestep, self.features))
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        return train_X, train_y, test_X, test_y


    def train_model(self, data_path: str, features: list, evaluate=False):
        train_X, train_y, test_X, test_y = self.prepare_data(data_path, features)
        # build network
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss="mae", optimizer="adam")

        history = model.fit(train_X, train_y, epochs=self.n_epoch, batch_size=72,
                 validation_data=(test_X, test_y), verbose=2, shuffle=True)

        # plot history and save model
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="test")
        plt.legend()
        date = datetime.today().strftime('%Y-%m-%d')
        if not os.path.exists(os.path.join(os.getcwd(), "plots")):
            os.makedirs(os.path.join(os.getcwd(), "plots"))
        plt.savefig(f"plots/date{date}_e{self.n_epoch}_ts{self.timestep}_errors.png")
        
        if evaluate:
            # make a prediction
            yhat = model.predict(test_X)

            # invert scaling for prediction
            yhat = yhat.reshape(-1, 1)
            yhat = np.repeat(yhat, repeats=self.features, axis=1)
            inv_yhat = self.train_scaler.inverse_transform(yhat)
            inv_yhat = inv_yhat[:,-self.predict_len:]

            # invert scaling for actual
            test_y = test_y.reshape((-1, 1))
            test_y = np.repeat(test_y, repeats=self.features, axis=1)
            inv_y = self.train_scaler.inverse_transform(test_y)
            inv_y = inv_y[:,-self.predict_len:]

            # calculate RMSE
            rmse = sqrt(mean_absolute_error(inv_y, inv_yhat))
            print("Test RMSE: %.3f" % rmse)

            # visualize results
            x_data = [i for i in range(1, len(inv_y)+1)]
            plt.figure(figsize=(15,5))
            plt.plot(x_data, inv_yhat,color='r', label= f"Predicted y")
            plt.plot(x_data, inv_y, color='b', label = f"Real y")
            plt.xlabel("date scale")
            plt.ylabel("Daily y")
            plt.legend()
            plt.savefig(f"plots/date{date}_e{self.n_epoch}_ts{self.timestep}_evaluate.png")

        self.model = model
        self.save_model()

        return history.history


    def __series_to_supervised(self, data: np.array, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = [], []
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg


    def __to_reframed_data(self, item_data, timestep, predict_len):
        # ensure all data is float
        item_data = item_data.astype("float32")

        # frame as supervised learning
        reframed_data = self.__series_to_supervised(item_data, timestep, predict_len)

        return reframed_data.values



if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # expect 78.0
    data = {
        "item": [1, 1, 1, 1, 1],
        "price": [1.2, 1.2, 1.2, 1.2, 1.2],
        "day_of_week": [5, 6, 0 ,1, 3],
        "quantity": [128, 171, 128, 99, 109]
        }

    # expect 135.0
    data = {
        "item": [1, 1, 1, 1, 1],
        "price": [1.2, 1.2, 1.2, 1.2, 1.2],
        "day_of_week": [6, 0 ,1, 3, 4],
        "quantity": [171, 128, 99, 109, 78]
        }

    # expect 35.0
    data = {
        "item": [3, 3, 3, 3, 3],
        "price": [1, 1, 1, 1, 1],
        "day_of_week": [6, 0 ,1, 2, 3],
        "quantity": [36, 18, 34, 23, 30]
        }

    # expect 11.0
    data = {
        "item": [3, 3, 3, 3, 3],
        "price": [1, 1, 1, 1, 1],
        "day_of_week": [3, 4, 5, 6, 1],
        "quantity": [26, 25, 34, 47, 20]
        }


    bakery_model = BakeryModel(timestep=5, predict_len=1, features=4)
    bakery_model.train_model(data_path="data/bakery_train_mul_quan.csv",
            features=["item", "price", "day_of_week", "quantity"], evaluate=True)
    bakery_model.load_model()
    val = bakery_model.predict(data, predict_variable="quantity")
    print(val)



