from math import sqrt
import pandas as pd
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout


# load train data and preview part of the data, note that the sales should be the last column, otherwiese 
# the some part of the codes need to modify
# TODO - paramize the sales index when split X and y, as well as retrive inv_yhat and inv_y
data = pd.read_csv("data/bakery_train.csv")
data.head(10)

# check if there are null values in any of the columns
data.info()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
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


# 先测试单个商品，后面再想办法扩展到多商品预测
items = data.drop(['date', 'item', 'Unnamed: 0'], axis=1)

# set  params
timesteps = 4
features = items.shape[1]
n_obs = timesteps*features
n_train = int(data.shape[0] * 0.8)
n_epoch = 150

values = items.values
# integer encode direction
encoder = LabelEncoder()
# TODO - encode all categorical variable into integer
# ensure all data is float
values = values.astype("float32")

# normalize feactures
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

# split into train and test sets
scaled_train = scaled_values[:n_train, :]
scaled_test = scaled_values[n_train:, :]

# frame as supervised learning
reframed_train = series_to_supervised(scaled_train, timesteps, 1)
reframed_test = series_to_supervised(scaled_test, timesteps, 1)
# drop columns we don't want to predict, when train on multiple lag timesteps, we don't apply this
## only varn(t) where n is the index of the column we want to predict: sales 
# reframed.drop(reframed.columns[[i for i  in range(-2, -features-1, -1)]], axis=1, inplace=True)
print(reframed_train.head())

reframed_train = reframed_train.values
reframed_test = reframed_test.values

# split into input and outputs
train_X, train_y = reframed_train[:, :n_obs], reframed_train[:, -1]
test_X, test_y = reframed_test[:, :n_obs], reframed_test[:, -1]
# reshape input to 3D [samples, timesteps, feactures]


train_X = train_X.reshape((train_X.shape[0], timesteps, features))
test_X = test_X.reshape((test_X.shape[0], timesteps, features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss="mae", optimizer="adam")

# fit network
history = model.fit(train_X, train_y, epochs=n_epoch, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="test")
plt.legend()
plt.savefig(f"plots/e{n_epoch}_ts{timesteps}_errors.png")
plt.show()

# make a prediction
yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], timesteps*features))

# invert scaling for forecast
inv_yhat = concatenate((test_X[:,-features:-1], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:,-features:-1], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

# calculate RMSE
rmse = sqrt(mean_absolute_error(inv_y, inv_yhat))
print("Test RMSE: %.3f" % rmse)

# visualize results
x_data = [i for i in range(1, len(inv_y)+1)]
plt.figure(figsize=(15,5))
plt.plot(x_data, inv_yhat, color='r', label= "Predicted sales")
plt.plot(x_data, inv_y, color='b', label = "Real sales")
plt.xlabel("Date")
plt.ylabel("Daily Sales")
plt.title("Daily Item1 Sales")
plt.legend()
plt.savefig(f"plots/e{n_epoch}_ts{timesteps}.png")
plt.show()