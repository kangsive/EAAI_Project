from math import sqrt
import pandas as pd
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# load train data and preview part of the data, note that the sales should be the last column, otherwiese 
# the some part of the codes need to modify
# TODO - paramize the sales index when split X and y, as well as retrive inv_yhat and inv_y
data = pd.read_csv("data/bakery_train_mul.csv")

# retrive data of first 3 items in store 1
item1 = data[data["item"]=="TRADITIONAL BAGUETTE"]
item2 = data[data["item"]=="COUPE"]
item3 = data[data["item"]=="BAGUETTE"]

item1["item"] = [1] * item1.shape[0]
item2["item"] = [2] * item2.shape[0]
item3["item"] = [3] * item3.shape[0]

print(item1.shape)
print(item2.shape)
print(item3.shape)

# set  params
timesteps = 4
n_epoch = 500

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


def to_reframed_data(item_data):
    # ensure all data is float
    item_data = item_data.astype("float32")

    # frame as supervised learning
    reframed_data = series_to_supervised(item_data, timesteps, 1)

    print(reframed_data.head())

    reframed_data = reframed_data.values

    global n_obs
    global features
    features = item_data.shape[1]
    n_obs = timesteps*features

    return reframed_data

scaler = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))

trains, tests = [], []
for item_data in [item1, item2, item3]:
    item_data = item_data.drop(['date', 'Unnamed: 0'], axis=1)
    n_train = int(item_data.shape[0] * 0.8)
    trains.append(item_data.values[:n_train,:])
    tests.append(item_data.values[n_train:,:])

train = concatenate(trains, axis=0)
test = concatenate(tests, axis=0)

# normalize feactures
train = scaler.fit_transform(train)
test = scaler2.fit_transform(test)

train = to_reframed_data(train)
test = to_reframed_data(test)

# split into input and outputs
train_X, train_y = train[:, :n_obs], train[:, -1]
test_X, test_y = test[:, :n_obs], test[:, -1]

# shuffle data
train_X, train_y = shuffle(train_X, train_y, random_state=12580)

print(train_X.shape)

# reshape input to 3D [samples, timesteps, feactures]
train_X = train_X.reshape((train_X.shape[0], timesteps, features))
test_X = test_X.reshape((test_X.shape[0], timesteps, features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss="mae", optimizer="adam")

# model = Sequential()
# model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
# model.add(Dropout(0.5)) 
# model.add(LSTM(32,return_sequences=False))
# model.add(Dense(1))
# model.compile(loss="mae", optimizer="adam")

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
print(inv_yhat.shape)
print(test.shape)

inv_yhat = scaler2.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:,-features:-1], test_y), axis=1)
inv_y = scaler2.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

# calculate RMSE
rmse = sqrt(mean_absolute_error(inv_y, inv_yhat))
print("Test RMSE: %.3f" % rmse)

# visualize results
x_data = [i for i in range(1, len(inv_y)+1)]
plt.figure(figsize=(15,5))
plt.plot(x_data, inv_yhat,color='r', label= "Predicted sales")
plt.plot(x_data, inv_y, color='b', label = "Real sales")
plt.xlabel("Date")
plt.ylabel("Daily Sales")
plt.title("Daily Item1 Sales")
plt.legend()
plt.savefig(f"plots/e{n_epoch}_ts{timesteps}.png")
plt.show()