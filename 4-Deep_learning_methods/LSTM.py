import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt



#https://keras.io/examples/timeseries/timeseries_weather_forecasting/
#https://www.researchgate.net/publication/315905153_Short-Term_Residential_Load_Forecasting_based_on_Resident_Behaviour_Learning

data_raw = pd.read_csv(filepath_or_buffer='../1-Data_generation/Outputs/dataset_national.csv')
data_features = data_raw.keys()
selected_features = ['tod', 'Load', 'Load_d1', 'Load_d7',
       'temperature', 'temperature_smooth_990', 'temperature_smooth_950',
       'temperature_max_smooth_990', 'temperature_max_smooth_950',
       'temperature_min_smooth_990', 'temperature_min_smooth_950', 'toy', 'day_type_jf',
       'day_type_week', 'period_hour_changed', 'week_number']
data = data_raw[selected_features]

train_begin = train_split = list(data_raw["date"]).index("2018-09-01 00:00:00+00:00")
train_split = list(data_raw["date"]).index("2022-09-01 00:00:00+00:00")
step = 24

past = 48*7
future = 48
learning_rate = 0.001
batch_size = 100
epochs = 10

def normalize(data, train_split):
    data_mean = data[train_begin:train_split].mean(axis=0)
    data_std = data[train_begin:train_split].std(axis=0)
    return (data - data_mean) / data_std

data_n = normalize(data, train_split)
train_data = data_n.loc[train_begin - 1 : train_split - 1].astype('float32')
val_data = data_n.loc[train_split:].astype('float32')

start = train_begin + past + future
end =  past + future + train_split

x_train = train_data.values
y_train = data_n.iloc[start:end]["Load"]

sequence_length = int(past / step)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

#validation dataset
x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end].values
y_val = data_n.iloc[label_start:]["Load"]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

for batch in dataset_train.take(1):
    inputs, targets = batch

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(512, return_sequences=True)(inputs)
lstm_out2 = keras.layers.LSTM(512, return_sequences=False)(lstm_out)
dense = keras.activations.linear(keras.layers.Dense(512)(lstm_out2))
outputs = keras.layers.Dense(1)(dense)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mae")
model.summary()

path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)


