#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 11:29:07 2025

@author: Yannig
"""

# packages import
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons # A simple synthetic dataset
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error

from viking_kalman import expectation_maximization
from viking_kalman import iterative_grid_search
from viking_kalman import statespace

import numpy as np
import matplotlib.pyplot as plt
import random

import pandas as pd
import matplotlib.pyplot as plt


# Device Configuration
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) backend on Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        print("MPS backend not available. Using CPU.")
    return device

device = get_device()





# Import Data

Data = pd.read_csv('Data/Outputs/dataset_national.csv')
current_directory = os.getcwd()

Data.shape

Data['date']= pd.to_datetime(Data['date'])
Data.set_index("date", inplace=True)
Data['Time']= np.arange(0, Data.shape[0])

covariate_names = Data.columns.tolist()

Data = pd.get_dummies(Data, columns=['day_type_week'], drop_first=True, dtype=int)
Data = pd.get_dummies(Data, columns=['period_hour_changed'], drop_first=True, dtype=int)




covariate_names = Data.columns.tolist()
covariate_columns = ['Load', 'Time', 'temperature',  'temperature_smooth_990',  'temperature_smooth_950', 'temperature_min_smooth_990',
                     'temperature_max_smooth_990',
                     'toy', 'day_type_week_1', 'day_type_week_2',
                     'day_type_week_3', 'day_type_week_4', 'day_type_week_5', 'day_type_week_6', 'period_holiday', 'period_summer', 'period_christmas']

covariate_columns = ['Load', 'Time', 'temperature', 'day_type_week_1', 'day_type_week_2',
'day_type_week_3', 'day_type_week_4', 'day_type_week_5', 'day_type_week_6', 'period_holiday', 'period_summer', 'period_christmas']




# Drop the last row 
Data = Data[covariate_columns]
last_index = Data.index[-1]
Data = Data.drop(last_index) 

a = Data[Data.index.astype(str) == "2013-01-15 00:00:00+00:00"].index[0]
actual_all_hour = Data.loc[a:,'Load']
prediction_all_hour_NN = actual_all_hour.copy()
prediction_all_hour_NN_static = actual_all_hour.copy()
prediction_all_hour_NN_dyn = actual_all_hour.copy()
prediction_all_hour_NN_em = actual_all_hour.copy()






def create_sequences(data, sequence_length):
        X, y = [], []
        #for i in range(len(data) - sequence_length):
        for i in range(len(data) - sequence_length):

            # X is the sequence of the last `sequence_length` days
            X.append(data[i:i + sequence_length])
            # y is the load of the next day (the target to predict)
            y.append(data[i + sequence_length, 0])  # Assuming 'Load' is the first column
        return np.array(X), np.array(y)
    
# Set the sequence length (e.g., using the last sequence_length days to predict the next)
sequence_length = 7

for j in range(48):
    hour = j // 2
    minute = (j % 2) * 30
    
    start_time = f'{hour:02d}:{minute:02d}'
    end_time = f'{hour:02d}:{minute+29:02d}'
    
    print('###############################################' + str(j))
    print('###############################################' + str(start_time)+ '###########'+str(end_time))

    
    Data_h = Data.between_time(start_time, end_time)
    
    #d1 = Data_h[Data_h.index.astype(str) ==  "2020-03-15 "+ start_time +":00+00:00"].index[0]
    #d2 = Data_h[Data_h.index.astype(str) ==  "2020-03-09 "+ start_time +":00+00:00"].index[0]

    d1 = Data_h[Data_h.index.astype(str) ==  "2022-09-01 "+ start_time +":00+00:00"].index[0]
    d2 = Data_h[Data_h.index.astype(str) ==  "2022-08-26 "+ start_time +":00+00:00"].index[0]

 
    
    mean_load_train = np.mean(Data_h.loc[:d1, 'Load'])
    std_load_train = np.std(Data_h.loc[:d1,'Load'])
    
    mean_load_test = np.mean(Data_h.loc[d2:, 'Load'])
    std_load_test = np.std(Data_h.loc[d2:,'Load'])
    
    Data_train = Data_h.loc[:d1, ]
    Data_test = Data_h.loc[d2:, ]
    
    scaler = StandardScaler()
    Data_train_normalized = scaler.fit_transform(Data_train)
    Data_test_normalized = scaler.fit_transform(Data_test)
    
    Data_train_normalized.shape
    Data_test_normalized.shape
    
     
    # Create the sequences
    X_train, y_train = create_sequences(Data_train_normalized, sequence_length)
    X_test, y_test = create_sequences(Data_test_normalized, sequence_length)
    
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train).unsqueeze(1)
    
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test).unsqueeze(1)
    
    
    X_train.shape
    X_test.shape
    
    
    # Create DataLoader
    batch_size = 7*10
    train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), shuffle=False, batch_size=batch_size)
    
        
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            # x shape: (batch_size, sequence_length, input_size)
        
            # Initialize hidden state and cell state with zeros
            h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
            # Pass through LSTM layer
            # The LSTM module returns a tuple: (output, (h_n, c_n))
            out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
            # We take the output from the last time step, which is contained in the 'out' tensor
            out = self.fc(out[:, -1, :])
        
            return out
    
 
    
    
    # Model, loss, and optimizer initialization
    input_size = X_train.shape[2]  # Number of features: electricity load, temp, humidity, wind_speed
    hidden_size = 200
    output_size = 1
    num_epochs = 100
    learning_rate = 0.001
    
    model = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_step_loss = []
    seed = 100
    if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        
            
    for epoch in range(num_epochs):
        for i, (sequences, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            train_step_loss.append(loss.item())
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    
    # Evaluation on train set
    model.eval()
    train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=False, batch_size=batch_size)
    predictions_train = []
    actuals_train = []
    with torch.no_grad():
        for sequences, labels in train_loader:
            outputs = model(sequences)
            predictions_train.extend(outputs.squeeze().tolist())
            actuals_train.extend(labels.squeeze().tolist())
            
    actuals_train_denormalized = np.array(actuals_train) * std_load_test + mean_load_test        
    predictions_train_denormalized = np.array(predictions_train) * std_load_train + mean_load_train
       
    rmse_train = np.sqrt(np.mean((predictions_train_denormalized - actuals_train_denormalized)**2))
    print(f'Final Root Mean Squared Error on Train Data: {rmse_train:.4f}')


    # Evaluation on test set
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for sequences, labels in test_loader:
            outputs = model(sequences)
            outputs_np = outputs.detach().cpu().numpy().flatten()
            predictions.extend(outputs_np)
            labels_np = labels.detach().cpu().numpy().flatten()
            actuals.extend(labels_np)
    
    # Denormalize predictions and actuals
    predictions_denormalized = np.array(predictions) * std_load_train + mean_load_train
    #predictions_denormalized = np.array(predictions) * std_load_test + mean_load_test
    actuals_denormalized = np.array(actuals) * std_load_test + mean_load_test
    
    
    # Calculate hourly RMSE
    rmse = np.sqrt(np.mean((predictions_denormalized - actuals_denormalized)**2))
    print(f'Final Root Mean Squared Error on Test Data: {rmse:.4f}')
   
    
    y_rnn = np.concatenate((predictions_train_denormalized, predictions_denormalized))
    y = np.concatenate((actuals_train_denormalized, actuals_denormalized))


    intercept = np.ones(len(y_rnn))
    intercept = intercept.reshape(-1, 1)  
    Xkal = np.hstack((intercept,y_rnn.reshape(-1, 1)))

    #### static Kalman 
    ssm = statespace.StateSpaceModel(Xkal, y, kalman_params= None)
    rnn_static = ssm.pred_mean
    rmse_static = root_mean_squared_error(y[-len(predictions_denormalized):], rnn_static[-len(predictions_denormalized):])
    print(f'rnn_static Root Mean Squared Error on Test Data: {rmse_static:.4f}')


    #### dynamic Kalman (grid search)
    l = iterative_grid_search(Xkal, y, q_list =  2.0 ** np.arange(-30, 1), p1 = 1, ncores=10) 
    ssm_dyn = statespace.StateSpaceModel(Xkal, y, kalman_params= l)
    rnn_dyn = ssm_dyn.pred_mean[-len(predictions_denormalized):]
    rmse_dyn = root_mean_squared_error(y[-len(predictions_denormalized):], rnn_dyn[-len(predictions_denormalized):])
    print(f'rnn_dyn Root Mean Squared Error on Test Data: {rmse_dyn:.4f}')


    #### dynamic Kalman (EM)
    l_em = expectation_maximization(Xkal, y, Q_init = np.eye(Xkal.shape[1]), p1 = 1, n_iter = 3*10^3) 
    ssm_dyn_em = statespace.StateSpaceModel(Xkal, y, kalman_params= l_em)
    rnn_dyn_em = ssm_dyn_em.pred_mean[-ltest:]
    rmse_dyn_em = root_mean_squared_error(y[-ltest:], rnn_dyn_em[-ltest:])
    print(f'rnn_dyn Root Mean Squared Error on Test Data: {rmse_dyn_em:.4f}')


    indices_to_update = actual_all_hour.between_time(start_time, end_time).index
    prediction_all_hour_NN.loc[indices_to_update] = y_rnn.flatten()
    prediction_all_hour_NN_static[indices_to_update] = ssm.pred_mean
    prediction_all_hour_NN_dyn[indices_to_update] = ssm_dyn.pred_mean
    prediction_all_hour_NN_em[indices_to_update] = ssm_dyn_em.pred_mean

 


concatenated_pred = pd.concat([actual_all_hour, prediction_all_hour_NN, prediction_all_hour_NN_static, 
                                prediction_all_hour_NN_dyn, prediction_all_hour_NN_em], axis=1)   
concatenated_pred.columns = ['Load', 'NN', 'Static', 'Dyn', 'Em']
concatenated_pred.to_csv('/Results/prediction_LSTM_newdates.txt', index=True, header=True)


#