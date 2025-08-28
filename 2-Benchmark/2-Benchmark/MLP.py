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
from viking_kalman import expectation_maximization

import numpy as np
import matplotlib.pyplot as plt
import random

import pandas as pd
import matplotlib.pyplot as plt
import math


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


covariate_columns = ['Load', 'Load_d1', 'Load_d7', 'temperature',  'temperature_smooth_990',  'temperature_smooth_950', 'temperature_min_smooth_990',
                     'temperature_max_smooth_990',
                     'toy', 'day_type_week_1', 'day_type_week_2',
                     'day_type_week_3', 'day_type_week_4', 'day_type_week_5', 'day_type_week_6', 'period_holiday', 'period_summer', 'period_christmas']



# Drop the last row 
Data = Data[covariate_columns]
last_index = Data.index[-1]
Data = Data.drop(last_index) 

actual_all_hour = Data.loc[:,'Load']
prediction_all_hour_NN = actual_all_hour.copy()
prediction_all_hour_NN_static = actual_all_hour.copy()
prediction_all_hour_NN_dyn = actual_all_hour.copy()
prediction_all_hour_NN_em = actual_all_hour.copy()




for j in range(48):
    hour = j // 2
    minute = (j % 2) * 30
    
    start_time = f'{hour:02d}:{minute:02d}'
    end_time = f'{hour:02d}:{minute+29:02d}'
    
    print('###############################################' + str(j))
    print('###############################################' + str(start_time)+ '###########'+str(end_time))

    
    Data_h = Data.between_time(start_time, end_time)
     
    d1 = Data_h[Data_h.index.astype(str) ==  "2022-09-01 "+ start_time +":00+00:00"].index[0]
    d2 = Data_h[Data_h.index.astype(str) ==  "2022-09-02 "+ start_time +":00+00:00"].index[0]

    
    mean_load_train = np.mean(Data_h.loc[:d1, 'Load'])
    std_load_train = np.std(Data_h.loc[:d1,'Load'])
    
    mean_load_test = np.mean(Data_h.loc[d2:, 'Load'])
    std_load_test = np.std(Data_h.loc[d2:,'Load'])
    
    Data_train = Data_h.loc[:d1, ]
    Data_test = Data_h.loc[d2:, ]
    
    # choice of the scaling function
    scaler = StandardScaler()
    
    # We convert data to torch tensors
    X0 = Data_train.drop('Load', axis=1)
    X0 = scaler.fit_transform(X0)
    X0 = X0.astype(np.float32)
    X0_tensor = torch.from_numpy(X0)
    
    Y0 = Data_train[['Load']]
    Y0 = scaler.fit_transform(Y0)
    Y0 = Y0.astype(np.float32)
    Y0_correct = Y0.reshape(-1, 1)
    Y0_tensor = torch.from_numpy(Y0_correct)
    
    
    class PytorchPerceptron(nn.Module):
       def __init__(self, input_shape, hidden_shape, output_shape, activation=torch.nn.Tanh(), p=0):
         #super().__init__()
         super(PytorchPerceptron, self).__init__()
         self.W1 = torch.nn.Linear(input_shape, hidden_shape)
         self.activation = activation
         self.dropout = nn.Dropout(p)
         self.W2 = torch.nn.Linear(hidden_shape, output_shape)
         ###
         self.initialize_weights()
         
       def forward(self, x):
         h = self.activation(self.W1(x))
         h = self.dropout(h)
         out = self.W2(h)
         ###
         return out
     
       def initialize_weights(self):
         torch.nn.init.xavier_uniform_(self.W1.weight, gain=nn.init.calculate_gain('tanh'))
         torch.nn.init.xavier_uniform_(self.W2.weight, gain=nn.init.calculate_gain('tanh'))
    
    
    ###set the attributes to the right value given the considered dataset
    input_shape = X0.shape[1]
    hidden_shape = 100
    output_shape = 1
    ###
    
    nb_epochs = 20000
    learning_rate = 0.01
    batch_size = 7*4
    
    
    model = PytorchPerceptron(input_shape, hidden_shape, output_shape).float()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    
    l = np.arange(0,len(Y0)).tolist()
    train_mask = np.random.choice(l, size=math.floor(len(Y0)*0.8), replace=False)
    test_mask = np.setdiff1d(l, train_mask)
    
    train_step_loss = []
    val_step_loss = []
    
    
    def pytorch_training(model, x_data, y_data, nb_epochs, learning_rate, criterion, optimizer):
      for epoch in range(0, nb_epochs):
        ### TODO: implemente the training strategy using pytorch functions
        # Forward pass
        y_pred = model(x_data)
        # Compute training Loss
        loss = criterion(y_pred[train_mask], y_data[train_mask])
        train_step_loss.append(loss.item())
        # Get validation losses
        validation_loss = criterion(y_pred[test_mask], y_data[test_mask])
        val_step_loss.append(validation_loss.item())
        # Zero gradient
        optimizer.zero_grad()
        # Back-propagation
        loss.backward()
        # One-step gradient
        optimizer.step()
        ###
        if epoch % 1000 == 0:
          print(f"Epoch {epoch} ===== loss {loss}")
      return y_pred, model
    
    y_pred, model = pytorch_training(model, X0_tensor, Y0_tensor, nb_epochs, learning_rate, criterion, optimizer)

    plt.figure(figsize=(10, 6)) # Set the figure size for better readability
    plt.plot(np.arange(0,nb_epochs), train_step_loss, linewidth=1, color='blue')
    plt.plot(np.arange(0,nb_epochs), val_step_loss, linewidth=1, color='red')
    
    plt.figure(figsize=(10, 6)) # Set the figure size for better readability
    plt.plot(np.arange(0,nb_epochs)[1000:], train_step_loss[1000:], linewidth=1, color='blue')
    plt.plot(np.arange(0,nb_epochs)[1000:], val_step_loss[1000:], linewidth=1, color='red')

    ####rmse training
    y_train_denormalized = y_pred.detach().numpy()*std_load_train+mean_load_train
    rmse_train = root_mean_squared_error(Data_train[['Load']], y_train_denormalized)
    print(f'Final Root Mean Squared Error on Train Data: {rmse_train:.4f}')

    
    ####rmse test
    # We convert test data to torch tensors
    X1 = Data_test.drop('Load', axis=1)
    X1 = scaler.fit_transform(X1)
    X1 = X1.astype(np.float32)
    X1_tensor = torch.from_numpy(X1)
    
    y_pred = model(X1_tensor)
    y_pred_denormalized = y_pred.detach().numpy()*std_load_train+mean_load_train
    rmse_test = root_mean_squared_error(Data_test[['Load']], y_pred_denormalized)
    print(f'Final Root Mean Squared Error on Test Data: {rmse_test:.4f}')

    #Oracle 
    y_pred_denormalized_oracle = y_pred.detach().numpy()*std_load_test+mean_load_test
    rmse_oracle = root_mean_squared_error(Data_test[['Load']], y_pred_denormalized_oracle)
    print(f'Final Root Mean Squared Error on Test Data Oracle: {rmse_oracle:.4f}')
   
    ##############Partie Kalman
    
    y_mlp = np.concatenate((y_train_denormalized, y_pred_denormalized))
    y = np.concatenate((Data_train[['Load']], Data_test[['Load']]))+0.1
    #y = y.reshape(-1, 1)  
    intercept = np.ones(len(y_mlp))
    intercept = intercept.reshape(-1, 1)  
    
    Xkal = np.hstack((intercept,y_mlp.reshape(-1, 1)))
    
    ltest = len(Data_test[['Load']].values)
    #### static Kalman 
    ssm = statespace.StateSpaceModel(Xkal, y, kalman_params= None)
    mlp_static = ssm.pred_mean
    rmse_static = root_mean_squared_error(y[-ltest:], mlp_static[-ltest:])
    print(f'mlp_static Root Mean Squared Error on Test Data: {rmse_static:.4f}')
    
    
    
    #### dynamic Kalman (grid search)
    l = iterative_grid_search(Xkal, y, q_list =  2.0 ** np.arange(-60, 1), p1 = 1, ncores=10) 
    ssm_dyn = statespace.StateSpaceModel(Xkal, y, kalman_params= l)
    mlp_dyn = ssm_dyn.pred_mean[-ltest:]
    rmse_dyn = root_mean_squared_error(y[-ltest:], mlp_dyn[-ltest:])
    print(f'mlp_dyn Root Mean Squared Error on Test Data: {rmse_dyn:.4f}')
    
    
    #### dynamic Kalman (EM)
    l_em = expectation_maximization(Xkal, y, Q_init = np.eye(Xkal.shape[1]), p1 = 1, n_iter = 10^3)  #######1.8s
    ssm_dyn_em = statespace.StateSpaceModel(Xkal, y, kalman_params= l_em)
    mlp_dyn_em = ssm_dyn_em.pred_mean[-ltest:]
    rmse_dyn_em = root_mean_squared_error(y[-ltest:], mlp_dyn_em[-ltest:])
    print(f'mlp_dyn_em Root Mean Squared Error on Test Data: {rmse_dyn_em:.4f}')
    
    
    indices_to_update = prediction_all_hour.between_time(start_time, end_time).index
    prediction_all_hour_NN.loc[indices_to_update] = y_mlp.flatten()
    prediction_all_hour_NN_static[indices_to_update] = ssm.pred_mean
    prediction_all_hour_NN_dyn[indices_to_update] = ssm_dyn.pred_mean
    prediction_all_hour_NN_em[indices_to_update] = ssm_dyn_em.pred_mean



concatenated_pred = pd.concat([actual_all_hour, prediction_all_hour_NN, prediction_all_hour_NN_static, 
                                prediction_all_hour_NN_dyn, prediction_all_hour_NN_em], axis=1)   
concatenated_pred.columns = ['Load', 'NN', 'Static', 'Dyn', 'Em']
  
concatenated_pred.to_csv('/Users/Yannig/Documents/These_Nathan/Mobility_data_review/Results/prediction_MLP_newdate_lag.txt', index=True, header=True)

