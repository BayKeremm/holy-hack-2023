from HeatPumpLMST import TimeSeriesNet
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime as dt
from torch.autograd import Variable 
import numpy as np


def readData(filename):
    data = pd.read_csv(filename, low_memory=False)
    data1 = data.copy()
    data1.loc[:, 'date'] = [dt.fromtimestamp(float(x)/1000) for x  in data1['timestamp']]
    input_cols = ['date', # date 
              'rt_temp', # room temperature
              'ambient_temp', # ambient temperature
              'rt_sp_heating', # setpoint
              'status_3wv'] # status_3wv==1 then domestic hot water is active 
                            # status_3wv==0 then space heating is active 


    output_cols = [col for col in data1.columns if 'input_kw' in col]  


    #process data
    data_ml = data1[input_cols+output_cols]
    data_ml = data_ml.set_index('date').astype(float)
    data_ml_resampled = data_ml.resample('24H').mean()
    df_filtered = data_ml_resampled.where(data_ml_resampled["hp_elec_input_kw"] < 10)
    df_filtered2 = data_ml_resampled.where(df_filtered["buh_elec_input_kw"] < 10)
    df_filtered3 = df_filtered2.dropna()

    return df_filtered3

def convertTensor(data):
    # Convert the dataframe to a PyTorch tensor
    data = Variable(torch.Tensor(data.to_numpy()))

    # Split the data into training and testing sets
    #in total 346 days of data
    train_data = data[:300, :]
    test_data = data[300:, :]
    return train_data, test_data

#def simulateValues(data_in):
    #return data_in

#def readWheatherData(filename):
    #data = pd.read_csv(filename, low_memory=False)
    #data1 = data.copy()
    #data1.loc[:, 'date'] = [dt.fromtimestamp(float(x)/1000) for x  in data1['timestamp']]
    #input_cols = ['date', # date 
              #'rt_temp', # room temperature
              #'ambient_temp', # ambient temperature
              #'rt_sp_heating', # setpoint
              #'status_3wv'] # status_3wv==1 then domestic hot water is active 
                            ## status_3wv==0 then space heating is active 
    #data_ml = data1[input_cols]
    #data_ml = data_ml.set_index('date').astype(float)
    #data_ml_resampled = data_ml.resample('24H').mean()
    #return data_ml_resampled
def simulateData(setpoint):
    # room temp, ambient temp, setpoint, on off 
    data = [ [13.4, 4.2, setpoint, 1 ],
  [14.1, 2.6, setpoint, 0 ],
  [12.7, 1.8, setpoint, 1 ],
  [10.5, 8.2, setpoint, 1 ],
  [10.9, 9.9, setpoint, 0 ],
  [17.3, 7.1, setpoint, 1 ],
  [16.5, 3.3, setpoint, 0 ],
  [15.1, 5.5, setpoint, 0 ],
  [18.0, 9.6, setpoint, 1 ],
  [ 11.2, 0.5, setpoint, 1 ],
  [ 12.8, 6.4, setpoint, 0 ],
  [ 13.9, 8.9, setpoint, 1 ],
  [ 10.6, 1.2, setpoint, 1 ],
  [ 11.5, 7.8, setpoint, 0 ],
  [ 16.2, 5.9, setpoint, 1 ]
]
    return torch.Tensor(data)

    pass
if __name__=='__main__':
    loadmodel = True
    #data_from_csv = readData("/Users/keremokyay/masters/holy_hack/challenge_and_workshop_hackaton_daikin/data/fs1_altherma_1y.csv")
    data_from_csv = readData("./fs1_altherma_1y.csv")
    train, test = convertTensor(data_from_csv)
    data_in = simulateData(20) # set point as input
    #print(data_in.shape)

    if loadmodel:
        model = TimeSeriesNet(input_dim=4, num_layers=1, output_dim=2,hidden_state=6)
        #model.load_state_dict(torch.load("/Users/keremokyay/masters/holy_hack/challenge_and_workshop_hackaton_daikin/codes/model50r.pt"))
        model.load_state_dict(torch.load("./model.pt"))
    else:
        model = TimeSeriesNet(input_dim=4, num_layers=1, output_dim=2, hidden_state=6)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    if not loadmodel:
        model.train(model=model, train_data=train, criterion=criterion, optimizer=optimizer, epochs=200, batch_size=32)

    model.test(model=model, criterion=criterion, test_data=test)
    test_mse, test_rmse, test_mae, test_r2, test_mape = model.evaluate(model=model, criterion=criterion, data=test)
    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print(f'Test MAE: {test_mae:.4f}')
    print(f'Test R-squared: {test_r2:.4f}')
    print(f'Test MAPE: {test_mape:.4f}')

    model.predict(model=model, data_in=data_in, criterion=criterion)


