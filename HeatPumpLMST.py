import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

class TimeSeriesNet(nn.Module):
    def __init__(self, input_dim, num_layers, output_dim, hidden_state):
        super(TimeSeriesNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_state, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_state, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out
    
    def train(self, model, train_data, criterion, optimizer, epochs, batch_size):
        for epoch in range(epochs):
            for i in range(0, train_data.shape[0], batch_size):
                # Get the batch of data
                batch_data = train_data[i:i+batch_size, :-2]
                batch_labels = train_data[i:i+batch_size, -2:]

                # Forward pass
                outputs = model(batch_data)

                # Compute the loss
                loss = criterion(outputs, batch_labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print the loss after every epoch
            if(epoch % 100 == 0):
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def test(self, model, criterion, test_data):
        # Test the model
        with torch.no_grad():
            # Get the test data
            test_inputs = test_data[:, :-2]
            test_labels = test_data[:, -2:]
            # Make predictions
            test_outputs = model(test_inputs)

            # Compute the test loss
            test_loss = criterion(test_outputs, test_labels)
            self.showPlot(test_labels, test_outputs)

        print(f'Test Loss: {test_loss.item():.4f}')

    def predict(self, model, criterion, data_in):
        # Test the model
        with torch.no_grad():
            # Get the test data
            test_inputs = data_in
            # Make predictions
            test_outputs = model(test_inputs)

            # Compute the test loss
            plt.plot(np.cumsum(test_outputs[:,1]), label='Predicted Data') #predicted plot
            plt.title('Time-Series Prediction for the next 15 days using simulated data')
            plt.xlabel("Days")
            plt.ylabel("cumulative heat pump kW")
            plt.legend()
            plt.show() 

    def showPlot(self, test_labels, test_outputs):
        #change 0 to 1 or the other way around to see backburner or heatpump
        plt.plot(test_labels[:,1], label='Predicted Data') #actual plot
        plt.plot(test_outputs[:,1], label='Predicted Data') #predicted plot
        plt.title('Time-Series Prediction')
        plt.xlabel("Days")
        plt.ylabel("heat pump kW")
        plt.legend()
        plt.show() 

    # Define the performance metrics
    def evaluate(self, model, criterion, data):
        inputs = data[:, :-2]
        labels = data[:, -2:]

        # Get the predictions
        with torch.no_grad():
            outputs = model(inputs)
    
        # Calculate the metrics
        mse = criterion(outputs, labels).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(outputs - labels)).item()
        r2 = r2_score(labels.numpy(), outputs.numpy())
        mape = torch.mean(torch.abs((outputs - labels) / labels)).item()

        return mse, rmse, mae, r2, mape

    
