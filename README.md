# Ex.No: 07  AUTO REGRESSIVE MODEL



### AIM:

To Implementat an Auto Regressive Model using Python

### ALGORITHM:


1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.


### PROGRAM

      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt
      from statsmodels.tsa.stattools import adfuller
      from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
      from statsmodels.tsa.ar_model import AutoReg
      from sklearn.metrics import mean_squared_error
      
      # Load the AirPassengers dataset
      data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\sem 3\time series\AirPassengers.csv",
                         parse_dates=['Month'], index_col='Month')
      
      # Explicitly set monthly frequency to avoid warnings
      data = data.asfreq('MS')
      
      # Check stationarity using Augmented Dickey-Fuller test
      result = adfuller(data['#Passengers'])
      print('ADF Statistic:', result[0])
      print('p-value:', result[1])
      
      # Train-test split (80% train, 20% test)
      x = int(0.8 * len(data))
      train_data = data.iloc[:x]
      test_data = data.iloc[x:]
      
      # Fit the AutoRegressive model
      lag_order = 13
      model = AutoReg(train_data['#Passengers'], lags=lag_order)
      model_fit = model.fit()
      
      # Plot ACF
      plt.figure(figsize=(10, 6))
      plot_acf(data['#Passengers'], lags=40, alpha=0.05)
      plt.title('Autocorrelation Function (ACF)')
      plt.show()
      
      # Plot PACF
      plt.figure(figsize=(10, 6))
      plot_pacf(data['#Passengers'], lags=40, alpha=0.05)
      plt.title('Partial Autocorrelation Function (PACF)')
      plt.show()
      
      # Generate predictions on the test set
      start = len(train_data)
      end = len(train_data) + len(test_data) - 1
      predictions = model_fit.predict(start=start, end=end, dynamic=False)
      
      # Compute Mean Squared Error
      mse = mean_squared_error(test_data['#Passengers'], predictions)
      print('Mean Squared Error (MSE):', mse)
      
      # Plot actual vs predicted values
      plt.figure(figsize=(12, 6))
      plt.plot(test_data.index, test_data['#Passengers'], label='Test Data - Number of passengers')
      plt.plot(test_data.index, predictions, label='Predictions - Number of passengers', linestyle='--')
      plt.xlabel('Date')
      plt.ylabel('Number of passengers')
      plt.title('AR Model Predictions vs Test Data')
      plt.legend()
      plt.grid()
      plt.show()




### OUTPUT:

<img width="809" height="598" alt="498127404-40e1921d-4cfc-473e-8762-098fbd9ebb1a" src="https://github.com/user-attachments/assets/b20dee4f-84f2-439a-84b2-b56e002ad9cd" />


<img width="908" height="563" alt="498127414-c79f91ee-bd5e-46b1-9c83-de4b8de90cc0" src="https://github.com/user-attachments/assets/095304e5-1a20-4865-8aa2-d3d465e89f74" />


<img width="1368" height="721" alt="498127431-fd7edda4-6fe4-4e65-9adf-0140569178fe" src="https://github.com/user-attachments/assets/4693d6fb-ca52-4ebb-bc6a-59cd47a2762c" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
