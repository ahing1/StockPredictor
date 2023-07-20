import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load data
company = "NVDA"

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)
 
data = yf.download(company, start = start, end = end, progress = False) 

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1)) # Reshape data to 1 column

prediction_days = 100 # How many days to look back to predict the next day

x_train = [] #predicted price
y_train = [] #actual price

for x in range(prediction_days, len(scaled_data)): # Loop through scaled data
    x_train.append(scaled_data[x - prediction_days:x, 0]) # Append scaled data from x - prediction
    y_train.append(scaled_data[x, 0]) # Append scaled data from x
    
x_train, y_train = np.array(x_train), np.array(y_train) # Convert to numpy arrays
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # Reshape data to 3 dimensions

#Build the model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1))) # Add LSTM layer
model.add(Dropout(0.2)) # Add dropout layer to prevent overfitting
model.add(LSTM(units=50, return_sequences=True)) 
model.add(Dropout(0.2)) 
model.add(LSTM(units=50)) 
model.add(Dropout(0.2)) 
model.add(Dense(units=1)) # Prediction of the next closing price

model.compile(optimizer='adam', loss='mean_squared_error') 
model.fit(x_train, y_train, epochs=25, batch_size=32) # Train the model

# Test the model accuracy on existing data

#Load test data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download(company, start = test_start, end = test_end, progress = False) 
actual_prices = test_data['Close'].values 

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_input = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_input = model_input.reshape(-1, 1)
model_input = scaler.transform(model_input) 

# Make predictions on test data
x_test = []

for x in range(prediction_days, len(model_input)):
    x_test.append(model_input[x - prediction_days:x, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices) # Undo scaling

#Output the test predictions
plt.plot(actual_prices, color='black', label=f'Actual {company} Price')
plt.plot(predicted_prices, color='green', label=f'Predicted {company} Price')
plt.title(f'{company} Share Price')
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

# Predict next day
real_data = [model_input[len(model_input) + 1 - prediction_days:len(model_input + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f'Prediction: {prediction}')