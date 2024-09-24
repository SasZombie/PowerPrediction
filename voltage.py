import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


def main()->None:

    file_path = 'data.txt'

    columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']

    data = pd.read_csv(file_path, sep=' ', names=columns, nrows=10000)

    print(data.head())

    features = data[['temperature', 'humidity', 'light', 'voltage']]

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    X = np.array(scaled_features)

    window_size = 50
    X_lstm = []
    y_lstm = []

    for i in range(window_size, len(X)):
        X_lstm.append(X[i-window_size:i])
        y_lstm.append(X[i])  

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    print(X_lstm.shape) 

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
    model.add(LSTM(units=50))

    model.add(Dense(units=4))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.summary()


    split = int(0.8 * len(X_lstm))
    X_train, X_test = X_lstm[:split], X_lstm[split:]
    y_train, y_test = y_lstm[:split], y_lstm[split:]


    history = model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    predicted_values = model.predict(X_test)

    predicted_values_rescaled = scaler.inverse_transform(predicted_values)
    y_test_rescaled = scaler.inverse_transform(y_test)

    features_list = ['Temperature', 'Humidity', 'Light', 'Voltage']

    plt.figure(figsize=(14, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)  
        plt.plot(range(len(y_test_rescaled)), y_test_rescaled[:, i], 'ro', label='Actual')  
        plt.plot(range(len(predicted_values_rescaled)), predicted_values_rescaled[:, i], 'go', label='Predicted')  
        plt.title(features_list[i])
        plt.xlabel('Time steps')
        plt.ylabel(features_list[i])
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()