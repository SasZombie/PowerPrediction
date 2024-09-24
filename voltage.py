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
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('voltage1.svg')
    plt.show()

    predicted_values = model.predict(X_test)
    title = "Predictions"

    predicted_values_rescaled = scaler.inverse_transform(predicted_values)
    y_test_rescaled = scaler.inverse_transform(y_test)
    features_list = ['Temperature', 'Humidity', 'Light', 'Voltage']
    future_step = 1900
    cutoff_factor = 500
    plt.figure(figsize=(14, 10))
    
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        
        labels = ["History", "True Future", "Predicted Future", "Prediction Histroy"]
        marker = [".-", "go", "ro", "bo"]
        
        #True History 
        plt.plot(range(len(y_test_rescaled) - cutoff_factor), y_test_rescaled[:-cutoff_factor, i], marker[0], label=labels[0])
        
        #Predicted History
        
        plt.plot(range(len(predicted_values_rescaled) - cutoff_factor), predicted_values_rescaled[:-cutoff_factor, i], marker[1], label=labels[3])

        #True Future Dot
        plt.plot([future_step], [y_test_rescaled[future_step, i]], marker[3], markersize=10, label=labels[1])
        
        #Predicted Fututre Dot
        plt.plot([future_step], [predicted_values_rescaled[future_step, i]], marker[2], markersize=10, label=labels[2])
        
        plt.title(features_list[i])
        plt.xlabel("Time steps")
        plt.ylabel(features_list[i])
        plt.legend()

    plt.suptitle(title)  
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()
    
if __name__ == "__main__":
    main()