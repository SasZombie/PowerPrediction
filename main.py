import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import keras
from PIL import Image

def visualize_loss(history, title):
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('plot.svg')
        plt.show()
        
        
def show_plot(plot_data, future, title):
    plt.figure(figsize=(10, 6))
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i == 1: 
            plt.plot([future], [plot_data[i]], marker[i], markersize=10, label=labels[i])
        elif i == 2:
            plt.plot([future], [plot_data[i]], marker[i], markersize=10, label=labels[i])
        else: 
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])

    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Days")
    plt.ylabel("Energy Consumption")
    plt.savefig('plot2.svg')
    plt.show()
    return


def create_sequences(data, past, future, step):
    X, y = [], []
    for i in range(past, len(data) - future):
        indices = range(i - past, i, step)
        X.append(data.iloc[indices].values)
        y.append(data.iloc[i + future].values)
    return np.array(X), np.array(y)

def main()->None:

    df = pd.read_csv("Data/db_power_consumption.csv")

    categorical_features = ['Area', 'Municipality', 'Use', 'Stratum']
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_features])

    scaler = StandardScaler()
    df['Consumption'] = scaler.fit_transform(df[['Consumption']])

    encoded_features_df = pd.DataFrame(encoded_features)
    encoded_features_df['Consumption'] = df['Consumption'].values

    split_fraction = 0.715
    train_split = int(split_fraction * int(encoded_features_df.shape[0]))

    train_data = encoded_features_df.iloc[:train_split]
    val_data = encoded_features_df.iloc[train_split:]
    
    past = 720
    future = 72
    step = 6

    x_train, y_train = create_sequences(train_data, past, future, step)
    x_val, y_val = create_sequences(val_data, past, future, step)

    inputs = keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1, activation='gelu')(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    path_checkpoint = "model_checkpoint.weights.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        save_weights_only=True,
        save_best_only=True,
    )
    
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=256,
        validation_data=(x_val, y_val),
        callbacks=[es_callback, modelckpt_callback],
    )

    visualize_loss(history, "Training and Validation Loss")

    num_sequences = 5
    for i in range(num_sequences):
        x, y = x_val[i], y_val[i]
        prediction = model.predict(np.expand_dims(x, axis=0))[0]
        show_plot(
            [x[:, -1], y, prediction],
            future,
            f"Single Step Prediction - Sequence {i+1}"
        )

if __name__ == "__main__":
    main()