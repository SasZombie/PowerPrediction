import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow import keras

# Function to visualize training and validation loss
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

# Function to plot predictions
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

# Function to create sequences for training and validation
def create_sequences(data, past, future, step, target_column):
    x = []
    y = []
    for i in range(past, len(data) - future, step):
        x.append(data.iloc[i - past:i].values)  
        y.append(data.iloc[i + future][target_column])  # Use the target column passed as a parameter

    x = np.array(x)
    y = np.array(y)
    return x, y

# Main function
def main()->None:
    # Load and preprocess the data
    df = pd.read_csv("household_data_60min_singleindex.csv").iloc[12025:12425]

    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df['cet_cest_timestamp'] = pd.to_datetime(df['cet_cest_timestamp'])

    # Drop the column you want to ignore
    df.drop(columns=['DE_KN_industrial3_area_offices'], inplace=True)

    # Select only numeric columns for scaling
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns

    # Check and handle constant columns (zero variance)
    constant_columns = [col for col in numeric_features if df[col].nunique() == 1]
    if constant_columns:
        print(f"Removing constant columns: {constant_columns}")
        df.drop(columns=constant_columns, inplace=True)

    # Replace infinite values and handle NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

    # Apply StandardScaler only to the remaining numeric columns
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # One-hot encode categorical features (interpolated column)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_interpolated = encoder.fit_transform(df[['interpolated']])
    encoded_interpolated_df = pd.DataFrame(encoded_interpolated, columns=encoder.get_feature_names_out(['interpolated']))

    # Combine scaled numeric features and encoded features
    final_df = pd.concat([df[numeric_features].reset_index(drop=True), encoded_interpolated_df.reset_index(drop=True)], axis=1)

    # Split the data into training and validation sets
    split_fraction = 0.715
    train_split = int(split_fraction * int(final_df.shape[0]))

    train_data = final_df.iloc[:train_split]
    val_data = final_df.iloc[train_split:]

    # Print the size of train_data and val_data
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")

  
    # Set past, future, and step values
    past = 50
    future = 10
    step = 3

    # Choose the correct target column for prediction
    target_column = 'DE_KN_residential1_grid_import'  # Replace with the correct target column

    # Create sequences
    if train_data.shape[0] <= past + future:
        print("Warning: The dataset is too small for the selected past and future values.")
        return

    x_train, y_train = create_sequences(train_data, past, future, step, target_column)
    x_val, y_val = create_sequences(val_data, past, future, step, target_column)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print(np.isnan(x_train).sum())  # Should be 0 if there's no NaN
    print(np.isnan(y_train).sum())  # Should be 0 if there's no NaN

# Similarly for validation data
    print(np.isnan(x_val).sum())
    print(np.isnan(y_val).sum())
    

    # Define the model
    inputs = keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1, activation='linear')(lstm_out)  # Change activation from 'gelu' to 'linear'

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    # Define callbacks for early stopping and model checkpoint
    path_checkpoint = "model_checkpoint.weights.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        save_weights_only=True,
        save_best_only=True,
    )

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=256,
        validation_data=(x_val, y_val),
        callbacks=[es_callback, modelckpt_callback],
        verbose=1
    )

    # Visualize the training and validation loss
    visualize_loss(history, "Training and Validation Loss")

    # Predict and plot results
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
