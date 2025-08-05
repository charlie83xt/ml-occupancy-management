from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

LOOK_BACK_WINDOW = 30
FORECAST_HORIZON = 5
NEPOCHS = 100

class DeepLearningClass:
    def __init__(self, X_scaled_df, y_scaled_target):
        self.look_back_window = LOOK_BACK_WINDOW
        self.forecast_horizon = FORECAST_HORIZON
        self.X_scaled = X_scaled_df
        self.y_scaled = y_scaled_target
        self.nepochs = NEPOCHS


    def deep_learning_prep(self):
        # Time-based train-test-split (e. g., 80%)
        splitting_length = len(self.X_scaled)
        split_idx = int(splitting_length * 0.8)
        X_train_scaled, X_test_scaled = self.X_scaled[:split_idx], self.X_scaled[split_idx:]
        y_train_scaled, y_test_scaled = self.y_scaled[:split_idx], self.y_scaled[split_idx:]

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def create_sequences(self, X_data, y_data, look_back_window, forecast_horizon):
        X_seq, y_seq = [], []
        for i in range(len(X_data) - look_back_window - forecast_horizon + 1):
            # Input sequence (look_back_window days as features)
            X_seq.append(X_data[i:(i + look_back_window), :])
            # Target sequence (forecast_horizon days of target)
            y_seq.append(y_data[(i + look_back_window):(i + look_back_window + forecast_horizon), 0])
        return np.array(X_seq), np.array(y_seq)
    
    def sequences(self):
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self.deep_learning_prep()
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled, self.look_back_window, self.forecast_horizon)
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled, self.look_back_window, self.forecast_horizon)

        return X_train_seq, y_train_seq, X_test_seq, y_test_seq

    def model_return(self):
        X_train_seq, y_train_seq, X_test_seq, y_test_seq = self.sequences()

        # Output shape for LSTMs should be (samples, timesteps, features)
        n_features = X_train_seq.shape[2]

        model = Sequential([
            LSTM(units=50, activation='relu', input_shape=(self.look_back_window, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, activation='relu'),
            Dropout(0.2),
            Dense(units=self.forecast_horizon) # predict FORECAST_HORIZON values
        ])
        model.compile(optimizer='adam', loss='mse')
        # Callbacks for better training
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
        print("\n--- Training LSTM Model. ---")
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=self.nepochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        return model, X_test_seq, y_test_seq
        
