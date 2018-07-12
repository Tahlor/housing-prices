from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy


# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()

    # Input
    model.add(Dense(50, input_dim=8, activation='relu'))

    # Hidden
    model.add(Dropout(0.5, noise_shape=None, seed=None))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5, noise_shape=None, seed=None))
    model.add(Dense(32, activation='relu'))

    # Output
    model.add(Dense(1, activation='relu'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_logarithmic_error'])
    return model


model = KerasRegressor(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
