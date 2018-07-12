import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb

def nn_train(x_train, x_test):
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)


    def vectorize(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1
        return results


    data = vectorize(data)
    targets = np.array(targets).astype("float32")
    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]
    model = models.Sequential()

    # Input - Layer
    model.add(layers.Dense(50, activation="relu", input_shape=(10000,)))

    # Hidden - Layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))

    # Output- Layer
    model.add(layers.Dense(1, activation="relu"))
    model.summary()

    # compiling the model
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_squared_logarithmic_error"]
    )

    results = model.fit(
        train_x, train_y,
        epochs=2,
        batch_size=500,
        validation_data=(test_x, test_y)
    )
    print("Test-Accuracy:", np.mean(results.history["mean_squared_logarithmic_error"]))

def nn_predict(x_test):
    pass

if False:
    # evaluate model with standardized dataset
    numpy.random.seed(seed)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
