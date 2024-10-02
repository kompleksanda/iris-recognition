import keras
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical


# Loading data to go through neural networks
X = pickle.load(open("data/X_iris_svd.p", "rb"))
Y = pickle.load(open("data/y_iris_svd.p", "rb"))

print('Shape of X:', X.shape, 'Shape of Y:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state =0)

Y_train = to_categorical(Y_train, 4)
Y_test = to_categorical(Y_test, 4)

print('Shape of X_train:', X_train.shape, 'Shape of Y_train:', Y_train.shape)

# Building recurrent neural network model
model_RNN = tf.keras.models.Sequential\
([
    tf.keras.layers.LSTM(10, input_shape=(X_train.shape[1], X_train.shape[2])),    # Takes number of timesteps & features as input
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
model_RNN.compile(optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'], loss='categorical_crossentropy')

# Training model
print("Training model...")
historyBF_RNN = model_RNN.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=50, epochs=10, shuffle = False)
model_RNN.summary()     # Displays parameters within model


# Training model
print("Training model...")
historyBF_RNN200 = model_RNN.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=50, epochs=10, shuffle = False)
model_RNN.summary()     # Displays parameters within model

# Prediction Stage
print("Evaluating model...")
loss, acc = model_RNN.evaluate(X_test, Y_test, verbose=1)
print("Evaluated accuracy: {:4.4f}%" .format(100*acc))

print("Making predictions...")
predictions = model_RNN.predict_classes(X_test, batch_size=50, verbose=1)
predictions = to_categorical(predictions, 4)
pred_acc = accuracy_score(predictions, Y_test)
print("Predicted accuracy: {:4.4f}%" .format(100*pred_acc))

# Saving Stage
print("Saving history of model without filter...")
pickle_out = open("Iris_SVD_RNN.pickle", "wb")
pickle.dump(historyBF_RNN200.history, pickle_out)
pickle_out.close()
