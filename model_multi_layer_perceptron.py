'''
This Python file loads the data from the pickle files
and carries out a MLP model to categorize falls from ADL's.
'''

import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split


# Loading data to go through neural networks
X = pickle.load(open("data/X_iris_svd.p", "rb"))
Y = pickle.load(open("data/y_iris_svd.p", "rb"))

print('Shape of X:', X.shape, 'Shape of Y:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state =0)

Y_train = to_categorical(Y_train, 4)
Y_test = to_categorical(Y_test, 4)


# Building basic neural network model
modelBF = tf.keras.models.Sequential\
([
    tf.keras.layers.Dense(32, input_shape=(X_train.shape[1],)),  # 1st layer
    tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2, activation='sigmoid') # Last Layer
])
modelBF.summary()     # Displays parameters within model
modelBF.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# Training model

print("Training model w/ data before filtering...")
historyBF = modelBF.fit(X, Y, validation_data=(X_test, Y_test), batch_size=20, epochs=50, shuffle = True, verbose=2)

# Prediction Stage
print("Evaluating models...")
loss1, acc1 = modelBF.evaluate(X1_test, Y1_test, verbose=1)
print("Evaluated Accuracy")
print("------------------")
print("Before Filter: {:4.4f}%" .format(100*acc1))

# Saving Stage
print("Saving history of model without filtering...")
pickle_out = open("model_Iris_mlp.pickle", "wb")
pickle.dump(historyBF.history, pickle_out)
pickle_out.close()