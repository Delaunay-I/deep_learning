import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./Codes/insurance_data.csv")
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:2], df.iloc[:,-1], test_size=0.2, random_state=25)

X_train_scaled = X_train.copy()
X_train_scaled.iloc[:, 0] = X_train_scaled.iloc[:, 0] / 100

X_test_scaled = X_test.copy()
X_test_scaled.iloc[:, 0] = X_test_scaled.iloc[:, 0] / 100

ft1 = X_train_scaled.iloc[:, 0]
ft2 = X_train_scaled.iloc[:, 1]

def log_loss(y_pred, y_true):
    eps = 1e-15
    y_pred_new = [max(item, eps) for item in y_pred]
    y_pred_new = [min(item, 1-eps) for item in y_pred_new]
    y_pred_new = np.array(y_pred_new)
    loss = -np.mean(y_true*np.log(y_pred_new) + (1 - y_true)* np.log(1-y_pred_new))
    return loss

def sigmoid(X):
    return 1/(1 + np.exp(-X))

# Gradient descent supporting two features
def GD(ft1, ft2, y_true, epochs, loss_threshold):
    # initializing weights
    w1 = w2 = 1
    bias = 0
    # number of samples
    n = len(ft1)

    # defining the learning rate
    lr = 0.5

    for iter in range(epochs):
        weighted_sum = ft1*w1 + ft2*w2 + bias
        y_pred = sigmoid(weighted_sum)
        loss = log_loss(y_pred, y_true)

        dw1 = (1/n) * np.dot(np.transpose(ft1), (y_pred - y_true))
        dw2 = (1/n) * np.dot(np.transpose(ft2), (y_pred - y_true))
        db =  np.mean(y_pred - y_true)

        w1 = w1 - lr*dw1
        w2 = w2 - lr*dw2
        bias = bias - lr*db

        print(f"epoch: {iter+1}, w1:{w1}, w2:{w2}, bias:{bias}, loss: {loss} ")

        if loss <= loss_threshold:
            break

    return w1, w2, bias


(w1, w2, bias) = GD(ft1, ft2, y_train, epochs=1000, loss_threshold=0.4631)


import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=400)

coef, intercept = model.get_weights()
print(f"tf coef: {coef}, intercept: {intercept[0]}\n self-implemented GD: {w1, w2, bias}")

# [[0.6396089 ]
#  [0.64082175]] [-0.42080283]