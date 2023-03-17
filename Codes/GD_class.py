import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./Codes/insurance_data.csv")
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:2], df.iloc[:,-1], test_size=0.2, random_state=25)

X_train_scaled = X_train.copy()
X_train_scaled.iloc[:, 0] = X_train_scaled.iloc[:, 0] / 100

X_test_scaled = X_test.copy()
X_test_scaled.iloc[:, 0] = X_test_scaled.iloc[:, 0] / 100



class myNN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0

    def fit(self, X, y, epochs, loss_threshold):
        self.GD(X, y, epochs, loss_threshold)
        print(f"Final weights and bias: w1: {self.w1}, w2: {self.w2}, bias: {self.bias}")

    def predict(self, X):
        weighted_sum = self.w1*X.iloc[:,0] + self.w2*X.iloc[:,1] + self.bias
        return self.sigmoid(weighted_sum)

    

    def GD(self, X, y_true, epochs, loss_threshold):
        # number of samples
        n = len(X.iloc[:,0])

        # defining the learning rate
        lr = 0.5

        for iter in range(epochs):
            y_pred = self.predict(X)
            loss = self.log_loss(y_pred, y_true)

            dw1 = (1/n) * np.dot(np.transpose(X.iloc[:,0]), (y_pred - y_true))
            dw2 = (1/n) * np.dot(np.transpose(X.iloc[:,1]), (y_pred - y_true))
            db =  np.mean(y_pred - y_true)

            self.w1 = self.w1 - lr*dw1
            self.w2 = self.w2 - lr*dw2
            self.bias = self.bias - lr*db

            if iter%50==0:
                print(f"epoch: {iter+1}, w1:{self.w1}, w2:{self.w2}, bias:{self.bias}, loss: {loss} ")

            if loss <= loss_threshold:
                print(f"epoch: {iter+1}, w1:{self.w1}, w2:{self.w2}, bias:{self.bias}, loss: {loss} ")
                break

        
    
    def log_loss(self, y_pred, y_true):
        eps = 1e-15
        y_pred_new = [max(item, eps) for item in y_pred]
        y_pred_new = [min(item, 1-eps) for item in y_pred_new]
        y_pred_new = np.array(y_pred_new)
        loss = -np.mean(y_true*np.log(y_pred_new) + (1 - y_true)* np.log(1-y_pred_new))
        return loss

    def sigmoid(self, X):
        return 1/(1 + np.exp(-X))
    

customModel = myNN()
customModel.fit(X_train_scaled, y_train, epochs=500, loss_threshold=0.4631)