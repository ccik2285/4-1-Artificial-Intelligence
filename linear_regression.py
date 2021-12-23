from keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import pandas as pd

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


# functions for data pre-processing
def norm_x(x):
    x2 = x.copy()
    x2 -= x2.mean(axis=0)
    x2 /= x2.std(axis=0)
    x0 = np.ones((x.shape[0], 1)) #x0 column for bias
    x2 = np.concatenate((x0, x2), axis=1)
    return x2


def norm_y(x):
    x2 = x.copy()
    x2 -= x2.mean()
    x2 /= x2.std()
    return x2


def r2_score(true, pred):
    r2 = skm.r2_score(true, pred)
    return round(r2, 6)


# lasso
def z_compute(X):
    z = np.sum(X * X, axis=0)  # Compute sum for each column
    return z


# class definition
class LinearRegression:
    def __init__(self, x, y):
        self.y_mean = y.mean()
        self.y_std = y.std()

        self.x = norm_x(x)
        self.y = norm_y(y).reshape(y.shape[0], -1)

        self.n = self.x.shape[1]
        self.m = self.x.shape[0]

        self.z = z = np.sum(self.x * self.x, axis=0)

    def denorm(self, h):
        return h * self.y_std + self.y_mean

    def GradDescent(self, h):
        return np.mean((h - self.y) * self.x, axis=0).reshape(-1, 1)

    def plot_cost(self, cost_val):
        cost_val = np.array(cost_val)
        plt.plot(cost_val)
        plt.xlabel("iter #")
        plt.ylabel("cost")
        plt.show()

    def ordinary(self, learning_rate):
        tolerance = 1e-3
        iter = 0
        iter_max = 1000

        cost_val = []  # cost value list for plotting
        w = np.zeros((self.n, 1))
        h = np.dot(self.x, w)

        while (True):
            cost = np.mean(np.square(h - self.y))  # mse
            cost_val.append(cost)
            # update
            w_update = w - learning_rate * LinearRegression.GradDescent(self, h)
            h = np.dot(self.x, w_update)

            if np.sum(abs(w_update - w)) < tolerance:
                break
            elif iter == iter_max:
                break
            else:
                w = w_update
                iter += 1

        h = LinearRegression.denorm(self, h)
        return h, w, cost_val

    def ridge(self, learning_rate, lmbda):
        tolerance = 1e-4
        iter = 0
        iter_max = 1000

        cost_val = []
        w = np.zeros((self.n, 1))
        b = w[0].reshape(1, -1)
        w_ = w[1:]
        h = np.dot(self.x, w)

        while (True):
            cost = np.mean(np.square(h - self.y)) + (lmbda / 2) * np.sum(np.square(w))
            cost_val.append(cost)
            d_cost = LinearRegression.GradDescent(self, h)

            # update
            b = b - d_cost[0]
            w_update = w_ - learning_rate * (d_cost[1:] + (lmbda / self.m) * np.sum(w))
            w = np.concatenate((b, w_update), axis=0)
            h = np.dot(self.x, w)

            if np.sum(abs(w_update - w_)) < tolerance:
                break
            elif iter == iter_max:
                break
            else:
                w_ = w_update
                iter += 1

        h = LinearRegression.denorm(self, h)
        return h, w, cost_val

    # lasso
    def rho_compute(self, y, X, w, j):
        X_k = np.delete(X, j, 1)
        w_k = np.delete(w, j)
        predict_k = np.matmul(X_k, w_k)
        rho_j = np.sum(X[:, j] * (y - predict_k))
        return rho_j
    #lasso
    def coordinate_descent(self, y, X, w, alpha, z):
        cost_val = []
        h = np.dot(X, w)

        tolerance = 1e-4
        max_step = 100.
        iteration = 0
        while (max_step > tolerance):
            cost = np.mean(np.square(h - y)) + alpha * np.sum(np.abs(w))
            cost_val.append(cost)
            
            iteration += 1
            old_weights = np.copy(w)
            for j in range(len(w)):
                rho_j = LinearRegression.rho_compute(self, y, X, w, j)
                if j == 0:
                    w[j] = rho_j / z[j]
                elif rho_j < -alpha * len(y):
                    w[j] = (rho_j + (alpha * len(y))) / z[j]
                elif rho_j > -alpha * len(y) and rho_j < alpha * len(y):
                    w[j] = 0.
                elif rho_j > alpha * len(y):
                    w[j] = (rho_j - (alpha * len(y))) / z[j]
                else:
                    w[j] = np.NaN

            step_sizes = abs(old_weights - w)
            max_step = step_sizes.max()
            h = np.dot(X, w)

        h = LinearRegression.denorm(self, h)
        return h, w, cost_val

    def prediction(self, x_test, parameter):
        x_test = norm_x(x_test)
        y_pred = np.dot(x_test, parameter)
        y_pred = LinearRegression.denorm(self, y_pred)
        return y_pred


# instance 선언
model = LinearRegression(x_train, y_train)

# ordinary_r2_score
print("Ordinary")
h_ols, w_ols, cost_val_ols = model.ordinary(1e-3)
y_pred_ols = model.prediction(x_test, w_ols)
print("train r2:", r2_score(y_train, h_ols), "\ttest r2:", r2_score(y_test, y_pred_ols), "\n")
model.plot_cost(cost_val_ols)

#Ridge_r2_score
print("Ridge")
h_ridge, w_ridge, cost_val_ridge = model.ridge(1e-3, 1e-1)
y_pred_ridge = model.prediction(x_test, w_ridge)
print("train r2:", r2_score(y_train, h_ridge), "\tr2 test:", r2_score(y_test, y_pred_ridge), "\n")
model.plot_cost(cost_val_ridge)

#Lasso_r2_score
print("Lasso")
X = norm_x(x_train)
y = norm_y(y_train)
w = np.zeros(X.shape[1], dtype = float)
z = z_compute(X)

h_lasso, w_lasso, cost_val_lasso = model.coordinate_descent(y,X,w,1e-1,z)
y_pred_lasso = model.prediction(x_test, w_lasso)
print("train r2:", r2_score(y_train, h_lasso), "\tr2 test:", r2_score(y_test, y_pred_lasso), "\n")
model.plot_cost(cost_val_lasso)


# 교차검증
# 전체 데이터 불러오기
(whole_x, whole_y), (void1,void2) = boston_housing.load_data(test_split=0)

# 트레인 데이터와 테스트 데이터 쌍 분리, 5개 

test_split = 0.2 # 이번 경우에 테스트 데이터의 비율을 0.2로 함 
k = (int)(1/test_split) # 테스트 데이터의 비율에 따라 k가 결정됨, 이번 경우 5
kfolds_size = np.floor(test_split * np.shape(whole_x)[0]).astype(int) # 101 (=506/5)


# test data의 인덱스를 계산하는 함수
def index_start(i):
  if (i == k):
    idx = np.shape(whole_x)[0] - (kfolds_size-1)
  else:
    idx = i * kfolds_size
  return idx

def index_end(i):
  if (i == k):
    idx = np.shape(whole_x)[0]
  else:
    idx = (i+1)*kfolds_size - 1
  return idx
    
# kfold 실행
def kfold(function, k):

    cv_accuracy = []
    if (function=='lr'):
        print("------linear regression-----")
    elif (function=='ridge'):
        print("----- ridge regression -----")
    elif (function=='lasso'):
        print("----- lasso regression -----")
    else:
        print("No such function")

    print("train size ", whole_x.shape[0]-kfolds_size, "\ttest size: ", kfolds_size, "\n")

    for i in range(k):
        # 데이터 분할
        indices = range(index_start(i), index_end(i)+1)

        fold_x_test = whole_x[index_start(i):index_end(i)+1].copy()
        fold_x_train = np.delete(whole_x, indices, axis=0)
        fold_y_test = whole_y[index_start(i):index_end(i)+1].copy()
        fold_y_train = np.delete(whole_y, indices, axis=0) 

        model_k = LinearRegression(fold_x_train, fold_y_train)

        # 함수별로 갈라지는 부분

        if (function=='lr'):
            h, w, cost_val = model_k.ordinary(1e-2)

        elif (function == 'ridge'):
            h, w, cost_val = model_k.ridge(1e-2, 1e-1)

        elif (function == 'lasso'):
            X = norm_x(fold_x_train)
            y = norm_y(fold_y_train)
            w = np.zeros(X.shape[1], dtype = float)
            z = z_compute(X)
            h, w, cost_val = model_k.coordinate_descent(y,X,w,1e-1,z)

        y_pred_k = model_k.prediction(fold_x_test, w)

        accuracy2 = r2_score(fold_y_train, h)
        accuracy = r2_score(fold_y_test, y_pred_k)
        print(i, "\ttrain r2 score:", accuracy2, "\ttest r2 score:", accuracy)

        cv_accuracy.append(accuracy)

    print('평균: ', round(np.mean(cv_accuracy), 6), "\n")

print("\nk fold cross validation")

kfold("lr", k)
kfold("ridge", k)
kfold("lasso", k)
