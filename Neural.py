from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
from keras.datasets import boston_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

mean = x_train.mean(axis = 0)
std = x_train.std(axis = 0)
x_train = (x_train - mean)/std
x_test = (x_test - mean)/std #정규화 과정
epoch = 20
model = Sequential()
model.add(Dense(13, activation='relu', input_shape=(13,), name="Hidden_Layer1"))
model.add(Dense(13, activation='relu', name="Hidden_Layer2")) #regression임을8 감안한 relu function 사용
model.add(Dense(1, name = "Output_Layer")) # output_layer
model.compile(optimizer=optimizers.Adam(learning_rate=0.01),loss='mse') #model의 생성 method등을 parameter로 가진다.
model.summary() #layer에 따른 prameter의 숫자, output의 모형을 출력해줌

out = model.fit(x_train, y_train, epochs=epoch,validation_data=(x_test, y_test),batch_size=30) #모델 구현 및 validation
y_predict = model.predict(x_test)
print("r2_score_train",r2_score(y_test, y_predict)) #r2_score
