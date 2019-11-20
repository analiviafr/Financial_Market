#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pylab as plt
import numpy as np


treino = 'petr4-treinamento.csv'
teste = 'petr4-teste.csv'

janela = 7

def windows(dado, tam_janela):
    entrada, saida = [], []
    
    for i in range(len(dado)-tam_janela-1):
        entrada.append(dado[i:i+tam_janela])
        saida.append(dado[i+tam_janela+1])

    return entrada, saida

df = pd.read_csv(treino)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10,5])
plt.plot(df['Adj Close'])
data = df['Adj Close'].values
new_data = []
for i in range(len(data)):
    if data[i] == data[i]:
        new_data.append(data[i])
train_data = np.array(new_data)
data_max = max(train_data)
data_min = min(train_data)
norm_train_data = (train_data - data_min)/(data_max - data_min)

df = pd.read_csv(teste)
data_test = df['Adj Close'].values
new_data = []
for i in range(len(data_test)):
    if data_test[i] == data_test[i]:
        new_data.append(data_test[i])
test_data = np.array(new_data)
norm_data_test = (test_data - data_min)/(data_max - data_min)

entrada, saida = windows(norm_train_data, janela)
entrada_teste, saida_teste = windows(norm_data_test, janela)

print(entrada, '\n', saida)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

batch_size = 16
epochs = 100

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(janela, )))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae'])

entrada = np.array(entrada)
saida = np.array(saida)

entrada_teste = np.array(entrada_teste)
saida_teste = np.array(saida_teste)

history = model.fit(entrada, saida,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)





y_test_pred = model.predict(entrada_teste)
y_train_pred = model.predict(entrada)

#Plote dos resultados de Teste e Treinamento
plt.figure(figsize = (15,5))
orig, = plt.plot(saida_teste, label="Original data")
pred, = plt.plot(y_test_pred, label="Forecasts")
plt.legend(handles=[orig, pred])
plt.title('Test')


plt.figure(figsize = (15,5))
orig, = plt.plot(saida, label="Original data")
pred, = plt.plot(y_train_pred, label="Forecasts")
plt.legend(handles=[orig, pred])
plt.title('Train')


