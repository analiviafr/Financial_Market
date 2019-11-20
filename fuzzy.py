# -*- coding: utf-8 -*-

fzz = 500
treino = 'OIBR3_treino.csv'
teste = 'OIBR3_teste.csv'

#petr4-teste.csv, petr4-treinamento.csv, OIBR3_treino.csv, OIBR3_teste.csv, MGLU3_teste.csv, MGLU3_treino.csv
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

#carregamento do conjunto de treinamento
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


from pyFTS.partitioners import Grid
fs = Grid.GridPartitioner(data=norm_train_data, npart=fzz)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15,5])
fs.plot(ax)

#Cria os conjuntos fuzzy
fuzzyfied = fs.fuzzyfy(norm_train_data, method='fuzzy', mode='sets')
print(fuzzyfied)

#Ordena os conjuntos fuzzy
from pyFTS.common import FLR
patterns = FLR.generate_non_recurrent_flrs(fuzzyfied)
print([str(k) for k in patterns])

#Treina o modelo com o conjunto fuzzy
from pyFTS.models import chen
model = chen.ConventionalFTS(partitioner=fs)
model.fit(norm_train_data)
print(model)


#carregamento do conjunto de teste
df = pd.read_csv(teste)
data_test = df['Adj Close'].values
new_data = []
for i in range(len(data_test)):
    if data_test[i] == data_test[i]:
        new_data.append(data_test[i])
test_data = np.array(new_data)
norm_data_test = (test_data - data_min)/(data_max - data_min)

#Plota as regras de associação do conjunto
from pyFTS.common import Util
Util.plot_rules(model, size=[15,5] , rules_by_axis=fzz)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15,5])


#Prediz o resultado (Teste)
forecasts_test_norm = model.predict(norm_data_test)
forecasts_norm = model.predict(norm_train_data)
forecasts_test = np.array(forecasts_test_norm) * (data_max - data_min) + data_min
forecasts = np.array(forecasts_norm) * (data_max - data_min) + data_min


from sklearn.metrics import mean_squared_error
#calculo da métrica
print('MSE Treinamento: ', mean_squared_error(norm_train_data, forecasts_norm))
print('MSE Teste: ', mean_squared_error(norm_data_test, forecasts_test_norm))

#Plote dos resultados de Teste e Treinamento
plt.figure(figsize = (15,5))
orig, = plt.plot(test_data, label="Original data")
pred, = plt.plot(forecasts_test, label="Forecasts")
plt.legend(handles=[orig, pred])
plt.title('Test')


plt.figure(figsize = (15,5))
orig, = plt.plot(train_data, label="Original data")
pred, = plt.plot(forecasts, label="Forecasts")
plt.legend(handles=[orig, pred])
plt.title('Train')





