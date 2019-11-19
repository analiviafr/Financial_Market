from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
#importação para gráficos:
import matplotlib.pyplot as plt

base = pd.read_csv('petr4-treinamento.csv')

#excluindo valores nulos:
base = base.dropna()

base_treinamento = base.iloc[:, 1:2].values

#normalização dos dados: 
#deixando dados entre 0 e 1
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

#previsores -> ultimos 90 valores
previsores = []
preco_real = []

#onde começa até onde termina
for i in range (90, 1242):
        previsores.append(base_treinamento_normalizada[i-90: i,0])
        preco_real.append(base_treinamento_normalizada[i, 0])


#passando para o formato numpy
previsores, preco_real=np.array(previsores), np.array(preco_real)

previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))
                                    #tamanho total           #90
        
#CRIANDO REDE NEURAL                            
regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape = (previsores.shape[1], 1)))
#numero de neuron
#indica se terá mais de uma camada e se os dados devem ser passados
#1 -> somente um atributo previsor
regressor.add(Dropout(0,5))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0,5))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0,5))

regressor.add(Dense(units=1, activation='sigmoid'))

regressor.compile(optimizer='rmsprop', loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

history = regressor.fit(previsores, preco_real, epochs=100, batch_size=32)

base_teste = pd.read_csv('petr4-teste.csv')
preco_real_teste = base_teste.iloc[:,1:2]. values
#pegou só a primeira coluna

base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)
#concatenando base original com teste, para pegar as 90 anteriores

entradas = base_completa[len(base_completa)-len(base_teste)-90:].values

entradas = entradas.reshape(-1,1)
entradas = normalizador.transform(entradas)

X_teste = []
for i in range (90, 112):
    X_teste.append(entradas[i-90:i,0])
    
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = regressor.predict(X_teste)
#desnomarlizar para melhorar visualização
previsoes = normalizador.inverse_transform(previsoes)

#media
previsoes.mean()
preco_real_teste.mean()

#GRÁFICO PARA COMPARAR RESULTADOS DO TESTE
plt.figure(figsize = (15,5))
plt.plot(preco_real_teste, color='red', label='Preco Real')
plt.plot(previsoes, color='blue', label='Previsoes')
plt.title('Previsão')
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend()
plt.show()

######################################################################

from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score, explained_variance_score, max_error


evs = explained_variance_score(preco_real_teste, previsoes)
print(evs) #1

r2 = r2_score(preco_real_teste, previsoes)
print(r2)#1

mse = mean_squared_error(preco_real_teste, previsoes)
print(mse) #0

mae = mean_absolute_error(preco_real_teste, previsoes)
print(mae) #0

#####################################################################
preco_real_treino=[]
for i in range (90, 1242):
        preco_real_treino.append(base_treinamento[i, 0])

previsoes_treino = regressor.predict(previsores)
previsoes_treino = normalizador.inverse_transform(previsoes_treino)

#GRÁFICO PARA COMPARAR RESULTADOS DO TREINO
plt.figure(figsize = (15,5))
plt.plot(preco_real_treino, color='red', label='Preco Real')
plt.plot(previsoes_treino, color='blue', label='Previsoes')
plt.title('Previsão')
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend()
plt.show()