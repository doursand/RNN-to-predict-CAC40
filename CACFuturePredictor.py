# -*- coding: utf-8 -*-

"""
Created on Fri Nov 16 21:29:00 2018

@author: andre_000
"""

# Partie 1 preparation des donnees

# Librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# jeu d'entrainement avec tout : ohlc
dataset_train = pd.read_csv("FutureCAC_Price_Train.csv")
training_set = dataset_train.iloc[:,1:5].values

# Features scaling entre 0 et 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
training_set_scaled = sc.fit_transform(training_set)

# creation de la structure avec 60 timesteps et 1 sortie
y_train=[]
for i in range(60,len(training_set)):
    y_train.append(training_set_scaled[i,0])
y_train = np.array(y_train)

X_train = []
for variable in range(0, 4):
    X = []
    for i in range(60, len(training_set)):
        X.append(training_set_scaled[i-60:i, variable])
    X, np.array(X)
    X_train.append(X)
X_train, np.array(X_train)

# conversion des listes en numpy array car demandé par Keras
X_train = np.array(X_train)    
y_train = np.array(y_train)

# Reshaping - input dim = nombre de variables d'entrée, ici 1 pour le moment (Last). Si on rajoute une autre dimension, alors augmenter
X_train = np.swapaxes(np.swapaxes(X_train, 0, 1), 1, 2)
#X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))

# Partie 2 construction du RNN

# Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# couche LSTM + couche Dropout (pour eviter sur entrainement) X_train.shape[1]=60 , 4 = nombres variables d'entree ohlc
regressor.add(LSTM(units=50, return_sequences=True,
                   input_shape=(X_train.shape[1], 4)))
regressor.add(Dropout(0.2))

# 2e couche LSTM + couche Dropout 
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 3e couche LSTM + couche Dropout 
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 4e couche LSTM + couche Dropout 
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))

# couche de sortie
regressor.add(Dense(units=1))

# compilation : adam ou RMSProp pour un RNN. Fonction de cout mean_squared_error marche pour un cas de regression comme celui ci, ou on essaye de predire une variable continue, ie le cours de l'action
regressor.compile(optimizer="adam",loss="mean_squared_error")

# Entrainement
regressor.fit(X_train,y_train,epochs=50, batch_size=32)

# Partie 3 prediction et visualisation

# Donnees de 2018

# reload ou save du regressor
"""from keras.models import load_model
from keras.models import save_model

regressor.load_weights("CACFutureTraining.h5")
"""

dataset_test = pd.read_csv("FutureCAC_Price_Test.csv")

# predictions pour 2018

dataset_total = pd.concat((dataset_train.iloc[:,1:5],dataset_test.iloc[:,1:5]),
                          axis=0)
real_stock_price = dataset_test.iloc[:,1].values

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = sc.transform(inputs)

X_test = []
for variable in range(0, 4):
    X = []
    for i in range(60, 80):
        X.append(training_set_scaled[i-60:i, variable])
    X, np.array(X)
    X_test.append(X)
X_test, np.array(X_test)

X_test=np.array(X_test)
X_test = np.swapaxes(np.swapaxes(X_test, 0, 1), 1, 2)

# Reshaping - input dim = nombre de variables d'entrée, ici 1 pour le moment (Last). Si on rajoute une autre dimension, alors augmenter
#X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))

# prediction + retour vers les vrais valeurs et non plus (0,1)
predicted_stock_price = regressor.predict(X_test)

# creation d'une matrice intermédiaire avec 4 colonnes de zéros
trainPredict_dataset_like = np.zeros(shape=(len(predicted_stock_price), 4) )
# rajout en première colonne du vecteur de résultats prédits
trainPredict_dataset_like[:,0] = predicted_stock_price[:,0]
# inversion et retour vers les valeurs non scalées
predicted_stock_price = sc.inverse_transform(trainPredict_dataset_like)[:,0]
# fin test

# visualisation
plt.plot(real_stock_price,color="red", label = "Valeur de clôture réelle du CAC Future")
plt.plot(predicted_stock_price,color="green", label = "Valeur de clôture prédite du CAC Future")
plt.title("prediction du CAC Future")
plt.xlabel("jour")
plt.ylabel("valeur de clôture")
plt.legend()
plt.show()

"""Il est possible d'enregistrer un modèle dans un fichier à l'aide de la méthode save .

Par exemple : classifier.save("filename.h5")

Ensuite on peut le charger avec load_model .

Par exemple : classifier = load_model("filename.h5")"""