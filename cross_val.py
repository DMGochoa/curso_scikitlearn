import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score,
    KFold
)

if __name__ == '__main__':
    path = './data/Felicidad/'
    # Importamos el dataset
    dataset = pd.read_csv(path + 'felicidad.csv')
    # Mostramos el reporte estadistico
    print(dataset.head())

    # Definicion de las variables y el target
    X = dataset.drop(['country', 'score', 'rank'], axis=1)
    y = dataset['score']

    model = DecisionTreeRegressor()
    score = cross_val_score(model, X, y, cv=7, scoring='neg_mean_squared_error')
    print(score) # nos arroja el error medio cuadratico para cada una de las pruebas que hizo

    # Para mejor interpretacion vamos a realizar el valor absoluto, la media y la desv std
    print(abs(np.mean(score)), abs(np.std(score)))

    # Ahora usando k-folds
    kf = KFold(n_splits=7, shuffle=True, random_state=42)
    print('Tamano de indices: ')
    i=0
    for train, test in kf.split(dataset):
        i+=1
        print('Particion numero: ' + str(i),len(train), len(test))