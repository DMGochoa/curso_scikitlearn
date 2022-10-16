import pandas as pd
import sklearn

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    path = './data/Felicidad/'
    # Importamos el dataset
    dataset = pd.read_csv(path + 'felicidad.csv')
    # Mostramos el reporte estadistico
    print(dataset.head())

    # Definicion de las variables y el target
    X = dataset.drop(['country', 'score', 'rank'], axis=1)
    y = dataset['score']
    # Definir el regresor
    reg = RandomForestRegressor()

    # Se establecen los rangos de los parametros
    parametros = {
        'n_estimators': range(4,16),
        'criterion':['squared_error', 'absolute_error', 'poisson'],
        'max_depth': range(2,11)
    }

    rand_est = RandomizedSearchCV(reg,
                                  parametros, 
                                  n_iter=20, 
                                  cv=3, 
                                  scoring='neg_mean_absolute_error').fit(X,y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print(rand_est.predict(X.loc[[0]]), dataset.loc[[0]]['score'])